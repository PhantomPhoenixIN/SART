#!/usr/bin/env python3
"""Profile fixed-length in-domain validation inference for compute-matched experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from src.evaluation.checkpoints import load_trained_model
from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.training.data import ManifestDataset, TranslationCollator


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profiler_flops(function) -> tuple[Any, int]:
    """Run a callable under PyTorch's operator FLOP estimator."""
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, with_flops=True) as profiler:
        output = function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    total_flops = sum(int(event.flops or 0) for event in profiler.key_averages())
    return output, total_flops


def _adapter_events(
    model: torch.nn.Module, enabled: bool
) -> tuple[list[Any], list[Any], list[Any]]:
    """Install CUDA event hooks to measure the source-side adapter contribution."""
    if not enabled or not isinstance(model, MidLayerSourceAdapter):
        return [], [], []
    starts: list[Any] = []
    ends: list[Any] = []

    def before(_module, _inputs):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        starts.append(start)

    def after(_module, _inputs, _output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        ends.append(end)

    first = model.adapter.register_forward_pre_hook(before)
    second = model.adapter.register_forward_hook(after)
    return starts, ends, [first, second]


def _call_prefill(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    region_batch: Any,
    budget: float,
    hard_threshold: float | None,
    top_k_fraction: float | None,
) -> Any:
    if isinstance(model, MidLayerSourceAdapter):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            region_batch=region_batch,
            budget=budget,
            hard_threshold=hard_threshold,
            top_k_fraction=top_k_fraction,
            use_cache=True,
            return_dict=True,
        )
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )


def _call_decode(
    model: torch.nn.Module,
    token: torch.Tensor,
    attention_mask: torch.Tensor,
    position: int,
    past_key_values: Any,
) -> Any:
    position_ids = torch.tensor([[position]], device=token.device, dtype=torch.long)
    if isinstance(model, MidLayerSourceAdapter):
        model.clear_route()
        return model(
            input_ids=token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return model(
        input_ids=token,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


def _attention_flops_upper(
    model: torch.nn.Module, source_length: int, decode_tokens: int
) -> tuple[int, int]:
    """Estimate dense QK/AV attention work for each query/key position pair."""
    base = model.causal_lm if isinstance(model, MidLayerSourceAdapter) else model
    config = base.config
    layers = int(getattr(config, "num_hidden_layers", getattr(config, "n_layer", 0)))
    hidden = int(getattr(config, "hidden_size", getattr(config, "n_embd", 0)))
    if not layers or not hidden:
        return 0, 0
    # Four multiply-add FLOPs per query/key channel for QK and AV. This dense estimate
    # intentionally bounds fused/causal attention implementations from above.
    prefill_flops = 4 * hidden * layers * source_length * source_length
    decode_flops = (
        4 * hidden * layers * sum(source_length + index for index in range(decode_tokens))
    )
    return prefill_flops, decode_flops


def profile_example(
    model: torch.nn.Module,
    item: dict[str, Any],
    *,
    device: torch.device,
    decode_tokens: int,
    budget: float,
    hard_threshold: float | None,
    top_k_fraction: float | None,
) -> dict[str, Any]:
    """Profile prefill and teacher-forced cached decode at a shared fixed length."""
    if decode_tokens < 1 or item["target_ids"].numel() == 0:
        raise ValueError("Validation target is empty or requested decode-token cap is invalid")
    decode_tokens = min(decode_tokens, item["target_ids"].shape[1])
    input_ids = item["input_ids"].to(device)
    attention_mask = item["attention_mask"].to(device)
    region_batch = item["region_batch"].to(device)
    source_length = int(attention_mask.sum().item())
    replay = item["target_ids"][0, :decode_tokens].to(device)

    # Warm model kernels and allocation paths outside the timed/profiler window.
    warm = _call_prefill(
        model,
        input_ids,
        attention_mask,
        region_batch,
        budget,
        hard_threshold,
        top_k_fraction,
    )
    warm_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1)
    _call_decode(model, replay[:1].view(1, 1), warm_mask, source_length, warm.past_key_values)
    del warm
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if isinstance(model, MidLayerSourceAdapter):
        model.clear_route()

    def prefill():
        return _call_prefill(
            model,
            input_ids,
            attention_mask,
            region_batch,
            budget,
            hard_threshold,
            top_k_fraction,
        )

    starts, ends, hooks = _adapter_events(model, device.type == "cuda")
    captured_adapter_inputs: list[tuple[Any, ...]] = []
    capture_hook = None
    if isinstance(model, MidLayerSourceAdapter):
        capture_hook = model.adapter.register_forward_pre_hook(
            lambda _module, inputs: captured_adapter_inputs.append(
                tuple(
                    value.detach() if isinstance(value, torch.Tensor) else value for value in inputs
                )
            )
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        pre_start = torch.cuda.Event(enable_timing=True)
        pre_end = torch.cuda.Event(enable_timing=True)
        pre_start.record()
    wall_start = time.perf_counter()
    prefill_output, prefill_operator_flops = _profiler_flops(prefill)
    if capture_hook is not None:
        capture_hook.remove()
    if device.type == "cuda":
        pre_end.record()
        torch.cuda.synchronize(device)
        prefill_ms = float(pre_start.elapsed_time(pre_end))
        module_ms = sum(
            float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
        )
    else:
        prefill_ms = (time.perf_counter() - wall_start) * 1000
        module_ms = 0.0

    past = prefill_output.past_key_values
    current_mask = attention_mask

    def decode():
        nonlocal past, current_mask
        output = None
        for index in range(decode_tokens):
            token = replay[index].view(1, 1)
            output = _call_decode(
                model,
                token,
                current_mask,
                source_length + index,
                past,
            )
            past = output.past_key_values
            current_mask = torch.cat([current_mask, torch.ones_like(current_mask[:, :1])], dim=-1)
        return output

    if device.type == "cuda":
        decode_start = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        decode_start.record()
    decode_wall_start = time.perf_counter()
    _, decode_operator_flops = _profiler_flops(decode)
    if device.type == "cuda":
        decode_end.record()
        torch.cuda.synchronize(device)
        decode_ms = float(decode_start.elapsed_time(decode_end))
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        decode_ms = (time.perf_counter() - decode_wall_start) * 1000
        peak_memory = 0
    for hook in hooks:
        hook.remove()
    if isinstance(model, MidLayerSourceAdapter):
        model.clear_route()
    recurrent_module_flops = 0
    if captured_adapter_inputs:
        adapter_inputs = captured_adapter_inputs[0]

        def profile_adapter():
            if len(adapter_inputs) == 1:
                return model.adapter(adapter_inputs[0])
            return model.adapter(
                adapter_inputs[0],
                adapter_inputs[1],
                budget=budget,
                hard_threshold=hard_threshold,
                top_k_fraction=top_k_fraction,
                return_diagnostics=True,
            )

        _, recurrent_module_flops = _profiler_flops(profile_adapter)
    profiler_flops = prefill_operator_flops + decode_operator_flops
    prefill_attention_flops, decode_attention_flops = _attention_flops_upper(
        model, source_length, decode_tokens
    )
    attention_flops = prefill_attention_flops + decode_attention_flops
    estimated_prefill_flops = prefill_operator_flops + prefill_attention_flops
    estimated_decode_flops = decode_operator_flops + decode_attention_flops
    return {
        "source_tokens": source_length,
        "decode_tokens": decode_tokens,
        "timing_device": device.type,
        "prefill_device_ms": prefill_ms,
        "recurrent_module_device_ms": module_ms,
        "cached_decode_device_ms": decode_ms,
        "end_to_end_fixed_device_ms": prefill_ms + decode_ms,
        "peak_vram_bytes": peak_memory,
        "profiler_operator_flops": profiler_flops,
        "prefill_profiler_operator_flops": prefill_operator_flops,
        "decode_profiler_operator_flops": decode_operator_flops,
        "recurrent_module_profiler_flops": recurrent_module_flops,
        "sdpa_attention_flops_upper_estimate": attention_flops,
        "estimated_total_flops": profiler_flops + attention_flops,
        "estimated_prefill_flops": estimated_prefill_flops,
        "estimated_recurrent_module_flops": recurrent_module_flops,
        "estimated_decode_flops": estimated_decode_flops,
        "estimated_flops_per_source_token": (profiler_flops + attention_flops)
        / max(1, source_length),
        "estimated_decode_flops_per_generated_token": estimated_decode_flops / decode_tokens,
    }


def profile_checkpoint(
    checkpoint: Path,
    validation_root: Path,
    *,
    sample_index: int = 0,
    sample_plan_path: Path | None = None,
    decode_tokens: int = 128,
    hard_threshold: float | None = None,
    top_k_fraction: float | None = None,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Profile a checkpoint using a selected in-domain validation program only."""
    device = torch.device(device_name)
    model, tokenizer, metadata = load_trained_model(checkpoint, device)
    dataset = ManifestDataset(
        validation_root, tokenizer, split="validation", max_sequence_length=None
    )
    architecture = metadata["architecture"]
    collator = TranslationCollator(
        tokenizer,
        architecture=architecture,
        random_seed=int(metadata["config"]["seed"]),
        generation_only=True,
    )
    model.eval()
    if sample_plan_path is None:
        sample_rows = [{"dataset_index": sample_index}]
        sample_plan_sha256 = None
    else:
        sample_plan = json.loads(sample_plan_path.read_text(encoding="utf-8"))
        sample_plan_sha256 = _file_sha256(sample_plan_path)
        manifest_path = validation_root / "manifest.json"
        if _file_sha256(manifest_path) != sample_plan.get("validation_manifest_sha256"):
            raise ValueError("Efficiency sample plan does not match the validation manifest")
        sample_rows = sample_plan.get("samples", [])
        if sample_plan.get("status") != "FROZEN_PREMODEL" or not sample_rows:
            raise ValueError("Efficiency profile sample plan is absent or not frozen")
    sample_results = []
    for sample in sample_rows:
        index = int(sample["dataset_index"])
        if index < 0 or index >= len(dataset):
            raise IndexError(f"Validation sample index {index} out of range 0..{len(dataset) - 1}")
        raw = dataset[index]
        if sample_plan_path is not None:
            row = raw["row"]
            direction = f"{row['source_language']}-to-{row['target_language']}"
            if (
                str(row["problem_id"]) != str(sample["problem_id"])
                or direction != sample["direction"]
                or row.get("source_sha256") != sample.get("source_sha256")
            ):
                raise ValueError(f"Efficiency sample plan identity mismatch at index {index}")
        batch = collator([raw])
        # Generation-only collation contains the source prompt; retain the
        # validation target separately for fixed-length cached decode replay.
        batch["target_ids"] = torch.tensor(raw["target_ids"], dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            measures = profile_example(
                model,
                batch,
                device=device,
                decode_tokens=decode_tokens,
                budget=float(metadata["config"]["expected_depth_budget"]),
                hard_threshold=hard_threshold,
                top_k_fraction=top_k_fraction,
            )
        measures["sample_index"] = index
        measures["problem_id"] = str(raw["row"]["problem_id"])
        measures["direction"] = f"{raw['row']['source_language']}-to-{raw['row']['target_language']}"
        sample_results.append(measures)
    base = model.causal_lm if isinstance(model, MidLayerSourceAdapter) else model
    layers = int(getattr(base.config, "num_hidden_layers", getattr(base.config, "n_layer", 0)))
    numeric_fields = [
        key for key, value in sample_results[0].items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    if len(sample_results) == 1:
        aggregate = dict(sample_results[0])
    else:
        aggregate = {
            key: float(statistics.median(float(row[key]) for row in sample_results))
            for key in numeric_fields
        }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapter_parameters = sum(parameter.numel() for parameter in model.adapter.parameters()) if isinstance(model, MidLayerSourceAdapter) else 0
    router_parameters = (
        sum(parameter.numel() for name, parameter in model.adapter.named_parameters() if "router" in name)
        if isinstance(model, MidLayerSourceAdapter) else 0
    )
    aggregate.update({
            "experiment_id": metadata["experiment_id"],
            "architecture": architecture,
            "backbone": metadata["backbone_path"],
            "checkpoint_sha256": _file_sha256(checkpoint),
            "checkpoint_global_step": metadata["checkpoint_global_step"],
            "validation_problem_ids": [row["problem_id"] for row in sample_results],
            "directions": sorted({row["direction"] for row in sample_results}),
            "decoder_layers": layers,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameters,
            "adapter_parameter_count": adapter_parameters,
            "router_parameter_count": router_parameters,
            "profile_protocol": "fixed-source-prefill + target-reference-token cached decode capped at 128 tokens",
            "requested_decode_token_cap": int(decode_tokens),
            "hard_threshold": hard_threshold,
            "top_k_fraction": top_k_fraction,
            "expected_depth_budget": float(metadata["config"]["expected_depth_budget"]),
            "external_data_used": False,
            "sample_count": len(sample_results),
            "sample_plan_sha256": sample_plan_sha256,
            "aggregation": "median_over_frozen_validation_profile_sample",
            "samples": sample_results,
        })
    aggregate["source_tokens_per_prefill_second"] = aggregate["source_tokens"] / max(aggregate["prefill_device_ms"] / 1000.0, 1e-9)
    aggregate["decode_tokens_per_second"] = aggregate["decode_tokens"] / max(aggregate["cached_decode_device_ms"] / 1000.0, 1e-9)
    aggregate["end_to_end_tokens_per_second"] = aggregate["decode_tokens"] / max(aggregate["end_to_end_fixed_device_ms"] / 1000.0, 1e-9)
    if isinstance(model, MidLayerSourceAdapter):
        model.close()
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-plan", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--hard-threshold", type=float)
    parser.add_argument("--top-k-fraction", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile_checkpoint(
        args.checkpoint,
        args.validation_root,
        sample_index=args.sample_index,
        sample_plan_path=args.sample_plan,
        decode_tokens=args.decode_tokens,
        hard_threshold=args.hard_threshold,
        top_k_fraction=args.top_k_fraction,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
