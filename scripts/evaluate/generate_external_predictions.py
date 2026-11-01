#!/usr/bin/env python3
"""Generate resumable one-candidate Pass@1 predictions for frozen task caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from src.data.translation_protocol import extract_generated_code
from src.evaluation.checkpoints import load_trained_model
from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.training.data import ManifestDataset, TranslationCollator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _tensor_record(value: torch.Tensor) -> Any:
    data = value.detach().float().cpu()
    return data.item() if data.numel() == 1 else data.squeeze(0).tolist()


def _record_path(root: Path, row: dict[str, Any]) -> Path:
    digest = hashlib.sha256(row["task_id"].encode()).hexdigest()[:20]
    return root / row["direction"] / f"{digest}.json"


def _verify_cache_files(cache_root: Path, manifest: dict[str, Any]) -> None:
    """Fail before generation if any committed structure shard changed or vanished."""
    entries = manifest.get("completed_shards", manifest.get("files", {}))
    for key, metadata in entries.items():
        path = Path(metadata["path"])
        if not path.is_absolute():
            path = Path.cwd() / path
        if (
            not path.is_file()
            or (metadata.get("bytes") is not None and path.stat().st_size != metadata.get("bytes"))
            or _sha256(path) != metadata.get("sha256")
        ):
            raise ValueError(f"Structure cache shard failed integrity check: {key} ({path})")


def _validate_frozen_protocol(
    policy_path: Path | None,
    benchmark: str,
    split: str,
    task_manifest_sha256: str,
    architecture: str,
    backbone_path: str,
    configured_epochs: int,
    max_recurrence: int,
    insertion_fraction: float,
    checkpoint_epoch: int,
    checkpoint_validation_loss: float | None,
    checkpoint_path: Path,
    max_new_tokens: int,
    eos_token_id: int | None,
    hard_threshold: float | None,
    top_k_fraction: float | None,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Require a validation-only freeze record before any external task generation."""
    if policy_path is None or not policy_path.is_file():
        raise ValueError("External generation is blocked until a frozen protocol record exists")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("status") != "FROZEN_PREMODEL":
        raise ValueError("Frozen protocol status must be FROZEN_PREMODEL")
    if policy.get("selection_metric") != "validation_target_loss":
        raise ValueError("Checkpoint/horizon selection must use in-domain validation loss")
    if policy.get("external_metrics_used_for_selection") is not False:
        raise ValueError("External metrics must not select horizons or checkpoints")
    if protocol_path is None:
        protocol_path = Path(__file__).resolve().parents[2] / "docs/TRANSLATION_PROTOCOL.md"
    if policy.get("translation_protocol_sha256") != _sha256(protocol_path):
        raise ValueError("Frozen prompt/translation protocol hash does not match the current file")
    manifest_key = f"{benchmark}:{split}"
    if policy.get("external_task_manifests", {}).get(manifest_key) != task_manifest_sha256:
        raise ValueError(f"External task manifest is not locked for {manifest_key}")

    revision = Path(backbone_path).name
    horizon = policy.get("training_horizon_epochs", {}).get(revision)
    if not isinstance(horizon, int) or horizon < 1 or configured_epochs != horizon:
        raise ValueError(
            f"Checkpoint training epoch count {configured_epochs} does not match frozen "
            f"horizon for backbone revision {revision!r}: {horizon!r}"
        )
    if (
        checkpoint_path.name != "best.pt"
        or checkpoint_validation_loss is None
        or not 1 <= checkpoint_epoch <= horizon
    ):
        raise ValueError(
            "External evaluation requires the validation-selected best.pt checkpoint "
            "from within the frozen horizon"
        )

    generation = policy.get("generation", {})
    expected_generation = {
        "do_sample": True,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "postprocess": "extract_generated_code_v1",
        "pass_at_1": {
            "do_sample": True,
            "uses_sample_index": 0,
            "num_samples": 1,
            "temperature": 0.8,
            "top_p": 0.95,
            "num_beams": 1,
            "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32",
        },
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        raise ValueError("Generation settings differ from the frozen protocol")
    eos_by_revision = generation.get("eos_token_id_by_revision", {})
    expected_eos = eos_by_revision.get(revision, generation.get("eos_token_id"))
    if expected_eos != eos_token_id:
        raise ValueError("EOS token differs from the frozen backbone-specific generation policy")
    routing_policies = policy.get("routing_policies", {})
    routing = routing_policies.get(revision, {}).get(architecture)
    # Legacy records keyed only by architecture remain readable. New freezes
    # must key by backbone revision because calibrated compute policies can
    # differ across model sizes and model families.
    if routing is None:
        routing = routing_policies.get(architecture)
    if not isinstance(routing, dict):
        raise ValueError(f"No frozen validation routing policy for architecture {architecture!r}")
    if (
        routing.get("hard_threshold") != hard_threshold
        or routing.get("top_k_fraction") != top_k_fraction
    ):
        raise ValueError("Routing settings differ from the frozen validation calibration")
    if (
        routing.get("max_recurrence") != max_recurrence
        or routing.get("insertion_fraction") != insertion_fraction
    ):
        raise ValueError("Recurrent depth/insertion settings differ from the frozen protocol")
    if hard_threshold is not None and top_k_fraction is not None:
        raise ValueError("hard_threshold and top_k_fraction are mutually exclusive")
    return policy


def _generate_one(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    region_batch: Any,
    model_metadata: dict[str, Any],
    max_new_tokens: int,
    eos_token_id: int,
    hard_threshold: float | None,
    top_k_fraction: float | None,
    *,
    sample_seed: int | None = None,
) -> dict[str, Any]:
    """Generate one candidate under the frozen greedy or sampled decoding policy."""
    sampled = sample_seed is not None
    if isinstance(model, MidLayerSourceAdapter):
        generator = torch.Generator(device=input_ids.device) if sampled else None
        if generator is not None:
            generator.manual_seed(sample_seed)
        generated, _ = model.generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            region_batch=region_batch,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            budget=float(model_metadata["config"]["expected_depth_budget"]),
            hard_threshold=hard_threshold,
            top_k_fraction=top_k_fraction,
            do_sample=sampled,
            temperature=0.8 if sampled else 1.0,
            top_p=0.95 if sampled else 1.0,
            generator=generator,
        )
        diagnostics = model._last_diagnostics
    elif sampled:
        cuda_devices = (
            [input_ids.device.index or torch.cuda.current_device()] if input_ids.is_cuda else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(sample_seed)
            if input_ids.is_cuda:
                torch.cuda.manual_seed_all(sample_seed)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=0.8,
                top_p=0.95,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        diagnostics = {}
    else:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        diagnostics = {}
    new_token_ids = generated[0, input_ids.shape[1] :]
    raw_output = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    eos = bool(
        new_token_ids.numel()
        and tokenizer.eos_token_id is not None
        and int(new_token_ids[-1]) == tokenizer.eos_token_id
    )
    return {
        "status": "generated",
        "generated_code": extract_generated_code(raw_output),
        "generation_tokens": int(new_token_ids.numel()),
        "reached_max_new_tokens": int(new_token_ids.numel()) >= max_new_tokens,
        "ended_with_eos": eos,
        "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
        "recurrence_stats": {
            name: _tensor_record(value)
            for name, value in diagnostics.items()
            if name
            in {"expected_depth", "gate_probabilities", "token_gate_probabilities", "mean_depth"}
        },
    }


def _generate_sampled_batch(
    model: Any,
    tokenizer: Any,
    batch: dict[str, Any],
    model_metadata: dict[str, Any],
    max_new_tokens: int,
    eos_token_id: int,
    hard_threshold: float | None,
    top_k_fraction: float | None,
    sample_seeds: list[int],
) -> list[dict[str, Any]]:
    """Generate one independently seeded candidate per prompt in one GPU batch."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    count = input_ids.shape[0]
    if len(sample_seeds) != count:
        raise ValueError("Batch needs one frozen seed per translation task")
    generators = [torch.Generator(device=input_ids.device).manual_seed(seed)
                  for seed in sample_seeds]
    if isinstance(model, MidLayerSourceAdapter):
        generated, _ = model.generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            region_batch=batch["region_batch"],
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            budget=float(model_metadata["config"]["expected_depth_budget"]),
            hard_threshold=hard_threshold,
            top_k_fraction=top_k_fraction,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            generator=generators,
        )
        diagnostics = model._last_diagnostics
    else:
        # HF's batched sampling uses one RNG stream, which changes a task's
        # output when an earlier task is resumed or a batch is retried. Decode
        # with a private generator for each row to keep task seeds independent.
        generated = input_ids
        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        finished = torch.zeros(count, dtype=torch.bool, device=input_ids.device)
        for step in range(max_new_tokens):
            if step == 0:
                positions = (current_mask.cumsum(-1) - 1).clamp_min(0)
            else:
                positions = (current_mask.sum(-1) - 1).clamp_min(0)[:, None]
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                position_ids=positions,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            if step == 0:
                last = attention_mask.sum(-1).clamp_min(1) - 1
                logits = outputs.logits[torch.arange(count, device=input_ids.device), last]
            else:
                logits = outputs.logits[:, -1, :]
            logits = logits / 0.8
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probabilities_sorted = torch.softmax(sorted_logits, dim=-1)
            remove_sorted = probabilities_sorted.cumsum(dim=-1) - probabilities_sorted >= 0.95
            remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
            remove_sorted[:, 0] = False
            remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
            probabilities = torch.softmax(
                logits.masked_fill(remove, torch.finfo(logits.dtype).min).float(), dim=-1
            )
            next_token = torch.stack([
                torch.multinomial(probabilities[i], 1, generator=generators[i])[0]
                for i in range(count)
            ])
            next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
                break
            current_ids = next_token[:, None]
            current_mask = torch.cat([current_mask, torch.ones_like(current_mask[:, :1])], dim=1)
            past_key_values = outputs.past_key_values
        diagnostics = {}

    rows = []
    for index in range(count):
        new_tokens = generated[index, input_ids.shape[1]:]
        eos_positions = (new_tokens == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_positions.numel():
            new_tokens = new_tokens[:int(eos_positions[0]) + 1]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        row_diagnostics = {}
        for name, value in diagnostics.items():
            if name not in {"expected_depth", "gate_probabilities",
                            "token_gate_probabilities", "mean_depth"}:
                continue
            selected = value[index] if isinstance(value, torch.Tensor) and value.ndim > 0 \
                and value.shape[0] == count else value
            row_diagnostics[name] = _tensor_record(selected)
        rows.append({
            "status": "generated",
            "generated_code": extract_generated_code(raw_output),
            "generation_tokens": int(new_tokens.numel()),
            "reached_max_new_tokens": int(new_tokens.numel()) >= max_new_tokens,
            "ended_with_eos": bool(eos_positions.numel()),
            "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
            "recurrence_stats": row_diagnostics,
        })
    return rows


def generate(
    benchmark: str,
    split: str,
    cache_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    max_new_tokens: int = 768,
    hard_threshold: float | None = None,
    top_k_fraction: float | None = None,
    device_name: str = "cuda",
    frozen_policy_path: Path | None = None,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Run deterministic source-only generation and resume completed task records."""
    if frozen_policy_path is None or not frozen_policy_path.is_file():
        raise ValueError("External generation is blocked until a frozen protocol record exists")
    cache_manifest_path = cache_root / split / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "COMPLETE":
        raise ValueError(f"External structural cache is not complete: {cache_manifest_path}")
    _verify_cache_files(cache_manifest_path.parent, cache_manifest)
    device = torch.device(device_name)
    checkpoint_hash = _sha256(checkpoint_path)
    model, tokenizer, model_metadata = load_trained_model(checkpoint_path, device)
    frozen_policy = _validate_frozen_protocol(
        frozen_policy_path,
        benchmark=benchmark,
        split=split,
        task_manifest_sha256=cache_manifest.get("input_manifest_sha256", ""),
        architecture=model_metadata["architecture"],
        backbone_path=model_metadata["backbone_path"],
        configured_epochs=int(model_metadata["config"]["epochs"]),
        max_recurrence=int(model_metadata["config"]["max_recurrence"]),
        insertion_fraction=float(model_metadata["config"]["insertion_fraction"]),
        checkpoint_epoch=int(model_metadata["checkpoint_epoch"]),
        checkpoint_validation_loss=model_metadata["checkpoint_validation_loss"],
        checkpoint_path=checkpoint_path,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        hard_threshold=hard_threshold,
        top_k_fraction=top_k_fraction,
    )
    assert frozen_policy_path is not None
    dataset = ManifestDataset(
        manifest_path=cache_root,
        tokenizer=tokenizer,
        split=split,
        max_sequence_length=None,
    )
    architecture = model_metadata["architecture"]
    collator = TranslationCollator(
        tokenizer,
        architecture=architecture,
        random_seed=int(model_metadata["config"]["seed"]),
        generation_only=True,
    )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if frozen_policy["generation"].get("batch_size") != batch_size:
        raise ValueError("Batch size differs from the frozen protocol")
    generation_config = {
        "batch_size": batch_size,
        "do_sample": True,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "postprocess": "extract_generated_code_v1",
        "pass_at_1": {
            "do_sample": True,
            "uses_sample_index": 0,
            "num_samples": 1,
            "temperature": 0.8,
            "top_p": 0.95,
            "num_beams": 1,
            "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32",
        },
        "hard_threshold": hard_threshold,
        "top_k_fraction": top_k_fraction,
    }
    run_manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "benchmark": benchmark,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "cache_manifest_sha256": _sha256(cache_manifest_path),
        "model": model_metadata,
        "frozen_protocol_sha256": _sha256(frozen_policy_path),
        "frozen_protocol_path": str(frozen_policy_path.resolve()),
        "frozen_protocol": frozen_policy,
        "generation_config": generation_config,
        "tasks_total": len(dataset),
        "predictions_root": str(output_root),
    }
    run_manifest_path = output_root / "run_manifest.json"
    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        for key in (
            "benchmark",
            "split",
            "checkpoint_sha256",
            "cache_manifest_sha256",
            "frozen_protocol_sha256",
            "generation_config",
        ):
            if old.get(key) != run_manifest.get(key):
                raise ValueError(f"Existing prediction run differs at {key}; use a new output root")
    _atomic_json(run_manifest_path, run_manifest)

    completed = 0
    failures = 0
    for first_index in range(0, len(dataset), batch_size):
        items = [dataset[index] for index in range(first_index, min(len(dataset), first_index + batch_size))]
        rows = [item["row"] for item in items]
        paths = [_record_path(output_root, row) for row in rows]
        existing = []
        for record_path in paths:
            if not record_path.exists():
                existing.append(None)
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                record.get("checkpoint_sha256") != checkpoint_hash
                or record.get("cache_manifest_sha256") != run_manifest["cache_manifest_sha256"]
                or record.get("generation_config") != generation_config
            ):
                raise ValueError(f"Stale or conflicting task result exists: {record_path}")
            existing.append(None if record.get("status") == "generation_error" else record)
        if all(record is not None for record in existing):
            completed += len(items)
            failures += sum(record.get("status") == "generation_error" for record in existing)
            continue

        eligible = [
            i for i, item in enumerate(items)
            if existing[i] is None
            and len(item["prompt_ids"]) + max_new_tokens <= model_metadata["max_position_embeddings"]
        ]
        candidates: dict[int, dict[str, Any]] = {}
        started = time.perf_counter()
        if eligible:
            eligible_items = [items[i] for i in eligible]
            batch = collator(eligible_items)
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            batch["region_batch"] = batch["region_batch"].to(device)
            seeds = [int.from_bytes(hashlib.sha256(
                f"{model_metadata['experiment_id']}|{rows[i]['task_id']}|0".encode()
            ).digest()[:4], "big") for i in eligible]
            try:
                with torch.inference_mode():
                    generated = _generate_sampled_batch(
                        model, tokenizer, batch, model_metadata, max_new_tokens,
                        tokenizer.eos_token_id, hard_threshold, top_k_fraction, seeds,
                    )
                candidates.update(zip(eligible, generated, strict=True))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # A memory fallback retains the exact task seed and records its mode.
                for local_index, item_index in enumerate(eligible):
                    single = collator([items[item_index]])
                    try:
                        with torch.inference_mode():
                            candidate = _generate_one(
                                model, tokenizer, single["input_ids"].to(device),
                                single["attention_mask"].to(device),
                                single["region_batch"].to(device), model_metadata,
                                max_new_tokens, tokenizer.eos_token_id,
                                hard_threshold, top_k_fraction, sample_seed=seeds[local_index],
                            )
                        candidate["batch_fallback"] = "cuda_oom_single"
                    except Exception as exc:
                        candidate = {"status": "generation_error", "generated_code": "",
                                     "error_type": type(exc).__name__, "error": str(exc)}
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                    candidates[item_index] = candidate
            except Exception as exc:
                for item_index in eligible:
                    candidates[item_index] = {
                        "status": "generation_error", "generated_code": "",
                        "error_type": type(exc).__name__, "error": str(exc),
                    }
            finally:
                if isinstance(model, MidLayerSourceAdapter):
                    model.clear_route()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        for index, (row, record_path, previous) in enumerate(zip(rows, paths, existing, strict=True)):
            if previous is not None:
                completed += 1
                failures += int(previous.get("status") == "generation_error")
                continue
            result: dict[str, Any] = {
                "schema_version": 1,
                "benchmark": benchmark, "split": split,
                "task_id": row["task_id"], "problem_id": row["problem_id"],
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "source_sha256": row["source_sha256"],
                "target_interface_sha256": hashlib.sha256(
                    row.get("target_interface", "").encode()
                ).hexdigest(),
                "checkpoint_sha256": checkpoint_hash,
                "cache_manifest_sha256": run_manifest["cache_manifest_sha256"],
                "prompt_tokens": len(items[index]["prompt_ids"]),
                "generation_config": generation_config,
                "target_reference_sha256": row.get("target_reference_sha256"),
                "target_test_sha256": row.get("target_test_sha256"),
                "test_count": row.get("test_count", 0),
            }
            candidate = candidates.get(index, {
                "status": "context_overflow", "generated_code": "",
                "generation_tokens": 0, "reached_max_new_tokens": False,
                "ended_with_eos": False,
            })
            result.update({"sample_index": 0, **candidate})
            result["sampled_candidates"] = [{"sample_index": 0, **candidate}]
            route = candidate.get("recurrence_stats", {}).get("expected_depth")
            # H4 uses per-region route diagnostics only for Full-SART. Avoid
            # making other architecture predictions depend on this analysis-only
            # serialization path; their task-level recurrence statistics remain saved.
            if (model_metadata["architecture"] == "full_sart"
                    and isinstance(model, MidLayerSourceAdapter)
                    and isinstance(route, list)):
                region_rows = []
                for region in row.get("structure", {}).get("regions", []):
                    region_id = int(region["region_id"])
                    if region_id >= len(route):
                        raise ValueError(
                            f"Expected-depth diagnostics do not cover source region {region_id} "
                            f"for {row['task_id']}"
                        )
                    region_rows.append({
                        "region_id": region_id,
                        "expected_depth": float(route[region_id]),
                        "source_features": {
                            key: region.get(key) for key in (
                                "region_type", "start_byte", "end_byte", "ast_depth",
                                "child_count", "cfg_degree", "dfg_degree", "region_length",
                            )
                        },
                    })
                result["region_diagnostics"] = {
                    "source_sha256": row["source_sha256"],
                    "expected_depth_source": "pass_at_1_sample_prefill",
                    "regions": region_rows,
                }
            result["latency_seconds"] = elapsed / max(1, len(eligible))
            _atomic_json(record_path, result)
            completed += 1
            failures += int(result.get("status") == "generation_error")
        if completed % 24 < len(items):
            run_manifest["tasks_completed"] = completed
            run_manifest["generation_errors"] = failures
            _atomic_json(run_manifest_path, run_manifest)
            print(f"completed {completed}/{len(dataset)} tasks", flush=True)
    run_manifest["status"] = "COMPLETE" if failures == 0 else "INCOMPLETE_GENERATION_ERRORS"
    run_manifest["tasks_completed"] = completed
    run_manifest["generation_errors"] = failures
    _atomic_json(run_manifest_path, run_manifest)
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--split", choices=["common_core", "full_directional"], default="common_core"
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--hard-threshold", type=float)
    parser.add_argument("--top-k-fraction", type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    result = generate(
        benchmark=args.benchmark,
        split=args.split,
        cache_root=args.cache_root,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        max_new_tokens=args.max_new_tokens,
        hard_threshold=args.hard_threshold,
        top_k_fraction=args.top_k_fraction,
        device_name=args.device,
        frozen_policy_path=args.frozen_protocol,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "tasks_total": result["tasks_total"],
                "tasks_completed": result.get("tasks_completed"),
                "generation_errors": result.get("generation_errors", 0),
            }
        )
    )


if __name__ == "__main__":
    main()
