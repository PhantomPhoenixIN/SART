#!/usr/bin/env python3
"""Measure one full-SART optimizer step with the requested upper layer unfrozen."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.training.data import ManifestDataset, TranslationCollator
from src.training.run_translation import RunConfig, _load_model


def profile(config_path: Path, output_path: Path, unfreeze_layers: int) -> dict[str, Any]:
    """Run a single optimizer update on the longest eligible training example."""
    config = RunConfig(**yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config = replace(
        config,
        experiment_id="memory-profile",
        epochs=1,
        micro_batch_size=1,
        gradient_accumulation=1,
        unfreeze_last_layers=unfreeze_layers,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("A100 memory profile must run in a GPU PBS allocation")
    device = torch.device("cuda:0")
    model, tokenizer = _load_model(config, device)
    train_data = ManifestDataset(
        Path(config.train_manifest), tokenizer, "train", config.max_sequence_length
    )

    longest_item = None
    longest_length = -1
    started_scan = time.perf_counter()
    for index in range(len(train_data)):
        item = train_data[index]
        length = len(item["input_ids"])
        if length > longest_length:
            longest_item = item
            longest_length = length
    if longest_item is None:
        raise RuntimeError("No eligible training examples found")
    data_scan_seconds = time.perf_counter() - started_scan

    batch = TranslationCollator(tokenizer, config.architecture, config.seed)([longest_item])
    labels = batch.pop("labels").to(device)
    batch.pop("task_ids")
    batch.pop("target_texts")
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    batch["region_batch"] = batch["region_batch"].to(device)

    adapter_model = model if isinstance(model, MidLayerSourceAdapter) else None
    backbone = adapter_model.causal_lm if adapter_model is not None else model
    backbone_ids = {id(parameter) for parameter in backbone.parameters() if parameter.requires_grad}
    backbone_parameters = []
    adapter_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (backbone_parameters if id(parameter) in backbone_ids else adapter_parameters).append(
            parameter
        )
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": config.learning_rate})
    if adapter_parameters:
        groups.append({"params": adapter_parameters, "lr": config.adapter_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    started_step = time.perf_counter()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                **batch,
                labels=labels,
                budget=config.expected_depth_budget,
                hard_threshold=config.hard_threshold,
                use_cache=False,
            )
            budget_loss = (
                adapter_model._last_diagnostics.get(
                    "budget_loss", labels.new_zeros((), dtype=torch.float32)
                )
                if adapter_model is not None
                else labels.new_zeros((), dtype=torch.float32)
            )
            loss = output.loss + config.budget_lambda * budget_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            backbone_parameters + adapter_parameters, config.max_grad_norm
        )
        optimizer.step()
        torch.cuda.synchronize(device)
        status = "PASS"
        error = None
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        status = "CUDA_OOM"
        error = str(exc)

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "status": status,
        "error": error,
        "experiment_type": "single_step_memory_profile_not_training_result",
        "architecture": config.architecture,
        "backbone_revision": Path(config.backbone_path).name,
        "unfreeze_last_layers": unfreeze_layers,
        "trainable_parameters": trainable_parameters,
        "profile_sequence_tokens": longest_length,
        "profile_target_tokens": int((labels != -100).sum()),
        "data_scan_seconds": data_scan_seconds,
        "optimizer_step_seconds": time.perf_counter() - started_step,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "loss": float(loss.detach().float().item()) if status == "PASS" else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(profile(args.config, args.output, args.unfreeze_layers), indent=2), flush=True)


if __name__ == "__main__":
    main()
