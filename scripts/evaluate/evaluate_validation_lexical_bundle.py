#!/usr/bin/env python3
"""Retrospectively score an in-domain checkpoint with the frozen lexical bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.evaluation.checkpoints import load_trained_model
from src.training.data import ManifestDataset
from src.training.run_translation import _validation_generation_metrics


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
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def evaluate(checkpoint_path: Path, output_path: Path, device_name: str = "cuda") -> dict[str, Any]:
    """Score the checkpoint's best validation-selected state on XLCoST validation only."""
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_hash = _sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_values = checkpoint["config"]
    validation_manifest = Path(config_values["validation_manifest"])
    if not validation_manifest.is_absolute():
        validation_manifest = project_root / validation_manifest
    manifest_path = validation_manifest / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    implementation_hash = hashlib.sha256(
        (
            _sha256(project_root / "src/evaluation/lexical_metrics.py")
            + _sha256(project_root / "src/training/run_translation.py")
        ).encode()
    ).hexdigest()
    del checkpoint
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            existing.get("checkpoint_sha256") == checkpoint_hash
            and existing.get("validation_manifest_sha256") == manifest_hash
            and existing.get("metric_implementation_sha256") == implementation_hash
        ):
            return existing
        raise ValueError("Existing validation metric record has different source/checkpoint hashes")

    from src.training.run_translation import RunConfig

    config = RunConfig(**config_values)
    path_fields = ("backbone_path", "train_manifest", "validation_manifest", "output_dir")
    resolved_paths = {
        field: str(
            path if path.is_absolute() else (project_root / path).resolve()
        )
        for field in path_fields
        if (path := Path(getattr(config, field)))
    }
    config = replace(config, **resolved_paths)
    device = torch.device(device_name)
    model, tokenizer, metadata = load_trained_model(
        checkpoint_path.resolve(), device, config_override=config
    )
    dataset = ManifestDataset(
        validation_manifest, tokenizer, "validation", config.max_sequence_length
    )
    metrics = _validation_generation_metrics(model, dataset, tokenizer, config, device)
    record = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": "xlcost_in_domain_validation_only",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "validation_manifest_sha256": manifest_hash,
        "metric_implementation_sha256": implementation_hash,
        "experiment_id": metadata["experiment_id"],
        "architecture": metadata["architecture"],
        "seed": int(config.seed),
        "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
        "checkpoint_validation_loss": metadata["checkpoint_validation_loss"],
        "generation_examples_per_direction": config.generation_examples_per_direction,
        "selection_metric": "validation_target_loss",
        "external_metrics_used": False,
        "metrics": metrics,
    }
    _atomic_json(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    record = evaluate(args.checkpoint, args.output, args.device)
    print(json.dumps(record["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
