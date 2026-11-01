#!/usr/bin/env python3
"""Calibrate recurrent hard-halting thresholds using only XLCoST validation data."""

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
import yaml
from torch.utils.data import Subset

from src.evaluation.checkpoints import load_trained_model
from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.training.data import ManifestDataset
from src.training.run_translation import _validate


def select_threshold(
    observations: list[dict[str, float]], budget: float, tolerance: float = 0.05
) -> dict[str, float]:
    """Choose minimum validation loss under the predeclared mean-depth constraint."""
    if not observations:
        raise ValueError("No validation threshold observations were supplied")
    if budget <= 0 or tolerance < 0:
        raise ValueError("budget must be positive and tolerance must be non-negative")
    feasible = [row for row in observations if row["mean_recurrent_depth"] <= budget + tolerance]
    if not feasible:
        raise ValueError("No threshold satisfies the predeclared recurrence-depth budget")
    selected = min(
        feasible,
        key=lambda row: (
            row["validation_target_loss"],
            row["mean_recurrent_depth"],
            row["threshold"],
        ),
    )
    return dict(selected)


def _validation_subset(dataset: ManifestDataset, seed: int, per_direction: int) -> list[int]:
    by_direction: dict[str, list[tuple[str, int]]] = {}
    for index in range(len(dataset)):
        row = dataset.row_metadata(index)
        direction = f"{row['source_language']}-to-{row['target_language']}"
        stable_key = hashlib.sha256(f"{seed}|{direction}|{row['problem_id']}".encode()).hexdigest()
        by_direction.setdefault(direction, []).append((stable_key, index))
    if len(by_direction) != 6 or any(len(rows) < per_direction for rows in by_direction.values()):
        raise ValueError("Validation calibration requires six directions with enough examples")
    return [index for rows in by_direction.values() for _, index in sorted(rows)[:per_direction]]


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate(
    checkpoint_path: Path,
    output_path: Path,
    *,
    thresholds: list[float],
    per_direction: int = 16,
    tolerance: float = 0.05,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Evaluate a fixed threshold grid on one deterministic in-domain subset."""
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Threshold grid must be non-empty and contain no duplicates")
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("Every hard threshold must be in [0, 1]")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    config_values = checkpoint["config"]
    from src.training.run_translation import RunConfig

    config = RunConfig(**config_values)
    if config.seed != 42:
        raise ValueError(
            "Frozen routing policies must be calibrated from the designated seed-42 checkpoint"
        )
    del checkpoint
    validation_manifest_path = Path(config.validation_manifest) / "manifest.json"
    validation_manifest_sha256 = hashlib.sha256(validation_manifest_path.read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(
        json.dumps(config_values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        expected = {
            "checkpoint_sha256": checkpoint_sha256,
            "threshold_grid": [float(x) for x in thresholds],
            "tolerance": tolerance,
            "target_mean_recurrent_depth": float(config.expected_depth_budget),
            "validation_manifest_sha256": validation_manifest_sha256,
            "config_sha256": config_sha256,
            "validation_examples_per_direction": per_direction,
        }
        for key, value in expected.items():
            actual = (
                existing.get("compute_constraint", {}).get(key)
                if key in {"tolerance", "target_mean_recurrent_depth"}
                else existing.get(key)
            )
            if actual != value:
                raise ValueError(f"Existing routing calibration differs at {key}")
        if existing.get("status") == "VALIDATION_CALIBRATED":
            return existing
    model, tokenizer, model_metadata = load_trained_model(
        checkpoint_path, torch.device(device_name)
    )
    adaptive_modes = {
        "mor_token",
        "sart",
        "sart_no_message",
        "region_adaptive_no_graph",
        "random_region",
    }
    if (
        not isinstance(model, MidLayerSourceAdapter)
        or getattr(model.adapter, "mode", None) not in adaptive_modes
    ):
        raise ValueError(
            "Routing calibration requires an adaptive recurrent architecture checkpoint"
        )
    dataset = ManifestDataset(
        Path(config.validation_manifest), tokenizer, "validation", config.max_sequence_length
    )
    selected = _validation_subset(dataset, config.seed, per_direction)
    validation_data = Subset(dataset, selected)
    validation_tasks = []
    for index in selected:
        task = dataset.row_metadata(index)
        validation_tasks.append(
            {
                "problem_id": task["problem_id"],
                "source_language": task["source_language"],
                "target_language": task["target_language"],
            }
        )
    record = {
        "schema_version": 1,
        "status": "RUNNING",
        "benchmark": "xlcost",
        "split": "validation",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_manifest_sha256": validation_manifest_sha256,
        "config_sha256": config_sha256,
        "experiment_id": model_metadata["experiment_id"],
        "architecture": model_metadata["architecture"],
        "seed": int(config.seed),
        "calibration_seed_policy": "seed42_policy_shared_across_seeds",
        "selection_metric": "validation_target_loss",
        "compute_constraint": {
            "target_mean_recurrent_depth": float(config.expected_depth_budget),
            "tolerance": tolerance,
        },
        "validation_tasks": validation_tasks,
        "validation_examples_per_direction": per_direction,
        "threshold_grid": [float(x) for x in thresholds],
        "observations": [],
        "external_metrics_used": False,
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "threshold_grid", "validation_tasks"):
            if existing.get(key) != record.get(key):
                raise ValueError(f"Existing routing calibration differs at {key}")
        record = existing
    observations_by_threshold = {
        float(row["threshold"]): row for row in record.get("observations", [])
    }
    if set(observations_by_threshold) - set(thresholds):
        raise ValueError("Existing calibration includes a threshold outside the frozen grid")
    record["status"] = "RUNNING"
    _atomic_json(output_path, record)
    for threshold in thresholds:
        threshold = float(threshold)
        if threshold in observations_by_threshold:
            continue
        threshold_config = replace(
            config,
            hard_threshold=float(threshold),
            generation_examples_per_direction=0,
        )
        loss, stats = _validate(
            model, validation_data, tokenizer, threshold_config, torch.device(device_name)
        )
        observations_by_threshold[threshold] = {
            "threshold": threshold,
            "validation_target_loss": float(loss),
            "mean_recurrent_depth": float(stats["mean_recurrent_depth"]),
            "target_tokens": int(stats["target_tokens"]),
        }
        record["observations"] = [
            observations_by_threshold[value]
            for value in thresholds
            if value in observations_by_threshold
        ]
        _atomic_json(output_path, record)
    observations = [observations_by_threshold[float(value)] for value in thresholds]
    selected_threshold = select_threshold(
        observations, config.expected_depth_budget, tolerance=tolerance
    )
    record["status"] = "VALIDATION_CALIBRATED"
    record["observations"] = observations
    record["selected"] = selected_threshold
    _atomic_json(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/routing_calibration.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Calibration JSON path. Defaults to "
            "reports/calibration/<backbone-revision>/<architecture>-seed42.json."
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    policy = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(policy.get("calibration_seed", -1)) != 42:
        raise ValueError("routing_calibration.yaml must designate seed 42")
    output_path = args.output
    if output_path is None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        config = checkpoint["config"]
        revision = Path(str(config["backbone_path"])).name
        architecture = str(config["architecture"])
        seed = int(config["seed"])
        if seed != 42:
            raise ValueError("Default calibration path is reserved for the seed-42 policy checkpoint")
        output_path = Path("reports/calibration") / revision / f"{architecture}-seed42.json"
        del checkpoint
    result = calibrate(
        args.checkpoint,
        output_path,
        thresholds=[float(value) for value in policy["thresholds"]],
        per_direction=int(policy["validation_examples_per_direction"]),
        tolerance=float(policy["compute_constraint"]["tolerance"]),
        device_name=args.device,
    )
    print(json.dumps(result["selected"], sort_keys=True))


if __name__ == "__main__":
    main()
