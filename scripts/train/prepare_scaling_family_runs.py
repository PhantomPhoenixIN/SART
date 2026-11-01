#!/usr/bin/env python3
"""Stage the deferred Qwen-14B and StarCoder2 backbone checks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from src.training.run_translation import RunConfig, _canonical_experiment_id

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/scaling_family.yaml"
REGISTRY = ROOT / "experiments/registry.csv"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
FIELDS = [
    "experiment_id", "architecture", "backbone", "seed", "dataset", "structure",
    "router", "Rmax", "compute_budget", "training_status", "evaluation_status",
    "checkpoint_path", "metrics_path", "git_commit", "config_path",
    "training_horizon_policy", "notes", "pbs_job_id",
]


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def study_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"no-commit+study:{digest.hexdigest()}"


def stage(backbone_key: str) -> list[str]:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    backbone = plan["backbones"][backbone_key]
    model_path = ROOT / backbone["path"]
    artifact_path = model_path / "artifact_manifest.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Pinned model snapshot is not staged: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("status") != "COMPLETE" or artifact.get("revision") != backbone["revision"]:
        raise ValueError("Model artifact is incomplete or differs from the pinned study revision")
    model_index = model_path / "model.safetensors.index.json"
    if model_index.is_file():
        index = json.loads(model_index.read_text(encoding="utf-8"))
        indexed_shards = set(index.get("weight_map", {}).values())
        manifest_files = artifact.get("files", {})
        missing = sorted(
            name for name in indexed_shards
            if name not in manifest_files
            or not (model_path / name).is_file()
            or (model_path / name).stat().st_size != manifest_files[name].get("size_bytes")
        )
        if missing:
            raise ValueError(
                "Pinned model artifact is marked COMPLETE but its safetensors index "
                f"has missing/incomplete shards: {', '.join(missing)}"
            )
    base_values = yaml.safe_load(
        (ROOT / "configs/training/pilot_sart_external_clean.yaml").read_text(encoding="utf-8")
    )
    default_max_sequence_length = int(base_values["max_sequence_length"])
    base_values.update(
        experiment_id="placeholder",
        backbone_path=backbone["path"],
        output_dir=backbone["output_dir"],
        epochs=int(plan["epochs"]),
        train_manifest=plan["train_manifest"],
        validation_manifest=plan["validation_manifest"],
        seed=int(plan["seed"]),
        max_sequence_length=int(backbone.get("max_sequence_length", base_values["max_sequence_length"])),
    )
    base = RunConfig(**base_values)
    fingerprint = study_fingerprint(
        [
            PLAN,
            Path(__file__).resolve(),
            ROOT / "src/training/run_translation.py",
            ROOT / "src/models/recurrent/region_module.py",
            ROOT / "src/models/recurrent/backbone_hook.py",
            ROOT / "docs/TRANSLATION_PROTOCOL.md",
            ROOT / plan["train_manifest"],
            ROOT / plan["validation_manifest"] / "manifest.json",
            artifact_path,
        ]
    )
    new_rows: list[dict[str, str]] = []
    for architecture in plan["architectures"]:
        config = replace(
            base,
            architecture=architecture,
            budget_lambda=0.0 if architecture == "dense" else base.budget_lambda,
            experiment_id="placeholder",
        )
        run_id = _canonical_experiment_id(config)
        config = replace(config, experiment_id=run_id)
        config_path = ROOT / "configs/generated/scaling_family" / backbone_key / f"{run_id}.yaml"
        atomic_write(config_path, yaml.safe_dump(config.__dict__, sort_keys=False))
        new_rows.append(
            {
                "experiment_id": run_id,
                "architecture": architecture,
                "backbone": backbone["registry_name"],
                "seed": str(plan["seed"]),
                "dataset": "XLCoST-program",
                "structure": "AST+CFG+DFG" if architecture == "full_sart" else "none",
                "router": {"dense": "none", "mor_style": "token-adaptive", "full_sart": "region-adaptive"}[architecture],
                "Rmax": str(config.max_recurrence),
                "compute_budget": "matched-by-validation-calibration",
                "training_status": "PLANNED",
                "evaluation_status": "PLANNED",
                "checkpoint_path": f"{config.output_dir}/{run_id}/best.pt",
                "metrics_path": f"{config.output_dir}/{run_id}/metrics.jsonl",
                "git_commit": fingerprint,
                "config_path": str(config_path.relative_to(ROOT)),
                "training_horizon_policy": backbone["horizon_policy"],
                "notes": (
                    "Seed-42 scaling/family validation; deferred until 7B core matrix is complete."
                    + (
                        f" StarCoder2 tokenizer context guard is {config.max_sequence_length} tokens "
                        "to retain every record in the frozen sample."
                        if config.max_sequence_length != default_max_sequence_length
                        else ""
                    )
                ),
                "pbs_job_id": "",
            }
        )

    with LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        with REGISTRY.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        existing = [
            row for row in existing
            if not (
                row.get("experiment_id", "").startswith("plan-")
                and row.get("backbone") == backbone["registry_name"]
                and row.get("architecture") in plan["architectures"]
            )
        ]
        by_id = {row["experiment_id"]: row for row in existing}
        for row in new_rows:
            if row["experiment_id"] in by_id:
                by_id[row["experiment_id"]].update(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"training_status", "evaluation_status", "pbs_job_id"}
                    }
                )
            else:
                by_id[row["experiment_id"]] = row
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(by_id.values())
        atomic_write(REGISTRY, buffer.getvalue())
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
    return [row["experiment_id"] for row in new_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["qwen14b", "starcoder2"], required=True)
    args = parser.parse_args()
    ids = stage(args.backbone)
    print(f"Staged {len(ids)} deferred {args.backbone} runs; none submitted to PBS")


if __name__ == "__main__":
    main()
