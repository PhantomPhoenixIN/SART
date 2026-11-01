#!/usr/bin/env python3
"""Stage the deferred nine-run Qwen2.5-Coder-1.5B comparison."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import yaml

from src.training.run_translation import RunConfig, _canonical_experiment_id

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/secondary_1p5b.yaml"
REGISTRY = ROOT / "experiments/registry.csv"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
CONFIG_DIR = ROOT / "configs/generated/confirmatory_1p5b"
MODEL_PATH = (
    "content/models/qwen2.5-coder-1.5b/df3ce67c0e24480f20468b6ef2894622d69eb73b"
)
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


def stage() -> list[str]:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    model_dir = ROOT / MODEL_PATH
    artifact = model_dir / "artifact_manifest.json"
    if not artifact.is_file():
        raise FileNotFoundError(f"Pinned model snapshot is not staged: {artifact}")
    metadata = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETE" or metadata.get("revision") != plan["revision"]:
        raise ValueError("Model artifact is incomplete or differs from the pinned study revision")
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
            artifact,
        ]
    )

    base_values = yaml.safe_load(
        (ROOT / "configs/training/pilot_sart_external_clean.yaml").read_text(encoding="utf-8")
    )
    base_values.update(
        experiment_id="placeholder",
        backbone_path=MODEL_PATH,
        output_dir="checkpoints/confirmatory_1p5b",
        epochs=int(plan["epochs"]),
        train_manifest=plan["train_manifest"],
        validation_manifest=plan["validation_manifest"],
    )
    base = RunConfig(**base_values)
    new_rows: list[dict[str, str]] = []
    for architecture in plan["architectures"]:
        for seed in plan["seeds"]:
            fixed = architecture == "dense"
            config = replace(
                base,
                architecture=architecture,
                seed=int(seed),
                budget_lambda=0.0 if fixed else base.budget_lambda,
                experiment_id="placeholder",
            )
            run_id = _canonical_experiment_id(config)
            config = replace(config, experiment_id=run_id)
            config_path = CONFIG_DIR / f"{run_id}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(config_path, yaml.safe_dump(config.__dict__, sort_keys=False))
            new_rows.append(
                {
                    "experiment_id": run_id,
                    "architecture": architecture,
                    "backbone": plan["backbone"],
                    "seed": str(seed),
                    "dataset": "XLCoST-program",
                    "structure": "AST+CFG+DFG" if architecture == "full_sart" else "none",
                    "router": {
                        "dense": "none",
                        "mor_style": "token-adaptive",
                        "full_sart": "region-adaptive",
                    }[architecture],
                    "Rmax": str(config.max_recurrence),
                    "compute_budget": "matched-by-validation-calibration",
                    "training_status": "PLANNED",
                    "evaluation_status": "PLANNED",
                    "checkpoint_path": f"{config.output_dir}/{run_id}/best.pt",
                    "metrics_path": f"{config.output_dir}/{run_id}/metrics.jsonl",
                    "git_commit": fingerprint,
                    "config_path": str(config_path.relative_to(ROOT)),
                    "training_horizon_policy": "secondary-shared-2-epoch-1.5B-deferred",
                    "notes": "Three-seed secondary model-size comparison; do not queue before core studies complete.",
                    "pbs_job_id": "",
                }
            )

    with LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        with REGISTRY.open(newline="", encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        by_id = {row["experiment_id"]: row for row in existing}
        for row in new_rows:
            if row["experiment_id"] in by_id:
                # A prior staged plan remains authoritative if its run has started.
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


if __name__ == "__main__":
    staged = stage()
    print(f"Staged {len(staged)} deferred 1.5B runs; none submitted to PBS")
