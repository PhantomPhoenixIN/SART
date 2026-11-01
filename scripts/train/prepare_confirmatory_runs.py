#!/usr/bin/env python3
"""Materialize hash-addressed 7B run configs only after the common horizon is frozen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from src.training.run_translation import RunConfig, _canonical_experiment_id

ROOT = Path(__file__).resolve().parents[2]
FIELDS = [
    "experiment_id",
    "architecture",
    "backbone",
    "seed",
    "dataset",
    "structure",
    "router",
    "Rmax",
    "compute_budget",
    "training_status",
    "evaluation_status",
    "checkpoint_path",
    "metrics_path",
    "git_commit",
    "config_path",
    "training_horizon_policy",
    "notes",
    "pbs_job_id",
]
STRUCTURE = {
    "dense": "none",
    "dense_param_match": "none",
    "uniform_token_loop": "none",
    "mor_style": "none",
    "uniform_region_loop": "semantic regions",
    "region_adaptive_no_graph": "semantic regions, no explicit structure",
    "random_region": "random contiguous regions",
    "shuffled_structure": "permuted AST+CFG+DFG",
    "structure_only": "AST+CFG+DFG, one pass",
    "ast_sart": "AST",
    "sart_no_scalar": "AST+CFG+DFG, no scalars",
    "sart_no_message": "AST+CFG+DFG, message passing disabled",
    "sart_attention_pool": "AST+CFG+DFG, attention pooling",
    "sart_no_language_pair": "AST+CFG+DFG, no language-pair condition",
    "sart_coarse_ontology": "AST+CFG+DFG, coarse ontology",
    "cfg_sart": "CFG",
    "dfg_sart": "DFG",
    "ast_cfg_sart": "AST+CFG",
    "ast_dfg_sart": "AST+DFG",
    "full_sart": "AST+CFG+DFG",
}
ROUTER = {
    "dense": "none",
    "dense_param_match": "none",
    "uniform_token_loop": "fixed token",
    "uniform_region_loop": "fixed region",
    "mor_style": "token-adaptive",
    "structure_only": "single pass",
    "random_region": "region-adaptive",
    "region_adaptive_no_graph": "content+direction region-adaptive",
}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _workspace_revision() -> str:
    """Combine the commit ID with a deterministic source/config working-tree digest."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    digest = hashlib.sha256()
    roots = (
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "experiments",
        ROOT / "run_on_astra",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
    )
    files = [path for root in roots for path in root.rglob("*") if path.is_file()]
    files.extend(ROOT / name for name in ("README.md", "pyproject.toml", "requirements.lock"))
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        if (
            "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or relative.parts[:2] == ("configs", "generated")
            or relative == Path("experiments/registry.csv")
        ):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    commit = result.stdout.strip() if result.returncode == 0 else "no-commit"
    return f"{commit}+worktree:{digest.hexdigest()}"


def prepare(
    horizon_path: Path,
    base_config_path: Path,
    registry_path: Path,
    config_dir: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Create one config and registry row for every confirmatory/screening run."""
    horizon = yaml.safe_load(horizon_path.read_text(encoding="utf-8"))
    epochs = horizon["confirmatory_7b"]["frozen_epochs"]
    if not isinstance(epochs, int) or not 1 <= epochs <= 5:
        raise ValueError("Freeze confirmatory_7b.frozen_epochs from the validation pilot first")
    base_values = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    base_values["experiment_id"] = "placeholder"
    base = RunConfig(**base_values)
    primary = yaml.safe_load((ROOT / "configs/experiments/primary.yaml").read_text())
    seeds = primary["confirmatory_seeds"]
    multi_seed = set(horizon["confirmatory_7b"]["three_seed_architectures"])
    screening = set(horizon["confirmatory_7b"]["single_seed_screening_architectures"])
    architectures = list(horizon["confirmatory_7b"]["architectures"])
    if set(architectures) != multi_seed | screening:
        raise ValueError("Every architecture must appear in exactly one seed-policy list")
    git_commit = _workspace_revision()
    rows = []
    for architecture in architectures:
        run_seeds = seeds if architecture in multi_seed else [42]
        for seed in run_seeds:
            fixed_depth_architectures = {
                "dense",
                "dense_param_match",
                "structure_only",
                "uniform_token_loop",
                "uniform_region_loop",
            }
            budget_lambda = (
                0.0
                if architecture in fixed_depth_architectures
                else base.budget_lambda
            )
            config = replace(
                base,
                experiment_id="placeholder",
                architecture=architecture,
                seed=int(seed),
                epochs=int(epochs),
                output_dir="checkpoints/confirmatory_7b",
                budget_lambda=budget_lambda,
            )
            experiment_id = _canonical_experiment_id(config)
            config = replace(config, experiment_id=experiment_id)
            config_path = config_dir / f"{experiment_id}.yaml"
            if not dry_run:
                _atomic_text(config_path, yaml.safe_dump(config.__dict__, sort_keys=False))
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "architecture": architecture,
                    "backbone": primary["backbones"]["primary"],
                    "seed": seed,
                    "dataset": "XLCoST-program",
                    "structure": STRUCTURE[architecture],
                    "router": ROUTER.get(architecture, "region-adaptive"),
                    "Rmax": config.max_recurrence,
                    "compute_budget": "matched-by-frozen-validation-calibration",
                    "training_status": "PLANNED",
                    "evaluation_status": "PLANNED",
                    "checkpoint_path": f"{config.output_dir}/{experiment_id}/best.pt",
                    "metrics_path": f"{config.output_dir}/{experiment_id}/metrics.jsonl",
                    "git_commit": git_commit,
                    "config_path": (
                        str(config_path.relative_to(ROOT))
                        if config_path.is_relative_to(ROOT)
                        else str(config_path)
                    ),
                    "training_horizon_policy": f"shared-{epochs}-epoch-7B-horizon",
                    "notes": "Three-seed confirmatory"
                    if architecture in multi_seed
                    else "Seed-42 screening only",
                }
            )
    if dry_run:
        return rows
    existing = []
    if registry_path.exists():
        with registry_path.open(encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    # Retain non-7B plans and any real run records; replace only pseudo-ID 7B plans.
    retained = [
        row
        for row in existing
        if not (
            row.get("experiment_id", "").startswith("plan-")
            and row.get("backbone") == primary["backbones"]["primary"]
            and "pilot" not in row.get("training_horizon_policy", "")
        )
    ]
    by_id = {row["experiment_id"]: row for row in retained}
    for row in rows:
        by_id[row["experiment_id"]] = row
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(by_id.values())
    _atomic_text(registry_path, buffer.getvalue())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=Path, default=ROOT / "configs/training/horizon.yaml")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/training/pilot_sart_external_clean.yaml",
    )
    parser.add_argument("--registry", type=Path, default=ROOT / "experiments/registry.csv")
    parser.add_argument(
        "--config-dir", type=Path, default=ROOT / "configs/generated/confirmatory_7b"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = prepare(
        args.horizon, args.base_config, args.registry, args.config_dir, dry_run=args.dry_run
    )
    print(
        json.dumps({"run_count": len(rows), "experiment_ids": [r["experiment_id"] for r in rows]})
    )


if __name__ == "__main__":
    main()
