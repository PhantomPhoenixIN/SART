#!/usr/bin/env python3
"""Analyze predeclared H5 structure controls on paired Common-Core problems."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.dispatch_external_predictions import model_runs  # noqa: E402
from scripts.evaluate.dispatch_external_scoring import result_root  # noqa: E402
from src.statistics.paired import build_outcome_tensor, hierarchical_bootstrap  # noqa: E402

PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
REGISTRY = ROOT / "experiments/registry.csv"
REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
CONTROLS = ("region_adaptive_no_graph", "random_region", "shuffled_structure")


def _score_rows(experiment_id: str) -> tuple[list[dict[str, Any]], Path]:
    root = result_root(REVISION, experiment_id, "xcodeeval", "common_core")
    manifest_path = root / "score_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "COMPLETE"
        or manifest.get("tasks_total") != 1992
        or manifest.get("tasks_completed") != 1992
        or manifest.get("candidate_scoring_version") != "pass_at_1_single_sample_v1"
    ):
        raise ValueError(f"Incomplete H5 score manifest: {manifest_path}")
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in root.rglob("*.json") if path.name != "score_manifest.json"]
    if len(rows) != 1992 or len({row["task_id"] for row in rows}) != 1992:
        raise ValueError(f"Incomplete H5 task scores: {root}")
    return rows, manifest_path


def analyze() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_PREMODEL" or protocol.get("protocol_version") != 3:
        raise ValueError("H5 requires frozen Protocol v3")
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        runs = model_runs(list(csv.DictReader(stream)))
    selected = [row for row in runs if row.get("seed") == "42"
                and row.get("architecture") in {"full_sart", *CONTROLS}
                and Path(str(row.get("backbone_path", ""))).name == REVISION]
    # The registry's backbone field may be a model name; resolve from the frozen config.
    if len(selected) != 4:
        selected = []
        for row in runs:
            if row.get("seed") != "42" or row.get("architecture") not in {"full_sart", *CONTROLS}:
                continue
            resolved = ROOT / row["checkpoint_path"]
            config = json.loads((resolved.parent / "resolved_config.json").read_text(encoding="utf-8"))
            if Path(str(config["backbone_path"])).name == REVISION:
                selected.append(row)
    if len(selected) != 4 or {row["architecture"] for row in selected} != {"full_sart", *CONTROLS}:
        raise ValueError("H5 requires Full-SART and three structure controls at 7B seed 42")
    by_architecture = {row["architecture"]: row for row in selected}
    rows = []
    manifest_hashes = {}
    flops = {}
    for architecture, row in by_architecture.items():
        score_rows, manifest_path = _score_rows(row["experiment_id"])
        rows.extend(score_rows)
        manifest_hashes[architecture] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        profile_path = ROOT / "reports/profiling" / REVISION / f"{row['experiment_id']}.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("experiment_id") != row["experiment_id"] or profile.get("sample_count") != 24:
            raise ValueError(f"H5 profile does not match {architecture}")
        flops[architecture] = float(profile["estimated_total_flops"])
    comparisons = {}
    for baseline in CONTROLS:
        tensor, problems, directions, seeds = build_outcome_tensor(
            rows, "full_sart", baseline,
            benchmark="xcodeeval", split="common_core", metric="pass_at_1",
        )
        if seeds != [42] or len(problems) != 332 or len(directions) != 6:
            raise ValueError(f"H5 comparison lacks paired six-direction coverage: {baseline}")
        bootstrap = hierarchical_bootstrap(tensor, replicates=10_000)
        relative_flops = flops["full_sart"] / flops[baseline] - 1
        comparisons[f"full_sart_vs_{baseline}"] = {
            "baseline": baseline,
            "seed": 42,
            "problem_count": len(problems),
            "pass_at_1_difference_percentage_points": bootstrap["difference_percentage_points"],
            "hierarchical_problem_bootstrap_95_percent_interval_percentage_points":
                bootstrap["ci95_percentage_points"],
            "relative_total_flops_difference": relative_flops,
            "within_five_percent_flops": abs(relative_flops) <= 0.05,
            "full_sart_pass_at_1": bootstrap["model_a_mean_task_score"],
            "baseline_pass_at_1": bootstrap["baseline_mean_task_score"],
        }
    return {
        "status": "analyzed",
        "hypothesis": "H5 true structure contributes beyond arbitrary grouping and graph-free routing",
        "benchmark": "xcodeeval", "split": "common_core",
        "backbone_revision": REVISION,
        "seed": 42,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "score_manifest_sha256": manifest_hashes,
        "comparisons": comparisons,
        "interpretation_limit": "Single-seed H5 controls are descriptive/secondary; matched-compute support requires the recorded ±5% FLOP tolerance. Confidence intervals resample problems but cannot represent seed variation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
