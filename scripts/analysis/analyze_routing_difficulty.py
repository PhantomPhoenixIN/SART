#!/usr/bin/env python3
"""Join Full-SART per-region routes to frozen source/reference difficulty and scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

PRIMARY_BACKBONE_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
CONFIRMATORY_SEEDS = {"13", "42", "87"}


def _json_rows(root: Path, pattern: str):
    for path in sorted(root.glob(pattern)):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            yield path, row


def analyze(
    scores_root: Path,
    predictions_root: Path,
    difficulty_path: Path,
    region_difficulty_path: Path,
    *,
    backbone_revision: str = PRIMARY_BACKBONE_REVISION,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    difficulty = json.loads(difficulty_path.read_text(encoding="utf-8"))
    region_difficulty = json.loads(region_difficulty_path.read_text(encoding="utf-8"))
    if difficulty.get("status") != "FROZEN_PREMODEL" or region_difficulty.get("status") != "FROZEN_PREMODEL":
        raise ValueError("H4 analysis requires frozen pre-model task and region difficulty artifacts")
    if difficulty.get("task_manifest_sha256") != region_difficulty.get("task_manifest_sha256"):
        raise ValueError("H3/H4 difficulty artifacts refer to different task manifests")
    tasks = {row["task_id"]: row for row in difficulty["records"]}
    regions = {row["task_id"]: row for row in region_difficulty["records"]}
    if len(tasks) != len(difficulty["records"]) or len(regions) != len(region_difficulty["records"]):
        raise ValueError("Duplicate task IDs in frozen difficulty artifacts")
    if tasks.keys() != regions.keys() or len(tasks) != 1992:
        raise ValueError("H4 requires the same 1,992 frozen tasks in both difficulty artifacts")
    predictions_by_digest: dict[str, dict[str, Any]] = {}
    pattern = f"{backbone_revision}/full_sart-seed*/xcodeeval/common_core/*/*.json"
    for path, row in _json_rows(predictions_root, pattern):
        if "region_diagnostics" not in row:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        predictions_by_digest[digest] = row

    xs: list[float] = []
    ys: list[float] = []
    cluster_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    problem_clusters: set[str] = set()
    seed_clusters: set[str] = set()
    pass_counts = {False: 0, True: 0}
    by_direction: dict[str, list[tuple[float, float]]] = defaultdict(list)
    matched_tasks = set()
    for _, score in _json_rows(scores_root, pattern):
        if (score.get("benchmark") != "xcodeeval" or score.get("split") != "common_core"
                or score.get("architecture") != "full_sart"
                or Path(str(score.get("backbone", ""))).name != backbone_revision):
            continue
        prediction = predictions_by_digest.get(score.get("prediction_record_sha256"))
        if prediction is None:
            continue
        task_id = score["task_id"]
        task, region_task = tasks.get(task_id), regions.get(task_id)
        route = prediction["region_diagnostics"]
        if task is None or region_task is None:
            raise ValueError(f"Scored prediction is absent from frozen difficulty artifacts: {task_id}")
        for key, value in (("source_sha256", task["source_sha256"]),
                            ("target_reference_sha256", task["target_reference_sha256"])):
            expected = region_task[key] if key == "source_sha256" else region_task["target_reference_sha256"]
            if score.get(key) != value or prediction.get(key) != value or expected != value:
                raise ValueError(f"Frozen code hash mismatch for {key}: {task_id}")
        if route.get("source_sha256") != task["source_sha256"]:
            raise ValueError(f"Region route belongs to different source code: {task_id}")
        route_rows = {int(row["region_id"]): row for row in route.get("regions", [])}
        if len(route_rows) != len(route.get("regions", [])):
            raise ValueError(f"Duplicate routed region IDs: {task_id}")
        pair = (str(score["seed"]), task_id)
        if pair in matched_tasks:
            raise ValueError(f"Duplicate Full-SART seed/task score: {pair}")
        matched_tasks.add(pair)
        for region in region_task["regions"]:
            region_id = int(region["source_region_id"])
            routed = route_rows.get(region_id)
            if routed is None:
                raise ValueError(f"Missing route for frozen source region {region_id}: {task_id}")
            if routed["source_features"].get("start_byte") != region["source_features"].get("start_byte"):
                raise ValueError(f"Region span mismatch for {task_id} region {region_id}")
            x, y = float(routed["expected_depth"]), float(region["structural_difficulty"])
            index = len(xs)
            xs.append(x)
            ys.append(y)
            problem = str(task["problem_id"])
            seed = str(score["seed"])
            cluster_rows[(problem, seed)].append(index)
            problem_clusters.add(problem)
            seed_clusters.add(seed)
            pass_counts[bool(score["pass_at_1"])] += 1
            by_direction[task["direction"]].append((x, y))
    if not xs:
        return {
            "status": "awaiting_scored_full_sart_predictions",
            "benchmark": "xcodeeval", "split": "common_core",
            "backbone_revision": backbone_revision,
            "matched_full_sart_experiment_task_pairs": 0,
            "per_region_observations": 0,
            "difficulty_artifact_sha256": hashlib.sha256(difficulty_path.read_bytes()).hexdigest(),
            "region_difficulty_artifact_sha256": hashlib.sha256(region_difficulty_path.read_bytes()).hexdigest(),
        }

    expected_pairs = {(seed, task_id) for seed in CONFIRMATORY_SEEDS for task_id in tasks}
    if matched_tasks != expected_pairs:
        missing = len(expected_pairs - matched_tasks)
        unexpected = len(matched_tasks - expected_pairs)
        raise ValueError(f"H4 requires all three Full-SART seeds and 1,992 tasks per seed: missing={missing}, unexpected={unexpected}")

    x_values = np.asarray(xs)
    y_values = np.asarray(ys)
    observed = spearmanr(x_values, y_values)
    unique_clusters = sorted(problem_clusters)
    seed_ids = sorted(seed_clusters)
    rng = random.Random(int(hashlib.sha256(b"SART-H4-xcodeeval-common-core").hexdigest()[:16], 16))
    bootstrap = []
    for _ in range(bootstrap_replicates):
        selected_clusters = [rng.choice(unique_clusters) for _ in unique_clusters]
        selected_seeds = [rng.choice(seed_ids) for _ in seed_ids]
        indexes = [
            i for cluster in selected_clusters for seed in selected_seeds
            for i in cluster_rows.get((cluster, seed), [])
        ]
        if len(indexes) < 3:
            continue
        rho = spearmanr(x_values[indexes], y_values[indexes]).statistic
        if np.isfinite(rho):
            bootstrap.append(float(rho))
    bootstrap.sort()
    ci = None if not bootstrap else [
        bootstrap[int(0.025 * (len(bootstrap) - 1))],
        bootstrap[int(0.975 * (len(bootstrap) - 1))],
    ]
    directions = {
        direction: {
            "region_count": len(values),
            "spearman_rho": float(spearmanr([v[0] for v in values], [v[1] for v in values]).statistic),
        }
        for direction, values in sorted(by_direction.items())
    }
    return {
        "status": "analyzed",
        "benchmark": "xcodeeval", "split": "common_core",
        "backbone_revision": backbone_revision,
        "hypothesis": "H4: Full-SART expected recurrent depth is positively associated with source-to-reference region structural difficulty",
        "matched_full_sart_experiment_task_pairs": len(matched_tasks),
        "per_region_observations": len(xs), "problem_cluster_count": len(unique_clusters),
        "seeds": seed_ids,
        "spearman_rho": float(observed.statistic), "naive_p_value_descriptive_only": float(observed.pvalue),
        "hierarchical_seed_problem_bootstrap_95_percent_interval": ci,
        "bootstrap_replicates": len(bootstrap), "direction_results": directions,
        "pass_at_1": {str(value): pass_counts[value] for value in (False, True)},
        "difficulty_artifact_sha256": hashlib.sha256(difficulty_path.read_bytes()).hexdigest(),
        "region_difficulty_artifact_sha256": hashlib.sha256(region_difficulty_path.read_bytes()).hexdigest(),
        "interpretation": "Association is retrospective and not causal. The region assignment is a structural proxy, not semantic translation alignment.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, default=Path("reports/analysis/xcodeeval_common_core_structural_difficulty.json"))
    parser.add_argument("--region-difficulty", type=Path, default=Path("reports/analysis/xcodeeval_common_core_region_difficulty.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-revision", default=PRIMARY_BACKBONE_REVISION)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    result = analyze(args.scores_root, args.predictions_root, args.difficulty, args.region_difficulty,
                     backbone_revision=args.backbone_revision,
                     bootstrap_replicates=args.bootstrap_replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "matched_full_sart_experiment_task_pairs", "per_region_observations")}, sort_keys=True))


if __name__ == "__main__":
    main()
