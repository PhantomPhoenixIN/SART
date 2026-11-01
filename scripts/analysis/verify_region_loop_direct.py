#!/usr/bin/env python3
"""Recompute exploratory paired Region Loop contrasts from Protocol v3 task scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.statistics.paired import (
    build_outcome_tensor,
    hierarchical_bootstrap,
    holm_bonferroni,
    paired_cluster_randomization,
)

REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"
COMPARATORS = ("uniform_token_loop", "dense", "full_sart", "mor_style")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for mode in ("uniform_region_loop", *COMPARATORS):
        for path in args.results_root.glob(
            f"{REVISION}/{mode}-seed*/xcodeeval/common_core/*/*.json"
        ):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "task_id" in value:
                rows.append({key: value[key] for key in (
                    "architecture", "seed", "benchmark", "split", "problem_id",
                    "direction", "pass_at_1",
                )})
    results = {"benchmark": "xcodeeval", "split": "common_core",
               "scope": "exploratory direct paired Region Loop comparisons", "comparisons": {}}
    p_values = {}
    for mode in COMPARATORS:
        tensor, problems, directions, seeds = build_outcome_tensor(
            rows, "uniform_region_loop", mode, benchmark="xcodeeval"
        )
        if seeds != [13, 42, 87] or len(problems) != 332 or len(directions) != 6:
            raise ValueError(f"Incomplete frozen Common-Core comparison: {mode}")
        bootstrap = hierarchical_bootstrap(tensor, replicates=10_000)
        randomization = paired_cluster_randomization(tensor, replicates=100_000)
        results["comparisons"][mode] = {
            "problem_count": len(problems), "task_pairs": int(tensor.shape[1] * tensor.shape[2] * tensor.shape[3]),
            "seeds": seeds, "region_loop_mean": float(tensor[0].mean()),
            "comparator_mean": float(tensor[1].mean()),
            "difference_pp": bootstrap["difference_percentage_points"],
            "ci95_pp": bootstrap["ci95_percentage_points"],
            "p_two_sided": randomization["p_value_two_sided"],
            "per_seed_difference_pp": {
                str(seed): float((tensor[0, index] - tensor[1, index]).mean() * 100)
                for index, seed in enumerate(seeds)
            },
        }
        p_values[mode] = randomization["p_value_two_sided"]
    results["holm_adjusted_p_values_exploratory_family"] = holm_bonferroni(p_values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
