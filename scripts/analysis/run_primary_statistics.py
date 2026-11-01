#!/usr/bin/env python3
"""Run the predeclared problem-clustered analysis on frozen benchmark scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.statistics.paired import (
    build_outcome_tensor,
    hierarchical_bootstrap,
    holm_bonferroni,
    mcnemar_by_direction_seed,
    paired_cluster_randomization,
)

PRIMARY_COMPARISONS = {
    "full_sart_vs_dense": ("full_sart", "dense"),
    "full_sart_vs_uniform_token_loop": ("full_sart", "uniform_token_loop"),
    "full_sart_vs_mor_style": ("full_sart", "mor_style"),
    "full_sart_vs_random_region": ("full_sart", "random_region"),
    "full_sart_vs_structure_only": ("full_sart", "structure_only"),
}
PRIMARY_BACKBONE_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"


def _read_score_rows(
    root: Path,
    *,
    benchmark: str = "xcodeeval",
    split: str = "common_core",
    backbone_revision: str = PRIMARY_BACKBONE_REVISION,
) -> list[dict[str, Any]]:
    rows = []
    architectures = {"full_sart", "dense", "uniform_token_loop", "mor_style",
                     "random_region", "structure_only"}
    for architecture in sorted(architectures):
        pattern = (
            f"{backbone_revision}/{architecture}-seed*/{benchmark}/{split}/*/*.json"
        )
        for path in sorted(root.glob(pattern)):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "pass_at_1" in value and "task_id" in value:
                rows.append({key: value[key] for key in (
                    "benchmark", "split", "backbone", "architecture", "seed",
                    "problem_id", "direction", "pass_at_1",
                )})
    return rows


def _complete_metric_problem_rows(
    rows: list[dict[str, Any]],
    model_a: str,
    model_b: str,
    *,
    benchmark: str,
    split: str,
    metric: str,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Keep only problems with an available paired metric for every seed/direction."""
    selected = [
        row
        for row in rows
        if row.get("benchmark") == benchmark
        and row.get("split") == split
        and row.get("architecture", row.get("model")) in {model_a, model_b}
    ]
    if not selected:
        return [], {"total_problem_count": 0, "complete_problem_count": 0,
                    "complete_problem_fraction": 0.0}
    seeds = sorted({int(row["seed"]) for row in selected})
    directions = sorted({str(row["direction"]) for row in selected})
    problems = sorted({str(row.get("problem_id", row.get("src_uid"))) for row in selected})
    if len(directions) != 6:
        raise ValueError(f"Expected all six translation directions, found {len(directions)}")
    model_ids = (model_a, model_b)
    indexed = {
        (
            row.get("architecture", row.get("model")),
            int(row["seed"]),
            str(row.get("problem_id", row.get("src_uid"))),
            str(row["direction"]),
        ): row
        for row in selected
    }
    complete = []
    for problem in problems:
        keys = [
            (model, seed, problem, direction)
            for model in model_ids
            for seed in seeds
            for direction in directions
        ]
        if all(
            key in indexed
            and isinstance(indexed[key].get(metric), (int, float, bool))
            for key in keys
        ):
            complete.append(problem)
    complete_set = set(complete)
    filtered = [
        row for row in selected
        if str(row.get("problem_id", row.get("src_uid"))) in complete_set
    ]
    total = len(problems)
    return filtered, {
        "total_problem_count": total,
        "complete_problem_count": len(complete),
        "complete_problem_fraction": len(complete) / total if total else 0.0,
    }


def analyze(
    rows: list[dict[str, Any]],
    *,
    benchmark: str = "xcodeeval",
    split: str = "common_core",
    backbone_revision: str = PRIMARY_BACKBONE_REVISION,
    bootstrap_replicates: int = 10_000,
    randomization_replicates: int = 100_000,
) -> dict[str, Any]:
    """Compute primary contrasts, clustered inference and secondary McNemar tables."""
    # The confirmatory question is the pinned Qwen2.5-Coder-7B core.  Filter
    # before architecture/seed indexing: the 1.5B matrix intentionally repeats
    # Dense, MoR-style and Full-SART seeds, while the family checks reuse those
    # architecture names on other backbones.
    rows = [
        row for row in rows
        if Path(str(row.get("backbone", ""))).name == backbone_revision
    ]
    results: dict[str, Any] = {
        "benchmark": benchmark,
        "split": split,
        "backbone_revision": backbone_revision,
        "primary_aggregate": "equal_weight_macro_over_six_directions",
        "cluster_unit": "problem_id",
        "comparisons": {},
    }
    p_values: dict[str, float] = {}
    for metric in ("pass_at_1",):
        if metric == "pass_ratio_at_5":
            matching = [
                row
                for row in rows
                if row.get("benchmark") == benchmark and row.get("split") == split
            ]
            if matching and all(row.get(metric) is None for row in matching):
                for comparison, (model_a, model_b) in PRIMARY_COMPARISONS.items():
                    results["comparisons"][f"{metric}:{comparison}"] = {
                        "metric": metric,
                        "status": "unavailable",
                        "reason": "benchmark_scoring_does_not_expose_per_test_ratios",
                        "model_a": model_a,
                        "model_b": model_b,
                    }
                continue
        for comparison, (model_a, model_b) in PRIMARY_COMPARISONS.items():
            metric_rows = rows
            availability = None
            if metric == "pass_ratio_at_5":
                metric_rows, availability = _complete_metric_problem_rows(
                    rows,
                    model_a,
                    model_b,
                    benchmark=benchmark,
                    split=split,
                    metric=metric,
                )
                if not availability["complete_problem_count"]:
                    results["comparisons"][f"{metric}:{comparison}"] = {
                        "metric": metric,
                        "status": "unavailable",
                        "reason": "no_complete_paired_problems_with_per_test_outcomes",
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric_availability": availability,
                    }
                    continue
            tensor, problems, directions, seeds = build_outcome_tensor(
                metric_rows,
                model_a,
                model_b,
                benchmark=benchmark,
                split=split,
                metric=metric,
            )
            bootstrap = hierarchical_bootstrap(tensor, replicates=bootstrap_replicates)
            randomization = paired_cluster_randomization(
                tensor, replicates=randomization_replicates
            )
            if metric == "pass_at_1":
                p_values[f"{metric}:{comparison}"] = randomization["p_value_two_sided"]
            by_direction = {}
            for index, direction in enumerate(directions):
                a = float(tensor[0, :, :, index].mean())
                b = float(tensor[1, :, :, index].mean())
                by_direction[direction] = {
                    "model_a_score": a,
                    "model_b_score": b,
                    "difference": a - b,
                }
            entry = {
                "metric": metric,
                "multiplicity_status": (
                    "confirmatory_holm_family"
                    if metric == "pass_at_1"
                    else "exploratory_not_in_holm_family"
                ),
                "model_a": model_a,
                "model_b": model_b,
                "problem_count": len(problems),
                "seeds": seeds,
                "directions": directions,
                "bootstrap": bootstrap,
                "effect_size_percentage_points": bootstrap["difference_percentage_points"],
                "paired_cluster_randomization": randomization,
                "direction_results": by_direction,
            }
            if availability is not None:
                entry["metric_availability"] = {
                    **availability,
                    "omitted_problem_count": (
                        availability["total_problem_count"]
                        - availability["complete_problem_count"]
                    ),
                }
            if metric == "pass_at_1":
                entry["relative_error_reduction"] = bootstrap["relative_error_reduction"]
            if metric == "pass_at_1":
                entry["mcnemar_by_direction_seed"] = mcnemar_by_direction_seed(
                    tensor, directions, seeds
                )
            results["comparisons"][f"{metric}:{comparison}"] = entry
    results["holm_adjusted_p_values"] = holm_bonferroni(p_values)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", default="xcodeeval")
    parser.add_argument("--split", default="common_core")
    parser.add_argument("--backbone-revision", default=PRIMARY_BACKBONE_REVISION)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--randomization-replicates", type=int, default=100_000)
    args = parser.parse_args()
    report = analyze(
        _read_score_rows(
            args.results_root,
            benchmark=args.benchmark,
            split=args.split,
            backbone_revision=args.backbone_revision,
        ),
        benchmark=args.benchmark,
        split=args.split,
        backbone_revision=args.backbone_revision,
        bootstrap_replicates=args.bootstrap_replicates,
        randomization_replicates=args.randomization_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["holm_adjusted_p_values"], sort_keys=True))


if __name__ == "__main__":
    main()
