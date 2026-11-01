#!/usr/bin/env python3
"""Join frozen validation compute profiles to xCodeEval Common-Core Pass@1 results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(root: Path, pattern: str = "**/*.json"):
    for path in sorted(root.glob(pattern)):
        if path.name in {"run_manifest.json", "score_manifest.json"}:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            yield path, row


def analyze(profiles_root: Path, scores_root: Path) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    for _path, row in _rows(profiles_root):
        if row.get("experiment_id") and "estimated_total_flops" in row:
            exp_id = str(row["experiment_id"])
            if exp_id in profiles:
                raise ValueError(f"Duplicate compute profile for experiment {exp_id}")
            profiles[exp_id] = row
    score_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for _path, row in _rows(scores_root, "*/*/xcodeeval/common_core/*/*.json"):
        if row.get("benchmark") != "xcodeeval" or row.get("split") != "common_core":
            continue
        if "pass_at_1" not in row or not row.get("experiment_id"):
            continue
        key = (str(row["backbone"]), str(row["architecture"]), str(row["experiment_id"]))
        score_groups[key].append({field: row[field] for field in (
            "experiment_id", "architecture", "backbone", "seed", "direction",
            "problem_id", "pass_at_1",
        )})
    if not score_groups:
        return {"status": "awaiting_xcodeeval_common_core_scores", "model_groups": []}

    summaries = []
    by_backbone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending = []
    for (backbone, architecture, experiment_id), rows in sorted(score_groups.items()):
        profile = profiles.get(experiment_id)
        directions = sorted({str(row["direction"]) for row in rows})
        if profile is None or len(directions) != 6:
            pending.append({
                "experiment_id": experiment_id,
                "backbone": backbone,
                "architecture": architecture,
                "profile_available": profile is not None,
                "direction_count": len(directions),
            })
            continue
        direction_scores = {}
        for direction in directions:
            values = [float(row["pass_at_1"]) for row in rows if row["direction"] == direction]
            direction_scores[direction] = sum(values) / len(values)
        summary = {
            "experiment_id": experiment_id,
            "architecture": architecture,
            "seed": int(rows[0]["seed"]),
            "backbone": backbone,
            "direction_count": len(direction_scores),
            "problem_count": len({str(row["problem_id"]) for row in rows}),
            "pass_at_1_macro_six_directions": sum(direction_scores.values()) / 6,
            "direction_pass_at_1": direction_scores,
            "estimated_total_flops": float(profile["estimated_total_flops"]),
            "estimated_prefill_flops": float(profile["estimated_prefill_flops"]),
            "estimated_recurrent_module_flops": float(profile["estimated_recurrent_module_flops"]),
            "estimated_decode_flops": float(profile["estimated_decode_flops"]),
            "end_to_end_fixed_device_ms": float(profile["end_to_end_fixed_device_ms"]),
            "prefill_device_ms": float(profile["prefill_device_ms"]),
            "recurrent_module_device_ms": float(profile["recurrent_module_device_ms"]),
            "cached_decode_device_ms": float(profile["cached_decode_device_ms"]),
            "peak_vram_bytes": int(profile["peak_vram_bytes"]),
            "estimated_flops_per_source_token": float(profile["estimated_flops_per_source_token"]),
            "estimated_decode_flops_per_generated_token": float(profile["estimated_decode_flops_per_generated_token"]),
            "parameter_count": int(profile.get("parameter_count", 0)),
            "trainable_parameter_count": int(profile.get("trainable_parameter_count", 0)),
            "adapter_parameter_count": int(profile.get("adapter_parameter_count", 0)),
            "router_parameter_count": int(profile.get("router_parameter_count", 0)),
            "profile_sample_count": int(profile.get("sample_count", 1)),
            "profile_sample_plan_sha256": profile.get("sample_plan_sha256"),
        }
        summaries.append(summary)
        by_backbone[backbone].append(summary)

    backbone_results = {}
    baselines = ("dense", "uniform_token_loop", "mor_style", "random_region", "structure_only")
    for backbone, rows in sorted(by_backbone.items()):
        arch_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            arch_groups[row["architecture"]].append(row)
        architecture_summary = {}
        for architecture, items in sorted(arch_groups.items()):
            architecture_summary[architecture] = {
                "seed_count": len(items),
                "seeds": sorted(item["seed"] for item in items),
                "pass_at_1_macro_mean_over_seeds": statistics.mean(item["pass_at_1_macro_six_directions"] for item in items),
                "median_estimated_total_flops": statistics.median(item["estimated_total_flops"] for item in items),
                "median_end_to_end_fixed_device_ms": statistics.median(item["end_to_end_fixed_device_ms"] for item in items),
                "median_peak_vram_bytes": statistics.median(item["peak_vram_bytes"] for item in items),
            }
        comparisons = {}
        sart = architecture_summary.get("full_sart")
        if sart:
            for baseline in baselines:
                other = architecture_summary.get(baseline)
                if not other:
                    continue
                relative_flops = (
                    sart["median_estimated_total_flops"] / other["median_estimated_total_flops"] - 1
                    if other["median_estimated_total_flops"] else None
                )
                comparisons[f"full_sart_vs_{baseline}"] = {
                    "pass_at_1_difference_percentage_points": 100 * (
                        sart["pass_at_1_macro_mean_over_seeds"] - other["pass_at_1_macro_mean_over_seeds"]
                    ),
                    "relative_total_flops_difference": relative_flops,
                    "within_five_percent_flops": relative_flops is not None and abs(relative_flops) <= 0.05,
                    "relative_end_to_end_latency_difference": (
                        sart["median_end_to_end_fixed_device_ms"] / other["median_end_to_end_fixed_device_ms"] - 1
                        if other["median_end_to_end_fixed_device_ms"] else None
                    ),
                }
        frontier = []
        for candidate in architecture_summary:
            item = architecture_summary[candidate]
            dominated = any(
                other != candidate
                and architecture_summary[other]["median_estimated_total_flops"] <= item["median_estimated_total_flops"]
                and architecture_summary[other]["pass_at_1_macro_mean_over_seeds"] >= item["pass_at_1_macro_mean_over_seeds"]
                and (
                    architecture_summary[other]["median_estimated_total_flops"] < item["median_estimated_total_flops"]
                    or architecture_summary[other]["pass_at_1_macro_mean_over_seeds"] > item["pass_at_1_macro_mean_over_seeds"]
                )
                for other in architecture_summary
            )
            if not dominated:
                frontier.append(candidate)
        backbone_results[backbone] = {
            "architectures": architecture_summary,
            "full_sart_comparisons": comparisons,
            "pareto_frontier_architectures": sorted(frontier),
        }
    return {
        "status": "analyzed" if summaries else "awaiting_profiles_for_scored_models",
        "benchmark": "xcodeeval", "split": "common_core",
        "model_groups": summaries,
        "backbone_results": backbone_results,
        "pending_experiment_groups": pending,
        "interpretation": "The profile is an in-domain 24-example median; accuracy comes from executable Common-Core scoring. FLOP and latency frontiers are descriptive; seed/problem uncertainty is reported in the primary statistics outputs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles-root", type=Path, default=Path("reports/profiling"))
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.profiles_root, args.scores_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "model_groups": len(report["model_groups"])}, sort_keys=True))


if __name__ == "__main__":
    main()
