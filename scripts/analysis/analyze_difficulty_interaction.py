#!/usr/bin/env python3
"""Analyze the preregistered SART–MoR Pass@1 interaction by frozen difficulty quartile."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PRIMARY_BACKBONE_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"


def read_scores(root: Path, backbone_revision: str = PRIMARY_BACKBONE_REVISION) -> list[dict[str, Any]]:
    rows = []
    for architecture in ("full_sart", "mor_style"):
        pattern = f"{backbone_revision}/{architecture}-seed*/xcodeeval/common_core/*/*.json"
        for path in sorted(root.glob(pattern)):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "pass_at_1" in value and "task_id" in value:
                rows.append({key: value[key] for key in (
                    "task_id", "benchmark", "split", "backbone", "architecture",
                    "seed", "problem_id", "direction", "pass_at_1",
                    "source_sha256", "target_reference_sha256",
                )})
    return rows


def analyze(
    scores: list[dict[str, Any]],
    difficulty_path: Path,
    *,
    backbone_revision: str = PRIMARY_BACKBONE_REVISION,
    replicates: int = 10_000,
) -> dict[str, Any]:
    frozen = json.loads(difficulty_path.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_PREMODEL":
        raise ValueError("Difficulty artifact is not frozen pre-model")
    by_task = {row["task_id"]: row for row in frozen["records"]}
    if len(by_task) != frozen.get("task_count"):
        raise ValueError("Difficulty task IDs are not unique or task_count differs")
    scores = [
        row for row in scores
        if row.get("benchmark") == frozen.get("benchmark")
        and row.get("split") == frozen.get("split")
        and row.get("architecture") in {"full_sart", "mor_style"}
        and Path(str(row.get("backbone", ""))).name == backbone_revision
    ]
    cells: dict[tuple[str, str, str, str], dict[str, bool]] = defaultdict(dict)
    for row in scores:
        task_id = row["task_id"]
        task = by_task.get(task_id)
        if task is None:
            raise ValueError(f"Scored task absent from frozen artifact: {task_id}")
        for score_key, frozen_key in (
            ("source_sha256", "source_sha256"),
            ("target_reference_sha256", "target_reference_sha256"),
        ):
            if row.get(score_key) != task.get(frozen_key):
                raise ValueError(f"{score_key} mismatch for {task_id}")
        model, seed = row["architecture"], str(row["seed"])
        # Artifact labels are e.g. Q1_easiest; the analysis keeps the compact Q1–Q4.
        quartile = str(task["difficulty_quartile"]).split("_", 1)[0]
        if quartile not in {"Q1", "Q2", "Q3", "Q4"}:
            raise ValueError(f"Unknown frozen difficulty quartile: {task['difficulty_quartile']}")
        key = (task["problem_id"], task["direction"], seed, quartile)
        if model in cells[key]:
            raise ValueError(f"Duplicate model/task/seed score for {key} ({model})")
        cells[key][model] = bool(row["pass_at_1"])

    # Retain only matched SART/MoR cells; the frozen Common-Core task set has the
    # same problem clusters and directions across quartiles.
    paired: dict[tuple[str, str, str, str], float] = {
        key: float(value["full_sart"]) - float(value["mor_style"])
        for key, value in cells.items()
        if set(value) == {"full_sart", "mor_style"}
    }
    if not paired:
        return {
            "status": "awaiting_paired_scores",
            "benchmark": frozen["benchmark"], "split": frozen["split"],
            "backbone_revision": backbone_revision,
            "matched_task_seed_cell_count": 0,
            "difficulty_artifact_sha256": hashlib.sha256(difficulty_path.read_bytes()).hexdigest(),
        }
    expected_seeds = {"13", "42", "87"}
    observed_seeds = {seed for _problem, _direction, seed, _quartile in paired}
    if observed_seeds != expected_seeds:
        raise ValueError(f"H3 requires all three confirmatory seeds; found {sorted(observed_seeds)}")
    expected_cells = {
        (row["problem_id"], row["direction"], seed,
         str(row["difficulty_quartile"]).split("_", 1)[0])
        for row in frozen["records"]
        for seed in expected_seeds
    }
    missing_cells = expected_cells - set(paired)
    if missing_cells:
        raise ValueError(f"H3 requires complete paired scores; missing {len(missing_cells)} task-seed cells")
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for (problem, direction, _seed, quartile), delta in paired.items():
        grouped[(direction, quartile)].append((problem, delta))
    direction_results: dict[str, Any] = {}
    quartiles = ("Q1", "Q2", "Q3", "Q4")
    for direction in sorted({d for d, _q in grouped}):
        means = {}
        for q in quartiles:
            vals = [v for _p, v in grouped.get((direction, q), [])]
            means[q] = sum(vals) / len(vals) if vals else None
        direction_results[direction] = {
            "matched_task_seed_cells_by_quartile": {
                q: len(grouped.get((direction, q), [])) for q in quartiles
            },
            "full_sart_minus_mor_pass_at_1": means,
            "q4_minus_q1": (
                means["Q4"] - means["Q1"]
                if means["Q4"] is not None and means["Q1"] is not None else None
            ),
        }
    contrasts = [v["q4_minus_q1"] for v in direction_results.values() if v["q4_minus_q1"] is not None]
    estimate = sum(contrasts) / len(contrasts) if contrasts else None

    # Hierarchically resample seed IDs and problem clusters; keep each problem's
    # direction/task cells together and use one seed draw across all quartiles.
    problem_ids = sorted({problem for problem, _d, _s, _q in paired})
    seed_ids = sorted({seed for _problem, _direction, seed, _quartile in paired})
    rng = random.Random(int(hashlib.sha256(b"SART-H3-xcodeeval-common-core").hexdigest()[:16], 16))
    draws = []
    if problem_ids and seed_ids and estimate is not None:
        for _ in range(replicates):
            sampled_problems = [rng.choice(problem_ids) for _ in problem_ids]
            sampled_seeds = [rng.choice(seed_ids) for _ in seed_ids]
            direction_effects = []
            for direction in direction_results:
                sampled_means = {}
                for q in ("Q1", "Q4"):
                    values = [
                        paired[(problem, direction, seed, q)]
                        for problem in sampled_problems
                        for seed in sampled_seeds
                        if (problem, direction, seed, q) in paired
                    ]
                    sampled_means[q] = sum(values) / len(values) if values else None
                if None not in sampled_means.values():
                    direction_effects.append(sampled_means["Q4"] - sampled_means["Q1"])
            if direction_effects:
                draws.append(sum(direction_effects) / len(direction_effects))
    draws.sort()
    ci = None
    if draws:
        ci = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    return {
        "status": "analyzed" if estimate is not None else "no_matched_scores",
        "benchmark": frozen["benchmark"],
        "split": frozen["split"],
        "backbone_revision": backbone_revision,
        "hypothesis": "H3: Full-SART minus MoR-style Pass@1 grows from Q1 to Q4",
        "difficulty_artifact_sha256": hashlib.sha256(difficulty_path.read_bytes()).hexdigest(),
        "matched_problem_count": len(problem_ids),
        "seeds": seed_ids,
        "matched_task_seed_cell_count": len(paired),
        "macro_equal_direction_q4_minus_q1": estimate,
        "hierarchical_seed_problem_bootstrap_95_percent_interval": ci,
        "bootstrap_replicates": len(draws),
        "direction_results": direction_results,
        "interpretation": "Positive Q4−Q1 supports the preregistered interaction; this is retrospective mechanism analysis, not model selection.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, default=Path("reports/analysis/xcodeeval_common_core_structural_difficulty.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone-revision", default=PRIMARY_BACKBONE_REVISION)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(
        read_scores(args.scores_root, args.backbone_revision), args.difficulty,
        backbone_revision=args.backbone_revision,
        replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "matched_problem_count", "macro_equal_direction_q4_minus_q1")}, sort_keys=True))


if __name__ == "__main__":
    main()
