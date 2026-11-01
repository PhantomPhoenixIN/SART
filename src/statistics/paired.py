"""Problem-clustered paired inference for deterministic code translation outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import binomtest


def build_outcome_tensor(
    rows: Iterable[dict[str, Any]],
    model_a: str,
    model_b: str,
    *,
    benchmark: str,
    split: str = "common_core",
    metric: str = "pass_at_1",
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Return complete paired outcomes for a binary or bounded task metric.

    Rows must contain one deterministic task outcome for each model, seed, problem and
    direction. Missing, duplicate, or non-binary observations fail explicitly.
    """
    selected = [
        row
        for row in rows
        if row.get("benchmark") == benchmark
        and row.get("split") == split
        and row.get("architecture", row.get("model")) in {model_a, model_b}
    ]
    if not selected:
        raise ValueError("No rows match the requested benchmark, split and model IDs")
    models = [model_a, model_b]
    seeds = sorted({int(row["seed"]) for row in selected})
    problems = sorted({str(row.get("problem_id", row.get("src_uid"))) for row in selected})
    directions = sorted({str(row["direction"]) for row in selected})
    if len(directions) != 6:
        raise ValueError(f"Expected all six translation directions, found {len(directions)}")
    indexes = {
        model: {value: i for i, value in enumerate(values)}
        for model, values in (
            ("architecture", models),
            ("seed", seeds),
            ("problem_id", problems),
            ("direction", directions),
        )
    }
    outcomes = np.full((2, len(seeds), len(problems), len(directions)), np.nan, dtype=np.float64)
    for row in selected:
        problem = str(row.get("problem_id", row.get("src_uid")))
        passed = row.get(metric)
        if not isinstance(passed, int | float | bool | np.number) or not 0 <= float(passed) <= 1:
            raise ValueError(f"{metric} must be a numeric task score in [0, 1]: {row}")
        index = (
            indexes["architecture"][row.get("architecture", row.get("model"))],
            indexes["seed"][int(row["seed"])],
            indexes["problem_id"][problem],
            indexes["direction"][str(row["direction"])],
        )
        if not np.isnan(outcomes[index]):
            raise ValueError(f"Duplicate deterministic task outcome: {index}")
        outcomes[index] = float(passed)
    missing = int(np.isnan(outcomes).sum())
    if missing:
        raise ValueError(f"Paired evaluation is incomplete: {missing} model/task cells missing")
    return outcomes, problems, directions, seeds


def macro_difference(outcomes: np.ndarray) -> float:
    """Compute model-A minus model-B task score, equally weighted by direction."""
    if outcomes.ndim != 4 or outcomes.shape[0] != 2:
        raise ValueError("outcomes must have shape [2, seeds, problems, directions]")
    return float((outcomes[0] - outcomes[1]).mean(axis=(0, 1)).mean())


def hierarchical_bootstrap(
    outcomes: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Resample seeds and problem IDs, retaining all associated directions."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    _, seed_count, problem_count, direction_count = outcomes.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    difference = outcomes[0] - outcomes[1]
    for replicate in range(replicates):
        sampled_seeds = rng.integers(seed_count, size=seed_count)
        sampled_problems = rng.integers(problem_count, size=problem_count)
        sample = difference[np.ix_(sampled_seeds, sampled_problems, np.arange(direction_count))]
        estimates[replicate] = sample.mean(axis=(0, 1)).mean()
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    point = macro_difference(outcomes)
    baseline = float(outcomes[1].mean(axis=(0, 1)).mean())
    return {
        "difference": point,
        "difference_percentage_points": point * 100.0,
        "ci95": [float(lower), float(upper)],
        "ci95_percentage_points": [float(lower * 100.0), float(upper * 100.0)],
        "relative_error_reduction": point / (1.0 - baseline) if baseline < 1.0 else None,
        "replicates": replicates,
        "bootstrap_seed": seed,
        "baseline_mean_task_score": baseline,
        "model_a_mean_task_score": float(outcomes[0].mean(axis=(0, 1)).mean()),
    }


def paired_cluster_randomization(
    outcomes: np.ndarray,
    *,
    replicates: int = 100_000,
    seed: int = 43,
) -> dict[str, Any]:
    """Two-sided paired sign-swap test with one swap per problem cluster."""
    if replicates < 1:
        raise ValueError("replicates must be positive")
    problem_effects = (outcomes[0] - outcomes[1]).mean(axis=(0, 2))
    observed = float(problem_effects.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, replicates, 10_000):
        count = min(10_000, replicates - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(problem_effects)))
        randomized = signs @ problem_effects / len(problem_effects)
        extreme += int(np.count_nonzero(np.abs(randomized) >= abs(observed)))
    p_value = (extreme + 1) / (replicates + 1)
    return {
        "difference": observed,
        "p_value_two_sided": p_value,
        "replicates": replicates,
        "randomization_seed": seed,
        "cluster_count": int(len(problem_effects)),
    }


def mcnemar_by_direction_seed(
    outcomes: np.ndarray, directions: list[str], seeds: list[int]
) -> list[dict[str, Any]]:
    """Report exact paired McNemar tests as secondary per-direction analyses."""
    rows = []
    for seed_index, seed_value in enumerate(seeds):
        for direction_index, direction in enumerate(directions):
            a = outcomes[0, seed_index, :, direction_index].astype(bool)
            b = outcomes[1, seed_index, :, direction_index].astype(bool)
            a_only = int(np.count_nonzero(a & ~b))
            b_only = int(np.count_nonzero(~a & b))
            discordant = a_only + b_only
            p_value = (
                float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
            )
            rows.append(
                {
                    "seed": seed_value,
                    "direction": direction,
                    "both_correct": int(np.count_nonzero(a & b)),
                    "a_correct_b_wrong": a_only,
                    "a_wrong_b_correct": b_only,
                    "both_wrong": int(np.count_nonzero(~a & ~b)),
                    "exact_mcnemar_p": p_value,
                }
            )
    return rows


def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values keyed by comparison name."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_value))
        adjusted[name] = running
    return adjusted
