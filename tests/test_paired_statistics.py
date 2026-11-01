"""Tests for paired problem-level outcome construction and inference."""

import pytest

from src.statistics.paired import (
    build_outcome_tensor,
    hierarchical_bootstrap,
    holm_bonferroni,
    paired_cluster_randomization,
)


def _rows(problem_count: int = 12) -> list[dict]:
    rows = []
    for architecture in ("full_sart", "dense"):
        for seed in (13, 42, 87):
            for problem in range(problem_count):
                for direction in (
                    "Python-to-Java",
                    "Java-to-Python",
                    "Python-to-C++",
                    "C++-to-Python",
                    "Java-to-C++",
                    "C++-to-Java",
                ):
                    passed = architecture == "full_sart" and problem % 3 == 0
                    rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "benchmark": "xcodeeval",
                            "split": "common_core",
                            "problem_id": f"p{problem}",
                            "direction": direction,
                            "pass_at_1": passed,
                            "pass_at_5": architecture == "full_sart"
                            and (passed or problem % 3 == 1),
                            "pass_ratio_at_5": 0.5 if passed else 0.1,
                        }
                    )
    return rows


def test_outcome_tensor_preserves_problem_clusters_and_direction_macro() -> None:
    outcomes, problems, directions, seeds = build_outcome_tensor(
        _rows(), "full_sart", "dense", benchmark="xcodeeval"
    )
    assert outcomes.shape == (2, 3, 12, 6)
    assert len(problems) == 12
    assert len(directions) == 6
    assert seeds == [13, 42, 87]


def test_outcome_tensor_rejects_missing_or_duplicate_observations() -> None:
    rows = _rows()
    rows.pop()
    with pytest.raises(ValueError, match="incomplete"):
        build_outcome_tensor(rows, "full_sart", "dense", benchmark="xcodeeval")
    with pytest.raises(ValueError, match="Duplicate"):
        build_outcome_tensor(_rows() + [_rows()[0]], "full_sart", "dense", benchmark="xcodeeval")


def test_outcome_tensor_supports_pass_at_5_and_pass_ratio() -> None:
    rows = _rows()
    outcomes, *_ = build_outcome_tensor(
        rows,
        "full_sart",
        "dense",
        benchmark="xcodeeval",
        metric="pass_at_5",
    )
    assert outcomes[0, :, 1, :].mean() == pytest.approx(1.0)
    assert outcomes[1, :, 1, :].mean() == pytest.approx(0.0)
    ratio, *_ = build_outcome_tensor(
        rows,
        "full_sart",
        "dense",
        benchmark="xcodeeval",
        metric="pass_ratio_at_5",
    )
    assert ratio.shape == outcomes.shape
    assert ratio.min() >= 0.0 and ratio.max() <= 1.0


def test_clustered_inference_is_deterministic_and_holm_monotone() -> None:
    outcomes, *_ = build_outcome_tensor(_rows(), "full_sart", "dense", benchmark="xcodeeval")
    bootstrap = hierarchical_bootstrap(outcomes, replicates=100, seed=5)
    randomization = paired_cluster_randomization(outcomes, replicates=1000, seed=6)
    assert bootstrap["difference"] == pytest.approx(1 / 3)
    assert bootstrap["ci95"][0] <= bootstrap["difference"] <= bootstrap["ci95"][1]
    assert randomization["cluster_count"] == 12
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}
