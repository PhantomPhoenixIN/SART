"""Tests for validation-only recurrent halting threshold selection."""

import pytest

from scripts.evaluate.calibrate_routing import select_threshold


def test_threshold_selector_optimizes_loss_only_within_depth_budget() -> None:
    observations = [
        {"threshold": 0.2, "validation_target_loss": 0.1, "mean_recurrent_depth": 4.2},
        {"threshold": 0.5, "validation_target_loss": 0.2, "mean_recurrent_depth": 3.01},
        {"threshold": 0.7, "validation_target_loss": 0.15, "mean_recurrent_depth": 2.9},
    ]
    selected = select_threshold(observations, budget=3.0, tolerance=0.05)
    assert selected["threshold"] == 0.7


def test_threshold_ties_prefer_lower_depth_then_lower_threshold() -> None:
    observations = [
        {"threshold": 0.4, "validation_target_loss": 0.1, "mean_recurrent_depth": 2.8},
        {"threshold": 0.5, "validation_target_loss": 0.1, "mean_recurrent_depth": 2.7},
    ]
    selected = select_threshold(observations, budget=3.0)
    assert selected["threshold"] == 0.5


def test_threshold_selector_rejects_grid_without_budget_feasible_point() -> None:
    observations = [{"threshold": 0.2, "validation_target_loss": 0.1, "mean_recurrent_depth": 4.2}]
    with pytest.raises(ValueError, match="satisfies"):
        select_threshold(observations, budget=3.0)
