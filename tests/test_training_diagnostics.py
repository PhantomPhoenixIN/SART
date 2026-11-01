import torch

from src.training.run_translation import _router_validation_stats


def test_router_validation_stats_reports_depth_mass_and_collapse():
    gates = [torch.tensor([[0.5, 0.5, 0.5]])]
    metrics = _router_validation_stats(gates, max_steps=3)
    assert metrics["mean_continue_probability"] == 0.5
    assert metrics["stop_after_first_fraction"] == 0.5
    assert metrics["reach_rmax_fraction"] == 0.25
    assert metrics["router_entropy"] > 0.69
    assert metrics["depth_mass_r1"] == 0.5
    assert metrics["depth_mass_r2"] == 0.25
    assert metrics["depth_mass_r3"] == 0.25


def test_router_validation_stats_empty_model_has_zero_diagnostics():
    metrics = _router_validation_stats([], max_steps=2)
    assert metrics["mean_continue_probability"] == 0.0
    assert metrics["depth_mass_r1"] == 0.0
    assert metrics["depth_mass_r2"] == 0.0
