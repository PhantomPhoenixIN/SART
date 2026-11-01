"""Verify that hard routing skips recurrent-core execution after halting."""

import torch

from src.models.recurrent.region_module import RecurrentRegionCore, RegionBatch


def _region_batch() -> RegionBatch:
    return RegionBatch(
        token_to_region=torch.tensor([[0, 0, 1, 1]]),
        region_mask=torch.tensor([[True, True]]),
        region_type=torch.tensor([[1, 4]]),
        ast_depth=torch.zeros((1, 2)),
        child_count=torch.zeros((1, 2)),
        cfg_degree=torch.zeros((1, 2)),
        dfg_degree=torch.zeros((1, 2)),
        region_length=torch.ones((1, 2)),
        edge_index=torch.empty((0, 3), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        source_language=torch.tensor([0]),
        target_language=torch.tensor([1]),
    )


def _core_calls(module: RecurrentRegionCore, hidden: torch.Tensor, **kwargs: object) -> int:
    calls = 0

    def count(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = module.core_input.register_forward_hook(count)
    with torch.inference_mode():
        module(hidden, _region_batch(), **kwargs)
    handle.remove()
    return calls


def test_region_hard_halting_skips_all_later_core_calls() -> None:
    module = RecurrentRegionCore(hidden_size=4, width=8, max_steps=4, mode="sart")
    module.router[-1].weight.data.zero_()
    module.router[-1].bias.data.fill_(-10.0)
    calls = _core_calls(
        module,
        torch.randn(1, 4, 4),
        hard_threshold=0.5,
        return_diagnostics=True,
    )
    assert calls == 1


def test_token_adaptive_hard_halting_skips_all_later_core_calls() -> None:
    module = RecurrentRegionCore(hidden_size=4, width=8, max_steps=4, mode="mor_token")
    module.token_router[-1].weight.data.zero_()
    module.token_router[-1].bias.data.fill_(-10.0)
    calls = _core_calls(
        module,
        torch.randn(1, 4, 4),
        hard_threshold=0.5,
        return_diagnostics=True,
    )
    assert calls == 1


def test_fixed_region_and_structure_only_depths_do_not_adapt() -> None:
    hidden = torch.randn(1, 4, 4)
    fixed = RecurrentRegionCore(hidden_size=4, width=8, max_steps=4, mode="uniform_region")
    structured = RecurrentRegionCore(hidden_size=4, width=8, max_steps=4, mode="structure_only")
    assert _core_calls(fixed, hidden) == 4
    assert _core_calls(structured, hidden) == 1


def test_top_k_routing_uses_a_fixed_per_example_active_fraction() -> None:
    probabilities = torch.tensor([0.2, 0.9, 0.4, 0.8])
    batch_indexes = torch.tensor([0, 0, 1, 1])
    decisions = RecurrentRegionCore._top_k_decisions(
        probabilities, batch_indexes, batch_size=2, fraction=0.5
    )
    assert decisions.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_top_k_token_routing_shrinks_each_recurrent_call() -> None:
    module = RecurrentRegionCore(hidden_size=4, width=8, max_steps=3, mode="mor_token")
    sizes: list[int] = []

    def record(_module, inputs, _output):
        sizes.append(inputs[0].shape[0])

    handle = module.core_input.register_forward_hook(record)
    with torch.inference_mode():
        module(torch.randn(1, 4, 4), _region_batch(), top_k_fraction=0.5)
    handle.remove()
    assert sizes == [4, 2, 1]
