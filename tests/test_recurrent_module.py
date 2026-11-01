from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.models.recurrent.region_module import (
    DenseCapacityAdapter,
    RecurrentRegionCore,
    RegionBatch,
)


def _region_batch(sequence_length: int = 5) -> RegionBatch:
    return RegionBatch(
        token_to_region=torch.tensor([[0, 0, 1, 1, -1][:sequence_length]]),
        region_mask=torch.tensor([[True, True]]),
        region_type=torch.tensor([[0, 5]]),
        ast_depth=torch.tensor([[0.0, 0.5]]),
        child_count=torch.tensor([[1.0, 0.0]]),
        cfg_degree=torch.zeros((1, 2)),
        dfg_degree=torch.zeros((1, 2)),
        region_length=torch.tensor([[2.0, 2.0]]),
        edge_index=torch.tensor([[0, 0, 1], [0, 1, 0]]),
        edge_type=torch.tensor([0, 1]),
        source_language=torch.tensor([0]),
        target_language=torch.tensor([1]),
    )


def test_adaptive_region_module_shapes_gradients_and_budget() -> None:
    module = RecurrentRegionCore(16, width=8, max_steps=3, mode="sart")
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    output, stats = module(hidden, _region_batch(), budget=2.0, return_diagnostics=True)
    assert output.shape == hidden.shape
    assert stats["expected_depth"].shape == (1, 2)
    assert stats["gate_probabilities"].shape == (1, 3, 2)
    assert torch.isfinite(stats["budget_loss"])
    (output.square().mean() + stats["budget_loss"]).backward()
    assert hidden.grad is not None
    assert module.router[-1].weight.grad is not None


def test_token_adaptive_core_uses_source_token_gates() -> None:
    module = RecurrentRegionCore(16, width=8, max_steps=3, mode="mor_token")
    hidden = torch.randn(1, 5, 16)
    output, stats = module(hidden, _region_batch(), return_diagnostics=True)
    assert output.shape == hidden.shape
    assert stats["token_gate_probabilities"].shape == (1, 3, 5)
    assert torch.equal(output[:, 4], hidden[:, 4])
    output.sum().backward()
    assert module.token_router[-1].weight.grad is not None


def test_no_message_ablation_is_invariant_to_edges() -> None:
    from dataclasses import replace

    module = RecurrentRegionCore(16, width=8, max_steps=3, mode="sart_no_message")
    hidden = torch.randn(1, 5, 16)
    batch = _region_batch()
    with_edges, _ = module(hidden, batch)
    no_edges = replace(
        batch,
        edge_index=torch.empty((0, 3), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
    )
    without_edges, _ = module(hidden, no_edges)
    assert torch.allclose(with_edges, without_edges)


def test_attention_pooling_is_differentiable() -> None:
    module = RecurrentRegionCore(
        16,
        width=8,
        max_steps=2,
        mode="sart",
        pooling_mode="attention",
    )
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    output, _ = module(hidden, _region_batch())
    output.square().mean().backward()
    assert output.shape == hidden.shape
    assert module.pool_attention is not None
    assert module.pool_attention.weight.grad is not None


def test_language_pair_ablation_ignores_direction_ids() -> None:
    from dataclasses import replace

    module = RecurrentRegionCore(
        16,
        width=8,
        max_steps=2,
        mode="sart",
        use_language_pair=False,
    )
    hidden = torch.randn(1, 5, 16)
    batch = _region_batch()
    changed_direction = replace(
        batch,
        source_language=torch.tensor([2]),
        target_language=torch.tensor([0]),
    )
    original, _ = module(hidden, batch)
    changed, _ = module(hidden, changed_direction)
    assert torch.allclose(original, changed)


def test_router_temperature_scales_gate_logits() -> None:
    module = RecurrentRegionCore(16, width=8, max_steps=2, mode="sart")
    hidden = torch.randn(1, 5, 16)
    _, normal = module(hidden, _region_batch(), return_diagnostics=True)
    module.router_temperature = 0.25
    _, cold = module(hidden, _region_batch(), return_diagnostics=True)
    normal_distance = (normal["gate_probabilities"] - 0.5).abs().mean()
    cold_distance = (cold["gate_probabilities"] - 0.5).abs().mean()
    assert cold_distance > normal_distance


def test_dense_capacity_adapter_shape() -> None:
    adapter = DenseCapacityAdapter(16, bottleneck=8)
    values = torch.randn(2, 7, 16)
    assert adapter(values).shape == values.shape


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, hidden: torch.Tensor):
        return (hidden + self.linear(hidden),)


class _DummyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_Block(), _Block(), _Block()])
        self.head = nn.Linear(8, 32)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        hidden = self.embed(input_ids)
        for layer in self.transformer.h:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.head(hidden))


def test_midlayer_hook_only_changes_mapped_source_tokens() -> None:
    model = _DummyLM()
    adapter = DenseCapacityAdapter(8, bottleneck=4)
    hook_model = MidLayerSourceAdapter(model, adapter, insertion_fraction=0.5)
    with torch.no_grad():
        adapter.scale.fill_(1.0)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    region_batch = _region_batch()
    ordinary = model(input_ids).logits
    adapted = hook_model(input_ids, region_batch=region_batch).logits
    assert ordinary.shape == adapted.shape
    assert not torch.equal(ordinary[:, :4], adapted[:, :4])
    assert torch.equal(ordinary[:, 4], adapted[:, 4])
    hook_model.close()
