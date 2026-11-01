from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.models.recurrent.region_module import (
    DenseCapacityAdapter,
    RecurrentRegionCore,
    RegionBatch,
)


class _IdentityLayer(nn.Module):
    def forward(self, hidden, **kwargs):
        return (hidden,)


class _ToyCausalLM(nn.Module):
    def __init__(self, hidden_dtype=torch.float32):
        super().__init__()
        self.hidden_dtype = hidden_dtype
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_IdentityLayer(), _IdentityLayer()])
        self.seen_position_ids = None

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        self.seen_position_ids = position_ids.detach().clone()
        hidden = input_ids.to(self.hidden_dtype).unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = torch.zeros((*input_ids.shape, 16), device=input_ids.device)
        logits.scatter_(2, (input_ids + 1).unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits, past_key_values=((torch.tensor(1),),))


def test_greedy_prefill_uses_each_right_padded_rows_last_real_token():
    base = _ToyCausalLM()
    wrapped = MidLayerSourceAdapter(base, DenseCapacityAdapter(hidden_size=1, bottleneck=2))
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    generated, _ = wrapped.generate_greedy(
        input_ids,
        attention_mask,
        region_batch=None,
        max_new_tokens=1,
        eos_token_id=15,
    )

    assert generated[:, -1].tolist() == [3, 6]
    assert base.seen_position_ids.tolist() == [[0, 1, 1], [0, 1, 2]]


def test_sampled_cached_decoding_is_reproducible_from_generator_seed():
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)

    def sample_once():
        base = _ToyCausalLM()
        wrapped = MidLayerSourceAdapter(base, DenseCapacityAdapter(hidden_size=1, bottleneck=2))
        generator = torch.Generator(device=input_ids.device).manual_seed(90210)
        generated, _ = wrapped.generate_greedy(
            input_ids,
            attention_mask,
            region_batch=None,
            max_new_tokens=1,
            eos_token_id=15,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            generator=generator,
        )
        return generated

    assert torch.equal(sample_once(), sample_once())


class _CheckpointCausalLM(_ToyCausalLM):
    def forward(self, input_ids, attention_mask, region_batch=None, **kwargs):
        hidden = input_ids.to(torch.float32).unsqueeze(-1).requires_grad_()
        for layer in self.model.layers:
            hidden = checkpoint(
                lambda value, current_layer=layer: current_layer(value)[0],
                hidden,
                use_reentrant=False,
            )
        return hidden.square().mean()


def test_route_survives_activation_checkpoint_recomputation():
    base = _CheckpointCausalLM()
    adapter = DenseCapacityAdapter(hidden_size=1, bottleneck=2)
    wrapped = MidLayerSourceAdapter(base, adapter)
    region_batch = RegionBatch(
        token_to_region=torch.tensor([[0, 1]]),
        region_mask=torch.tensor([[True, True]]),
        region_type=torch.zeros((1, 2), dtype=torch.long),
        ast_depth=torch.zeros((1, 2)),
        child_count=torch.zeros((1, 2)),
        cfg_degree=torch.zeros((1, 2)),
        dfg_degree=torch.zeros((1, 2)),
        region_length=torch.ones((1, 2)),
        edge_index=torch.empty((0, 3), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        source_language=torch.zeros((1,), dtype=torch.long),
        target_language=torch.ones((1,), dtype=torch.long),
    )
    loss = wrapped(
        input_ids=torch.tensor([[2, 3]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        region_batch=region_batch,
    )
    loss.backward()
    assert adapter.scale.grad is not None
    wrapped.clear_route()


def test_bfloat16_backbone_hidden_is_compatible_with_fp32_sart_adapter():
    base = _ToyCausalLM(hidden_dtype=torch.bfloat16)
    adapter = RecurrentRegionCore(hidden_size=1, width=4, max_steps=1, mode="sart")
    wrapped = MidLayerSourceAdapter(base, adapter)
    regions = RegionBatch(
        token_to_region=torch.tensor([[0, 1]]),
        region_mask=torch.tensor([[True, True]]),
        region_type=torch.zeros((1, 2), dtype=torch.long),
        ast_depth=torch.zeros((1, 2)),
        child_count=torch.zeros((1, 2)),
        cfg_degree=torch.zeros((1, 2)),
        dfg_degree=torch.zeros((1, 2)),
        region_length=torch.ones((1, 2)),
        edge_index=torch.empty((0, 3), dtype=torch.long),
        edge_type=torch.empty((0,), dtype=torch.long),
        source_language=torch.zeros((1,), dtype=torch.long),
        target_language=torch.ones((1,), dtype=torch.long),
    )

    generated, _ = wrapped.generate_greedy(
        torch.tensor([[1, 2]]),
        torch.ones((1, 2), dtype=torch.long),
        region_batch=regions,
        max_new_tokens=1,
        eos_token_id=15,
    )

    assert generated.shape == (1, 3)
    assert wrapped._last_diagnostics
