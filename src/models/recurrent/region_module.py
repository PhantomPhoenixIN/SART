"""Source-side token/region recurrent computation for controlled SART baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

RoutingMode = Literal[
    "uniform_token",
    "uniform_region",
    "mor_token",
    "random_region",
    "region_adaptive_no_graph",
    "sart_no_message",
    "sart",
    "structure_only",
]
PoolingMode = Literal["mean", "attention"]

LANGUAGES = {"Python": 0, "Java": 1, "C++": 2}
REGION_TYPES = {
    "PROGRAM": 0,
    "FUNCTION": 1,
    "PARAMETER": 2,
    "DECLARATION": 3,
    "CONDITIONAL": 4,
    "LOOP": 5,
    "SWITCH": 6,
    "BRANCH": 7,
    "ASSIGNMENT": 8,
    "MUTATION": 9,
    "CALL": 10,
    "CONSTRUCTOR": 11,
    "MEMBER_ACCESS": 12,
    "INDEX_ACCESS": 13,
    "ARITHMETIC": 14,
    "COMPARISON": 15,
    "BOOLEAN_EXPRESSION": 16,
    "RETURN": 17,
    "BREAK": 18,
    "CONTINUE": 19,
    "EXCEPTION": 20,
    "OTHER_STRUCTURAL": 21,
}
RELATIONS = {
    "AST_PARENT": 0,
    "AST_CHILD": 1,
    "CFG_NEXT": 2,
    "CFG_TRUE": 3,
    "CFG_FALSE": 4,
    "CFG_BACKEDGE": 5,
    "DFG_DEF_USE": 6,
    "DFG_USE_DEF": 7,
}


@dataclass
class RegionBatch:
    """Padded per-example region features and typed directed graph edges."""

    token_to_region: Tensor  # [batch, source_length], -1 for prefix/unmapped
    region_mask: Tensor  # [batch, regions]
    region_type: Tensor  # [batch, regions]
    ast_depth: Tensor  # [batch, regions], normalized float
    child_count: Tensor  # [batch, regions], normalized float
    cfg_degree: Tensor  # [batch, regions], normalized float
    dfg_degree: Tensor  # [batch, regions], normalized float
    region_length: Tensor  # [batch, regions], normalized float
    edge_index: Tensor  # [edges, 3] = (batch, source-region, target-region)
    edge_type: Tensor  # [edges]
    source_language: Tensor  # [batch]
    target_language: Tensor  # [batch]

    def to(self, device: torch.device | str) -> RegionBatch:
        """Move every tensor to a target device."""
        return RegionBatch(**{name: value.to(device) for name, value in vars(self).items()})


class GraphMessagePassing(nn.Module):
    """Typed edge attention over sparse AST/CFG/DFG region relationships."""

    def __init__(self, width: int, relation_count: int = len(RELATIONS)) -> None:
        super().__init__()
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.relation = nn.Embedding(relation_count, width)
        self.relation_value = nn.Embedding(relation_count, width)
        self.output = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        regions: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        active_mask: Tensor | None = None,
    ) -> Tensor:
        batch, count, width = regions.shape
        if edge_index.numel() == 0:
            return torch.zeros_like(regions)
        batch_index, source_index, target_index = edge_index.unbind(dim=-1)
        if active_mask is not None:
            keep = active_mask[batch_index, source_index]
            batch_index = batch_index[keep]
            source_index = source_index[keep]
            target_index = target_index[keep]
            edge_type = edge_type[keep]
            if edge_type.numel() == 0:
                return torch.zeros_like(regions)
        query = self.query(regions[batch_index, source_index])
        key = self.key(regions[batch_index, target_index]) + self.relation(edge_type)
        value = self.value(regions[batch_index, target_index]) + self.relation_value(edge_type)
        score = (query * key).sum(-1) / (width**0.5)
        # Normalize messages independently for every destination region.
        flat_destination = batch_index * count + source_index
        maxima = torch.full((batch * count,), -torch.inf, device=regions.device, dtype=score.dtype)
        maxima.scatter_reduce_(0, flat_destination, score, reduce="amax", include_self=True)
        exponentials = torch.exp(score - maxima[flat_destination])
        denominator = torch.zeros_like(maxima)
        denominator.scatter_add_(0, flat_destination, exponentials)
        weights = exponentials / denominator[flat_destination].clamp_min(1e-9)
        # Attention weights are accumulated in fp32 for numerical stability;
        # explicitly accumulate values in the same dtype (autocast may leave
        # `value` in bf16 while the softmax path is fp32).
        messages = torch.zeros(
            (batch * count, width), device=regions.device, dtype=torch.float32
        )
        messages.index_add_(
            0, flat_destination, value.float() * weights.float().unsqueeze(-1)
        )
        messages = messages.view(batch, count, width).to(regions.dtype)
        return self.norm(self.output(messages))


class RecurrentRegionCore(nn.Module):
    """Shared recurrent transformation with optional graph and adaptive routing."""

    def __init__(
        self,
        hidden_size: int,
        width: int = 1024,
        max_steps: int = 6,
        mode: RoutingMode = "sart",
        relation_count: int = len(RELATIONS),
        region_type_count: int = len(REGION_TYPES),
        adapter_scale_init: float = 1e-3,
        pooling_mode: PoolingMode = "mean",
        use_language_pair: bool = True,
        router_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")
        if router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        self.hidden_size = hidden_size
        self.width = width
        self.max_steps = max_steps
        self.mode = mode
        self.pooling_mode = pooling_mode
        self.use_language_pair = use_language_pair
        self.router_temperature = router_temperature
        self.pool_attention = (
            nn.Linear(hidden_size, 1, bias=False) if pooling_mode == "attention" else None
        )
        self.input_projection = nn.Linear(hidden_size, width)
        self.output_projection = nn.Linear(width, hidden_size)
        self.graph = GraphMessagePassing(width, relation_count)
        self.region_type_embedding = nn.Embedding(region_type_count, width)
        self.source_language_embedding = nn.Embedding(3, width)
        self.target_language_embedding = nn.Embedding(3, width)
        self.step_embedding = nn.Embedding(max_steps, width)
        scalar_count = 5
        self.feature_projection = nn.Linear(scalar_count, width)
        self.core_input = nn.Linear(width * 4, width * 2)
        self.core_output = nn.Linear(width * 2, width)
        self.core_norm = nn.LayerNorm(width)
        self.router = nn.Sequential(
            nn.LayerNorm(width * 4 + scalar_count),
            nn.Linear(width * 4 + scalar_count, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.token_router = nn.Sequential(
            nn.LayerNorm(width * 3),
            nn.Linear(width * 3, max(128, width // 4)),
            nn.GELU(),
            nn.Linear(max(128, width // 4), 1),
        )
        self.adapter_scale = nn.Parameter(torch.tensor(adapter_scale_init))

    def _pool_tokens(self, hidden: Tensor, token_to_region: Tensor, region_count: int) -> Tensor:
        batch, _, hidden_size = hidden.shape
        indexes = token_to_region.clamp_min(0)
        mask = (token_to_region >= 0).to(hidden.dtype)
        if self.pool_attention is not None:
            flat_index = (
                torch.arange(batch, device=hidden.device).unsqueeze(1) * region_count + indexes
            ).reshape(-1)
            # Keep the segmented softmax and its accumulation in FP32. Under
            # autocast, hidden states may be BF16 while the attention scores
            # are FP32; scatter_reduce_ requires matching source/destination
            # dtypes, and FP32 is also more stable for the exponentials.
            logits = self.pool_attention(hidden).squeeze(-1).reshape(-1).float()
            valid = mask.reshape(-1).bool()
            maxima = logits.new_full((batch * region_count,), -torch.inf)
            maxima.scatter_reduce_(
                0, flat_index[valid], logits[valid], reduce="amax", include_self=True
            )
            exponentials = torch.zeros_like(logits)
            exponentials[valid] = torch.exp(logits[valid] - maxima[flat_index[valid]])
            denominator = logits.new_zeros((batch * region_count,))
            denominator.scatter_add_(0, flat_index, exponentials)
            weights = exponentials / denominator[flat_index].clamp_min(1e-9)
            sums = logits.new_zeros((batch * region_count, hidden_size))
            sums.index_add_(
                0,
                flat_index,
                hidden.reshape(-1, hidden_size).float() * weights.unsqueeze(-1),
            )
            return sums.view(batch, region_count, hidden_size).to(hidden.dtype)
        sums = hidden.new_zeros((batch, region_count, hidden_size))
        sums.scatter_add_(
            1, indexes.unsqueeze(-1).expand(-1, -1, hidden_size), hidden * mask.unsqueeze(-1)
        )
        counts = hidden.new_zeros((batch, region_count, 1))
        counts.scatter_add_(1, indexes.unsqueeze(-1), mask.unsqueeze(-1))
        return sums / counts.clamp_min(1.0)

    @staticmethod
    def _scatter_regions(deltas: Tensor, token_to_region: Tensor) -> Tensor:
        batch, sequence_length = token_to_region.shape
        hidden_size = deltas.shape[-1]
        indexes = token_to_region.clamp_min(0)
        gathered = deltas.gather(1, indexes.unsqueeze(-1).expand(-1, -1, hidden_size))
        return gathered * (token_to_region >= 0).unsqueeze(-1).to(gathered.dtype)

    def _structural_features(self, batch: RegionBatch) -> Tensor:
        return torch.stack(
            [
                batch.ast_depth,
                torch.log1p(batch.child_count),
                torch.log1p(batch.cfg_degree),
                torch.log1p(batch.dfg_degree),
                torch.log1p(batch.region_length),
            ],
            dim=-1,
        )

    def _token_self_attention(self, states: Tensor) -> Tensor:
        """Apply content-only recurrent self-attention to one active token set."""
        if states.shape[0] == 0:
            return states
        query = self.graph.query(states)
        key = self.graph.key(states)
        value = self.graph.value(states)
        weights = torch.softmax(query @ key.transpose(0, 1) / (self.width**0.5), dim=-1)
        return self.graph.norm(self.graph.output(weights @ value))

    @staticmethod
    def _top_k_decisions(
        probabilities: Tensor,
        batch_indexes: Tensor,
        batch_size: int,
        fraction: float,
    ) -> Tensor:
        """Keep a stable per-example top fraction of currently active units."""
        decisions = torch.zeros_like(probabilities)
        for batch_index in range(batch_size):
            candidates = (batch_indexes == batch_index).nonzero(as_tuple=False).flatten()
            keep_count = min(len(candidates), math.ceil(len(candidates) * fraction))
            if keep_count:
                order = torch.argsort(probabilities[candidates], descending=True, stable=True)
                decisions[candidates[order[:keep_count]]] = 1
        return decisions

    def _forward_token_mode(
        self,
        hidden: Tensor,
        batch: RegionBatch,
        budget: float | None,
        hard_threshold: float | None,
        top_k_fraction: float | None,
        return_diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply the shared recurrent transformation to mapped source tokens."""
        token_valid = batch.token_to_region >= 0
        safe_region = batch.token_to_region.clamp_min(0)
        token_types = batch.region_type.gather(
            1,
            safe_region.clamp_max(batch.region_type.shape[1] - 1),
        )
        # Token-level MoR receives token content and translation direction only; structural
        # region labels are intentionally hidden from this baseline.
        type_embedding = torch.zeros(
            (*token_types.shape, self.width),
            device=hidden.device,
            dtype=self.input_projection.weight.dtype,
        )
        pair = self._language_pair(batch)
        state0 = self.input_projection(hidden)
        state = state0
        active = token_valid.to(hidden.dtype)
        gates: list[Tensor] = []
        probability_log: list[Tensor] = []
        for step in range(self.max_steps):
            step_embedding = self.step_embedding.weight[step].view(1, 1, -1)
            if self.mode == "mor_token" and (
                hard_threshold is not None or top_k_fraction is not None
            ):
                active_indexes = (active > 0).nonzero(as_tuple=False)
                gate_values = torch.zeros_like(active)
                if active_indexes.numel() == 0:
                    gates.append(gate_values)
                    probability_log.append(gate_values)
                    continue
                batch_indexes, token_indexes = active_indexes.unbind(dim=-1)
                current = state[batch_indexes, token_indexes]
                message = torch.zeros_like(current)
                for batch_index in range(hidden.shape[0]):
                    selected = batch_indexes == batch_index
                    if bool(selected.any()):
                        message[selected] = self._token_self_attention(current[selected]).to(
                            message.dtype
                        )
                pair_values = pair.expand(-1, hidden.shape[1], -1)[batch_indexes, token_indexes]
                recurrent_input = torch.cat(
                    [current, message, torch.zeros_like(current), pair_values],
                    dim=-1,
                )
                transformed = self.core_norm(
                    current + self.core_output(F.gelu(self.core_input(recurrent_input)))
                ) + step_embedding.view(1, -1)
                probabilities = torch.sigmoid(
                    self.token_router(
                        torch.cat([current, torch.zeros_like(current), pair_values], dim=-1)
                    ).squeeze(-1)
                    / self.router_temperature
                )
                if top_k_fraction is not None:
                    decisions = self._top_k_decisions(
                        probabilities, batch_indexes, hidden.shape[0], top_k_fraction
                    )
                else:
                    decisions = (probabilities >= float(hard_threshold)).to(active.dtype)
                gate_values[batch_indexes, token_indexes] = decisions
                probability_values = torch.zeros_like(active)
                probability_values[batch_indexes, token_indexes] = probabilities.to(
                    dtype=probability_values.dtype
                )
                state = state.clone()
                # Hard-threshold calibration uses indexed updates; keep the recurrent
                # state in the backbone activation dtype (often BF16) before scatter.
                state[batch_indexes, token_indexes] = transformed.to(dtype=state.dtype)
                active = active.clone()
                active[batch_indexes, token_indexes] = decisions
                gates.append(gate_values)
                probability_log.append(probability_values)
                continue
            message = torch.zeros_like(state)
            for batch_index in range(hidden.shape[0]):
                selected = token_valid[batch_index].nonzero(as_tuple=False).flatten()
                if selected.numel():
                    message[batch_index, selected] = self._token_self_attention(
                        state[batch_index, selected]
                    ).to(message.dtype)
            recurrent_input = torch.cat(
                [state, message, type_embedding, pair.expand_as(state)], dim=-1
            )
            transformed = (
                self.core_norm(state + self.core_output(F.gelu(self.core_input(recurrent_input))))
                + step_embedding
            )
            if self.mode == "uniform_token":
                gate = torch.ones_like(active)
                probabilities = gate
            else:
                probabilities = torch.sigmoid(
                    self.token_router(
                        torch.cat([state, type_embedding, pair.expand_as(state)], -1)
                    ).squeeze(-1)
                    / self.router_temperature
                )
                gate = probabilities
                if hard_threshold is not None:
                    gate = (gate >= hard_threshold).to(gate.dtype)
            state = state + (active * gate).unsqueeze(-1) * (transformed - state)
            active = active * gate
            gates.append(gate)
            probability_log.append(probabilities)
        delta = self.output_projection(state - state0)
        output = hidden + self.adapter_scale * delta * token_valid.unsqueeze(-1).to(delta.dtype)
        survival = [torch.ones_like(active)]
        for step in range(1, self.max_steps):
            survival.append(torch.stack(gates[:step], dim=0).prod(dim=0))
        depth = torch.stack(survival, dim=0).sum(dim=0) * token_valid.to(hidden.dtype)
        mean_depth = depth.sum() / token_valid.sum().clamp_min(1)
        diagnostics = {
            "expected_depth": depth,
            "mean_depth": mean_depth,
            "token_gate_probabilities": torch.stack(probability_log, dim=1),
            "budget_loss": (mean_depth - float(budget)).square()
            if budget is not None
            else hidden.new_zeros(()),
        }
        if not return_diagnostics:
            diagnostics = {"budget_loss": diagnostics["budget_loss"]}
        return output, diagnostics

    def forward(
        self,
        hidden: Tensor,
        batch: RegionBatch,
        budget: float | None = None,
        hard_threshold: float | None = None,
        top_k_fraction: float | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Refine source representations and scatter deltas into backbone token states."""
        if top_k_fraction is not None and not 0.0 <= top_k_fraction <= 1.0:
            raise ValueError("top_k_fraction must be between zero and one")
        if hard_threshold is not None and top_k_fraction is not None:
            raise ValueError("hard_threshold and top_k_fraction are mutually exclusive")
        if top_k_fraction is not None and self.mode not in {
            "mor_token",
            "sart",
            "sart_no_message",
            "region_adaptive_no_graph",
            "random_region",
        }:
            raise ValueError("top-k routing applies only to adaptive routing modes")
        if self.mode in {"uniform_token", "mor_token"}:
            return self._forward_token_mode(
                hidden, batch, budget, hard_threshold, top_k_fraction, return_diagnostics
            )
        features = self._structural_features(batch)
        type_embedding = self.region_type_embedding(batch.region_type)
        if self.mode == "region_adaptive_no_graph":
            type_embedding = torch.zeros_like(type_embedding)
        language_pair = self._language_pair(batch)
        step_indices = torch.arange(self.max_steps, device=hidden.device)
        pooled = self._pool_tokens(hidden, batch.token_to_region, batch.region_mask.shape[1])
        z0 = self.input_projection(pooled)
        z = z0
        alive = batch.region_mask.to(hidden.dtype)
        gate_log: list[Tensor] = []
        probability_log: list[Tensor] = []
        active_log: list[Tensor] = []
        mode = self.mode

        recurrent_steps = 1 if mode == "structure_only" else self.max_steps
        for step in range(recurrent_steps):
            sparse_hard_route = (
                hard_threshold is not None or top_k_fraction is not None
            ) and mode in {
                "sart",
                "sart_no_message",
                "region_adaptive_no_graph",
                "random_region",
            }
            active_mask = alive > 0 if sparse_hard_route else None
            if mode in {"sart", "structure_only"}:
                message = self.graph(z, batch.edge_index, batch.edge_type, active_mask=active_mask)
            else:
                message = torch.zeros_like(z)
            step_embedding = self.step_embedding(step_indices[step]).view(1, 1, -1)
            routed_features = (
                torch.zeros_like(features) if self.mode == "region_adaptive_no_graph" else features
            )
            if mode == "structure_only":
                gates = torch.zeros_like(alive)
            elif mode in {
                "sart",
                "sart_no_message",
                "region_adaptive_no_graph",
                "random_region",
            }:
                if sparse_hard_route:
                    gates = torch.zeros_like(alive)
                else:
                    if mode == "region_adaptive_no_graph":
                        route_type = torch.zeros_like(type_embedding)
                        route_features = torch.zeros_like(features)
                        route_input = torch.cat(
                            [
                                z,
                                message,
                                route_type,
                                language_pair.expand_as(z),
                                route_features,
                            ],
                            dim=-1,
                        )
                    else:
                        route_input = torch.cat(
                            [
                                z,
                                message,
                                type_embedding,
                                language_pair.expand_as(z),
                                features,
                            ],
                            dim=-1,
                        )
                    gates = torch.sigmoid(
                        self.router(route_input).squeeze(-1) / self.router_temperature
                    )
                    if hard_threshold is not None:
                        gates = (gates >= hard_threshold).to(gates.dtype)
            else:  # uniform_region
                gates = torch.ones_like(alive)

            adaptive_modes = {
                "sart",
                "sart_no_message",
                "region_adaptive_no_graph",
                "random_region",
            }
            if sparse_hard_route:
                active_indexes = (alive > 0).nonzero(as_tuple=False)
                if active_indexes.numel() == 0:
                    gate_log.append(torch.zeros_like(alive))
                    probability_log.append(torch.zeros_like(alive))
                    continue
                batch_indexes, region_indexes = active_indexes.unbind(dim=-1)
                current = z[batch_indexes, region_indexes]
                message_current = message[batch_indexes, region_indexes]
                type_current = type_embedding[batch_indexes, region_indexes]
                language_current = language_pair.expand_as(z)[batch_indexes, region_indexes]
                feature_current = routed_features[batch_indexes, region_indexes]
                projected_features = self.feature_projection(feature_current)
                recurrent_input = torch.cat(
                    [current, message_current, type_current, language_current], dim=-1
                )
                transformed = self.core_norm(
                    current + self.core_output(F.gelu(self.core_input(recurrent_input)))
                )
                transformed = transformed + step_embedding.view(1, -1) + projected_features
                route_features = features[batch_indexes, region_indexes]
                if mode == "region_adaptive_no_graph":
                    route_features = torch.zeros_like(route_features)
                route_input = torch.cat(
                    [current, message_current, type_current, language_current, route_features],
                    dim=-1,
                )
                probabilities = torch.sigmoid(
                    self.router(route_input).squeeze(-1) / self.router_temperature
                )
                if top_k_fraction is not None:
                    decisions = self._top_k_decisions(
                        probabilities, batch_indexes, hidden.shape[0], top_k_fraction
                    )
                else:
                    decisions = (probabilities >= float(hard_threshold)).to(alive.dtype)
                z = z.clone()
                # Calibration uses hard-threshold indexed scatter; preserve
                # the recurrent activation dtype when copying projected state.
                z[batch_indexes, region_indexes] = transformed.to(dtype=z.dtype)
                alive = alive.clone()
                alive[batch_indexes, region_indexes] = decisions
                gates = torch.zeros_like(alive)
                gates[batch_indexes, region_indexes] = decisions
                probability_values = torch.zeros_like(alive)
                probability_values[batch_indexes, region_indexes] = probabilities.to(
                    dtype=probability_values.dtype
                )
                probability_log.append(probability_values)
            else:
                projected_features = self.feature_projection(routed_features)
                recurrent_input = torch.cat(
                    [z, message, type_embedding, language_pair.expand_as(z)], dim=-1
                )
                transformed = self.core_norm(
                    z + self.core_output(F.gelu(self.core_input(recurrent_input)))
                )
                transformed = transformed + step_embedding + projected_features
                probability_log.append(gates)

            if mode in adaptive_modes:
                if not sparse_hard_route:
                    effective = alive * gates
                    z = z + effective.unsqueeze(-1) * (transformed - z)
                    alive = alive * gates
            elif mode == "structure_only":
                if step == 0:
                    z = transformed
            else:
                z = torch.where(batch.region_mask.unsqueeze(-1), transformed, z)
            gate_log.append(gates)

        refined = self.output_projection(z - z0)
        delta = self._scatter_regions(refined, batch.token_to_region)
        output = hidden + self.adapter_scale * delta
        expected_depth = torch.stack(
            [torch.ones_like(batch.region_mask, dtype=hidden.dtype)]
            + [
                torch.stack(gate_log[:step], dim=0).prod(dim=0) for step in range(1, self.max_steps)
            ],
            dim=0,
        ).sum(dim=0)
        valid = batch.region_mask.to(hidden.dtype)
        mean_depth = (expected_depth * valid).sum() / valid.sum().clamp_min(1.0)
        diagnostics = {
            "expected_depth": expected_depth,
            "mean_depth": mean_depth,
            "gate_probabilities": torch.stack(probability_log, dim=1),
            "budget_loss": (mean_depth - float(budget)).square()
            if budget is not None
            else hidden.new_zeros(()),
        }
        if active_log:
            diagnostics["token_gate_probabilities"] = torch.stack(active_log, dim=1)
        if not return_diagnostics:
            diagnostics = {"budget_loss": diagnostics["budget_loss"]}
        return output, diagnostics

    def _language_pair(self, batch: RegionBatch) -> Tensor:
        """Return the shared language-pair condition or zeros for its ablation."""
        if not self.use_language_pair:
            reference = self.source_language_embedding.weight
            return reference.new_zeros((batch.source_language.shape[0], 1, self.width))
        return (
            self.source_language_embedding(batch.source_language)
            + self.target_language_embedding(batch.target_language)
        ).unsqueeze(1)


class DenseCapacityAdapter(nn.Module):
    """Non-recurrent dense residual adapter used for parameter-capacity matching."""

    def __init__(self, hidden_size: int, bottleneck: int = 1024) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scale = nn.Parameter(torch.tensor(1e-3))

    def forward(self, hidden: Tensor) -> Tensor:
        """Return dense residual adapter output."""
        return hidden + self.scale * self.up(self.activation(self.down(hidden)))
