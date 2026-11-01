"""Mid-backbone source-only integration for Qwen2.5 and StarCoder2 decoder models."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

from src.models.recurrent.region_module import (
    DenseCapacityAdapter,
    RecurrentRegionCore,
    RegionBatch,
)


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the decoder block list in supported Hugging Face causal LMs."""
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, nn.ModuleList):
            return candidate
    # PEFT wraps the base causal LM under base_model.model; search named modules so
    # its additional wrapper level does not hide Qwen's .model.layers / StarCoder's .h.
    for name, candidate in model.named_modules():
        if isinstance(candidate, nn.ModuleList) and name.rsplit(".", 1)[-1] in {
            "layers",
            "h",
        }:
            return candidate
    raise TypeError(f"Unsupported causal LM decoder layout: {type(model).__name__}")


class MidLayerSourceAdapter(nn.Module):
    """Register a source-prefix-only adapter at one fixed decoder insertion point."""

    def __init__(
        self,
        causal_lm: nn.Module,
        adapter: RecurrentRegionCore | DenseCapacityAdapter,
        insertion_fraction: float = 0.5,
        max_source_length: int = 2048,
    ) -> None:
        super().__init__()
        if not 0.0 < insertion_fraction < 1.0:
            raise ValueError("insertion_fraction must be between zero and one")
        self.causal_lm = causal_lm
        self.adapter = adapter
        object.__setattr__(self, "_decoder_layers", decoder_layers(causal_lm))
        self.insertion_index = min(
            len(self._decoder_layers) - 1, int(len(self._decoder_layers) * insertion_fraction)
        )
        self.max_source_length = max_source_length
        self._active_batch: RegionBatch | None = None
        self._active_budget: float | None = None
        self._active_threshold: float | None = None
        self._active_top_k_fraction: float | None = None
        self._last_diagnostics: dict[str, Tensor] = {}
        self._hook = self._decoder_layers[self.insertion_index].register_forward_hook(
            self._apply_adapter
        )

    def close(self) -> None:
        """Remove the decoder hook."""
        self._hook.remove()

    @contextmanager
    def route(
        self,
        regions: RegionBatch | None,
        budget: float | None = None,
        threshold: float | None = None,
        top_k_fraction: float | None = None,
    ) -> Iterator[None]:
        """Set request-local source structure for one forward/generation call."""
        previous = (
            self._active_batch,
            self._active_budget,
            self._active_threshold,
            self._active_top_k_fraction,
        )
        self._active_batch = regions
        self._active_budget = budget
        self._active_threshold = threshold
        self._active_top_k_fraction = top_k_fraction
        try:
            yield
        finally:
            (
                self._active_batch,
                self._active_budget,
                self._active_threshold,
                self._active_top_k_fraction,
            ) = previous

    @staticmethod
    def _replace_hidden(output, hidden: Tensor):
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        if isinstance(output, list):
            return [hidden, *output[1:]]
        return hidden

    def _apply_adapter(self, module: nn.Module, inputs, output):
        if isinstance(output, tuple | list):
            hidden = output[0]
        else:
            hidden = output
        if not isinstance(hidden, Tensor):
            raise TypeError("decoder block output does not begin with a hidden-state tensor")
        if isinstance(self.adapter, DenseCapacityAdapter):
            sequence = hidden.shape[1]
            if sequence > self.max_source_length:
                return output
            adapter_dtype = next(self.adapter.parameters()).dtype
            adapted = self.adapter(hidden.to(dtype=adapter_dtype)).to(dtype=hidden.dtype)
            mask = self._active_batch.token_to_region >= 0 if self._active_batch else None
            if mask is None:
                return output
            mask = mask[:, :sequence].to(hidden.device).unsqueeze(-1)
            hidden = torch.where(mask, adapted, hidden)
            return self._replace_hidden(output, hidden)

        batch = self._active_batch
        if batch is None or hidden.shape[1] != batch.token_to_region.shape[1]:
            # During autoregressive decode, the one-token hidden state belongs to a new target
            # token. Only the full source/prompt prefill receives structural recurrence.
            return output
        adapter_dtype = next(self.adapter.parameters()).dtype
        adapted, diagnostics = self.adapter(
            hidden.to(dtype=adapter_dtype),
            batch.to(hidden.device),
            budget=self._active_budget,
            hard_threshold=self._active_threshold,
            top_k_fraction=self._active_top_k_fraction,
            return_diagnostics=True,
        )
        self._last_diagnostics = diagnostics
        return self._replace_hidden(output, adapted.to(dtype=hidden.dtype))

    def forward(
        self,
        *args,
        region_batch: RegionBatch | None = None,
        budget: float | None = None,
        hard_threshold: float | None = None,
        top_k_fraction: float | None = None,
        **kwargs,
    ):
        """Call the LM and retain routing through activation-checkpoint recomputation.

        Gradient checkpointing may re-run decoder hooks during backward, after the base
        forward returns. The router therefore stays active until the trainer calls
        :meth:`clear_route` after backward.
        """
        self._active_batch = region_batch
        self._active_budget = budget
        self._active_threshold = hard_threshold
        self._active_top_k_fraction = top_k_fraction
        try:
            return self.causal_lm(*args, **kwargs)
        except BaseException:
            self.clear_route()
            raise

    def clear_route(self) -> None:
        """Clear request-local routing after backward or validation."""
        self._active_batch = None
        self._active_budget = None
        self._active_threshold = None
        self._active_top_k_fraction = None

    def generate_greedy(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        region_batch: RegionBatch,
        max_new_tokens: int,
        eos_token_id: int,
        budget: float | None = None,
        hard_threshold: float | None = None,
        top_k_fraction: float | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Cached decoding with source structure applied only during prefill."""
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")
        generated = input_ids
        current_ids = input_ids
        current_mask = attention_mask
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        scores = []
        for step in range(max_new_tokens):
            if step == 0:
                # Right-padded prompt batches end at different source positions. Explicit
                # positions keep RoPE positions aligned with each unpadded sequence.
                position_ids = (current_mask.cumsum(-1) - 1).clamp_min(0)
            else:
                # The current token is the final non-padding position for its row, even
                # though the shared KV cache also contains the batch's padding slots.
                position_ids = (current_mask.sum(-1) - 1).clamp_min(0)[:, None]
            with self.route(
                region_batch if step == 0 else None,
                budget,
                hard_threshold,
                top_k_fraction,
            ):
                outputs = self.causal_lm(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            if step == 0:
                last_prompt_index = attention_mask.sum(-1).clamp_min(1) - 1
                logits = outputs.logits[
                    torch.arange(input_ids.shape[0], device=input_ids.device),
                    last_prompt_index,
                ]
            else:
                logits = outputs.logits[:, -1, :]
            if do_sample:
                logits = logits / temperature
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_remove = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                sorted_remove[:, 0] = False
                remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
                logits = logits.masked_fill(remove, torch.finfo(logits.dtype).min)
                probabilities = torch.softmax(logits.float(), dim=-1)
                if isinstance(generator, list):
                    if len(generator) != probabilities.shape[0]:
                        raise ValueError("One seeded generator is required per batch row")
                    next_token = torch.stack([
                        torch.multinomial(probabilities[i], 1, generator=generator[i])[0]
                        for i in range(probabilities.shape[0])
                    ])
                else:
                    next_token = torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    ).squeeze(-1)
            else:
                next_token = logits.argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_token_id), next_token
            )
            scores.append(next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_token_id)
            if bool(finished.all()):
                break
            current_ids = next_token[:, None]
            current_mask = torch.cat([current_mask, torch.ones_like(current_mask[:, :1])], dim=1)
            past_key_values = outputs.past_key_values
        return generated, torch.stack(scores, dim=1) if scores else input_ids[:, :0]
