"""Load a trained PEFT/recurrent SART checkpoint for deterministic inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.recurrent.backbone_hook import MidLayerSourceAdapter
from src.training.run_translation import RunConfig, _load_model


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    config_override: RunConfig | None = None,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    """Restore frozen backbone plus every trained parameter from one checkpoint.

    The loader requires every trainable checkpoint tensor to match the current model
    and rejects unexpected keys. Frozen pretrained backbone keys are expected absent.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_override or RunConfig(**checkpoint["config"])
    model, tokenizer = _load_model(config, device)
    expected = set(checkpoint["trainable_state"])
    incompatible = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Checkpoint has unexpected tensors: {incompatible.unexpected_keys[:8]}")
    missing_trainable = expected.intersection(incompatible.missing_keys)
    if missing_trainable:
        names = sorted(missing_trainable)[:8]
        raise ValueError(f"Checkpoint trainable tensors did not load: {names}")
    base = model.causal_lm if isinstance(model, MidLayerSourceAdapter) else model
    if hasattr(base, "gradient_checkpointing_disable"):
        base.gradient_checkpointing_disable()
    base.config.use_cache = True
    model.eval()
    max_positions = int(
        getattr(base.config, "max_position_embeddings", getattr(base.config, "n_positions", 0))
    )
    if max_positions <= 0:
        raise ValueError("Backbone config does not expose a finite context length")
    if isinstance(model, MidLayerSourceAdapter):
        # Training caps sequences at 2,048; test inputs use the backbone's full context.
        model.max_source_length = max_positions
    metadata = {
        "experiment_id": config.experiment_id,
        "architecture": config.architecture,
        "backbone_path": config.backbone_path,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_loss": checkpoint.get("validation_loss"),
        "max_position_embeddings": max_positions,
        "config": checkpoint["config"],
    }
    return model, tokenizer, metadata
