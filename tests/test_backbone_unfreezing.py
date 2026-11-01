"""Tests for the confirmatory higher-backbone-trainability control."""

from __future__ import annotations

import hashlib
import json

import pytest
from torch import nn

from src.training.run_translation import (
    RunConfig,
    _config_hash,
    _config_matches_saved_run,
    _discover_resume_experiment_id,
    _saved_config_hash,
    _unfreeze_last_decoder_layers,
)


class TinyDecoder(nn.Module):
    """Minimal HF-shaped decoder used to test layer selection without model downloads."""

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(3, 3) for _ in range(layer_count)])


class TinyPeftWrapper(nn.Module):
    """Minimal PEFT `base_model.model` nesting around an HF-shaped causal LM."""

    def __init__(self, layer_count: int) -> None:
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = TinyDecoder(layer_count)


def _config(**kwargs) -> RunConfig:
    values = {
        "experiment_id": "unit-test",
        "architecture": "dense",
        "backbone_path": "qwen_revision",
        "train_manifest": "train.json",
        "validation_manifest": "validation.json",
        "output_dir": "checkpoints",
    }
    values.update(kwargs)
    return RunConfig(**values)


def test_unfreeze_selects_only_top_decoder_layers() -> None:
    model = TinyDecoder(layer_count=4)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _unfreeze_last_decoder_layers(model, 1)
    assert all(not p.requires_grad for p in model.model.layers[0].parameters())
    assert all(not p.requires_grad for p in model.model.layers[2].parameters())
    assert all(p.requires_grad for p in model.model.layers[3].parameters())


def test_unfreeze_finds_decoder_layers_through_peft_wrapper() -> None:
    model = TinyPeftWrapper(layer_count=3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _unfreeze_last_decoder_layers(model, 1)
    layers = model.base_model.model.model.layers
    assert all(not p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())


@pytest.mark.parametrize("count", [-1, 5])
def test_unfreeze_rejects_invalid_layer_count(count: int) -> None:
    with pytest.raises(ValueError):
        _unfreeze_last_decoder_layers(TinyDecoder(layer_count=4), count)


def test_added_default_settings_preserve_legacy_resume_compatibility() -> None:
    config = _config()
    saved = {
        key: value
        for key, value in vars(config).items()
        if key not in {"insertion_fraction", "router_temperature", "unfreeze_last_layers"}
    }
    legacy_payload = dict(saved)
    legacy_payload.pop("experiment_id")
    expected = hashlib.sha256(
        json.dumps(legacy_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _saved_config_hash(saved) == expected
    assert _config_matches_saved_run(config, saved)


def test_nondefault_added_setting_does_not_match_legacy_run() -> None:
    saved = {
        key: value
        for key, value in vars(_config()).items()
        if key not in {"insertion_fraction", "router_temperature", "unfreeze_last_layers"}
    }
    assert not _config_matches_saved_run(_config(unfreeze_last_layers=1), saved)


def test_auto_run_discovers_unique_legacy_checkpoint_directory(tmp_path) -> None:
    config = _config(output_dir=str(tmp_path))
    saved = {
        key: value
        for key, value in vars(config).items()
        if key not in {"insertion_fraction", "router_temperature", "unfreeze_last_layers"}
    }
    saved["experiment_id"] = "legacy-id"
    run_dir = tmp_path / "legacy-id"
    run_dir.mkdir()
    (run_dir / "resolved_config.json").write_text(json.dumps(saved), encoding="utf-8")
    (run_dir / "last.pt").write_bytes(b"checkpoint-exists")
    assert _discover_resume_experiment_id(config) == "legacy-id"


def test_unfreeze_setting_changes_experiment_hash() -> None:
    assert _config_hash(_config(unfreeze_last_layers=1)) != _config_hash(_config())
