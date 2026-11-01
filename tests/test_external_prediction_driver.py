"""Integrity and record-path tests for frozen external prediction runs."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate.generate_external_predictions import (
    _record_path,
    _validate_frozen_protocol,
    _verify_cache_files,
    generate,
)


def test_cache_shard_integrity_is_checked(tmp_path: Path) -> None:
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"source":"x"}\n', encoding="utf-8")
    metadata = {
        "path": str(shard),
        "bytes": shard.stat().st_size,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }
    _verify_cache_files(tmp_path, {"status": "COMPLETE", "files": {"d/0": metadata}})
    shard.write_text('{"source":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        _verify_cache_files(tmp_path, {"files": {"d/0": metadata}})


def test_task_result_path_is_stable_and_cluster_safe(tmp_path: Path) -> None:
    row = {"task_id": "C++-to-Java/problem/with/slash", "direction": "C++-to-Java"}
    first = _record_path(tmp_path, row)
    second = _record_path(tmp_path, row)
    assert first == second
    assert first.parent == tmp_path / "C++-to-Java"
    assert len(first.stem) == 20


def _frozen_policy(tmp_path: Path) -> tuple[Path, Path]:
    protocol = tmp_path / "translation_protocol.md"
    protocol.write_text("frozen prompt format\n", encoding="utf-8")
    policy = tmp_path / "frozen_external_protocol.json"
    policy.write_text(
        json.dumps(
            {
                "status": "FROZEN_PREMODEL",
                "selection_metric": "validation_target_loss",
                "external_metrics_used_for_selection": False,
                "translation_protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
                "external_task_manifests": {"xcodeeval:common_core": "manifest_hash"},
                "training_horizon_epochs": {"qwen_revision": 3},
                "generation": {
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": 768,
                    "eos_token_id": 2,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "repetition_penalty": 1.0,
                    "postprocess": "extract_generated_code_v1",
                    "pass_at_5": {
                        "do_sample": True,
                        "num_samples": 5,
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "num_beams": 1,
                        "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32",
                    },
                },
                "routing_policies": {
                    "full_sart": {
                        "hard_threshold": 0.5,
                        "top_k_fraction": None,
                        "max_recurrence": 6,
                        "insertion_fraction": 0.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return policy, protocol


def _validate_policy(policy: Path | None, protocol: Path, **overrides):
    arguments = {
        "policy_path": policy,
        "benchmark": "xcodeeval",
        "split": "common_core",
        "task_manifest_sha256": "manifest_hash",
        "architecture": "full_sart",
        "backbone_path": "/models/qwen_revision",
        "configured_epochs": 3,
        "max_recurrence": 6,
        "insertion_fraction": 0.5,
        "checkpoint_epoch": 2,
        "checkpoint_validation_loss": 0.05,
        "checkpoint_path": Path("/checkpoints/best.pt"),
        "max_new_tokens": 768,
        "eos_token_id": 2,
        "hard_threshold": 0.5,
        "top_k_fraction": None,
        "protocol_path": protocol,
    }
    arguments.update(overrides)
    return _validate_frozen_protocol(**arguments)


def test_external_policy_accepts_matching_validation_freeze(tmp_path: Path) -> None:
    policy, protocol = _frozen_policy(tmp_path)
    assert _validate_policy(policy, protocol)["status"] == "FROZEN_PREMODEL"


def test_external_policy_uses_backbone_specific_routing_calibration(tmp_path: Path) -> None:
    policy, protocol = _frozen_policy(tmp_path)
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["routing_policies"]["qwen_revision"] = {
        "full_sart": {
            "hard_threshold": 0.7,
            "top_k_fraction": None,
            "max_recurrence": 6,
            "insertion_fraction": 0.5,
        }
    }
    policy.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_policy(policy, protocol, hard_threshold=0.7)["status"] == "FROZEN_PREMODEL"


def test_external_policy_rejects_missing_or_unfrozen_record(tmp_path: Path) -> None:
    policy, protocol = _frozen_policy(tmp_path)
    with pytest.raises(ValueError, match="blocked until"):
        _validate_policy(None, protocol)
    value = json.loads(policy.read_text(encoding="utf-8"))
    value["status"] = "PILOT_ONLY"
    policy.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="FROZEN_PREMODEL"):
        _validate_policy(policy, protocol)


def test_generation_blocks_missing_policy_before_loading_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="blocked until a frozen protocol"):
        generate(
            benchmark="xcodeeval",
            split="common_core",
            cache_root=tmp_path / "missing_cache",
            checkpoint_path=tmp_path / "missing_checkpoint.pt",
            output_root=tmp_path / "predictions",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("configured_epochs", 5, "does not match frozen horizon"),
        ("hard_threshold", 0.7, "Routing settings differ"),
        ("max_new_tokens", 512, "Generation settings differ"),
        ("eos_token_id", None, "Generation settings differ"),
        ("task_manifest_sha256", "different_manifest", "task manifest is not locked"),
        ("max_recurrence", 4, "Recurrent depth/insertion settings differ"),
        ("insertion_fraction", 0.333333, "Recurrent depth/insertion settings differ"),
        ("checkpoint_path", Path("/checkpoints/last.pt"), "best.pt checkpoint"),
    ],
)
def test_external_policy_rejects_protocol_mismatches(
    tmp_path: Path, field: str, value, message: str
) -> None:
    policy, protocol = _frozen_policy(tmp_path)
    with pytest.raises(ValueError, match=message):
        _validate_policy(policy, protocol, **{field: value})
