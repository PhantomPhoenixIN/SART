"""Guardrails for hash-addressed confirmatory training configuration generation."""

import csv
from pathlib import Path

import pytest
import yaml

from scripts.train.prepare_confirmatory_runs import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_run_generation_refuses_unfrozen_horizon(tmp_path: Path) -> None:
    horizon = {
        "confirmatory_7b": {
            "frozen_epochs": None,
            "architectures": ["dense"],
            "three_seed_architectures": ["dense"],
            "single_seed_screening_architectures": [],
        }
    }
    horizon_path = tmp_path / "horizon.yaml"
    horizon_path.write_text(yaml.safe_dump(horizon))
    with pytest.raises(ValueError, match="Freeze confirmatory"):
        prepare(
            horizon_path,
            ROOT / "configs/training/pilot_dense_external_clean.yaml",
            tmp_path / "registry.csv",
            tmp_path / "configs",
            dry_run=True,
        )


def test_run_generation_creates_hash_ids_and_preserves_pilot_registry_rows(tmp_path: Path) -> None:
    horizon_path = ROOT / "configs/training/horizon.yaml"
    horizon = yaml.safe_load(horizon_path.read_text())
    horizon["confirmatory_7b"]["frozen_epochs"] = 3
    temporary_horizon = tmp_path / "horizon.yaml"
    temporary_horizon.write_text(yaml.safe_dump(horizon))
    registry = tmp_path / "registry.csv"
    registry.write_text(
        "experiment_id,architecture,backbone,seed,training_horizon_policy\n"
        "plan-pilot,dense,Qwen/Qwen2.5-Coder-7B,42,up-to-5-epoch-validation-pilot\n"
        "plan-old,dense,Qwen/Qwen2.5-Coder-7B,13,shared-7B-horizon\n"
        "plan-14b,dense,Qwen/Qwen2.5-Coder-14B,42,2-epoch\n",
        encoding="utf-8",
    )
    config_dir = tmp_path / "configs"
    rows = prepare(
        temporary_horizon,
        ROOT / "configs/training/pilot_dense_external_clean.yaml",
        registry,
        config_dir,
    )
    assert len(rows) == 40
    assert len({row["experiment_id"] for row in rows}) == 40
    assert all(row["experiment_id"].startswith(row["architecture"] + "-seed") for row in rows)
    assert all((config_dir / f"{row['experiment_id']}.yaml").is_file() for row in rows)
    with registry.open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    ids = {row["experiment_id"] for row in saved}
    assert "plan-pilot" in ids
    assert "plan-old" not in ids
    assert "plan-14b" in ids
    assert ids.issuperset(row["experiment_id"] for row in rows)
