#!/usr/bin/env python3
"""Preflight and freeze the external-evaluation policy after training/calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("xcodeeval", "humanevalx", "transcoder_test", "avatar_tc")
SPLITS = ("common_core", "full_directional")
ADAPTIVE_ROUTERS = {"region-adaptive", "content+direction region-adaptive", "token-adaptive"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_calibrations(root: Path) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if value.get("status") == "VALIDATION_CALIBRATED":
            value["_path"] = str(path.resolve())
            rows.append(value)
    return rows


def preflight(
    registry_path: Path,
    calibration_root: Path,
    cache_root: Path,
    translation_protocol: Path,
) -> dict[str, Any]:
    with registry_path.open(newline="", encoding="utf-8") as stream:
        all_runs = list(csv.DictReader(stream))
    runs = [
        row for row in all_runs
        if row.get("training_horizon_policy") != "up-to-5-epoch-validation-pilot"
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]
    missing: list[str] = []
    for row in runs:
        checkpoint = ROOT / (row.get("checkpoint_path") or "")
        if row.get("training_status") != "TRAINED":
            missing.append(f"run not TRAINED: {row['experiment_id']} ({row['training_status']})")
        if not checkpoint.is_file() or checkpoint.name != "best.pt":
            missing.append(f"best checkpoint absent: {row['experiment_id']} ({checkpoint})")

    cache_hashes: dict[str, dict[str, str]] = {}
    for benchmark in BENCHMARKS:
        cache_hashes[benchmark] = {}
        for split in SPLITS:
            path = cache_root / benchmark / split / "manifest.json"
            key = f"{benchmark}:{split}"
            if not path.is_file():
                missing.append(f"external structural cache manifest absent: {path}")
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "COMPLETE":
                missing.append(f"external structural cache incomplete: {key}")
            digest = manifest.get("input_manifest_sha256")
            if not digest:
                missing.append(f"cache lacks frozen task-manifest digest: {key}")
            else:
                cache_hashes[benchmark][split] = digest

    calibrations = _read_calibrations(calibration_root)
    needed_calibrations: list[dict[str, Any]] = []
    revisions: dict[str, dict[str, Any]] = {}
    routing_policies: dict[str, dict[str, Any]] = {}
    # Only seed 42 defines each architecture/revision policy; all other seeds
    # receive that same policy by the predeclared Decision 97.
    policy_runs: dict[tuple[str, str], dict[str, Any]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for row in runs:
        if row.get("seed") != "42":
            continue
        checkpoint = ROOT / (row.get("checkpoint_path") or "")
        if not checkpoint.is_file():
            continue
        # mmap keeps this metadata audit from copying large model state tensors.
        import torch

        value = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        config = value.get("config", {})
        revision = Path(str(config.get("backbone_path", ""))).name
        if not revision:
            missing.append(f"checkpoint lacks pinned backbone revision: {row['experiment_id']}")
            continue
        arch = row["architecture"]
        key = (revision, arch)
        if key in policy_runs:
            previous = policy_runs[key]
            if previous["experiment_id"] != row["experiment_id"]:
                missing.append(f"ambiguous seed-42 run for {revision}/{arch}")
            continue
        policy_runs[key] = row
        if revision not in revisions:
            revisions[revision] = {"horizon": int(config["epochs"]), "backbone_path": str(config["backbone_path"])}
        elif revisions[revision]["horizon"] != int(config["epochs"]):
            missing.append(f"inconsistent horizon within backbone revision {revision}")
        router = row.get("router", "none")
        routing_policies.setdefault(revision, {})[arch] = {
            "hard_threshold": None,
            "top_k_fraction": None,
            "max_recurrence": int(config.get("max_recurrence", 1)),
            "insertion_fraction": float(config.get("insertion_fraction", 0.5)),
        }
        if router in ADAPTIVE_ROUTERS:
            ckpt_hash = sha256(checkpoint)
            candidates = [
                item for item in calibrations
                if item.get("architecture") == arch
                and int(item.get("seed", -1)) == 42
                and item.get("checkpoint_sha256") == ckpt_hash
                and Path(str(item.get("checkpoint", ""))).resolve() == checkpoint.resolve()
            ]
            if len(candidates) != 1:
                missing.append(f"one seed-42 validation calibration required for {revision}/{arch}; found {len(candidates)}")
                needed_calibrations.append({"backbone_revision": revision, "architecture": arch, "checkpoint": str(checkpoint)})
            else:
                selected = candidates[0].get("selected", {})
                routing_policies[revision][arch]["hard_threshold"] = float(selected["threshold"])
                routing_policies[revision][arch]["calibration_sha256"] = sha256(Path(candidates[0]["_path"]))
                routing_policies[revision][arch]["calibration_path"] = candidates[0]["_path"]
                routing_policies[revision][arch]["calibration_seed"] = 42
        loaded[row["experiment_id"]] = config

    # Every planned backbone/architecture family must have its seed-42 anchor.
    expected_keys = set()
    for row in runs:
        if row.get("seed") == "42":
            config = loaded.get(row["experiment_id"])
            if config:
                expected_keys.add((Path(str(config.get("backbone_path", ""))).name, row["architecture"]))
    missing_keys = expected_keys - set(policy_runs)
    for revision, arch in sorted(missing_keys):
        missing.append(f"seed-42 checkpoint missing for policy key {revision}/{arch}")

    return {
        "ready": not missing,
        "status": "READY_TO_FREEZE" if not missing else "NOT_READY",
        "registered_model_runs": len(runs),
        "trained_model_runs": sum(row.get("training_status") == "TRAINED" for row in runs),
        "missing_prerequisites": missing,
        "required_calibrations": needed_calibrations,
        "training_horizon_epochs": {revision: meta["horizon"] for revision, meta in revisions.items()},
        "backbone_paths": {revision: meta["backbone_path"] for revision, meta in revisions.items()},
        "routing_policies": routing_policies,
        "external_task_manifests": {
            f"{benchmark}:{split}": digest
            for benchmark, splits in cache_hashes.items()
            for split, digest in splits.items()
        },
        "translation_protocol_sha256": sha256(translation_protocol),
        "translation_protocol_path": str(translation_protocol.resolve()),
        "cache_root": str(cache_root.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    parser.add_argument("--calibration-root", type=Path, default=Path("reports/calibration"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/structural/external"))
    parser.add_argument("--translation-protocol", type=Path, default=Path("docs/TRANSLATION_PROTOCOL.md"))
    parser.add_argument("--preflight-report", type=Path, default=Path("reports/progress/external_protocol_preflight.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation/frozen_external_protocol.json"))
    parser.add_argument("--preflight-only", action="store_true", help="Write readiness evidence without freezing the protocol")
    args = parser.parse_args()
    report = preflight(args.registry, args.calibration_root, args.cache_root, args.translation_protocol)
    args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["ready"]:
        print(json.dumps({"status": report["status"], "missing_count": len(report["missing_prerequisites"]), "required_calibrations": len(report["required_calibrations"])}, sort_keys=True))
        return
    if args.preflight_only:
        print(json.dumps({"status": report["status"], "missing_count": 0, "required_calibrations": 0}, sort_keys=True))
        return
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == "FROZEN_PREMODEL":
            raise FileExistsError(f"Refusing to overwrite frozen protocol: {args.output}")
    generation = {
        "do_sample": True, "num_beams": 1, "max_new_tokens": 768,
        "temperature": 0.8, "top_p": 0.95, "repetition_penalty": 1.0,
        "postprocess": "extract_generated_code_v1",
        "batch_size": 2,
        "pass_at_1": {"do_sample": True, "uses_sample_index": 0,
                       "num_samples": 1, "temperature": 0.8,
                       "top_p": 0.95, "num_beams": 1,
                       "seed_rule": "sha256(experiment_id|task_id|sample_index) mod 2^32"},
        "eos_token_id_by_revision": {},
    }
    from transformers import AutoTokenizer

    for revision, path in report["backbone_paths"].items():
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        generation["eos_token_id_by_revision"][revision] = tokenizer.eos_token_id
    record = {
        "protocol_version": 3,
        "amendment": "2026-10-12 user-directed before external scoring: one seeded stochastic candidate per task; Pass@1 only.",
        "schema_version": 1, "status": "FROZEN_PREMODEL",
        "selection_metric": "validation_target_loss",
        "external_metrics_used_for_selection": False,
        "training_horizon_epochs": report["training_horizon_epochs"],
        "translation_protocol_sha256": report["translation_protocol_sha256"],
        "external_task_manifests": report["external_task_manifests"],
        "generation": generation,
        "routing_policies": report["routing_policies"],
        "routing_calibration_rule": "seed-42 validation calibration per architecture/backbone revision, shared across seeds",
        "calibration_constraint": "validation_target_loss among thresholds with realized mean recurrent depth <= configured budget + 0.05",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "backbone_revisions": len(record["training_horizon_epochs"]), "routing_policies": sum(len(v) for v in record["routing_policies"].values())}, sort_keys=True))


if __name__ == "__main__":
    main()
