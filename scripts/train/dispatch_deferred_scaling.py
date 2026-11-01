#!/usr/bin/env python3
"""Dispatch staged scaling runs in remaining GPU slots, within a GPU cap."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
ATTEMPTS = ROOT / "experiments/pbs_attempts.csv"
WORKER = ROOT / "run_on_astra/confirmatory_7b_worker.pbs"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/deferred_scaling_dispatcher.log"
CALIBRATION_PLAN = ROOT / "reports/progress/routing_calibration_plan.json"
CALIBRATION_JOBS = ROOT / "reports/progress/routing_calibration_jobs.csv"
ACTIVE_STATES = {"R", "Q", "H", "W"}
GPU_OCCUPIED_STATES = ACTIVE_STATES | {"E", "T"}
DEFERRED_POLICIES = {
    "shared-2-epoch-qwen14b-horizon",
    "shared-2-epoch-starcoder2-7b-horizon",
    "secondary-shared-2-epoch-1.5B-deferred",
}
POLICY_ORDER = {
    "shared-2-epoch-qwen14b-horizon": 0,
    "shared-2-epoch-starcoder2-7b-horizon": 1,
    "secondary-shared-2-epoch-1.5B-deferred": 2,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scheduled_gpu_cap(weekday_cap: int, weekend_cap: int) -> int:
    return weekend_cap if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() >= 5 else weekday_cap


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    fd, name = tempfile.mkstemp(prefix=f"{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def queue_states(user: str) -> dict[str, str]:
    result = subprocess.run(["qstat", "-u", user], text=True, capture_output=True, check=True)
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 11 and fields[0].endswith(".hn01") and fields[2] == "gpuq":
            states[fields[0]] = fields[9]
    return states


def calibration_backlog(states: dict[str, str]) -> str | None:
    """Return why a free GPU should go to validation calibration first."""
    if not CALIBRATION_PLAN.is_file() or not CALIBRATION_JOBS.is_file():
        return None
    try:
        plan = json.loads(CALIBRATION_PLAN.read_text(encoding="utf-8"))
        _, jobs = read_csv(CALIBRATION_JOBS)
    except (OSError, json.JSONDecodeError):
        return "calibration plan/status is unreadable; waiting for calibration dispatcher"

    registry_sha = hashlib.sha256(REGISTRY.read_bytes()).hexdigest()
    if plan.get("registry_sha256") != registry_sha:
        return "calibration plan is stale for the current registry; waiting for refresh"

    jobs_by_key: dict[str, list[dict[str, str]]] = {}
    for job in jobs:
        jobs_by_key.setdefault(job.get("calibration_key", ""), []).append(job)
    for task in plan.get("required_calibrations", []):
        key = f"{task['backbone_revision']}__{task['architecture']}__seed42"
        matching = jobs_by_key.get(key, [])
        if any(job.get("status") == "COMPLETE" for job in matching):
            continue
        # A submitted calibration that disappeared from qstat must first be
        # reconciled by the calibration dispatcher before scaling takes over.
        if any(
            job.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
            for job in matching
        ):
            return f"required calibration {key} is active or awaiting reconciliation"
        return f"required calibration {key} is pending"
    return None


def checkpoint_status(row: dict[str, str]) -> dict[str, Any] | None:
    checkpoint = ROOT / row.get("checkpoint_path", "")
    path = checkpoint.parent / "status.jsonl"
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        value = json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def exit_status(job_id: str) -> int | None:
    result = subprocess.run(["qstat", "-xf", job_id], text=True, capture_output=True, check=False)
    match = re.search(r"^\s*Exit_status\s*=\s*(-?\d+)\s*$", result.stdout, re.M)
    return int(match.group(1)) if match else None


def append_log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_model_artifact(config_path: Path) -> None:
    """Refuse PBS submission unless every indexed model shard is staged."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = ROOT / str(config["backbone_path"])
    manifest_path = model_path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model artifact manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Model artifact is not COMPLETE: {manifest_path}")
    expected_revision = model_path.name
    if manifest.get("revision") != expected_revision:
        raise ValueError(
            f"Model artifact revision mismatch for {model_path}: "
            f"{manifest.get('revision')} != {expected_revision}"
        )
    files = manifest.get("files", {})
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        required = set(index.get("weight_map", {}).values())
    else:
        required = {path.name for path in model_path.glob("*.safetensors")}
    if not required:
        raise ValueError(f"No model weight files found under {model_path}")
    missing = sorted(
        name for name in required
        if name not in files
        or not (model_path / name).is_file()
        or (model_path / name).stat().st_size != files[name].get("size_bytes")
    )
    if missing:
        raise ValueError(
            f"Model artifact manifest has missing/incomplete indexed weights: {missing}"
        )


def reconcile(
    rows: list[dict[str, str]], attempts: list[dict[str, str]], states: dict[str, str]
) -> bool:
    attempts_by_job = {entry.get("pbs_job_id", ""): entry for entry in attempts}
    changed = False
    for row in rows:
        job_id = row.get("pbs_job_id", "")
        if not job_id or row.get("training_status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
            continue
        state = states.get(job_id)
        if state in ACTIVE_STATES:
            row["training_status"] = "RUNNING" if state == "R" else "QUEUED"
            changed = True
            continue
        details = checkpoint_status(row)
        status = str((details or {}).get("status", ""))
        code = exit_status(job_id)
        attempt = attempts_by_job.get(job_id)
        if status == "TRAINED" and code in {0, None}:
            row["training_status"] = "TRAINED"
            if attempt:
                attempt["status"] = "COMPLETE"
            append_log(f"reconciled {job_id} {row['experiment_id']} as TRAINED (exit={code})")
        elif status == "PREEMPTED" or (code is not None and code != 0):
            row["training_status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            if attempt:
                attempt["status"] = "PREEMPTED" if status == "PREEMPTED" else "FAILED_RETRYABLE"
            append_log(f"reconciled {job_id} {row['experiment_id']} as retryable (exit={code})")
        else:
            # PBS may publish exit_status=0 just before the worker's atomic
            # TRAINED status file becomes visible. Keep the row pending until
            # application state is durable; never launch a duplicate retry.
            continue
        changed = True
    return changed


def submit(row: dict[str, str], attempts: list[dict[str, str]], attempt_fields: list[str]) -> str:
    run_id = row["experiment_id"]
    config = ROOT / row["config_path"]
    if not config.is_file():
        raise FileNotFoundError(config)
    validate_model_artifact(config)
    attempt_number = 1 + sum(
        record.get("experiment_id") == run_id and record.get("attempt", "").isdigit()
        for record in attempts
    )
    log_dir = ROOT / "logs/deferred_scaling"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run_id}.attempt{attempt_number}"
    command = [
        "qsub", "-N", f"sart_{run_id[:10]}",
        "-o", str(log_dir / f"{stem}.out"),
        "-e", str(log_dir / f"{stem}.err"),
        "-v", f"SART_CONFIG={config.relative_to(ROOT)},SART_RUN_ID={run_id},SART_ATTEMPT={attempt_number}",
        str(WORKER),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().splitlines()[-1]
    attempts.append({
        "experiment_id": run_id,
        "attempt": str(attempt_number),
        "pbs_job_id": job_id,
        "status": "SUBMITTED",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "deferred scaling dispatch",
    })
    row["training_status"] = "SUBMITTED"
    row["pbs_job_id"] = job_id
    return job_id


def dispatch_once(cap: int, user: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fields, rows = read_csv(REGISTRY)
        attempt_fields, attempts = read_csv(ATTEMPTS)
        states = queue_states(user)
        if reconcile(rows, attempts, states):
            write_csv_atomic(REGISTRY, fields, rows)
            write_csv_atomic(ATTEMPTS, attempt_fields, attempts)

        active = {
            job_id for job_id, state in states.items()
            if state in GPU_OCCUPIED_STATES
        }
        deferred = [
            row for row in rows
            if row.get("training_horizon_policy") in DEFERRED_POLICIES
            and row.get("training_status") != "CANCELLED_SUPERSEDED"
        ]
        expected_counts = {
            "shared-2-epoch-qwen14b-horizon": 3,
            "shared-2-epoch-starcoder2-7b-horizon": 3,
            "secondary-shared-2-epoch-1.5B-deferred": 9,
        }
        actual_counts = {
            policy: sum(row.get("training_horizon_policy") == policy for row in deferred)
            for policy in expected_counts
        }
        if actual_counts != expected_counts:
            append_log(f"deferred: scaling registry mismatch expected={expected_counts} actual={actual_counts}")
            return
        pending: list[dict[str, str]] = []
        for policy in sorted(DEFERRED_POLICIES, key=POLICY_ORDER.__getitem__):
            policy_rows = [
                row for row in deferred if row["training_horizon_policy"] == policy
            ]
            pending.extend(
                row for row in policy_rows
                if row.get("training_status") in {"PLANNED", "FAILED_RETRYABLE"}
            )
        pending.sort(key=lambda row: (
            POLICY_ORDER[row.get("training_horizon_policy", "")],
            row.get("architecture", ""),
            int(row.get("seed", "0")),
        ))
        slots = max(0, cap - len(active))
        if slots:
            reason = calibration_backlog(states)
            if reason:
                if "active or awaiting reconciliation" in reason:
                    # The calibration is already submitted (and therefore has
                    # its place in PBS's queue). Keep one project slot open for
                    # the next calibration while allowing deferred training to
                    # queue/run in the remaining free slots.
                    slots = max(0, slots - 1)
                    append_log(
                        f"deferred: reserving 1 free GPU slot for active calibration; "
                        f"dispatching in {slots} remaining slot(s); {reason}"
                    )
                else:
                    append_log(f"deferred: yielding {slots} free GPU slot(s) to calibration dispatcher; {reason}")
                    return
        submitted = []
        for row in pending[:slots]:
            submitted.append((row["experiment_id"], submit(row, attempts, attempt_fields)))
        if submitted:
            write_csv_atomic(REGISTRY, fields, rows)
            write_csv_atomic(ATTEMPTS, attempt_fields, attempts)
            append_log(f"cap={cap} active={len(active)} submitted={submitted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--weekend-cap", type=int, default=4)
    parser.add_argument("--user", default=__import__("getpass").getuser())
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.cap < 1 or args.weekend_cap < args.cap or args.interval_seconds < 10:
        parser.error("--cap must be positive, --weekend-cap must be >= --cap, and interval must be at least 10 seconds")
    append_log(f"dispatcher started weekday_cap={args.cap} weekend_cap={args.weekend_cap} timezone=Asia/Kolkata interval_seconds={args.interval_seconds}")
    while True:
        try:
            dispatch_once(scheduled_gpu_cap(args.cap, args.weekend_cap), args.user)
        except Exception as exc:
            append_log(f"dispatcher error {type(exc).__name__}: {exc}")
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
