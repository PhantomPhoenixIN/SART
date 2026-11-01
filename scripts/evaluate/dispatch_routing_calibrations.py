#!/usr/bin/env python3
"""Schedule seed-42 routing calibrations as their anchor checkpoints become ready."""

from __future__ import annotations

import argparse
import csv
import fcntl
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

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
ATTEMPTS = ROOT / "reports/progress/routing_calibration_jobs.csv"
PLAN = ROOT / "reports/progress/routing_calibration_plan.json"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/routing_calibration_dispatcher.log"
WORKER = ROOT / "run_on_astra/routing_calibration_worker.pbs"
ACTIVE_STATES = {"R", "Q", "H", "W"}
GPU_OCCUPIED_STATES = ACTIVE_STATES | {"E", "T"}
PILOT_POLICY = "up-to-5-epoch-validation-pilot"
FROZEN_PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scheduled_gpu_cap(weekday_cap: int, weekend_cap: int) -> int:
    return weekend_cap if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() >= 5 else weekday_cap


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_csv(path: Path, default_fields: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return default_fields, []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or default_fields), list(reader)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def exit_status(job_id: str) -> int | None:
    result = subprocess.run(["qstat", "-xf", job_id], text=True, capture_output=True, check=False)
    match = re.search(r"^\s*Exit_status\s*=\s*(-?\d+)\s*$", result.stdout, re.M)
    return int(match.group(1)) if match else None


def _calibration_key(item: dict[str, Any]) -> str:
    return f"{item['backbone_revision']}__{item['architecture']}__seed42"


def _valid_calibration(path: Path, checkpoint: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("status") != "VALIDATION_CALIBRATED":
        return False
    digest = file_sha256(checkpoint)
    return (
        record.get("checkpoint_sha256") == digest
        and int(record.get("seed", -1)) == 42
    )


def _trained_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("training_horizon_policy") != PILOT_POLICY
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]


def reconcile(jobs: list[dict[str, str]], states: dict[str, str]) -> bool:
    changed = False
    for row in jobs:
        job_id = row.get("pbs_job_id", "")
        if not job_id or row.get("status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
            continue
        state = states.get(job_id)
        if state in ACTIVE_STATES:
            row["status"] = "RUNNING" if state == "R" else "QUEUED"
            changed = True
            continue
        code = exit_status(job_id)
        calibration = Path(row["output_path"])
        checkpoint = ROOT / row["checkpoint_path"]
        if code in {0, None} and _valid_calibration(calibration, checkpoint):
            row["status"] = "COMPLETE"
            log(f"calibration complete {row['calibration_key']} job={job_id}")
        elif code is not None:
            row["status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            log(f"calibration retryable {row['calibration_key']} job={job_id} exit={code}")
        else:
            continue
        changed = True
    return changed


def submit_task(
    task: dict[str, Any],
    job_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
) -> dict[str, str]:
    checkpoint = Path(str(task["checkpoint"]))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file() or checkpoint.name != "best.pt":
        raise FileNotFoundError(f"Seed-42 best checkpoint missing: {checkpoint}")
    output_path = ROOT / "reports/calibration" / str(task["backbone_revision"]) / f"{task['architecture']}-seed42.json"
    key = _calibration_key(task)
    previous = [row for row in job_rows if row.get("calibration_key") == key]
    attempt_number = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
    matching = [
        item for item in run_rows
        if (ROOT / item.get("checkpoint_path", "")).resolve() == checkpoint.resolve()
    ]
    if len(matching) != 1:
        raise ValueError(f"Expected one registry run for calibration checkpoint {checkpoint}")
    run_id = matching[0]["experiment_id"]
    log_dir = ROOT / "logs/routing_calibrations"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment_key = key.replace("/", "_")
    stem = f"{environment_key}.attempt{attempt_number}"
    command = [
        "qsub", "-N", f"sart_cal_{task['architecture'][:8]}",
        "-o", str(log_dir / f"{stem}.out"),
        "-e", str(log_dir / f"{stem}.err"),
        "-v", (
            f"SART_CHECKPOINT={checkpoint.relative_to(ROOT)},"
            f"SART_CALIBRATION_OUTPUT={output_path.relative_to(ROOT)},"
            f"SART_CALIBRATION_KEY={environment_key},SART_ATTEMPT={attempt_number}"
        ),
        str(WORKER),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().splitlines()[-1]
    row = {
        "calibration_key": key,
        "backbone_revision": str(task["backbone_revision"]),
        "architecture": str(task["architecture"]),
        "experiment_id": run_id,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "output_path": str(output_path.resolve()),
        "attempt": str(attempt_number),
        "pbs_job_id": job_id,
        "status": "SUBMITTED",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    return row


def dispatch_once(cap: int, user: str) -> None:
    if FROZEN_PROTOCOL.exists():
        frozen = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
        if frozen.get("status") != "FROZEN_PREMODEL":
            raise ValueError(f"Refusing to act on an unexpected protocol file: {FROZEN_PROTOCOL}")
        log("calibration dispatcher stopping: protocol is already frozen")
        return
    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        registry_fields, rows = read_csv(REGISTRY, [])
        trained_runs = _trained_runs(rows)
        registry_sha = file_sha256(REGISTRY)

        job_fields = [
            "calibration_key", "backbone_revision", "architecture", "experiment_id", "checkpoint_path",
            "output_path", "attempt", "pbs_job_id", "status", "submitted_utc",
        ]
        job_fields, jobs = read_csv(ATTEMPTS, job_fields)
        states = queue_states(user)
        if reconcile(jobs, states):
            write_csv_atomic(ATTEMPTS, job_fields, jobs)

        preflight_cmd = [
            str(ROOT / "env/bin/python"), "scripts/evaluate/freeze_external_protocol.py",
            "--output", str(FROZEN_PROTOCOL), "--preflight-only",
        ]
        freeze_cmd = [
            str(ROOT / "env/bin/python"), "scripts/evaluate/freeze_external_protocol.py",
            "--output", str(FROZEN_PROTOCOL),
        ]
        if PLAN.is_file():
            saved_plan = json.loads(PLAN.read_text(encoding="utf-8"))
        else:
            saved_plan = {}
        if saved_plan.get("registry_sha256") != registry_sha:
            subprocess.run(preflight_cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            report = json.loads((ROOT / "reports/progress/external_protocol_preflight.json").read_text())
            all_runs_trained = (
                len(trained_runs) == 55
                and all(row.get("training_status") == "TRAINED" for row in trained_runs)
            )
            if all_runs_trained:
                validation_refresh = subprocess.run(
                    [str(ROOT / "env/bin/python"), "scripts/analysis/summarize_in_domain_validation.py"],
                    cwd=ROOT, text=True, capture_output=True, check=True,
                )
                log(f"refreshed in-domain validation summary after all 55 training runs: {validation_refresh.stdout.strip()}")
                subprocess.run(preflight_cmd, cwd=ROOT, text=True, capture_output=True, check=True)
                report = json.loads((ROOT / "reports/progress/external_protocol_preflight.json").read_text())

            trained_checkpoints = {
                (ROOT / row.get("checkpoint_path", "")).resolve()
                for row in trained_runs
                if row.get("training_status") == "TRAINED"
            }
            eligible = []
            for task in report.get("required_calibrations", []):
                checkpoint = Path(str(task.get("checkpoint", "")))
                if not checkpoint.is_absolute():
                    checkpoint = ROOT / checkpoint
                if checkpoint.resolve() in trained_checkpoints and checkpoint.is_file():
                    eligible.append(task)
            saved_plan = {
                "schema_version": 1,
                "registry_sha256": registry_sha,
                "required_calibrations": eligible,
                "eligible_calibrations": len(eligible),
                "preflight_required_calibrations": len(report.get("required_calibrations", [])),
                "created_utc": now(),
            }
            PLAN.parent.mkdir(parents=True, exist_ok=True)
            PLAN.write_text(json.dumps(saved_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(
                "calibration eligibility refreshed: "
                f"{len(eligible)}/{len(report.get('required_calibrations', []))} "
                "required seed-42 checkpoints are TRAINED and present"
            )
        required = saved_plan.get("required_calibrations", [])

        active = {
            job_id for job_id, state in states.items()
            if state in GPU_OCCUPIED_STATES
        }
        active.update(
            row["pbs_job_id"] for row in jobs
            if row.get("pbs_job_id")
            and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        )
        completed = {row.get("calibration_key") for row in jobs if row.get("status") == "COMPLETE"}
        known = {
            row.get("calibration_key") for row in jobs
            if row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        }
        pending = [
            task for task in required
            if _calibration_key(task) not in completed
            and _calibration_key(task) not in known
        ]
        slots = max(0, cap - len(active))
        submitted = []
        for task in pending[:slots]:
            job = submit_task(task, jobs, rows)
            jobs.append(job)
            submitted.append(job)
            write_csv_atomic(ATTEMPTS, job_fields, jobs)
        if submitted:
            log(f"cap={cap} active={len(active)} submitted={[(r['calibration_key'], r['pbs_job_id']) for r in submitted]}")
        elif not pending and not active and not FROZEN_PROTOCOL.exists():
            result = subprocess.run(preflight_cmd, cwd=ROOT, text=True, capture_output=True, check=True)
            report = json.loads((ROOT / "reports/progress/external_protocol_preflight.json").read_text())
            if report.get("ready"):
                subprocess.run(freeze_cmd, cwd=ROOT, text=True, capture_output=True, check=True)
                log("all calibrations complete; frozen external protocol created")
            else:
                log(f"no pending calibrations; preflight={report.get('status')}, missing={len(report.get('missing_prerequisites', []))}; {result.stdout.strip()}")


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
    log(f"dispatcher started weekday_cap={args.cap} weekend_cap={args.weekend_cap} timezone=Asia/Kolkata interval_seconds={args.interval_seconds}")
    while True:
        try:
            dispatch_once(scheduled_gpu_cap(args.cap, args.weekend_cap), args.user)
        except Exception as exc:
            log(f"dispatcher error {type(exc).__name__}: {exc}")
        if args.once or FROZEN_PROTOCOL.exists():
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
