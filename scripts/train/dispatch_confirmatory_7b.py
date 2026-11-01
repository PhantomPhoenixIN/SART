#!/usr/bin/env python3
"""Keep at most a small number of confirmatory jobs in PBS at once.

The dispatcher counts this project's running and queued GPU jobs together. When
a job exits, it reconciles the checkpoint status into the experiment registry,
then submits only enough planned runs to refill the configured concurrency cap.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
ATTEMPTS = ROOT / "experiments/pbs_attempts.csv"
LOG = ROOT / "logs/confirmatory_7b/dispatcher.log"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
ACTIVE_STATES = {"R", "Q", "H", "W"}
GPU_OCCUPIED_STATES = ACTIVE_STATES | {"E", "T"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scheduled_gpu_cap(weekday_cap: int, weekend_cap: int) -> int:
    return weekend_cap if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() >= 5 else weekday_cap


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def queue_states(user: str) -> dict[str, str]:
    result = subprocess.run(
        ["qstat", "-u", user], text=True, capture_output=True, check=True
    )
    states: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 11 and fields[0].endswith(".hn01") and fields[2] == "gpuq":
            states[fields[0]] = fields[9]
    return states


def checkpoint_status(row: dict[str, str]) -> dict[str, object] | None:
    experiment_id = row.get("experiment_id", "")
    path = ROOT / "checkpoints/confirmatory_7b" / experiment_id / "status.jsonl"
    if not path.is_file():
        return None
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        value = json.loads(lines[-1]) if lines else None
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def exit_status(job_id: str) -> int | None:
    result = subprocess.run(
        ["qstat", "-xf", job_id], text=True, capture_output=True, check=False
    )
    match = re.search(r"^\s*Exit_status\s*=\s*(-?\d+)\s*$", result.stdout, re.M)
    return int(match.group(1)) if match else None


def append_log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def reconcile(
    rows: list[dict[str, str]], attempts: list[dict[str, str]], states: dict[str, str]
) -> bool:
    attempts_by_job = {entry.get("pbs_job_id", ""): entry for entry in attempts}
    changed = False
    for row in rows:
        job_id = row.get("pbs_job_id", "")
        # QUEUED is also a live managed state: once PBS removes a queued job
        # from qstat, reconcile its final accounting/application status instead
        # of leaving a stale QUEUED registry row forever.
        if not job_id or row.get("training_status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
            continue
        state = states.get(job_id)
        if state in ACTIVE_STATES:
            row["training_status"] = "QUEUED" if state != "R" else "RUNNING"
            changed = True
            continue

        details = checkpoint_status(row)
        status = str((details or {}).get("status", ""))
        code = exit_status(job_id)
        attempt = attempts_by_job.get(job_id)
        if status == "TRAINED" and code in {0, None}:
            row["training_status"] = "TRAINED"
            if attempt is not None:
                attempt["status"] = "COMPLETE"
            append_log(
                f"reconciled {job_id} ({row['experiment_id']}) as TRAINED; "
                f"exit_status={code}"
            )
        elif status == "PREEMPTED":
            row["training_status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            if attempt is not None:
                attempt["status"] = "PREEMPTED"
            append_log(
                f"reconciled {job_id} ({row['experiment_id']}) as PREEMPTED; checkpoint retained"
            )
        elif code is not None and code != 0:
            row["training_status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            if attempt is not None:
                attempt["status"] = "FAILED_RETRYABLE"
            append_log(
                f"reconciled {job_id} ({row['experiment_id']}) exit_status={code}; retryable"
            )
        else:
            # Do not submit a replacement when the scheduler's final state is
            # unknown. A later dispatcher pass can reconcile it when qstat
            # accounting becomes available.
            continue
        changed = True
    return changed


def dispatch_once(cap: int, user: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        registry_fields, rows = read_csv(REGISTRY)
        attempt_fields, attempts = read_csv(ATTEMPTS)
        states = queue_states(user)
        changed = reconcile(rows, attempts, states)
        if changed:
            write_csv_atomic(REGISTRY, registry_fields, rows)
            write_csv_atomic(ATTEMPTS, attempt_fields, attempts)

        active = {
            job_id for job_id, state in states.items()
            if state in GPU_OCCUPIED_STATES
        }
        free_slots = max(0, cap - len(active))
        if free_slots:
            result = subprocess.run(
                [
                    str(ROOT / "env/bin/python"),
                    str(ROOT / "scripts/train/submit_confirmatory_7b.py"),
                    "--limit",
                    str(free_slots),
                    "--priority-full-sart",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            output = result.stdout.strip().replace("\n", " | ")
            if output:
                append_log(f"cap={cap} active_before_submit={len(active)} {output}")


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
        except Exception as exc:  # keep monitoring alive; preserve exact failure
            append_log(f"dispatcher error {type(exc).__name__}: {exc}")
        if args.once:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
