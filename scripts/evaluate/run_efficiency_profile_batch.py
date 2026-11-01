#!/usr/bin/env python3
"""Profile consecutive unclaimed models within one PBS GPU allocation."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import dispatch_efficiency_profiles as dispatch  # noqa: E402

MIN_REMAINING_SECONDS = 600
USABLE_WALLTIME_SECONDS = 95 * 3600


def _tasks(protocol: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    with dispatch.REGISTRY.open(newline="", encoding="utf-8") as stream:
        runs = dispatch.model_runs(list(csv.DictReader(stream)))
    if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
        raise ValueError("Profile batch requires all 55 trained models")
    return dispatch.tasks_for_runs(runs, protocol, plan)


def _assert_active_claim(key: str, job_id: str) -> None:
    with dispatch.LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_SH)
        _, jobs = dispatch.read_csv(dispatch.JOBS, [])
        matching = [row for row in jobs
                    if row.get("experiment_id") == key
                    and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}]
        if len(matching) != 1 or matching[0].get("pbs_job_id") != job_id:
            raise ValueError(f"Profile {key} is not exclusively claimed by {job_id}")


def _transition_and_claim(
    current_id: str,
    job_id: str,
    success: bool,
    attempted: set[str],
    allow_next: bool,
) -> dict[str, Any] | None:
    with dispatch.LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        protocol = json.loads(dispatch.PROTOCOL.read_text(encoding="utf-8"))
        plan = json.loads(dispatch.PROFILE_PLAN.read_text(encoding="utf-8"))
        tasks = _tasks(protocol, plan)
        by_id = {task["experiment_id"]: task for task in tasks}
        if current_id not in by_id:
            raise ValueError(f"Unknown profile task {current_id}")
        fields = [
            "experiment_id", "backbone", "revision", "architecture", "seed",
            "checkpoint_path", "checkpoint_sha256", "sample_plan_sha256", "output_path",
            "attempt", "pbs_job_id", "status", "submitted_utc",
        ]
        fields, jobs = dispatch.read_csv(dispatch.JOBS, fields)
        current = next((row for row in reversed(jobs)
                        if row.get("experiment_id") == current_id
                        and row.get("pbs_job_id") == job_id), None)
        if current is None or current.get("status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
            raise ValueError(f"No active profile ledger claim for {current_id} in {job_id}")
        plan_digest = dispatch.sha256(dispatch.PROFILE_PLAN)
        verified = dispatch.profile_complete(
            ROOT / by_id[current_id]["output"], by_id[current_id], plan_digest
        )
        current["status"] = "COMPLETE" if success and verified else "FAILED_RETRYABLE"
        attempted.add(current_id)
        next_task = None
        if allow_next:
            complete = {task["experiment_id"] for task in tasks
                        if dispatch.profile_complete(ROOT / task["output"], task, plan_digest)}
            active = {row["experiment_id"] for row in jobs
                      if row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}}
            next_task = next((task for task in tasks
                              if task["experiment_id"] not in complete | active | attempted), None)
            if next_task is not None:
                key = next_task["experiment_id"]
                previous = [row for row in jobs if row.get("experiment_id") == key]
                attempt = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
                jobs.append({
                    "experiment_id": key,
                    "backbone": next_task["backbone"],
                    "revision": next_task["revision"],
                    "architecture": next_task["architecture"],
                    "seed": str(next_task["seed"]),
                    "checkpoint_path": next_task["checkpoint"],
                    "checkpoint_sha256": next_task["checkpoint_sha256"],
                    "sample_plan_sha256": plan_digest,
                    "output_path": next_task["output"],
                    "attempt": str(attempt),
                    "pbs_job_id": job_id,
                    "status": "RUNNING",
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                })
        dispatch.write_csv_atomic(dispatch.JOBS, fields, jobs)
        return next_task


def main() -> None:
    job_id = os.environ["PBS_JOBID"]
    current_id = os.environ["SART_INITIAL_PROFILE"]
    protocol = json.loads(dispatch.PROTOCOL.read_text(encoding="utf-8"))
    plan = json.loads(dispatch.PROFILE_PLAN.read_text(encoding="utf-8"))
    task_by_id = {task["experiment_id"]: task for task in _tasks(protocol, plan)}
    if current_id not in task_by_id:
        raise ValueError(f"Initial profile is not in frozen matrix: {current_id}")
    deadline = time.monotonic() + USABLE_WALLTIME_SECONDS
    attempted: set[str] = set()
    while current_id is not None:
        _assert_active_claim(current_id, job_id)
        task = task_by_id[current_id]
        remaining = deadline - time.monotonic()
        if remaining <= MIN_REMAINING_SECONDS:
            _transition_and_claim(current_id, job_id, False, attempted, False)
            break
        command = [
            sys.executable, "scripts/evaluate/profile_inference_cost.py",
            "--checkpoint", task["checkpoint"],
            "--validation-root", str(dispatch.VALIDATION_ROOT.relative_to(ROOT)),
            "--sample-plan", str(dispatch.PROFILE_PLAN.relative_to(ROOT)),
            "--decode-tokens", "128", "--device", "cuda",
            "--output", task["output"],
        ]
        if task["hard_threshold"] is not None:
            command.extend(["--hard-threshold", str(task["hard_threshold"])])
        if task["top_k_fraction"] is not None:
            command.extend(["--top-k-fraction", str(task["top_k_fraction"])])
        print(f"{datetime.now(timezone.utc).isoformat()} {job_id} starting {current_id}", flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, check=False,
                                    timeout=max(1, remaining - MIN_REMAINING_SECONDS))
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False
        next_task = _transition_and_claim(
            current_id, job_id, success, attempted,
            deadline - time.monotonic() > MIN_REMAINING_SECONDS,
        )
        print(f"{datetime.now(timezone.utc).isoformat()} {job_id} finished "
              f"{current_id} success={success} next="
              f"{next_task['experiment_id'] if next_task else None}", flush=True)
        current_id = next_task["experiment_id"] if next_task else None


if __name__ == "__main__":
    main()
