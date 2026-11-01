#!/usr/bin/env python3
"""Use one PBS GPU allocation for consecutive unclaimed prediction runs."""

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

from scripts.evaluate import dispatch_external_predictions as dispatch  # noqa: E402
MIN_REMAINING_SECONDS = 600
USABLE_WALLTIME_SECONDS = 95 * 3600


def _tasks(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    with dispatch.REGISTRY.open(newline="", encoding="utf-8") as stream:
        runs = dispatch.model_runs(list(csv.DictReader(stream)))
    if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
        raise ValueError("Batch worker requires all 55 canonical models trained")
    return dispatch.task_list(runs, protocol)


def _assert_active_claim(key: str, job_id: str) -> None:
    with dispatch.LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_SH)
        _, jobs = dispatch.read_csv(dispatch.JOBS, [])
        matching = [row for row in jobs
                    if row.get("prediction_key") == key
                    and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}]
        if len(matching) != 1 or matching[0].get("pbs_job_id") != job_id:
            raise ValueError(f"Prediction task {key} is not exclusively claimed by {job_id}")


def _transition_and_claim(
    current_key: str,
    job_id: str,
    success: bool,
    attempted: set[str],
    allow_next: bool,
) -> dict[str, Any] | None:
    """Commit one result and claim the next task under the dispatcher lock."""
    with dispatch.LOCK.open("a+") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX)
        protocol = json.loads(dispatch.PROTOCOL.read_text(encoding="utf-8"))
        if protocol.get("status") != "FROZEN_PREMODEL" or protocol.get("protocol_version") != 3:
            raise ValueError("Batch worker requires the frozen one-candidate protocol")
        tasks = _tasks(protocol)
        fields = [
            "prediction_key", "experiment_id", "backbone", "architecture", "seed",
            "benchmark", "split", "checkpoint_path", "output_root", "attempt",
            "pbs_job_id", "status", "submitted_utc",
        ]
        fields, jobs = dispatch.read_csv(dispatch.JOBS, fields)
        current = next((row for row in reversed(jobs)
                        if row.get("prediction_key") == current_key
                        and row.get("pbs_job_id") == job_id), None)
        if current is None or current.get("status") not in {"SUBMITTED", "QUEUED", "RUNNING"}:
            raise ValueError(f"No active ledger claim for {current_key} in {job_id}")
        current_task = next(task for task in tasks if task["prediction_key"] == current_key)
        verified = dispatch.output_complete(ROOT / current_task["output_root"])
        current["status"] = "COMPLETE" if success and verified else "FAILED_RETRYABLE"
        attempted.add(current_key)
        next_task = None
        if allow_next:
            complete_keys = {
                task["prediction_key"] for task in tasks
                if dispatch.output_complete(ROOT / task["output_root"])
            }
            active_keys = {
                row["prediction_key"] for row in jobs
                if row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
            }
            next_task = next((task for task in tasks
                              if task["prediction_key"] not in complete_keys
                              and task["prediction_key"] not in active_keys
                              and task["prediction_key"] not in attempted), None)
            if next_task is not None:
                key = next_task["prediction_key"]
                previous = [row for row in jobs if row.get("prediction_key") == key]
                attempt = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
                jobs.append({
                    "prediction_key": key,
                    "experiment_id": next_task["experiment_id"],
                    "backbone": next_task["backbone"],
                    "architecture": next_task["architecture"],
                    "seed": str(next_task["seed"]),
                    "benchmark": next_task["benchmark"],
                    "split": next_task["split"],
                    "checkpoint_path": next_task["checkpoint"],
                    "output_root": str((ROOT / next_task["output_root"]).resolve()),
                    "attempt": str(attempt),
                    "pbs_job_id": job_id,
                    "status": "RUNNING",
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                })
        dispatch.write_csv_atomic(dispatch.JOBS, fields, jobs)
        return next_task


def main() -> None:
    job_id = os.environ["PBS_JOBID"]
    current_key = os.environ["SART_INITIAL_KEY"]
    protocol = json.loads(dispatch.PROTOCOL.read_text(encoding="utf-8"))
    task_by_key = {task["prediction_key"]: task for task in _tasks(protocol)}
    if current_key not in task_by_key:
        raise ValueError(f"Initial task is not in the frozen matrix: {current_key}")
    deadline = time.monotonic() + USABLE_WALLTIME_SECONDS
    attempted: set[str] = set()
    while current_key is not None:
        _assert_active_claim(current_key, job_id)
        task = task_by_key[current_key]
        remaining = deadline - time.monotonic()
        if remaining <= MIN_REMAINING_SECONDS:
            # No generator was started. Release the claim for the next allocation.
            next_task = _transition_and_claim(current_key, job_id, False, attempted, False)
            assert next_task is None
            break
        command = [
            sys.executable, "scripts/evaluate/generate_external_predictions.py",
            "--benchmark", task["benchmark"],
            "--split", task["split"],
            "--cache-root", task["cache_root"],
            "--checkpoint", task["checkpoint"],
            "--frozen-protocol", str(dispatch.PROTOCOL.relative_to(ROOT)),
            "--output-root", task["output_root"],
            "--max-new-tokens", str(task["max_new_tokens"]),
            "--batch-size", str(protocol["generation"]["batch_size"]),
            "--device", "cuda",
        ]
        if task["hard_threshold"] is not None:
            command.extend(["--hard-threshold", str(task["hard_threshold"])])
        if task["top_k_fraction"] is not None:
            command.extend(["--top-k-fraction", str(task["top_k_fraction"])])
        print(f"{datetime.now(timezone.utc).isoformat()} {job_id} starting {current_key}", flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, check=False,
                                    timeout=max(1, remaining - MIN_REMAINING_SECONDS))
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False
        next_task = _transition_and_claim(
            current_key, job_id, success, attempted,
            deadline - time.monotonic() > MIN_REMAINING_SECONDS,
        )
        print(f"{datetime.now(timezone.utc).isoformat()} {job_id} finished "
              f"{current_key} success={success} next="
              f"{next_task['prediction_key'] if next_task else None}", flush=True)
        current_key = next_task["prediction_key"] if next_task else None


if __name__ == "__main__":
    main()
