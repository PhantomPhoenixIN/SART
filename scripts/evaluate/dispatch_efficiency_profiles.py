#!/usr/bin/env python3
"""Schedule frozen validation efficiency profiles after external predictions finish."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / "experiments/registry.csv"
JOBS = ROOT / "reports/progress/efficiency_profile_jobs.csv"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/efficiency_profile_dispatcher.log"
WORKER = ROOT / "run_on_astra/efficiency_profile_worker.pbs"
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
PROFILE_PLAN = ROOT / "configs/evaluation/efficiency_profile_sample.json"
VALIDATION_ROOT = ROOT / "content/datasets/xlcost/structural_external_clean"
ACTIVE_STATES = {"R", "Q", "H", "W"}
GPU_OCCUPIED_STATES = ACTIVE_STATES | {"E", "T"}
PILOT_POLICY = "up-to-5-epoch-validation-pilot"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
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


def read_csv(path: Path, fields: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return fields, []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or fields), list(reader)


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


def model_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("training_horizon_policy") != PILOT_POLICY
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]


def output_path(revision: str, experiment_id: str) -> Path:
    return ROOT / "reports/profiling" / revision / f"{experiment_id}.json"


def profile_complete(path: Path, task: dict[str, Any], plan_digest: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("experiment_id") == task["experiment_id"]
        and value.get("checkpoint_sha256") == task["checkpoint_sha256"]
        and value.get("sample_plan_sha256") == plan_digest
        and value.get("sample_count") == task["sample_count"]
        and value.get("external_data_used") is False
        and len(value.get("samples", [])) == task["sample_count"]
    )


def tasks_for_runs(runs: list[dict[str, str]], protocol: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    policies = protocol.get("routing_policies", {})
    for row in runs:
        checkpoint = ROOT / row["checkpoint_path"]
        config_path = ROOT / row["config_path"]
        if checkpoint.name != "best.pt" or not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Trained run lacks best checkpoint/config: {row['experiment_id']}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        revision = Path(str(config["backbone_path"])).name
        architecture = row["architecture"]
        policy = policies.get(revision, {}).get(architecture)
        if not isinstance(policy, dict):
            raise ValueError(f"Frozen routing policy missing for {revision}/{architecture}")
        tasks.append({
            "experiment_id": row["experiment_id"],
            "architecture": architecture,
            "seed": int(row["seed"]),
            "backbone": row["backbone"],
            "revision": revision,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "hard_threshold": policy.get("hard_threshold"),
            "top_k_fraction": policy.get("top_k_fraction"),
            "sample_count": int(plan["sample_count"]),
            "output": str(output_path(revision, row["experiment_id"]).relative_to(ROOT)),
        })
    return tasks


def predictions_complete(runs: list[dict[str, str]], protocol: dict[str, Any]) -> bool:
    # Reuse the prediction dispatcher's canonical task construction and completion
    # check so profiling cannot take GPU capacity from unfinished benchmark inference.
    from scripts.evaluate.dispatch_external_predictions import (
        output_complete as prediction_complete,
        output_root as prediction_root,
        task_list,
    )

    tasks = task_list(runs, protocol)
    return all(
        prediction_complete(prediction_root(
            task["revision"], task["experiment_id"], task["benchmark"], task["split"]
        ))
        for task in tasks
    )


def scores_complete(runs: list[dict[str, str]], protocol: dict[str, Any]) -> bool:
    """Keep GPU profiling from overlapping the CPU-intensive scoring phase."""
    from scripts.evaluate.dispatch_external_scoring import result_root, score_complete
    from scripts.evaluate.dispatch_external_predictions import task_list

    return all(
        score_complete(result_root(
            task["revision"], task["experiment_id"], task["benchmark"], task["split"]
        ))
        for task in task_list(runs, protocol)
    )


def submit(task: dict[str, Any], jobs: list[dict[str, str]]) -> dict[str, str]:
    key = str(task["experiment_id"])
    previous = [row for row in jobs if row.get("experiment_id") == key]
    attempt = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
    safe_key = key.replace("/", "_")
    log_dir = ROOT / "logs/efficiency_profiles"
    log_dir.mkdir(parents=True, exist_ok=True)
    variables = [
        f"SART_CHECKPOINT={task['checkpoint']}",
        f"SART_VALIDATION_ROOT={VALIDATION_ROOT.relative_to(ROOT)}",
        f"SART_SAMPLE_PLAN={PROFILE_PLAN.relative_to(ROOT)}",
        f"SART_PROFILE_OUTPUT={task['output']}",
        f"SART_PROFILE_KEY={safe_key}",
        f"SART_INITIAL_PROFILE={key}",
        f"SART_ATTEMPT={attempt}",
        f"SART_DECODE_TOKENS=128",
        f"SART_HARD_THRESHOLD={'' if task['hard_threshold'] is None else task['hard_threshold']}",
        f"SART_TOP_K_FRACTION={'' if task['top_k_fraction'] is None else task['top_k_fraction']}",
    ]
    command = [
        "qsub", "-N", "sart_profile",
        "-o", str(log_dir / f"{safe_key}.attempt{attempt}.out"),
        "-e", str(log_dir / f"{safe_key}.attempt{attempt}.err"),
        "-v", ",".join(variables), str(WORKER),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().splitlines()[-1]
    return {
        "experiment_id": key,
        "backbone": task["backbone"],
        "revision": task["revision"],
        "architecture": task["architecture"],
        "seed": str(task["seed"]),
        "checkpoint_path": task["checkpoint"],
        "checkpoint_sha256": task["checkpoint_sha256"],
        "sample_plan_sha256": sha256(PROFILE_PLAN),
        "output_path": task["output"],
        "attempt": str(attempt),
        "pbs_job_id": job_id,
        "status": "SUBMITTED",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }


def reconcile(jobs: list[dict[str, str]], states: dict[str, str], plan_digest: str) -> bool:
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
        output = ROOT / row["output_path"]
        task = {
            "experiment_id": row["experiment_id"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "sample_count": 24,
        }
        if profile_complete(output, task, plan_digest) and code in {0, None}:
            row["status"] = "COMPLETE"
            log(f"profile complete {row['experiment_id']} job={job_id}")
        elif code is not None:
            row["status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            log(f"profile retryable {row['experiment_id']} job={job_id} exit={code}")
        else:
            continue
        changed = True
    return changed


def dispatch_once(cap: int, user: str) -> bool:
    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not PROTOCOL.is_file():
            log("deferred: frozen external protocol is absent")
            return False
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if protocol.get("status") != "FROZEN_PREMODEL":
            raise ValueError("Efficiency profile requires the frozen pre-model protocol")
        if protocol.get("translation_protocol_sha256") != sha256(ROOT / "docs/TRANSLATION_PROTOCOL.md"):
            raise ValueError("Translation protocol changed after freeze; refusing profile submission")
        if not PROFILE_PLAN.is_file() or not VALIDATION_ROOT.joinpath("manifest.json").is_file():
            raise FileNotFoundError("Frozen profile plan or validation manifest is absent")
        plan = json.loads(PROFILE_PLAN.read_text(encoding="utf-8"))
        if plan.get("status") != "FROZEN_PREMODEL" or plan.get("sample_count") != 24:
            raise ValueError("Efficiency profile plan must be the frozen 24-example plan")

        rows = list(csv.DictReader(REGISTRY.open(newline="", encoding="utf-8")))
        runs = model_runs(rows)
        if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
            log(f"deferred: all55 model runs must be TRAINED; observed {sum(r.get('training_status') == 'TRAINED' for r in runs)}/{len(runs)}")
            return False
        if not predictions_complete(runs, protocol):
            log("deferred: waiting for all frozen external prediction outputs before profiling")
            return False
        if not scores_complete(runs, protocol):
            log("deferred: waiting for all frozen external scores before GPU profiling")
            return False

        tasks = tasks_for_runs(runs, protocol, plan)
        fields = [
            "experiment_id", "backbone", "revision", "architecture", "seed",
            "checkpoint_path", "checkpoint_sha256", "sample_plan_sha256", "output_path",
            "attempt", "pbs_job_id", "status", "submitted_utc",
        ]
        fields, jobs = read_csv(JOBS, fields)
        states = queue_states(user)
        plan_digest = sha256(PROFILE_PLAN)
        if reconcile(jobs, states, plan_digest):
            write_csv_atomic(JOBS, fields, jobs)

        complete = {
            task["experiment_id"] for task in tasks
            if profile_complete(ROOT / task["output"], task, plan_digest)
        }
        active_keys = {
            row["experiment_id"] for row in jobs
            if row.get("pbs_job_id") and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        }
        pending = [task for task in tasks if task["experiment_id"] not in complete | active_keys]
        active = {job for job, state in states.items() if state in GPU_OCCUPIED_STATES}
        active.update(
            row["pbs_job_id"] for row in jobs
            if row.get("pbs_job_id") and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        )
        slots = max(0, cap - len(active))
        submitted = []
        for task in pending[:slots]:
            job = submit(task, jobs)
            jobs.append(job)
            submitted.append(job)
            write_csv_atomic(JOBS, fields, jobs)
        if submitted:
            log(f"cap={cap} active={len(active)} submitted={[(row['experiment_id'], row['pbs_job_id']) for row in submitted]}")
        if len(complete) == len(tasks) and not active:
            log(f"all {len(tasks)} frozen efficiency profiles are complete")
            return True
        return False


def scheduled_gpu_cap(weekday_cap: int, weekend_cap: int) -> int:
    return weekend_cap if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() >= 5 else weekday_cap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--weekend-cap", type=int, default=4)
    parser.add_argument("--user", default=__import__("getpass").getuser())
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.cap < 1 or args.weekend_cap < args.cap or args.interval_seconds < 10:
        parser.error("--cap must be positive, weekend cap >= cap, interval >= 10 seconds")
    log(f"dispatcher started weekday_cap={args.cap} weekend_cap={args.weekend_cap} interval_seconds={args.interval_seconds}")
    while True:
        try:
            done = dispatch_once(scheduled_gpu_cap(args.cap, args.weekend_cap), args.user)
        except Exception as exc:
            log(f"dispatcher error {type(exc).__name__}: {exc}")
            done = False
        if args.once or done:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
