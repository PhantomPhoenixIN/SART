#!/usr/bin/env python3
"""Schedule frozen external predictions for all trained runs under a GPU cap."""

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

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
JOBS = ROOT / "reports/progress/external_prediction_jobs_v3.csv"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/external_prediction_dispatcher.log"
WORKER = ROOT / "run_on_astra/external_prediction_worker.pbs"
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
ACTIVE_STATES = {"R", "Q", "H", "W"}
GPU_OCCUPIED_STATES = ACTIVE_STATES | {"E", "T"}
PILOT_POLICY = "up-to-5-epoch-validation-pilot"
BENCHMARKS = ("xcodeeval", "transcoder_test", "humanevalx", "avatar_tc")
SPLITS = ("common_core", "full_directional")
BENCHMARK_ORDER = {name: index for index, name in enumerate(BENCHMARKS)}
BACKBONE_ORDER = {
    "Qwen2.5-Coder-7B": 0,
    "Qwen2.5-Coder-14B": 1,
    "starcoder2-7b": 2,
    "Qwen2.5-Coder-1.5B": 3,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def scheduled_gpu_cap(weekday_cap: int, weekend_cap: int) -> int:
    return weekend_cap if datetime.now(ZoneInfo("Asia/Kolkata")).weekday() >= 5 else weekday_cap


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


def model_runs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("training_horizon_policy") != PILOT_POLICY
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]


def prediction_key(experiment_id: str, benchmark: str, split: str) -> str:
    return f"{experiment_id}::{benchmark}::{split}"


def output_root(revision: str, experiment_id: str, benchmark: str, split: str) -> Path:
    return ROOT / "reports/predictions/protocol_v3" / revision / experiment_id / benchmark / split


def output_complete(path: Path) -> bool:
    manifest = path / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "COMPLETE"
        and value.get("generation_errors", 0) == 0
        and value.get("tasks_completed") == value.get("tasks_total")
        and isinstance(value.get("tasks_total"), int)
        and value["tasks_total"] > 0
    )


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
        output = Path(row["output_root"])
        code = exit_status(job_id)
        if output_complete(output) and code in {0, None}:
            row["status"] = "COMPLETE"
            log(f"prediction complete {row['prediction_key']} job={job_id}")
        elif code is not None:
            row["status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            log(f"prediction retryable {row['prediction_key']} job={job_id} exit={code}")
        else:
            continue
        changed = True
    return changed


def task_list(runs: list[dict[str, str]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for row in runs:
        if row.get("training_status") != "TRAINED":
            raise ValueError(f"Prediction gate: model is not TRAINED: {row['experiment_id']}")
        checkpoint = ROOT / row["checkpoint_path"]
        resolved_path = checkpoint.parent / "resolved_config.json"
        if not checkpoint.is_file() or checkpoint.name != "best.pt" or not resolved_path.is_file():
            raise FileNotFoundError(f"Best checkpoint or resolved config missing for {row['experiment_id']}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        revision = Path(str(resolved["backbone_path"])).name
        architecture = row["architecture"]
        policy = protocol.get("routing_policies", {}).get(revision, {}).get(architecture)
        if not isinstance(policy, dict):
            raise ValueError(f"No frozen routing policy for {revision}/{architecture}")
        for benchmark in BENCHMARKS:
            for split in SPLITS:
                manifest_key = f"{benchmark}:{split}"
                expected = protocol.get("external_task_manifests", {}).get(manifest_key)
                manifest_path = ROOT / "data/structural/external" / benchmark / split / "manifest.json"
                # The protocol freezes the canonical task-manifest hash.  The
                # structural cache manifest is a separate artifact that embeds
                # that hash as `input_manifest_sha256`; comparing the cache
                # manifest file's own hash would reject every valid cache.
                if not expected or not manifest_path.is_file():
                    raise ValueError(f"Frozen cache manifest mismatch for {manifest_key}")
                cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if cache_manifest.get("input_manifest_sha256") != expected:
                    raise ValueError(f"Frozen task-manifest mismatch for {manifest_key}")
                tasks.append({
                    "prediction_key": prediction_key(row["experiment_id"], benchmark, split),
                    "experiment_id": row["experiment_id"],
                    "backbone": row["backbone"],
                    "revision": revision,
                    "architecture": architecture,
                    "checkpoint": str(checkpoint.relative_to(ROOT)),
                    "benchmark": benchmark,
                    "split": split,
                    "cache_root": str((ROOT / "data/structural/external" / benchmark).relative_to(ROOT)),
                    "output_root": str(output_root(revision, row["experiment_id"], benchmark, split).relative_to(ROOT)),
                    "hard_threshold": policy.get("hard_threshold"),
                    "top_k_fraction": policy.get("top_k_fraction"),
                    "max_new_tokens": protocol["generation"]["max_new_tokens"],
                    "seed": int(row["seed"]),
                })
    initial_architectures = {"full_sart", "dense", "mor_style", "uniform_token_loop"}
    confirmatory_architectures = initial_architectures | {"random_region", "structure_only"}
    def priority(row: dict[str, Any]) -> tuple[Any, ...]:
        primary = (
            row["revision"] == "0396a76181e127dfc13e5c5ec48a8cee09938b02"
            and row["benchmark"] == "xcodeeval"
            and row["split"] == "common_core"
        )
        architecture = row["architecture"]
        if primary and architecture in initial_architectures and row["seed"] == 42:
            tier = 0
        elif primary and architecture in confirmatory_architectures:
            tier = 1
        elif primary:
            tier = 2
        else:
            tier = 3
        initial_order = {"full_sart": 0, "dense": 1, "mor_style": 2,
                         "uniform_token_loop": 3, "random_region": 4,
                         "structure_only": 5}
        return (
            tier,
            initial_order.get(architecture, 99),
            0 if row["seed"] == 42 else row["seed"],
            BACKBONE_ORDER.get(row["backbone"].split("/")[-1], 99),
            BENCHMARK_ORDER[row["benchmark"]],
            0 if row["split"] == "common_core" else 1,
            row["experiment_id"],
        )
    tasks.sort(key=priority)
    return tasks


def submit(task: dict[str, Any], jobs: list[dict[str, str]]) -> dict[str, str]:
    checkpoint = ROOT / task["checkpoint"]
    destination = ROOT / task["output_root"]
    key = str(task["prediction_key"])
    previous = [row for row in jobs if row.get("prediction_key") == key]
    attempt = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
    safe_key = key.replace("::", "_").replace("/", "_")
    log_dir = ROOT / "logs/external_predictions"
    log_dir.mkdir(parents=True, exist_ok=True)
    variables = [
        f"SART_CHECKPOINT={task['checkpoint']}",
        f"SART_BENCHMARK={task['benchmark']}",
        f"SART_SPLIT={task['split']}",
        f"SART_CACHE_ROOT={task['cache_root']}",
        f"SART_PROTOCOL={PROTOCOL.relative_to(ROOT)}",
        f"SART_OUTPUT_ROOT={task['output_root']}",
        f"SART_PREDICTION_KEY={safe_key}",
        f"SART_INITIAL_KEY={key}",
        f"SART_ATTEMPT={attempt}",
        f"SART_MAX_NEW_TOKENS={task['max_new_tokens']}",
        f"SART_HARD_THRESHOLD={'' if task['hard_threshold'] is None else task['hard_threshold']}",
        f"SART_TOP_K_FRACTION={'' if task['top_k_fraction'] is None else task['top_k_fraction']}",
    ]
    command = [
        "qsub", "-N", f"sart_pred_{task['benchmark'][:4]}",
        "-o", str(log_dir / f"{safe_key}.attempt{attempt}.out"),
        "-e", str(log_dir / f"{safe_key}.attempt{attempt}.err"),
        "-v", ",".join(variables), str(WORKER),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().splitlines()[-1]
    return {
        "prediction_key": key,
        "experiment_id": task["experiment_id"],
        "backbone": task["backbone"],
        "architecture": task["architecture"],
        "seed": str(task["seed"]),
        "benchmark": task["benchmark"],
        "split": task["split"],
        "checkpoint_path": task["checkpoint"],
        "output_root": str(destination.resolve()),
        "attempt": str(attempt),
        "pbs_job_id": job_id,
        "status": "SUBMITTED",
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }


def dispatch_once(cap: int, user: str) -> bool:
    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not PROTOCOL.is_file():
            log("deferred: frozen external protocol does not exist")
            return False
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if protocol.get("status") != "FROZEN_PREMODEL":
            raise ValueError("External prediction requires a FROZEN_PREMODEL protocol")
        rows = list(csv.DictReader(REGISTRY.open(newline="", encoding="utf-8")))
        runs = model_runs(rows)
        if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
            log(f"deferred: 55 trained models required; observed {sum(r.get('training_status') == 'TRAINED' for r in runs)}/{len(runs)}")
            return False
        expected_protocol_hash = protocol.get("translation_protocol_sha256")
        translation_protocol = ROOT / "docs/TRANSLATION_PROTOCOL.md"
        if expected_protocol_hash != sha256(translation_protocol):
            raise ValueError("Frozen translation protocol hash changed; no external jobs submitted")

        fields = [
            "prediction_key", "experiment_id", "backbone", "architecture", "seed",
            "benchmark", "split", "checkpoint_path", "output_root", "attempt",
            "pbs_job_id", "status", "submitted_utc",
        ]
        fields, jobs = read_csv(JOBS, fields)
        states = queue_states(user)
        if reconcile(jobs, states):
            write_csv_atomic(JOBS, fields, jobs)
        tasks = task_list(runs, protocol)
        complete_keys = {
            task["prediction_key"] for task in tasks if output_complete(ROOT / task["output_root"])
        }
        complete_keys.update(row["prediction_key"] for row in jobs if row.get("status") == "COMPLETE")
        active_keys = {
            row["prediction_key"] for row in jobs
            if row.get("pbs_job_id")
            and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        }
        pending = [
            task for task in tasks
            if task["prediction_key"] not in complete_keys
            and task["prediction_key"] not in active_keys
        ]
        active = {
            job_id for job_id, state in states.items()
            if state in GPU_OCCUPIED_STATES
        }
        active.update(
            row["pbs_job_id"] for row in jobs
            if row.get("pbs_job_id")
            and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        )
        slots = max(0, cap - len(active))
        submitted = []
        for task in pending[:slots]:
            job = submit(task, jobs)
            jobs.append(job)
            submitted.append(job)
            write_csv_atomic(JOBS, fields, jobs)
        if submitted:
            log(f"cap={cap} active={len(active)} submitted={[(row['prediction_key'], row['pbs_job_id']) for row in submitted]}")
        if len(complete_keys) == len(tasks) and not active:
            log(f"all {len(tasks)} frozen external prediction runs are complete")
            return True
        return False


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
            done = dispatch_once(scheduled_gpu_cap(args.cap, args.weekend_cap), args.user)
        except Exception as exc:
            log(f"dispatcher error {type(exc).__name__}: {exc}")
            done = False
        if args.once or done:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
