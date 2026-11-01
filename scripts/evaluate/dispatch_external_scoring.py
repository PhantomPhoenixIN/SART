#!/usr/bin/env python3
"""Schedule resumable CPU scoring for completed frozen external predictions."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.dispatch_external_predictions import (  # noqa: E402
    model_runs,
    output_root as prediction_root,
    task_list,
)

REGISTRY = ROOT / "experiments/registry.csv"
JOBS = ROOT / "reports/progress/external_scoring_jobs_v3.csv"
LOCK = ROOT / "logs/confirmatory_7b/dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/external_scoring_dispatcher.log"
WORKER = ROOT / "run_on_astra/external_scoring_worker.pbs"
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
ACTIVE_STATES = {"R", "Q", "H", "W"}
MANIFESTS = {
    ("xcodeeval", "common_core"): "data/manifests/xcodeeval_common_core_manifest.json",
    ("xcodeeval", "full_directional"): "data/manifests/xcodeeval_full_direction_manifest.json",
    ("transcoder_test", "common_core"): "data/manifests/transcoder_test_manifest.json",
    ("transcoder_test", "full_directional"): "data/manifests/transcoder_test_manifest.json",
    ("humanevalx", "common_core"): "data/manifests/humanevalx_manifest.json",
    ("humanevalx", "full_directional"): "data/manifests/humanevalx_manifest.json",
    ("avatar_tc", "common_core"): "data/manifests/avatar_tc_manifest.json",
    ("avatar_tc", "full_directional"): "data/manifests/avatar_tc_manifest.json",
}
XCODE_SELECTION = ROOT / "data/manifests/xcodeeval_reference_selection.json"
MAX_AUTOMATIC_RETRIES_PER_SOURCE = 3


def scorer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT / "scripts/evaluate/score_external_predictions.py",
        ROOT / "scripts/evaluate/preflight_transcoder_test_references.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        if len(fields) >= 11 and fields[0].endswith(".hn01"):
            states[fields[0]] = fields[9]
    return states


def exit_status(job_id: str) -> int | None:
    result = subprocess.run(["qstat", "-xf", job_id], text=True, capture_output=True, check=False)
    match = re.search(r"^\s*Exit_status\s*=\s*(-?\d+)\s*$", result.stdout, re.M)
    return int(match.group(1)) if match else None


def score_key(experiment_id: str, benchmark: str, split: str) -> str:
    return f"{experiment_id}::{benchmark}::{split}"


def result_root(revision: str, experiment_id: str, benchmark: str, split: str) -> Path:
    return ROOT / "reports/results/protocol_v3" / revision / experiment_id / benchmark / split


def score_complete(path: Path) -> bool:
    manifest = path / "score_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "COMPLETE"
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
        code = exit_status(job_id)
        output = Path(row["output_root"])
        if score_complete(output) and code in {0, None}:
            row["status"] = "COMPLETE"
            log(f"scoring complete {row['score_key']} job={job_id}")
        elif code is not None:
            row["status"] = "FAILED_RETRYABLE"
            row["pbs_job_id"] = ""
            log(f"scoring retryable {row['score_key']} job={job_id} exit={code}")
        else:
            continue
        changed = True
    return changed


def submit(task: dict[str, Any], jobs: list[dict[str, str]], source_sha256: str) -> dict[str, str]:
    benchmark = task["benchmark"]
    split = task["split"]
    expid = task["experiment_id"]
    key = score_key(expid, benchmark, split)
    previous = [row for row in jobs if row.get("score_key") == key]
    attempt = 1 + max((int(row.get("attempt", "0")) for row in previous), default=0)
    safe_key = key.replace("::", "_").replace("/", "_")
    output = result_root(task["revision"], expid, benchmark, split)
    predictions = prediction_root(task["revision"], expid, benchmark, split)
    manifest = ROOT / MANIFESTS[(benchmark, split)]
    log_dir = ROOT / "logs/external_scoring"
    log_dir.mkdir(parents=True, exist_ok=True)
    variables = [
        f"SART_BENCHMARK={benchmark}",
        f"SART_SPLIT={split}",
        f"SART_MANIFEST={manifest.relative_to(ROOT)}",
        f"SART_PREDICTIONS={predictions.relative_to(ROOT)}",
        f"SART_OUTPUT_ROOT={output.relative_to(ROOT)}",
        f"SART_SCORING_KEY={safe_key}",
        f"SART_ATTEMPT={attempt}",
    ]
    if benchmark == "xcodeeval":
        variables.append(f"SART_XCODE_SELECTION={XCODE_SELECTION.relative_to(ROOT)}")
    command = [
        "qsub", "-N", f"sart_score_{benchmark[:4]}",
        "-o", str(log_dir / f"{safe_key}.attempt{attempt}.out"),
        "-e", str(log_dir / f"{safe_key}.attempt{attempt}.err"),
        "-v", ",".join(variables), str(WORKER),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().splitlines()[-1]
    return {
        "score_key": key,
        "experiment_id": expid,
        "backbone": task["backbone"],
        "architecture": task["architecture"],
        "seed": str(task["seed"]),
        "benchmark": benchmark,
        "split": split,
        "prediction_root": str(predictions.resolve()),
        "output_root": str(output.resolve()),
        "attempt": str(attempt),
        "pbs_job_id": job_id,
        "status": "SUBMITTED",
        "scorer_source_sha256": source_sha256,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }


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
            raise ValueError("Scoring requires a FROZEN_PREMODEL protocol")
        rows = list(csv.DictReader(REGISTRY.open(newline="", encoding="utf-8")))
        runs = model_runs(rows)
        if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
            log("deferred: all55 registered model runs must be TRAINED")
            return False
        tasks = task_list(runs, protocol)
        fields = [
            "score_key", "experiment_id", "backbone", "architecture", "seed", "benchmark",
            "split", "prediction_root", "output_root", "attempt", "pbs_job_id", "status",
            "submitted_utc", "scorer_source_sha256",
        ]
        fields, jobs = read_csv(JOBS, fields)
        if "scorer_source_sha256" not in fields:
            fields.append("scorer_source_sha256")
        source_sha256 = scorer_source_sha256()
        states = queue_states(user)
        if reconcile(jobs, states):
            write_csv_atomic(JOBS, fields, jobs)

        pending = []
        for task in tasks:
            pred_path = prediction_root(
                task["revision"], task["experiment_id"], task["benchmark"], task["split"]
            )
            pred_manifest_path = pred_path / "run_manifest.json"
            if not pred_manifest_path.is_file():
                continue
            pred_manifest = json.loads(pred_manifest_path.read_text(encoding="utf-8"))
            if pred_manifest.get("status") != "COMPLETE":
                continue
            if (
                pred_manifest.get("benchmark") != task["benchmark"]
                or pred_manifest.get("split") != task["split"]
                or pred_manifest.get("checkpoint") is None
                or pred_manifest.get("frozen_protocol_sha256") != _sha256_protocol(PROTOCOL)
            ):
                raise ValueError(f"Prediction manifest integrity mismatch: {pred_manifest_path}")
            result = result_root(task["revision"], task["experiment_id"], task["benchmark"], task["split"])
            key = score_key(task["experiment_id"], task["benchmark"], task["split"])
            if score_complete(result) or any(row.get("score_key") == key and row.get("status") == "COMPLETE" for row in jobs):
                continue
            active_or_unresolved = any(
                row.get("score_key") == key
                and row.get("pbs_job_id")
                and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
                for row in jobs
            )
            failures_on_source = sum(
                row.get("score_key") == key
                and row.get("status") == "FAILED_RETRYABLE"
                and row.get("scorer_source_sha256") == source_sha256
                for row in jobs
            )
            if failures_on_source >= MAX_AUTOMATIC_RETRIES_PER_SOURCE:
                continue
            if not active_or_unresolved:
                pending.append(task)

        registered_ids = {row.get("pbs_job_id", "") for row in jobs}
        active = {
            job_id for job_id, state in states.items()
            if state in ACTIVE_STATES and job_id in registered_ids
        }
        active.update(
            row["pbs_job_id"] for row in jobs
            if row.get("pbs_job_id") and row.get("status") in {"SUBMITTED", "QUEUED", "RUNNING"}
        )
        slots = max(0, cap - len(active))
        submitted = []
        for task in pending[:slots]:
            job = submit(task, jobs, source_sha256)
            jobs.append(job)
            submitted.append(job)
            write_csv_atomic(JOBS, fields, jobs)
        if submitted:
            log(f"cpu_cap={cap} active={len(active)} submitted={[(row['score_key'], row['pbs_job_id']) for row in submitted]}")
        if len([row for row in tasks if score_complete(result_root(row["revision"], row["experiment_id"], row["benchmark"], row["split"]))]) == len(tasks) and not active:
            log(f"all {len(tasks)} external score runs are complete")
            return True
        return False


def _sha256_protocol(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=4)
    parser.add_argument("--user", default=__import__("getpass").getuser())
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.cap < 1 or args.interval_seconds < 10:
        parser.error("--cap must be positive and interval must be at least 10 seconds")
    log(f"dispatcher started cpu_cap={args.cap} interval_seconds={args.interval_seconds}")
    while True:
        try:
            done = dispatch_once(args.cap, args.user)
        except Exception as exc:
            log(f"dispatcher error {type(exc).__name__}: {exc}")
            done = False
        if args.once or done:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
