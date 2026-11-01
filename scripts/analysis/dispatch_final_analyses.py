#!/usr/bin/env python3
"""Wait for complete frozen scores/profiles, then run the final analysis suite."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
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

REGISTRY = ROOT / "experiments/registry.csv"
PROTOCOL = ROOT / "reports/evaluation/frozen_external_protocol.json"
PROFILE_JOBS = ROOT / "reports/progress/efficiency_profile_jobs.csv"
LOCK = ROOT / "logs/confirmatory_7b/final_analysis_dispatcher.lock"
LOG = ROOT / "logs/confirmatory_7b/final_analysis_dispatcher.log"
SCORES_ROOT = ROOT / "reports/results/protocol_v3"
PREDICTIONS_ROOT = ROOT / "reports/predictions/protocol_v3"
PROFILES_ROOT = ROOT / "reports/profiling"
PILOT_POLICY = "up-to-5-epoch-validation-pilot"
PRIMARY_REVISION = "0396a76181e127dfc13e5c5ec48a8cee09938b02"

OUTPUTS = {
    "external_summary": ROOT / "reports/analysis/external_benchmark_summary.json",
    "primary": ROOT / "reports/statistics/xcodeeval_common_core_primary.json",
    "efficiency": ROOT / "reports/analysis/xcodeeval_common_core_efficiency_frontier.json",
    "h3": ROOT / "reports/analysis/xcodeeval_common_core_h3_difficulty_interaction.json",
    "h4": ROOT / "reports/analysis/xcodeeval_common_core_h4_routing_difficulty.json",
    "h5": ROOT / "reports/analysis/xcodeeval_common_core_h5_structure_controls.json",
    "final_report": ROOT / "reports/SART_CT_research_report.md",
}
COMPLETE_MARKER = ROOT / "experiments/PROJECT_COMPLETE"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"{now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def model_runs() -> list[dict[str, str]]:
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        row for row in rows
        if row.get("training_horizon_policy") != PILOT_POLICY
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]


def complete_score(path: Path) -> bool:
    manifest = path / "score_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "COMPLETE"
        and isinstance(value.get("tasks_total"), int)
        and value.get("tasks_total", 0) > 0
        and value.get("tasks_completed") == value.get("tasks_total")
    )


def all_scores_complete(runs: list[dict[str, str]], protocol: dict[str, Any]) -> bool:
    from scripts.evaluate.dispatch_external_predictions import task_list
    from scripts.evaluate.dispatch_external_scoring import result_root

    for task in task_list(runs, protocol):
        if not complete_score(result_root(
            task["revision"], task["experiment_id"], task["benchmark"], task["split"]
        )):
            return False
    return True


def all_profiles_complete(runs: list[dict[str, str]]) -> bool:
    if not PROFILE_JOBS.is_file():
        return False
    with PROFILE_JOBS.open(newline="", encoding="utf-8") as stream:
        jobs = list(csv.DictReader(stream))
    by_experiment = {row.get("experiment_id", ""): row for row in jobs}
    if len(by_experiment) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
        return False
    for run in runs:
        job = by_experiment.get(run["experiment_id"])
        if not job or job.get("status") != "COMPLETE":
            return False
        output = ROOT / job["output_path"]
        if not output.is_file():
            return False
        try:
            value = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            value.get("experiment_id") != run["experiment_id"]
            or value.get("sample_count") != 24
            or len(value.get("samples", [])) != 24
            or value.get("external_data_used") is not False
        ):
            return False
    return True


def output_valid(name: str) -> bool:
    if name == "final_report":
        try:
            body = OUTPUTS[name].read_text(encoding="utf-8")
        except OSError:
            return False
        return (
            "**Status:** complete artifact-backed report" in body
            and "All 55 canonical model runs and 440 benchmark/split prediction and score runs are complete" in body
            and "## Reproducibility and limitations" in body
            and "## Results and discussion" in body
            and "pending" not in body.lower()
        )
    try:
        value = json.loads(OUTPUTS[name].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if name == "primary":
        return value.get("backbone_revision") == PRIMARY_REVISION and len(value.get("comparisons", {})) == 5
    if name == "external_summary":
        markdown = ROOT / "reports/analysis/external_benchmark_summary.md"
        if not markdown.is_file():
            return False
        markdown_text = markdown.read_text(encoding="utf-8")
        return (
            value.get("status") == "COMPLETE"
            and value.get("canonical_model_runs") == 55
            and value.get("prediction_score_runs") == 440
            and len(value.get("detail_rows", [])) == 440
            and markdown_text.count("## ") == 8
            and "440/440 prediction and score runs reconciled" in markdown_text
        )
    if name == "efficiency":
        return value.get("status") == "analyzed" and len(value.get("model_groups", [])) == 55
    if name in {"h3", "h4"}:
        return value.get("status") == "analyzed" and value.get("backbone_revision") == PRIMARY_REVISION
    if name == "h5":
        return (
            value.get("status") == "analyzed"
            and value.get("backbone_revision") == PRIMARY_REVISION
            and len(value.get("comparisons", {})) == 3
        )
    return False


def outputs_complete() -> bool:
    return all(output_valid(name) for name in OUTPUTS)


def commands() -> list[tuple[str, list[str]]]:
    py = sys.executable
    return [
        ("external_summary", [
            py, "scripts/analysis/summarize_external_results.py",
            "--json-output", str(OUTPUTS["external_summary"].relative_to(ROOT)),
            "--markdown-output", "reports/analysis/external_benchmark_summary.md",
        ]),
        ("primary", [
            py, "scripts/analysis/run_primary_statistics.py",
            "--results-root", "reports/results/protocol_v3",
            "--backbone-revision", PRIMARY_REVISION,
            "--output", str(OUTPUTS["primary"].relative_to(ROOT)),
        ]),
        ("efficiency", [
            py, "scripts/analysis/analyze_efficiency_frontier.py",
            "--profiles-root", "reports/profiling",
            "--scores-root", "reports/results/protocol_v3",
            "--output", str(OUTPUTS["efficiency"].relative_to(ROOT)),
        ]),
        ("h3", [
            py, "scripts/analysis/analyze_difficulty_interaction.py",
            "--scores-root", "reports/results/protocol_v3",
            "--backbone-revision", PRIMARY_REVISION,
            "--output", str(OUTPUTS["h3"].relative_to(ROOT)),
        ]),
        ("h4", [
            py, "scripts/analysis/analyze_routing_difficulty.py",
            "--scores-root", "reports/results/protocol_v3",
            "--predictions-root", "reports/predictions/protocol_v3",
            "--backbone-revision", PRIMARY_REVISION,
            "--output", str(OUTPUTS["h4"].relative_to(ROOT)),
        ]),
        ("h5", [
            py, "scripts/analysis/analyze_structure_controls.py",
            "--output", str(OUTPUTS["h5"].relative_to(ROOT)),
        ]),
        ("final_report", [
            py, "scripts/analysis/assemble_final_report.py",
        ]),
    ]


def mark_project_complete() -> None:
    if not outputs_complete():
        raise ValueError("Completion marker requires every final analysis and report")
    marker = {
        "status": "COMPLETE",
        "completed_utc": now(),
        "canonical_model_runs": 55,
        "prediction_score_runs": 440,
        "efficiency_profiles": 55,
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "artifact_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in OUTPUTS.items()
        },
    }
    COMPLETE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".PROJECT_COMPLETE.", dir=COMPLETE_MARKER.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, COMPLETE_MARKER)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_analyses() -> bool:
    for name, command in commands():
        if output_valid(name):
            log(f"{name} output already passes completion checks; skipping")
            continue
        log(f"starting {name}: {' '.join(command)}")
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.stdout:
            log(f"{name} stdout: {result.stdout.strip()}")
        if result.stderr:
            log(f"{name} stderr: {result.stderr.strip()}")
        if result.returncode:
            log(f"{name} failed exit={result.returncode}; will retry on next cycle")
            return False
        log(f"{name} command completed exit=0")
    return outputs_complete()


def dispatch_once() -> bool:
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
            raise ValueError("Final analysis requires the frozen pre-model protocol")
        runs = model_runs()
        if len(runs) != 55 or any(row.get("training_status") != "TRAINED" for row in runs):
            log(f"deferred: all55 runs must be TRAINED; observed {sum(r.get('training_status') == 'TRAINED' for r in runs)}/{len(runs)}")
            return False
        if not all_scores_complete(runs, protocol):
            log("deferred: waiting for all frozen benchmark score manifests")
            return False
        if not all_profiles_complete(runs):
            log("deferred: waiting for55 validated efficiency profiles")
            return False
        if outputs_complete():
            log("all final statistical, efficiency and mechanism outputs are complete")
            mark_project_complete()
            return True
        done = run_analyses()
        if done:
            mark_project_complete()
        return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval_seconds < 10:
        parser.error("--interval-seconds must be at least 10")
    log(f"final-analysis dispatcher started interval_seconds={args.interval_seconds}")
    while True:
        try:
            done = dispatch_once()
        except Exception as exc:
            log(f"dispatcher error {type(exc).__name__}: {exc}")
            done = False
        if args.once or done:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
