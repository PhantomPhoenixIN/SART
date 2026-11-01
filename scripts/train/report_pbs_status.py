"""Print concise registry and PBS status for the confirmatory 7B matrix."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"


def active_jobs(user: str) -> dict[str, str]:
    result = subprocess.run(
        ["qstat", "-u", user], text=True, capture_output=True, check=True
    )
    states = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 11 and parts[0].endswith(".hn01") and parts[2] == "gpuq":
            states[parts[0]] = parts[9]
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=__import__("getpass").getuser())
    args = parser.parse_args()
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    runs = [
        row for row in rows
        if row.get("training_horizon_policy") == "shared-2-epoch-7B-horizon"
        and row.get("training_status") != "CANCELLED_SUPERSEDED"
    ]
    queue = active_jobs(args.user)
    by_arch: dict[str, Counter[str]] = defaultdict(Counter)
    for row in runs:
        status = row.get("training_status", "UNKNOWN")
        job_state = queue.get(row.get("pbs_job_id", ""))
        if job_state == "R":
            status = "RUNNING"
        elif job_state in {"Q", "H", "W"}:
            status = "QUEUED" if job_state != "H" else "HELD"
        by_arch[row.get("architecture", "unknown")][status] += 1
    print(f"7B confirmatory runs: {len(runs)}")
    totals = Counter()
    for counts in by_arch.values():
        totals.update(counts)
    print("Overall: " + ", ".join(f"{key}={value}" for key, value in sorted(totals.items())))
    print("Architecture breakdown:")
    for architecture in sorted(by_arch):
        counts = by_arch[architecture]
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        print(f"  {architecture}: {summary}")

    active_rows = [
        row for row in runs
        if queue.get(row.get("pbs_job_id", "")) in {"R", "Q", "H", "W"}
    ]
    checkpoint_paths = {
        row["experiment_id"]: (ROOT / row.get("checkpoint_path", "")).parent / "last.pt"
        for row in active_rows
    }
    fresh_mtimes: dict[Path, float] = {}
    existing_paths = [path.resolve() for path in checkpoint_paths.values() if path.is_file()]
    if existing_paths:
        result = subprocess.run(
            ["stat", "--cached=never", "-c", "%n|%Y", *map(str, existing_paths)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                try:
                    filename, mtime = line.rsplit("|", 1)
                    fresh_mtimes[Path(filename)] = float(mtime)
                except (ValueError, OSError):
                    continue
    print("Active run checkpoints:")
    for row in sorted(active_rows, key=lambda item: (item["architecture"], int(item["seed"]))):
        last_path = checkpoint_paths[row["experiment_id"]]
        if last_path.is_file():
            resolved_path = last_path.resolve()
            mtime = fresh_mtimes.get(resolved_path, last_path.stat().st_mtime)
            saved = datetime.fromtimestamp(mtime).astimezone()
            saved_text = saved.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            saved_text = "no durable last.pt"
        metrics_path = last_path.parent / "metrics.jsonl"
        latest_validation = None
        if metrics_path.is_file():
            try:
                for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines()):
                    if line.strip():
                        latest_validation = json.loads(line)
                        break
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                latest_validation = None
        if latest_validation:
            validation_text = (
                f"epoch={latest_validation.get('epoch')} "
                f"step={latest_validation.get('global_step')}"
            )
        else:
            validation_text = "no validation record yet"
        print(
            f"  {row['architecture']}-seed{row['seed']} "
            f"job={row.get('pbs_job_id') or 'unknown'} "
            f"last_saved={saved_text}; latest_validation={validation_text}"
        )


if __name__ == "__main__":
    main()
