"""Submit all not-yet-submitted frozen 7B confirmatory runs to PBS."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "experiments/registry.csv"
ATTEMPTS = ROOT / "experiments/pbs_attempts.csv"
WORKER = ROOT / "run_on_astra/confirmatory_7b_worker.pbs"
LOG_DIR = ROOT / "logs/confirmatory_7b"


def write_registry(rows: list[dict[str, str]], fields: list[str]) -> None:
    fields = list(fields)
    for field in ("pbs_job_id",):
        if field not in fields:
            fields.append(field)
    fd, name = tempfile.mkstemp(prefix="registry.", suffix=".tmp", dir=REGISTRY.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, REGISTRY)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--priority-full-sart",
        action="store_true",
        help="Submit the three predeclared Full-SART confirmatory seeds before other pending runs.",
    )
    args = parser.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    pending = [
        row for row in rows
        if row.get("training_horizon_policy") == "shared-2-epoch-7B-horizon"
        and row.get("training_status") in {"PLANNED", "FAILED_RETRYABLE"}
    ]
    if args.priority_full_sart:
        seed_order = {42: 0, 13: 1, 87: 2}
        pending.sort(
            key=lambda row: (
                row.get("architecture") != "full_sart",
                row.get("training_status") != "FAILED_RETRYABLE",
                seed_order.get(int(row.get("seed", -1)), 3),
            )
        )
    if args.limit is not None:
        pending = pending[: args.limit]
    submitted = []
    attempt_fields = ["experiment_id", "attempt", "pbs_job_id", "status", "submitted_utc", "notes"]
    attempt_rows: list[dict[str, str]] = []
    if ATTEMPTS.exists():
        with ATTEMPTS.open(newline="", encoding="utf-8") as stream:
            attempt_reader = csv.DictReader(stream)
            attempt_fields = list(attempt_reader.fieldnames or attempt_fields)
            attempt_rows = list(attempt_reader)
    for field in ("submitted_utc", "notes"):
        if field not in attempt_fields:
            attempt_fields.append(field)
    for row in pending:
        run_id = row["experiment_id"]
        attempt_number = 1 + sum(
            record.get("experiment_id") == run_id
            and record.get("attempt", "").isdigit()
            for record in attempt_rows
        )
        config = ROOT / row["config_path"]
        if not config.is_file():
            raise FileNotFoundError(config)
        log_stem = f"{run_id}.attempt{attempt_number}"
        command = [
            "qsub", "-N", f"sart_{run_id[:10]}",
            "-o", str(LOG_DIR / f"{log_stem}.out"),
            "-e", str(LOG_DIR / f"{log_stem}.err"),
            "-v", f"SART_CONFIG={config.relative_to(ROOT)},SART_RUN_ID={run_id},SART_ATTEMPT={attempt_number}",
            str(WORKER),
        ]
        if args.dry_run:
            print(" ".join(command))
            continue
        # The interactive shell may inherit a PBS TMPDIR from an old job that
        # has already been removed. qsub uses TMPDIR for its temporary script,
        # so do not pass that stale job-local path to the PBS client.
        qsub_env = os.environ.copy()
        for variable in ("TMPDIR", "TMP", "TEMP"):
            qsub_env.pop(variable, None)
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, env=qsub_env
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"qsub failed (exit {result.returncode}): {detail}")
        job_id = result.stdout.strip().splitlines()[-1]
        previous = row.get("pbs_job_id", "")
        attempt_rows.append(
            {
                "experiment_id": run_id,
                "attempt": str(attempt_number),
                "pbs_job_id": job_id,
                "status": "SUBMITTED",
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
                "notes": f"supersedes {previous}" if previous else "initial submission",
            }
        )
        with ATTEMPTS.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=attempt_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(attempt_rows)
        row["training_status"] = "SUBMITTED"
        row["pbs_job_id"] = job_id
        submitted.append((run_id, job_id))
        write_registry(rows, fields)
        print(f"{run_id},{job_id}", flush=True)
    if args.dry_run:
        print(f"dry-run: {len(pending)} jobs")
    else:
        print(f"submitted: {len(submitted)} jobs")


if __name__ == "__main__":
    main()
