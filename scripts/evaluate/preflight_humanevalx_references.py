#!/usr/bin/env python3
"""Resume-safe repeated canonical preflight for the three HumanEval-X languages."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.preprocess.audit_humanevalx_raw import _test_count
from src.evaluation.xcodeeval_preflight import run_function_harness_reference

ROOT = Path(__file__).resolve().parents[2]
REVISION = "62c78627f3072a1454fa0cb0184737cafe5e4198"
HARNESS_VERSION = 3
DATA_ROOT = ROOT / "content/datasets/humanevalx" / REVISION / "data"
LANGUAGES = {"Python": "python", "Java": "java", "C++": "cpp"}
RUNTIMES = {"Python": "Python 3.11", "Java": "Java 17", "C++": "GNU C++11"}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_rows(language: str) -> list[dict[str, Any]]:
    path = DATA_ROOT / LANGUAGES[language] / "data/humaneval.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def preflight(
    output_root: Path,
    repeats: int = 3,
    timeout_seconds: float = 30.0,
    workers: int = 6,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    tasks = [(language, row) for language in LANGUAGES for row in _load_rows(language)]
    counts_by_language = {language: Counter() for language in LANGUAGES}
    completed = 0
    progress_lock = threading.Lock()

    def run_one(task: tuple[str, dict[str, Any]]) -> tuple[str, str]:
        language, row = task
        uid = row["task_id"].split("/", 1)[-1]
        test = row["test"]
        source = row["prompt"] + row["canonical_solution"]
        entry_point = None
        if language == "Python":
            declaration_match = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", row["prompt"])
            if declaration_match is None:
                raise ValueError(f"HumanEval-X declaration has no function: {row['task_id']}")
            entry_point = declaration_match.group(1)
        task_path = output_root / language / f"{uid}.json"
        source_hash = _sha256(row["prompt"] + row["canonical_solution"])
        test_hash = _sha256(test)
        if task_path.is_file():
            existing = json.loads(task_path.read_text(encoding="utf-8"))
            if (
                existing.get("source_sha256") == source_hash
                and existing.get("test_sha256") == test_hash
                and existing.get("repeats") == repeats
                and existing.get("runtime_label") == RUNTIMES[language]
                and existing.get("harness_version") == HARNESS_VERSION
            ):
                return language, existing["status"]

        diagnostic = None
        try:
            result, sandbox = run_function_harness_reference(
                source,
                test,
                language,
                RUNTIMES[language],
                interface_source=row["prompt"],
                entry_point=entry_point,
                test_preamble=(
                    "".join(
                        line + "\n"
                        for line in row["prompt"].splitlines()
                        if line.lstrip().startswith("import ")
                    )
                    if language == "Java"
                    else ""
                ),
                repeats=repeats,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            result, sandbox = None, None
            diagnostic = f"{type(exc).__name__}: {exc}"
        status = result["status"] if result else "runner_error"
        record = {
            "schema_version": 1,
            "harness_version": HARNESS_VERSION,
            "benchmark": "HumanEval-X",
            "dataset_revision": REVISION,
            "task_id": uid,
            "language": language,
            "runtime_label": RUNTIMES[language],
            "source_sha256": source_hash,
            "test_sha256": test_hash,
            "test_count": _test_count(test, language),
            "repeats": repeats,
            "status": status,
            "compile_status": result.get("compile_status") if result else None,
            "all_passed": result.get("all_passed") if result else False,
            "flaky": result.get("flaky") if result else None,
            "program_outcomes": result.get("program_outcomes") if result else None,
            "sandbox_returncode": sandbox.returncode if sandbox else None,
            "sandbox_timed_out": sandbox.timed_out if sandbox else None,
            "sandbox_output_limited": sandbox.output_limited if sandbox else None,
            "sandbox_stderr": (
                sandbox.stderr.decode("utf-8", errors="replace")[:1000] if sandbox else ""
            ),
            "driver_diagnostic": result.get("diagnostic") if result else None,
            "exception_diagnostic": diagnostic,
        }
        _atomic_json(task_path, record)
        return language, status

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            language, status = future.result()
            with progress_lock:
                counts_by_language[language][status] += 1
                completed += 1
                if completed % 10 == 0:
                    summary = {name: dict(counts) for name, counts in counts_by_language.items()}
                    print(
                        f"completed {completed}/{len(tasks)}; counts={summary}",
                        flush=True,
                    )
    for language in LANGUAGES:
        summaries[language] = {
            "tasks": sum(counts_by_language[language].values()),
            "counts": dict(counts_by_language[language]),
        }
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/humanevalx_reference_preflight")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            preflight(args.output_root, args.repeats, args.timeout_seconds, args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
