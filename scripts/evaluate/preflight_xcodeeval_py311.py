#!/usr/bin/env python3
"""Resume-safe canonical-reference checks for the pinned xCodeEval test split."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/manifests/xcodeeval_reference_selection.json"
DEFAULT_OUTPUT = ROOT / "outputs/xcodeeval_reference_preflight_py311"
MANIFEST: dict[str, Any] = {}
MANIFEST_HASH = ""
OUTPUT_ROOT = Path()
REPEATS = 3
TIMEOUT = 6.0
RUNNER_HASH = ""


def _initialize(manifest_path: str, output_root: str, repeats: int, timeout: float) -> None:
    global MANIFEST, MANIFEST_HASH, OUTPUT_ROOT, REPEATS, TIMEOUT, RUNNER_HASH
    manifest_file = Path(manifest_path)
    MANIFEST = json.loads(manifest_file.read_text(encoding="utf-8"))
    MANIFEST_HASH = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    OUTPUT_ROOT = Path(output_root)
    REPEATS = repeats
    TIMEOUT = timeout
    runner_digest = hashlib.sha256()
    for relative in (
        "src/evaluation/execeval_task_driver_py311.py",
        "src/evaluation/xcodeeval_preflight_py311.py",
        "content/tools/sandbox/seccomp_exec",
    ):
        runner_digest.update((ROOT / relative).read_bytes())
    RUNNER_HASH = runner_digest.hexdigest()
    sys.path.insert(0, str(ROOT))


def _result_path(language: str, src_uid: str) -> Path:
    return OUTPUT_ROOT / language.replace("+", "p") / f"{src_uid}.json"


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as temporary:
        json.dump(record, temporary, ensure_ascii=False, separators=(",", ":"))
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _already_complete(path: Path, input_hash: str) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("manifest_sha256") == MANIFEST_HASH
        and record.get("input_sha256") == input_hash
        and record.get("complete") is True
    )


def _run_one(task: tuple[str, str]) -> dict[str, Any]:
    language, src_uid = task
    selected = MANIFEST["selected"][language][src_uid]
    candidates = MANIFEST["candidate_pool"][language][src_uid]
    path = _result_path(language, src_uid)
    input_hash = hashlib.sha256(
        json.dumps(
            {
                "candidate_hashes": [candidate["source_sha256"] for candidate in candidates],
                "test_hash": selected["raw_test_sha256"],
                "repeats": REPEATS,
                "timeout": TIMEOUT,
                "runner_sha256": RUNNER_HASH,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if _already_complete(path, input_hash):
        return {"skipped": True, "language": language, "src_uid": src_uid}

    from src.evaluation.xcodeeval_preflight_py311 import run_reference

    attempts: list[dict[str, Any]] = []
    winner = None
    for candidate in candidates:
        response = None
        outer = None
        for infrastructure_try in range(3):
            response, outer = run_reference(
                candidate["source_code"],
                language,
                candidate["runtime_label"],
                selected["tests"],
                repeats=REPEATS,
                case_timeout_seconds=TIMEOUT,
            )
            if response is not None:
                break
            attempts.append(
                {
                    "code_uid": candidate["code_uid"],
                    "source_sha256": candidate["source_sha256"],
                    "runtime_label": candidate["runtime_label"],
                    "status": "runner_error",
                    "infrastructure_try": infrastructure_try + 1,
                    "returncode": outer.returncode,
                    "timed_out": outer.timed_out,
                    "output_limited": outer.output_limited,
                    "stderr": outer.stderr.decode("utf-8", errors="replace")[:500],
                }
            )
        if response is None:
            record = {
                "complete": False,
                "manifest_sha256": MANIFEST_HASH,
                "input_sha256": input_hash,
                "runner_sha256": RUNNER_HASH,
                "language": language,
                "src_uid": src_uid,
                "attempts": attempts,
                "status": "retryable_runner_error",
            }
            _atomic_json(path, record)
            return record
        attempts.append(
            {
                "code_uid": candidate["code_uid"],
                "source_sha256": candidate["source_sha256"],
                "runtime_label": candidate["runtime_label"],
                "status": response.get("status"),
                "compile_status": response.get("compile_status"),
                "all_passed": response.get("all_passed", False),
                "flaky": response.get("flaky", False),
                "outcomes": response.get("outcomes"),
                "diagnostic": response.get("diagnostic"),
            }
        )
        if response.get("all_passed") is True and response.get("flaky") is False:
            winner = candidate
            break

    record = {
        "complete": True,
        "manifest_sha256": MANIFEST_HASH,
        "input_sha256": input_hash,
        "runner_sha256": RUNNER_HASH,
        "language": language,
        "src_uid": src_uid,
        "test_count": selected["test_count"],
        "repeat_count": REPEATS,
        "attempts": attempts,
        "valid": winner is not None,
        "status": "passed" if winner is not None else "no_passing_reference",
        "selected_code_uid": winner["code_uid"] if winner else None,
        "selected_source_sha256": winner["source_sha256"] if winner else None,
        "selected_runtime_label": winner["runtime_label"] if winner else None,
    }
    _atomic_json(path, record)
    return record


def _tasks(language: str | None, src_uid: str | None, limit: int | None) -> list[tuple[str, str]]:
    tasks = [
        (lang, uid)
        for lang in ("Python", "Java", "C++")
        if language is None or lang == language
        for uid in sorted(MANIFEST["candidate_pool"][lang])
        if src_uid is None or uid == src_uid
    ]
    return tasks[:limit] if limit is not None else tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-timeout", type=float, default=6.0)
    parser.add_argument("--language", choices=("Python", "Java", "C++"))
    parser.add_argument("--src-uid")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 1 or args.case_timeout <= 0:
        parser.error("workers, repeats, and timeout must be positive")

    _initialize(str(args.manifest), str(args.output), args.repeats, args.case_timeout)
    tasks = _tasks(args.language, args.src_uid, args.max_tasks)
    done = passed = failed = skipped = retryable = 0
    print(
        json.dumps(
            {
                "manifest_sha256": MANIFEST_HASH,
                "candidate_reference_tasks": len(tasks),
                "workers": args.workers,
                "repeats": args.repeats,
                "case_timeout_seconds": args.case_timeout,
                "output": str(args.output),
            }
        ),
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize,
        initargs=(str(args.manifest), str(args.output), args.repeats, args.case_timeout),
    ) as executor:
        futures = [executor.submit(_run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            skipped += bool(result.get("skipped"))
            passed += result.get("status") == "passed"
            failed += result.get("status") == "no_passing_reference"
            retryable += result.get("status") == "retryable_runner_error"
            if done % 10 == 0 or done == len(tasks):
                print(
                    json.dumps(
                        {
                            "done": done,
                            "total": len(tasks),
                            "passed": passed,
                            "no_passing_reference": failed,
                            "skipped": skipped,
                            "retryable_runner_errors": retryable,
                        }
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "done": done,
                "passed": passed,
                "no_passing_reference": failed,
                "skipped": skipped,
                "retryable_runner_errors": retryable,
            }
        ),
        flush=True,
    )
    if retryable:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
