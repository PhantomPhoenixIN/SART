#!/usr/bin/env python3
"""Repeatedly test pinned AVATAR-TC Python and Java references on official I/O cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess.audit_external_leakage import _official_python_detokenizer  # noqa: E402
from src.evaluation.xcodeeval_preflight import run_reference  # noqa: E402
from src.parsing.tree_sitter_adapter import parse_program  # noqa: E402

REVISION = "c02d3b4ef856c676c3924dd720824e581a30f1a5"
AVATAR = ROOT / "content/datasets/avatar_tc/official_repo/AVATAR-TC"
TEST_TABLES = (
    "aizu",
    "atcoder",
    "codeforces",
    "codejam",
    "geeksforgeeks",
    "leetcode",
    "projecteuler",
)
LANGUAGES = ("Python", "Java")
RUNTIMES = {"Python": "Python 3.11", "Java": "Java 14"}
HARNESS_VERSION = 2
COMPARISON_POLICY = "avatar_casefold_whitespace"
PAIR_SHIM = """package javafx.util;
public final class Pair<K, V> {
    private final K key;
    private final V value;
    public Pair(K key, V value) { this.key = key; this.value = value; }
    public K getKey() { return key; }
    public V getValue() { return value; }
    @Override public boolean equals(Object other) {
        if (!(other instanceof Pair)) return false;
        Pair<?, ?> pair = (Pair<?, ?>) other;
        return java.util.Objects.equals(key, pair.key)
            && java.util.Objects.equals(value, pair.value);
    }
    @Override public int hashCode() { return java.util.Objects.hash(key, value); }
}
"""


def _digest(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _detokenize_java(source: str) -> str:
    """Reconstruct Java lexer tokens while restoring AVATAR's string-space marker."""
    tokens = source.split()
    output: list[str] = []
    quote: str | None = None
    escaped = False
    for token in tokens:
        if quote is not None:
            if token == "▁":
                output.append(" ")
                continue
            output.append(token)
            for char in token:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            continue
        if token == "▁":
            output.append("▁")
            continue
        if token.startswith(("'", '"')):
            delimiter = token[0]
            if len(token) > 1 and token.endswith(delimiter):
                output.append(token.replace("▁", " "))
                continue
            output.append(token)
            quote = delimiter
            continue
        if (token.startswith('"') and token.endswith('"') and len(token) > 1) or (
            token.startswith("'") and token.endswith("'") and len(token) > 1
        ):
            output.append(token.replace("▁", " "))
            continue
        output.append(token)
        output.append(" ")
    return "".join(output).strip()


def _output_options(value: str | list[str]) -> list[str]:
    """Normalize one reference output or its list of accepted alternatives."""
    return value if isinstance(value, list) else [value]


def load_data() -> tuple[list[str], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    ids = (AVATAR / "test.java-python.id").read_text(encoding="utf-8").splitlines()
    encoded = {
        "Java": (AVATAR / "test.java-python.java").read_text(encoding="utf-8").splitlines(),
        "Python": (AVATAR / "test.java-python.python").read_text(encoding="utf-8").splitlines(),
    }
    if any(len(encoded[lang]) != len(ids) for lang in LANGUAGES):
        raise ValueError("AVATAR-TC language and ID splits are not aligned")
    python_detokenize = _official_python_detokenizer(ROOT)
    code = {lang: {} for lang in LANGUAGES}
    for index, uid in enumerate(ids):
        code["Python"][uid] = python_detokenize(encoded["Python"][index])
        code["Java"][uid] = _detokenize_java(encoded["Java"][index])
    tests: dict[str, list[dict[str, str]]] = {}
    for table_name in TEST_TABLES:
        table = json.loads((AVATAR / f"io_testcases_{table_name}.json").read_text(encoding="utf-8"))
        overlap = tests.keys() & table.keys()
        if overlap:
            raise ValueError(f"duplicate test IDs across AVATAR tables: {sorted(overlap)[:5]}")
        tests.update(
            {
                uid: [
                    {
                        "input": pair[0],
                        "output": _output_options(pair[1]),
                    }
                    for pair in pairs
                ]
                for uid, pairs in table.items()
            }
        )
    return ids, code, tests


def run_preflight(output_root: Path, repeats: int = 3, workers: int = 3) -> dict[str, Any]:
    ids, code, tests = load_data()
    common_ids = sorted(set(ids) & tests.keys())
    tasks = [(language, uid) for language in LANGUAGES for uid in common_ids]
    counts = {language: Counter() for language in LANGUAGES}
    completed = 0
    lock = threading.Lock()

    def run_one(task: tuple[str, str]) -> tuple[str, str]:
        language, uid = task
        source = code[language][uid]
        test_rows = tests[uid]
        syntax = parse_program(source, language)
        path = output_root / language / f"{uid}.json"
        source_hash = _digest(source)
        test_hash = _digest(json.dumps(test_rows, ensure_ascii=False, separators=(",", ":")))
        if path.is_file():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if (
                    old.get("harness_version") in {1, HARNESS_VERSION}
                    and old.get("source_sha256") == source_hash
                    and old.get("test_sha256") == test_hash
                    and old.get("repeats") == repeats
                    and old.get("runtime_label") == RUNTIMES[language]
                    and old.get("status")
                    in {"passed", "failed", "compile_failure", "parse_failure"}
                    and (old.get("status") != "failed" or old.get("flaky") is False)
                ):
                    old["harness_version"] = HARNESS_VERSION
                    old["comparison_policy"] = COMPARISON_POLICY
                    _atomic_json(path, old)
                    return language, old["status"]
            except (OSError, json.JSONDecodeError):
                pass
        if not syntax.valid:
            result = None
            sandbox = None
            status = "parse_failure"
            diagnostic = f"Tree-sitter errors={syntax.error_count}, root={syntax.root_type}"
        else:
            auxiliary = (
                {"javafx/util/Pair.java": PAIR_SHIM}
                if language == "Java" and "javafx.util.Pair" in source
                else {}
            )
            try:
                result, sandbox = run_reference(
                    source,
                    language,
                    RUNTIMES[language],
                    test_rows,
                    repeats=repeats,
                    case_timeout_seconds=8.0,
                    auxiliary_sources=auxiliary,
                    comparison_policy=COMPARISON_POLICY,
                )
                status = result["status"] if result else "runner_error"
                diagnostic = (
                    result.get("diagnostic") or result.get("reason", "")
                    if result
                    else sandbox.stderr.decode(errors="replace")[:1000]
                )
            except Exception as exc:
                result, sandbox = None, None
                status = "runner_error"
                diagnostic = f"{type(exc).__name__}: {exc}"
        row = {
            "schema_version": 1,
            "harness_version": HARNESS_VERSION,
            "benchmark": "AVATAR-TC",
            "dataset_revision": REVISION,
            "problem_id": uid,
            "language": language,
            "runtime_label": RUNTIMES[language],
            "comparison_policy": COMPARISON_POLICY,
            "source_sha256": source_hash,
            "test_sha256": test_hash,
            "test_count": len(test_rows),
            "repeats": repeats,
            "status": status,
            "all_passed": status == "passed",
            "flaky": result.get("flaky") if result else None,
            "attempts": result.get("attempts") if result else None,
            "diagnostic": diagnostic,
            "sandbox_returncode": sandbox.returncode if sandbox else None,
        }
        _atomic_json(path, row)
        return language, status

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            language, status = future.result()
            with lock:
                counts[language][status] += 1
                completed += 1
                if completed % 25 == 0:
                    print(f"completed {completed}/{len(tasks)} counts={counts}", flush=True)
    return {lang: {"tasks": sum(c.values()), "counts": dict(c)} for lang, c in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/avatar_tc_reference_preflight")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.output, args.repeats, args.workers), indent=2))


if __name__ == "__main__":
    main()
