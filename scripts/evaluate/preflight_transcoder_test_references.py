#!/usr/bin/env python3
"""Run the pinned TransCoder-Test f_gold reference and official tests repeatedly."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess.audit_transcoder_test import LANGS, _case_count, _ids  # noqa: E402
from src.evaluation.xcodeeval_preflight import run_complete_program_reference  # noqa: E402
from src.parsing.tree_sitter_adapter import parse_tree  # noqa: E402

REVISION = "1c21887dc55a88ea6535e669952d419256fbc24a"
DATA_ROOT = ROOT / "content/datasets/transcoder-test"
REPOSITORY = DATA_ROOT / "official_repo"
RAW = DATA_ROOT / REVISION / "raw/TransCoder_tokenized_test_set_functions"
TESTS = REPOSITORY / "data/evaluation/geeks_for_geeks_successful_test_scripts"
EXTENSIONS = {"Python": ".py", "Java": ".java", "C++": ".cpp"}
RUNTIMES = {"Python": "Python 3.11", "Java": "Java 8", "C++": "GNU C++11"}
HARNESS_VERSION = 2

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


def _digest(source: str) -> str:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(row, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _node_name(node, source: bytes, language: str) -> bytes | None:
    if language in {"Python", "Java"}:
        name = node.child_by_field_name("name")
        return name.text if name is not None else None
    declarator = node.child_by_field_name("declarator")
    if declarator is None:
        return None
    stack = [declarator]
    while stack:
        current = stack.pop()
        if current.type == "identifier":
            return current.text
        stack.extend(reversed(current.children))
    return None


def _extract_f_gold(program: str, language: str) -> str:
    raw = program.encode("utf-8")
    root = parse_tree(program, language).root_node
    expected = "function_definition" if language != "Java" else "method_declaration"
    stack = [root]
    matches = []
    while stack:
        node = stack.pop()
        if node.type == expected and _node_name(node, raw, language) == b"f_gold":
            matches.append(node)
        stack.extend(reversed(node.children))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one f_gold in {language} script; found {len(matches)}")
    node = matches[0]
    function = raw[node.start_byte : node.end_byte].decode("utf-8")
    name_match = re.search(r"\bf_gold\b", function)
    if name_match is None:
        raise ValueError("Tree-sitter found f_gold but text did not")
    return function[: name_match.start()] + "f_filled" + function[name_match.end() :]


def _insert_candidate(script: str, candidate: str, language: str) -> str:
    pattern = r"(?m)^\s*(?:#|//)\s*TOFILL\s*$"
    patched, count = re.subn(pattern, lambda _match: candidate, script, count=1)
    if count != 1:
        raise ValueError(f"expected one official TOFILL marker; got {count}")
    return patched


def _result_path(output: Path, language: str, uid: str) -> Path:
    return output / language.replace("+", "p") / f"{uid}.json"


def run_preflight(
    output: Path, repeats: int = 3, timeout_seconds: float = 30.0, workers: int = 3
) -> dict[str, Any]:
    ids_by_language = {
        language: _ids(RAW / f"test.{short}.shuf.test.tok")
        for short, (language, _) in LANGS.items()
    }
    tasks: list[tuple[str, str]] = []
    for short, (language, extension) in LANGS.items():
        folder = TESTS / short
        for uid in sorted(ids_by_language[language]):
            test_path = folder / f"{uid}{extension}"
            if test_path.is_file():
                tasks.append((language, uid))
    counts: dict[str, Counter[str]] = {
        language: Counter() for language, _extension in LANGS.values()
    }
    completed = 0
    lock = threading.Lock()

    def run_one(task: tuple[str, str]) -> tuple[str, str]:
        language, uid = task
        short = next(key for key, (name, _) in LANGS.items() if name == language)
        extension = EXTENSIONS[language]
        test_path = TESTS / short / f"{uid}{extension}"
        original = test_path.read_text(encoding="utf-8")
        try:
            candidate = _extract_f_gold(original, language)
            executable = _insert_candidate(original, candidate, language)
            test_count = _case_count(test_path, short)
            runtime = RUNTIMES[language]
            source_hash = _digest(candidate)
            test_hash = _digest(original)
        except Exception as exc:
            result, sandbox, test_count, source_hash, test_hash, runtime = (
                None,
                None,
                None,
                None,
                _digest(original),
                RUNTIMES[language],
            )
            setup_error = f"{type(exc).__name__}: {exc}"
        else:
            setup_error = None
        result_file = _result_path(output, language, uid)
        if setup_error is None and result_file.is_file():
            try:
                prior = json.loads(result_file.read_text(encoding="utf-8"))
                if (
                    prior.get("harness_version") == HARNESS_VERSION
                    and prior.get("source_sha256") == source_hash
                    and prior.get("test_sha256") == test_hash
                    and prior.get("repeats") == repeats
                    and prior.get("runtime_label") == runtime
                    and prior.get("status")
                    in {"passed", "reference_test_failure", "compile_failure"}
                ):
                    return language, prior["status"]
            except (OSError, json.JSONDecodeError):
                pass
        if setup_error is None:
            auxiliary = (
                {"javafx/util/Pair.java": PAIR_SHIM}
                if language == "Java" and "javafx.util.Pair" in original
                else {}
            )
            result, sandbox = run_complete_program_reference(
                executable,
                language,
                runtime,
                repeats=repeats,
                timeout_seconds=timeout_seconds,
                auxiliary_sources=auxiliary,
            )
            setup_error = None
        if setup_error is not None:
            status = "setup_failure"
        elif result is None:
            status = "runner_error"
        else:
            status = result["status"]
            samples = [item.get("stdout_sample", "") for item in result.get("program_outcomes", [])]
            if status != "passed":
                pass
            elif not samples:
                status = "harness_unrecognized"
            elif any(
                (match := re.search(r"#\s*Results\s*:\s*(\d+)\s*,\s*(\d+)", sample, re.IGNORECASE))
                is None
                or int(match.group(1)) != int(match.group(2))
                or int(match.group(2)) != test_count
                for sample in samples
            ):
                status = "reference_test_failure"
        row = {
            "schema_version": 1,
            "harness_version": HARNESS_VERSION,
            "benchmark": "TransCoder-Test",
            "dataset_revision": REVISION,
            "problem_id": uid,
            "language": language,
            "runtime_label": runtime,
            "source_sha256": source_hash,
            "test_sha256": test_hash,
            "test_count": test_count,
            "repeats": repeats,
            "status": status,
            "all_passed": status == "passed",
            "flaky": result.get("flaky") if result else None,
            "program_outcomes": result.get("program_outcomes") if result else None,
            "diagnostic": (
                result.get("diagnostic") or result.get("reason", "")
                if result
                else setup_error or sandbox.stderr.decode(errors="replace")[:1000]
                if sandbox
                else setup_error
            ),
        }
        _atomic_json(result_file, row)
        return language, status

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            language, status = future.result()
            with lock:
                counts[language][status] += 1
                completed += 1
                if completed % 20 == 0:
                    print(f"completed {completed}/{len(tasks)} counts={counts}", flush=True)
    return {
        language: {"tasks": sum(value.values()), "counts": dict(value)}
        for language, value in counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/transcoder_test_reference_preflight")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    print(
        json.dumps(
            run_preflight(args.output, args.repeats, args.timeout_seconds, args.workers), indent=2
        )
    )


if __name__ == "__main__":
    main()
