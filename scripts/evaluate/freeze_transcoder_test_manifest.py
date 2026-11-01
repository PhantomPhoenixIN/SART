#!/usr/bin/env python3
"""Freeze TransCoder-Test problem-level manifests from repeated official-reference runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate.preflight_transcoder_test_references import (  # noqa: E402
    EXTENSIONS,
    HARNESS_VERSION,
    LANGS,
    RAW,
    RUNTIMES,
    TESTS,
    _digest,
    _extract_f_gold,
)
from scripts.preprocess.audit_transcoder_test import _case_count, _ids  # noqa: E402

DIRECTIONS = (
    ("Python", "Java"),
    ("Java", "Python"),
    ("Python", "C++"),
    ("C++", "Python"),
    ("Java", "C++"),
    ("C++", "Java"),
)
LANGUAGES = tuple(language for language, _extension in LANGS.values())
REVISION = "1c21887dc55a88ea6535e669952d419256fbc24a"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _result_path(root: Path, language: str, uid: str) -> Path:
    return root / language.replace("+", "p") / f"{uid}.json"


def _interface(function: str, language: str) -> str:
    if language == "Python":
        return function.splitlines()[0]
    brace = function.find("{")
    if brace < 0:
        raise ValueError(f"No function body in {language} reference")
    return function[:brace].rstrip() + ";"


def _precision(count: int) -> dict[str, Any]:
    return {
        "statistical_unit": "problem ID cluster; six related directions retained",
        "worst_case_95_percent_ci_half_width_pp": round(100 * 1.96 / math.sqrt(count), 2)
        if count
        else None,
        "approximate_80_percent_power_mde_pp": round(100 * 2.80 / math.sqrt(count), 2)
        if count
        else None,
        "assumption": (
            "Conservative bounded paired difference; realized precision depends on paired outcomes."
        ),
    }


def freeze(dataset_root: Path, results_root: Path) -> dict[str, Any]:
    raw_ids = {
        language: _ids(RAW / f"test.{short}.shuf.test.tok")
        for short, (language, _) in LANGS.items()
    }
    valid: dict[str, set[str]] = {language: set() for language in LANGUAGES}
    records: dict[str, dict[str, dict[str, Any]]] = {language: {} for language in LANGUAGES}
    exclusions: dict[str, list[dict[str, str]]] = {language: [] for language in LANGUAGES}
    available_ids: dict[str, set[str]] = {language: set() for language in LANGUAGES}
    for short, (language, _) in LANGS.items():
        for uid in raw_ids[language]:
            test_path = TESTS / short / f"{uid}{EXTENSIONS[language]}"
            if not test_path.is_file():
                continue
            available_ids[language].add(uid)
            record_path = _result_path(results_root, language, uid)
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Missing result {language}/{uid}: {exc}") from exc
            source = _extract_f_gold(test_path.read_text(encoding="utf-8"), language)
            if (
                record.get("harness_version") != HARNESS_VERSION
                or record.get("dataset_revision") != REVISION
                or record.get("source_sha256") != _digest(source)
                or record.get("test_sha256") != _digest(test_path.read_text(encoding="utf-8"))
                or record.get("test_count") != _case_count(test_path, short)
                or record.get("repeats") != 3
            ):
                raise RuntimeError(f"Stale/incomplete canonical result: {language}/{uid}")
            records[language][uid] = record
            if record.get("status") == "passed" and record.get("all_passed") is True:
                valid[language].add(uid)
            else:
                exclusions[language].append(
                    {
                        "problem_id": uid,
                        "status": record.get("status", "unknown"),
                        "reason": record.get("diagnostic") or record.get("status", "unknown"),
                    }
                )
    common_candidates = set.intersection(*(available_ids[language] for language in LANGUAGES))
    common = sorted(
        common_candidates & set.intersection(*(valid[language] for language in LANGUAGES))
    )

    common_dirs: dict[str, Any] = {}
    full_dirs: dict[str, Any] = {}
    for source, target in DIRECTIONS:
        full_ids = sorted(common_candidates & valid[source] & valid[target])
        for ids, destination in ((common, common_dirs), (full_ids, full_dirs)):
            tasks = []
            test_counts = []
            for uid in ids:
                target_path = (
                    TESTS
                    / next(short for short, (name, _) in LANGS.items() if name == target)
                    / f"{uid}{EXTENSIONS[target]}"
                )
                source_path = (
                    TESTS
                    / next(short for short, (name, _) in LANGS.items() if name == source)
                    / f"{uid}{EXTENSIONS[source]}"
                )
                target_script = target_path.read_text(encoding="utf-8")
                source_script = source_path.read_text(encoding="utf-8")
                source_reference = _extract_f_gold(source_script, source)
                source_code = source_reference.replace("f_filled", "f_gold", 1)
                target_code = _extract_f_gold(target_script, target)
                target_reference = target_code.replace("f_filled", "f_gold", 1)
                count = _case_count(
                    target_path, next(short for short, (name, _) in LANGS.items() if name == target)
                )
                test_counts.append(count)
                tasks.append(
                    {
                        "problem_id": uid,
                        "source_language": source,
                        "target_language": target,
                        "source_code": source_code,
                        "source_sha256": _digest(source_code),
                        "target_reference_sha256": _digest(target_reference),
                        "target_interface": _interface(target_code, target),
                        "target_interface_sha256": _digest(_interface(target_code, target)),
                        "target_test_sha256": _sha256_file(target_path),
                        "target_test_count": count,
                        "source_runtime": RUNTIMES[source],
                        "target_runtime": RUNTIMES[target],
                    }
                )
            destination[f"{source}-to-{target}"] = {
                "unique_problems": len(ids),
                "translation_tasks": len(ids),
                "hidden_test_inputs": sum(test_counts),
                "mean_tests_per_problem": statistics.mean(test_counts) if test_counts else None,
                "median_tests_per_problem": statistics.median(test_counts) if test_counts else None,
                "minimum_tests_per_problem": min(test_counts) if test_counts else None,
                "maximum_tests_per_problem": max(test_counts) if test_counts else None,
                "tasks": tasks,
            }
    distinct_common_tests = {
        language: sum(records[language][uid]["test_count"] for uid in common)
        for language in LANGUAGES
    }
    return {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "benchmark": "TransCoder-Test",
        "dataset_revision": REVISION,
        "source_selection_rule": (
            "One official f_gold function per problem and source language, from the pinned "
            "matching benchmark test script; the release offers no alternative submissions."
        ),
        "target_interface_rule": (
            "Official target f_gold signature with only its symbol replaced by f_filled, "
            "matching the official TOFILL harness contract."
        ),
        "harness_rule": (
            "Duplicate canonical f_gold as f_filled at TOFILL; require the official reported "
            "pass count to equal audited hidden-test inputs across three stable runs."
        ),
        "runtimes": RUNTIMES,
        "sandbox": "Local rootless ExecEval-compatible fallback; not official ExecEval.",
        "common_core_problem_ids": common,
        "common_core": {
            "unique_problems": len(common),
            "directional_tasks": len(common) * 6,
            "distinct_tests_by_target_language": distinct_common_tests,
            "hidden_test_executions_one_pass": sum(
                info["hidden_test_inputs"] for info in common_dirs.values()
            ),
            "precision": _precision(len(common)),
            "directions": common_dirs,
        },
        "full_directional": full_dirs,
        "available_test_script_counts": {
            language: len(available_ids[language]) for language in LANGUAGES
        },
        "passing_reference_counts": {language: len(valid[language]) for language in LANGUAGES},
        "exclusions_by_language": exclusions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("content/datasets/transcoder-test")
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("outputs/transcoder_test_reference_preflight")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/transcoder_test_manifest.json")
    )
    args = parser.parse_args()
    value = freeze(args.dataset_root, args.results_root)
    _atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "common_core_problems": len(value["common_core_problem_ids"]),
                "common_core_tasks": value["common_core"]["directional_tasks"],
                "passing_reference_counts": value["passing_reference_counts"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
