#!/usr/bin/env python3
"""Freeze AVATAR-TC Java/Python eligibility from repeated official I/O checks."""

from __future__ import annotations

import argparse
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

from scripts.evaluate.preflight_avatar_tc_references import (  # noqa: E402
    COMPARISON_POLICY,
    HARNESS_VERSION,
    LANGUAGES,
    REVISION,
    RUNTIMES,
    _digest,
    load_data,
)

DIRECTIONS = (("Java", "Python"), ("Python", "Java"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _precision(count: int) -> dict[str, Any]:
    return {
        "statistical_unit": "src_uid/problem ID; both directions remain clustered",
        "worst_case_95_percent_ci_half_width_pp": round(100 * 1.96 / math.sqrt(count), 2)
        if count
        else None,
        "approximate_80_percent_power_mde_pp": round(100 * 2.80 / math.sqrt(count), 2)
        if count
        else None,
        "assumption": (
            "Conservative bounded paired difference; observed precision depends on paired outcomes."
        ),
    }


def freeze(results_root: Path) -> dict[str, Any]:
    ids, code, tests = load_data()
    unique_ids = sorted(set(ids) & tests.keys())
    records: dict[str, dict[str, dict[str, Any]]] = {language: {} for language in LANGUAGES}
    eligible: dict[str, set[str]] = {language: set() for language in LANGUAGES}
    exclusions: dict[str, list[dict[str, str]]] = {language: [] for language in LANGUAGES}

    for language in LANGUAGES:
        for uid in unique_ids:
            path = results_root / language / f"{uid}.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            expected_test_hash = _digest(
                json.dumps(tests[uid], ensure_ascii=False, separators=(",", ":"))
            )
            if (
                row.get("harness_version") != HARNESS_VERSION
                or row.get("comparison_policy") != COMPARISON_POLICY
                or row.get("dataset_revision") != REVISION
                or row.get("source_sha256") != _digest(code[language][uid])
                or row.get("test_sha256") != expected_test_hash
                or row.get("test_count") != len(tests[uid])
                or row.get("repeats") != 3
                or row.get("runtime_label") != RUNTIMES[language]
            ):
                raise RuntimeError(f"Stale or incomplete canonical result: {language}/{uid}")
            records[language][uid] = row
            valid = (
                row.get("status") == "passed"
                and row.get("all_passed") is True
                and row.get("flaky") is False
            )
            if valid:
                eligible[language].add(uid)
            else:
                exclusions[language].append(
                    {
                        "problem_id": uid,
                        "status": str(row.get("status", "unknown")),
                        "reason": str(row.get("diagnostic") or row.get("status", "unknown")),
                    }
                )

    common = sorted(set.intersection(*(eligible[language] for language in LANGUAGES)))
    common_tasks: dict[str, Any] = {}
    full_tasks: dict[str, Any] = {}
    for source, target in DIRECTIONS:
        pair_ids = sorted(eligible[source] & eligible[target])
        for chosen, destination in ((common, common_tasks), (pair_ids, full_tasks)):
            task_rows = []
            counts = []
            for uid in chosen:
                test_hash = _digest(
                    json.dumps(tests[uid], ensure_ascii=False, separators=(",", ":"))
                )
                counts.append(len(tests[uid]))
                task_rows.append(
                    {
                        "src_uid": uid,
                        "source_language": source,
                        "target_language": target,
                        "source_code": code[source][uid],
                        "source_sha256": _digest(code[source][uid]),
                        "target_reference_sha256": _digest(code[target][uid]),
                        "target_test_sha256": test_hash,
                        "test_count": len(tests[uid]),
                        "tests": tests[uid],
                        "source_runtime": RUNTIMES[source],
                        "target_runtime": RUNTIMES[target],
                    }
                )
            destination[f"{source}-to-{target}"] = {
                "unique_problems": len(chosen),
                "translation_tasks": len(chosen),
                "hidden_test_inputs": sum(counts),
                "mean_tests_per_problem": statistics.mean(counts) if counts else None,
                "median_tests_per_problem": statistics.median(counts) if counts else None,
                "minimum_tests_per_problem": min(counts) if counts else None,
                "maximum_tests_per_problem": max(counts) if counts else None,
                "precision": _precision(len(chosen)),
                "tasks": task_rows,
            }

    return {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "benchmark": "AVATAR-TC",
        "dataset_revision": REVISION,
        "harness_version": HARNESS_VERSION,
        "comparison_policy": COMPARISON_POLICY,
        "runtimes": RUNTIMES,
        "license_note": "Separate AVATAR-TC artifact license was not independently verified.",
        "sandbox": "Local rootless compatibility runner; not official ExecEval.",
        "source_selection_rule": (
            "One official test-split implementation per problem and source language."
        ),
        "eligibility_rule": (
            "Both the source- and target-language official programs parse and each canonical "
            "implementation passes every published I/O case in three stable runs."
        ),
        "common_problem_ids": common,
        "common_core": {
            "unique_problems": len(common),
            "directional_tasks": len(common) * len(DIRECTIONS),
            "hidden_test_executions_one_pass": sum(
                task["hidden_test_inputs"] for task in common_tasks.values()
            ),
            "precision": _precision(len(common)),
            "directions": common_tasks,
        },
        "full_directional": full_tasks,
        "available_problems": len(unique_ids),
        "passing_references": {language: len(eligible[language]) for language in LANGUAGES},
        "exclusions_by_language": exclusions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("outputs/avatar_tc_reference_preflight")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/avatar_tc_manifest.json")
    )
    args = parser.parse_args()
    manifest = freeze(args.results_root)
    _atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "common_problems": len(manifest["common_problem_ids"]),
                "common_tasks": manifest["common_core"]["directional_tasks"],
                "passing_references": manifest["passing_references"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
