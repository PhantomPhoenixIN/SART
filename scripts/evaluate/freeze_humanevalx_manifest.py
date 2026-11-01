#!/usr/bin/env python3
"""Freeze HumanEval-X canonical-reference eligibility before model inference."""

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

from scripts.preprocess.audit_humanevalx_raw import _test_count  # noqa: E402

LANGUAGES = ("Python", "Java", "C++")
FOLDERS = {"Python": "python", "Java": "java", "C++": "cpp"}
DIRECTIONS = (
    ("Python", "Java"),
    ("Java", "Python"),
    ("Python", "C++"),
    ("C++", "Python"),
    ("Java", "C++"),
    ("C++", "Java"),
)
REVISION = "62c78627f3072a1454fa0cb0184737cafe5e4198"
HARNESS_VERSION = 3


def digest_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def precision(problem_count: int) -> dict[str, Any]:
    return {
        "statistical_unit": "problem ID; related translation directions remain clustered",
        "worst_case_95_percent_ci_half_width_pp": round(100 * 1.96 / math.sqrt(problem_count), 2)
        if problem_count
        else None,
        "worst_case_80_percent_power_mde_pp": round(100 * 2.80 / math.sqrt(problem_count), 2)
        if problem_count
        else None,
        "assumption": (
            "Conservative bounded paired difference at problem-cluster level; actual precision "
            "depends on model discordance and covariance."
        ),
    }


def summarize(
    ids: list[str], target: str, rows: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    counts = [_test_count(rows[target][uid]["test"], target) for uid in ids]
    return {
        "unique_problems": len(ids),
        "translation_tasks": len(ids),
        "hidden_test_inputs": sum(counts),
        "mean_tests_per_problem": statistics.mean(counts) if counts else None,
        "median_tests_per_problem": statistics.median(counts) if counts else None,
        "minimum_tests_per_problem": min(counts) if counts else None,
        "maximum_tests_per_problem": max(counts) if counts else None,
        "precision": precision(len(ids)),
    }


def freeze(dataset_root: Path, results_root: Path) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    result_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for language in LANGUAGES:
        data_path = dataset_root / FOLDERS[language] / "data/humaneval.jsonl"
        rows[language] = {}
        for line in data_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            uid = row["task_id"].split("/", 1)[-1]
            rows[language][uid] = row
            result_path = results_root / language / f"{uid}.json"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            expected_source = digest_text(row["prompt"] + row["canonical_solution"])
            expected_test = digest_text(row["test"])
            if (
                record.get("harness_version") != HARNESS_VERSION
                or record.get("dataset_revision") != REVISION
                or record.get("source_sha256") != expected_source
                or record.get("test_sha256") != expected_test
                or record.get("repeats") != 3
            ):
                raise RuntimeError(f"Stale/incomplete preflight record: {language}/{uid}")
            result_rows.setdefault(language, {})[uid] = record
    common_ids = set.intersection(*(set(rows[lang]) for lang in LANGUAGES))
    valid = {
        lang: {
            uid
            for uid, record in result_rows[lang].items()
            if record.get("status") == "passed"
            and record.get("all_passed") is True
            and record.get("flaky") is False
        }
        for lang in LANGUAGES
    }
    common = sorted(
        common_ids & set.intersection(*(valid[lang] for lang in LANGUAGES)),
        key=lambda x: int(x),
    )
    all_dirs: dict[str, Any] = {}
    common_dirs: dict[str, Any] = {}
    for source, target in DIRECTIONS:
        eligible = sorted(common_ids & valid[source] & valid[target], key=lambda x: int(x))
        for _label, ids, destination in (
            ("common_core", common, common_dirs),
            ("full_directional", eligible, all_dirs),
        ):
            tasks = []
            for uid in ids:
                row = rows[target][uid]
                reference = result_rows[target][uid]
                tasks.append(
                    {
                        "problem_id": uid,
                        "source_language": source,
                        "target_language": target,
                        "source_code_sha256": digest_text(
                            rows[source][uid]["prompt"] + rows[source][uid]["canonical_solution"]
                        ),
                        "target_reference_sha256": reference["source_sha256"],
                        "target_test_sha256": reference["test_sha256"],
                        "test_count": _test_count(row["test"], target),
                        "runtime": reference["runtime_label"],
                    }
                )
            destination[f"{source}-to-{target}"] = {**summarize(ids, target, rows), "tasks": tasks}
    exclusions = {}
    for language in LANGUAGES:
        exclusions[language] = {
            "eligible_reference_count": len(valid[language]),
            "excluded": [
                {
                    "problem_id": uid,
                    "status": record.get("status"),
                    "reason": record.get("driver_diagnostic")
                    or record.get("exception_diagnostic")
                    or next(
                        (
                            outcome.get("stderr_sample", "")
                            for outcome in record.get("program_outcomes") or []
                            if outcome.get("stderr_sample")
                        ),
                        record.get("status", "unknown"),
                    ),
                }
                for uid, record in sorted(
                    result_rows[language].items(), key=lambda item: int(item[0])
                )
                if uid not in valid[language]
            ],
        }
    return {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "benchmark": "HumanEval-X",
        "dataset_revision": REVISION,
        "harness_version": HARNESS_VERSION,
        "runtime_policy": {
            "Python": "Private CPython 3.11; assertions enabled",
            "Java": "Private Temurin JDK 21 runtime and compiler with --release 17",
            "C++": "Pinned GNU C++11 compiler, static libstdc++/libgcc",
            "repeats": 3,
            "sandbox": (
                "Local rootless ExecEval-compatible namespace sandbox; not official service"
            ),
        },
        "eligibility_rule": (
            "Source and target canonical references both pass all official hidden tests in "
            "three stable runs. Common-Core also requires all three language references to "
            "pass for the problem."
        ),
        "common_core_problem_ids": common,
        "common_core": common_dirs,
        "full_directional": all_dirs,
        "exclusions_by_reference_language": exclusions,
        "precision": precision(len(common)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("content/datasets/humanevalx") / REVISION / "data",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("outputs/humanevalx_reference_preflight")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/humanevalx_manifest.json")
    )
    args = parser.parse_args()
    manifest = freeze(args.dataset_root, args.results_root)
    atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "common_core_problems": len(manifest["common_core_problem_ids"]),
                "directional_counts": {
                    name: item["unique_problems"]
                    for name, item in manifest["full_directional"].items()
                },
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
