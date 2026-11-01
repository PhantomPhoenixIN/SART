#!/usr/bin/env python3
"""Freeze xCodeEval eligibility manifests and pre-result precision estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

LANGUAGES = ("Python", "Java", "C++")
DIRECTIONS = (
    ("Python", "Java"),
    ("Java", "Python"),
    ("Python", "C++"),
    ("C++", "Python"),
    ("Java", "C++"),
    ("C++", "Java"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def result_path(output: Path, language: str, src_uid: str) -> Path:
    return output / language.replace("+", "p") / f"{src_uid}.json"


def read_results(
    selection: dict[str, Any],
    selection_hash: str,
    output: Path,
    python311_output: Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for language in LANGUAGES:
        language_output = (
            python311_output if language == "Python" and python311_output is not None else output
        )
        results[language] = {}
        for src_uid in selection["candidate_pool"][language]:
            path = result_path(language_output, language, src_uid)
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing.append(f"{language}:{src_uid}")
                continue
            if row.get("manifest_sha256") != selection_hash or row.get("complete") is not True:
                missing.append(f"{language}:{src_uid}")
                continue
            results[language][src_uid] = row
    if missing:
        raise RuntimeError(
            f"Cannot freeze eligibility: {len(missing)} canonical references incomplete; "
            f"examples={missing[:10]}"
        )
    return results


def coverage(ids: list[str], target: str, selection: dict[str, Any]) -> dict[str, Any]:
    counts = [selection["selected"][target][uid]["test_count"] for uid in ids]
    return {
        "unique_problems": len(ids),
        "translation_tasks": len(ids),
        "hidden_test_inputs": sum(counts),
        "mean_tests_per_problem": statistics.mean(counts) if counts else None,
        "median_tests_per_problem": statistics.median(counts) if counts else None,
        "minimum_tests_per_problem": min(counts) if counts else None,
        "maximum_tests_per_problem": max(counts) if counts else None,
        "precision": {
            "statistical_unit": "problem_id; six directions stay clustered",
            "worst_case_95_percent_ci_half_width_pp": round(100 * 1.96 / (len(ids) ** 0.5), 2)
            if ids
            else None,
            "worst_case_80_percent_power_mde_pp": round(100 * 2.80 / (len(ids) ** 0.5), 2)
            if ids
            else None,
            "assumption": (
                "Conservative bounded paired difference in [-1,1] at the problem-cluster "
                "level; actual precision depends on model discordance/covariance."
            ),
        },
    }


def task_records(
    ids: list[str], source: str, target: str, selection: dict[str, Any], results: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    for uid in ids:
        ref = results[target][uid]
        selected = selection["selected"][target][uid]
        records.append(
            {
                "src_uid": uid,
                "source_language": source,
                "target_language": target,
                "source_code_uid": results[source][uid]["selected_code_uid"],
                "source_sha256": results[source][uid]["selected_source_sha256"],
                "target_reference_code_uid": ref["selected_code_uid"],
                "target_reference_sha256": ref["selected_source_sha256"],
                "target_runtime_label": ref["selected_runtime_label"],
                "test_count": selected["test_count"],
                "raw_test_sha256": selected["raw_test_sha256"],
            }
        )
    return records


def freeze(
    selection_path: Path,
    results_path: Path,
    output_dir: Path,
    python311_results_path: Path | None = None,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = sha256(selection_path)
    results = read_results(selection, selection_hash, results_path, python311_results_path)
    valid = {
        language: {
            uid
            for uid, row in results[language].items()
            if row.get("valid") is True and row.get("status") == "passed"
        }
        for language in LANGUAGES
    }
    common = sorted(set.intersection(*(valid[language] for language in LANGUAGES)))
    common_manifest = {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "dataset_revision": selection["dataset_revision"],
        "selection_manifest_sha256": selection_hash,
        "runtime_policy": (
            "Pinned ExecEval-compatible rootless namespace runner; references pass every "
            "official hidden input in three stable repetitions. Python-language references "
            "use the private CPython3.11 rerun where applicable; Java and C++ use the "
            "recorded per-task runtimes. This runner is not the official ExecEval service."
        ),
        "eligibility_rule": (
            "Common-Core requires a passing canonical reference in Python, Java, and C++ "
            "for the same src_uid."
        ),
        "source_selection_rule": selection["selection_rule"],
        "unique_problem_ids": common,
        "common_core_coverage": {},
        "directions": {},
    }
    for source, target in DIRECTIONS:
        ids = common
        common_manifest["directions"][f"{source}-to-{target}"] = {
            **coverage(ids, target, selection),
            "tasks": task_records(ids, source, target, selection, results),
        }
    common_manifest["common_core_coverage"] = {
        "unique_problems": len(common),
        "directional_tasks": len(common) * 6,
        "hidden_test_executions_one_pass": sum(
            info["hidden_test_inputs"] for name, info in common_manifest["directions"].items()
        ),
        "per_target_reference_tests": {
            target: sum(selection["selected"][target][uid]["test_count"] for uid in common)
            for target in LANGUAGES
        },
        "precision": {
            "statistical_unit": "src_uid cluster, retaining all six directions",
            "worst_case_95_percent_ci_half_width_pp": round(100 * 1.96 / (len(common) ** 0.5), 2)
            if common
            else None,
            "worst_case_80_percent_power_mde_pp": round(100 * 2.80 / (len(common) ** 0.5), 2)
            if common
            else None,
        },
    }
    full_manifest = {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "dataset_revision": selection["dataset_revision"],
        "selection_manifest_sha256": selection_hash,
        "eligibility_rule": (
            "Full-Directional tasks require stable passing canonical references for the "
            "selected source and target language implementations of the same src_uid."
        ),
        "source_selection_rule": selection["selection_rule"],
        "directions": {},
    }
    for source, target in DIRECTIONS:
        ids = sorted(valid[source] & valid[target])
        full_manifest["directions"][f"{source}-to-{target}"] = {
            **coverage(ids, target, selection),
            "tasks": task_records(ids, source, target, selection, results),
        }
    atomic_json(output_dir / "xcodeeval_common_core_manifest.json", common_manifest)
    atomic_json(output_dir / "xcodeeval_full_direction_manifest.json", full_manifest)
    return {
        "common_core_problems": len(common),
        "common_core_tasks": len(common) * 6,
        "valid_reference_counts": {lang: len(valid[lang]) for lang in LANGUAGES},
        "full_direction_counts": {
            direction: value["translation_tasks"]
            for direction, value in full_manifest["directions"].items()
        },
        "common_manifest_sha256": sha256(output_dir / "xcodeeval_common_core_manifest.json"),
        "full_manifest_sha256": sha256(output_dir / "xcodeeval_full_direction_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=Path("data/manifests/xcodeeval_reference_selection.json")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("outputs/xcodeeval_reference_preflight")
    )
    parser.add_argument(
        "--python311-results",
        type=Path,
        default=Path("outputs/xcodeeval_reference_preflight_py311"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.selection, args.results, args.output_dir, args.python311_results), indent=2
        )
    )


if __name__ == "__main__":
    main()
