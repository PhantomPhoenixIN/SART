#!/usr/bin/env python3
"""Verify test/reference assets map exactly to the frozen external manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_sha256(text: str) -> str:
    return _sha256(text.replace("\r\n", "\n").replace("\r", "\n"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(manifest_root: Path, report_path: Path) -> dict[str, Any]:
    from scripts.evaluate.preflight_avatar_tc_references import load_data as load_avatar_data
    from scripts.evaluate.preflight_transcoder_test_references import (
        EXTENSIONS as transcoder_extensions,
    )
    from scripts.evaluate.preflight_transcoder_test_references import (
        TESTS as transcoder_tests,
    )
    from scripts.evaluate.score_external_predictions import (
        _load_humaneval_rows,
        _manifest_tasks,
    )

    results: dict[str, Any] = {}

    xcode_selection_path = manifest_root / "xcodeeval_reference_selection.json"
    xcode_selection = _read_json(xcode_selection_path)
    xcode_selection_hash = hashlib.sha256(xcode_selection_path.read_bytes()).hexdigest()
    humanoeval_rows = _load_humaneval_rows()
    avatar_ids, avatar_code, avatar_tests = load_avatar_data()
    avatar_uid_set = set(avatar_ids)

    configs = {
        "xcodeeval": [
            ("common_core", manifest_root / "xcodeeval_common_core_manifest.json"),
            ("full_directional", manifest_root / "xcodeeval_full_direction_manifest.json"),
        ],
        "humanevalx": [
            ("common_core", manifest_root / "humanevalx_manifest.json"),
            ("full_directional", manifest_root / "humanevalx_manifest.json"),
        ],
        "transcoder_test": [
            ("common_core", manifest_root / "transcoder_test_manifest.json"),
            ("full_directional", manifest_root / "transcoder_test_manifest.json"),
        ],
        "avatar_tc": [
            ("common_core", manifest_root / "avatar_tc_manifest.json"),
            ("full_directional", manifest_root / "avatar_tc_manifest.json"),
        ],
    }

    for benchmark, split_rows in configs.items():
        results[benchmark] = {}
        for split, manifest_path in split_rows:
            manifest = _read_json(manifest_path)
            if benchmark == "xcodeeval" and xcode_selection_hash != manifest[
                "selection_manifest_sha256"
            ]:
                raise ValueError("xCodeEval selection manifest hash mismatch")
            tasks = _manifest_tasks(benchmark, manifest, split)
            checked_tests = 0
            for task_key, task in tasks.items():
                target_language = task["target_language"]
                if benchmark == "xcodeeval":
                    selected = xcode_selection["selected"][target_language][task["src_uid"]]
                    target_reference = next(
                        candidate
                        for candidate in xcode_selection["candidate_pool"][target_language][
                            task["src_uid"]
                        ]
                        if candidate["code_uid"] == task["target_reference_code_uid"]
                    )
                    if selected["raw_test_sha256"] != task["raw_test_sha256"]:
                        raise ValueError(f"xCodeEval test digest mismatch for {task_key}")
                    if target_reference["source_sha256"] != task["target_reference_sha256"]:
                        raise ValueError(f"xCodeEval reference digest mismatch for {task_key}")
                    if target_reference["runtime_label"] != task["target_runtime_label"]:
                        raise ValueError(f"xCodeEval target runtime mismatch for {task_key}")
                    if selected["test_count"] != task["test_count"]:
                        raise ValueError(f"xCodeEval test count mismatch for {task_key}")
                    checked_tests += len(selected["tests"])
                elif benchmark == "humanevalx":
                    problem_id = str(task["problem_id"])
                    source = humanoeval_rows[(task["source_language"], problem_id)]
                    target = humanoeval_rows[(target_language, problem_id)]
                    if (
                        _sha256(source["prompt"] + source["canonical_solution"])
                        != task["source_code_sha256"]
                    ):
                        raise ValueError(f"HumanEval-X source digest mismatch for {task_key}")
                    if (
                        _sha256(target["prompt"] + target["canonical_solution"])
                        != task["target_reference_sha256"]
                    ):
                        raise ValueError(f"HumanEval-X target digest mismatch for {task_key}")
                    if _sha256(target["test"]) != task["target_test_sha256"]:
                        raise ValueError(f"HumanEval-X test digest mismatch for {task_key}")
                    checked_tests += int(task["test_count"])
                elif benchmark == "transcoder_test":
                    short = {"Python": "python", "Java": "java", "C++": "cpp"}[target_language]
                    test_path = (
                        transcoder_tests
                        / short
                        / f"{task['problem_id']}{transcoder_extensions[target_language]}"
                    )
                    test_source = test_path.read_text(encoding="utf-8")
                    if _sha256(test_source) != task["target_test_sha256"]:
                        raise ValueError(f"TransCoder-Test test digest mismatch for {task_key}")
                    if _sha256(task["source_code"]) != task["source_sha256"]:
                        raise ValueError(f"TransCoder-Test source digest mismatch for {task_key}")
                    if _sha256(task["target_interface"]) != task["target_interface_sha256"]:
                        raise ValueError(
                            f"TransCoder-Test interface digest mismatch for {task_key}"
                        )
                    checked_tests += int(task["target_test_count"])
                else:
                    uid = task["src_uid"]
                    if uid not in avatar_uid_set or uid not in avatar_tests:
                        raise ValueError(f"AVATAR-TC source/test ID missing for {task_key}")
                    code = avatar_code[target_language][uid]
                    if _normalized_sha256(task["source_code"]) != task["source_sha256"]:
                        raise ValueError(f"AVATAR-TC source digest mismatch for {task_key}")
                    if _normalized_sha256(code) != task["target_reference_sha256"]:
                        raise ValueError(f"AVATAR-TC target digest mismatch for {task_key}")
                    serialized = json.dumps(
                        task["tests"], ensure_ascii=False, separators=(",", ":")
                    )
                    if _sha256(serialized) != task["target_test_sha256"]:
                        raise ValueError(f"AVATAR-TC test digest mismatch for {task_key}")
                    checked_tests += int(task["test_count"])
            results[benchmark][split] = {
                "tasks": len(tasks),
                "unique_problems": len(
                    {task.get("src_uid", task.get("problem_id")) for task in tasks.values()}
                ),
                "directions": len({key.split("/", 1)[0] for key in tasks}),
                "test_inputs_across_tasks": checked_tests,
                "status": "PASS",
            }

    lines = [
        "# Frozen external scorer-input audit",
        "",
        (
            "This pre-model audit binds the frozen task manifest to the executable "
            "reference/test assets used by the scorer."
        ),
        "No trained-model predictions or scores are inspected.",
        "",
        "| Benchmark | Set | Tasks | Problems | Directions | Test inputs across tasks | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for benchmark, splits in results.items():
        for split, row in splits.items():
            lines.append(
                f"| {benchmark} | {split} | {row['tasks']} | {row['unique_problems']} | "
                f"{row['directions']} | {row['test_inputs_across_tasks']} | {row['status']} |"
            )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/frozen_scoring_input_audit.md")
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest_root, args.report), indent=2))


if __name__ == "__main__":
    main()
