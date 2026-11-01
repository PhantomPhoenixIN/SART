#!/usr/bin/env python3
"""Exercise each external scoring adapter using one frozen canonical solution."""

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


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate(selection: dict[str, Any], language: str, uid: str, code_uid: str) -> dict[str, Any]:
    return next(
        candidate
        for candidate in selection["candidate_pool"][language][uid]
        if candidate["code_uid"] == code_uid
    )


def smoke(manifest_root: Path, repeats: int = 3) -> dict[str, Any]:
    from scripts.evaluate.preflight_avatar_tc_references import load_data as load_avatar_data
    from scripts.evaluate.preflight_transcoder_test_references import (
        EXTENSIONS as transcoder_extensions,
    )
    from scripts.evaluate.preflight_transcoder_test_references import (
        TESTS as transcoder_tests,
    )
    from scripts.evaluate.preflight_transcoder_test_references import (
        _extract_f_gold,
    )
    from scripts.evaluate.score_external_predictions import (
        _load_humaneval_rows,
        _manifest_tasks,
        _score_task,
    )

    results = {}

    x_manifest_path = manifest_root / "xcodeeval_common_core_manifest.json"
    x_manifest = json.loads(x_manifest_path.read_text(encoding="utf-8"))
    x_selection = json.loads(
        (manifest_root / "xcodeeval_reference_selection.json").read_text(encoding="utf-8")
    )
    x_tasks = _manifest_tasks("xcodeeval", x_manifest, "common_core")
    x_key = next(key for key in sorted(x_tasks) if key.startswith("C++-to-Java/"))
    x_task = x_tasks[x_key]
    x_reference = _candidate(
        x_selection, "Java", x_task["src_uid"], x_task["target_reference_code_uid"]
    )
    x_task["_source_sha256"] = x_task["source_sha256"]
    x_task["_target_interface_sha256"] = _sha256("")
    x_prediction = {
        "task_id": x_key,
        "status": "generated",
        "generated_code": x_reference["source_code"],
        "source_sha256": x_task["source_sha256"],
        "target_interface_sha256": _sha256(""),
        "target_reference_sha256": x_task["target_reference_sha256"],
        "target_test_sha256": x_task["raw_test_sha256"],
    }
    results["xcodeeval"] = _score_task(
        "xcodeeval",
        x_task,
        x_prediction,
        x_selection["selected"]["Java"][x_task["src_uid"]]["tests"],
        repeats,
    )

    human_rows = _load_humaneval_rows()
    h_manifest = json.loads((manifest_root / "humanevalx_manifest.json").read_text())
    h_tasks = _manifest_tasks("humanevalx", h_manifest, "common_core")
    h_key = next(key for key in sorted(h_tasks) if key.startswith("Python-to-Java/"))
    h_task = h_tasks[h_key]
    h_target = human_rows[("Java", str(h_task["problem_id"]))]
    h_reference = h_target["prompt"] + h_target["canonical_solution"]
    h_task["_source_sha256"] = h_task["source_code_sha256"]
    h_task["_target_interface_sha256"] = _sha256(h_target["prompt"])
    h_prediction = {
        "task_id": h_key,
        "status": "generated",
        "generated_code": h_reference,
        "source_sha256": h_task["source_code_sha256"],
        "target_interface_sha256": h_task["_target_interface_sha256"],
        "target_reference_sha256": h_task["target_reference_sha256"],
        "target_test_sha256": h_task["target_test_sha256"],
    }
    results["humanevalx"] = _score_task("humanevalx", h_task, h_prediction, h_target, repeats)

    t_manifest = json.loads((manifest_root / "transcoder_test_manifest.json").read_text())
    t_tasks = _manifest_tasks("transcoder_test", t_manifest, "common_core")
    t_key = next(key for key in sorted(t_tasks) if key.startswith("C++-to-Java/"))
    t_task = t_tasks[t_key]
    t_path = transcoder_tests / "java" / f"{t_task['problem_id']}{transcoder_extensions['Java']}"
    t_script = t_path.read_text(encoding="utf-8")
    t_reference = _extract_f_gold(t_script, "Java")
    t_task["_source_sha256"] = t_task["source_sha256"]
    t_task["_target_interface_sha256"] = _sha256(t_task["target_interface"])
    t_prediction = {
        "task_id": t_key,
        "status": "generated",
        "generated_code": t_reference,
        "source_sha256": t_task["source_sha256"],
        "target_interface_sha256": t_task["_target_interface_sha256"],
        "target_reference_sha256": t_task["target_reference_sha256"],
        "target_test_sha256": t_task["target_test_sha256"],
    }
    results["transcoder_test"] = _score_task(
        "transcoder_test", t_task, t_prediction, t_script, repeats
    )

    a_manifest = json.loads((manifest_root / "avatar_tc_manifest.json").read_text())
    a_tasks = _manifest_tasks("avatar_tc", a_manifest, "common_core")
    a_key = next(key for key in sorted(a_tasks) if key.startswith("Java-to-Python/"))
    a_task = a_tasks[a_key]
    _ids, avatar_code, avatar_tests = load_avatar_data()
    a_reference = avatar_code["Python"][a_task["src_uid"]]
    a_task["_source_sha256"] = a_task["source_sha256"]
    a_task["_target_interface_sha256"] = _sha256("")
    a_task["_target_requires_pair_shim"] = False
    a_prediction = {
        "task_id": a_key,
        "status": "generated",
        "generated_code": a_reference,
        "source_sha256": a_task["source_sha256"],
        "target_interface_sha256": _sha256(""),
        "target_reference_sha256": a_task["target_reference_sha256"],
        "target_test_sha256": a_task["target_test_sha256"],
    }
    results["avatar_tc"] = _score_task(
        "avatar_tc", a_task, a_prediction, avatar_tests[a_task["src_uid"]], repeats
    )
    for benchmark, result in results.items():
        if not result.get("pass_at_1"):
            raise RuntimeError(f"Canonical scorer smoke failed for {benchmark}: {result}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/scorer_reference_smoke.json")
    )
    args = parser.parse_args()
    results = smoke(args.manifest_root, args.repeats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value["status"] for key, value in results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
