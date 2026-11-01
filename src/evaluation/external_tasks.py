"""Load frozen external benchmark manifests into one source-only inference schema.

This module deliberately exposes neither target reference code nor hidden tests to
the generation path. Those assets stay in benchmark-specific evaluators.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _xcode_program_index(root: Path) -> dict[str, dict[str, str]]:
    """Index official xCodeEval test programs by stable language/code UID."""
    index: dict[str, dict[str, str]] = {}
    for path in sorted((root / "code_translation" / "test").glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                index[f"{row['lang_cluster']}:{row['code_uid']}"] = {
                    "source": row["source_code"],
                    "language": row["lang_cluster"],
                }
    return index


def _humaneval_index(
    root: Path, languages: set[str]
) -> dict[tuple[str, str], dict[str, str]]:
    """Index HumanEval-X prompt-plus-canonical source programs by language/task."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    language_dirs = {"Python": "python", "Java": "java", "C++": "cpp"}
    for language, directory in language_dirs.items():
        if language not in languages:
            continue
        path = root / "data" / directory / "data" / "humaneval.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                index[(language, str(row["task_id"]).split("/")[-1])] = row
    return index


def _direction_records(
    benchmark: str,
    split_name: str,
    directions: dict[str, Any],
    xcode_programs: dict[str, dict[str, str]] | None,
    humaneval_rows: dict[tuple[str, str], dict[str, str]] | None,
) -> Iterator[dict[str, Any]]:
    for direction, direction_data in sorted(directions.items()):
        tasks = direction_data.get("tasks", []) if isinstance(direction_data, dict) else []
        for task in tasks:
            src_lang = task["source_language"]
            tgt_lang = task["target_language"]
            if benchmark == "xcodeeval":
                assert xcode_programs is not None
                selected = xcode_programs[f"{src_lang}:{task['source_code_uid']}"]
                source = selected["source"]
                problem_id = str(task["src_uid"])
                interface = ""
            elif benchmark == "humanevalx":
                assert humaneval_rows is not None
                source_row = humaneval_rows[(src_lang, str(task["problem_id"]))]
                target_row = humaneval_rows[(tgt_lang, str(task["problem_id"]))]
                source = source_row["prompt"] + source_row["canonical_solution"]
                problem_id = str(task["problem_id"])
                interface = target_row["declaration"]
            else:
                source = task["source_code"]
                problem_id = str(task["src_uid"] if "src_uid" in task else task["problem_id"])
                interface = str(task.get("target_interface", ""))
            source_hash = _sha256(source)
            expected_hash = task.get("source_sha256", task.get("source_code_sha256"))
            if expected_hash is not None and source_hash != expected_hash:
                raise ValueError(
                    f"Frozen {benchmark} source hash mismatch for {direction}/{problem_id}: "
                    f"expected {expected_hash}, got {source_hash}"
                )
            # The target/reference fields intentionally contain hashes and test counts only.
            yield {
                "benchmark": benchmark,
                "split": split_name,
                "direction": direction,
                "task_id": f"{direction}/{problem_id}",
                "problem_id": problem_id,
                "source_language": src_lang,
                "target_language": tgt_lang,
                "source": source,
                "source_sha256": source_hash,
                "target_interface": interface,
                "target_reference_sha256": task.get("target_reference_sha256"),
                "target_test_sha256": task.get("target_test_sha256", task.get("raw_test_sha256")),
                "test_count": int(task.get("target_test_count", task.get("test_count", 0))),
            }


def load_external_tasks(
    benchmark: str,
    manifest_path: Path,
    split: str = "common_core",
    xcode_root: Path | None = None,
    humanevalx_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load frozen tasks for generation, verifying source hashes and hiding labels.

    ``benchmark`` is one of ``xcodeeval``, ``humanevalx``, ``transcoder_test`` or
    ``avatar_tc``. For the first two benchmarks, raw source programs are recovered
    from their pinned official dataset snapshots. Other manifests include source code.
    """
    aliases = {"transcoder": "transcoder_test", "avatar": "avatar_tc"}
    benchmark = aliases.get(benchmark, benchmark)
    if benchmark not in {"xcodeeval", "humanevalx", "transcoder_test", "avatar_tc"}:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if benchmark == "xcodeeval" and "directions" in manifest:
        # xCodeEval freezes common-core and full-direction sets as separate files.
        directions = manifest["directions"]
    else:
        directions = manifest[split]
    if benchmark in {"transcoder_test", "avatar_tc"} and "directions" in directions:
        directions = directions["directions"]
    xcode_programs = (
        _xcode_program_index(xcode_root)
        if benchmark == "xcodeeval" and xcode_root
        else None
    )
    humaneval_rows = (
        _humaneval_index(
            humanevalx_root,
            {
                task["source_language"]
                for direction in directions.values()
                for task in direction.get("tasks", [])
            }
            | {
                task["target_language"]
                for direction in directions.values()
                for task in direction.get("tasks", [])
            },
        )
        if benchmark == "humanevalx" and humanevalx_root
        else None
    )
    if benchmark == "xcodeeval" and xcode_programs is None:
        raise ValueError("xcode_root is required for xCodeEval task loading")
    if benchmark == "humanevalx" and humaneval_rows is None:
        raise ValueError("humanevalx_root is required for HumanEval-X task loading")
    return list(
        _direction_records(
            benchmark, split, directions, xcode_programs, humaneval_rows
        )
    )
