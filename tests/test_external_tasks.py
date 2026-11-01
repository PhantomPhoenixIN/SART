"""Tests for source-only external benchmark task loading."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.preprocess.prepare_external_structures import _structure_row
from src.evaluation.external_tasks import load_external_tasks


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_transcoder_rows_hide_reference_and_tests(tmp_path: Path) -> None:
    source = "int f_gold(){return 1;}"
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "common_core": {
                "directions": {
                    "C++-to-Java": {
                        "tasks": [
                            {
                                "problem_id": "p1",
                                "source_language": "C++",
                                "target_language": "Java",
                                "source_code": source,
                                "source_sha256": _hash(source),
                                "target_interface": "static int f_filled();",
                                "target_reference_sha256": "reference-hash",
                                "target_test_sha256": "test-hash",
                                "target_test_count": 4,
                            }
                        ]
                    }
                }
            }
        },
    )
    rows = load_external_tasks("transcoder_test", manifest)
    assert rows[0]["source"] == source
    assert rows[0]["problem_id"] == "p1"
    assert rows[0]["test_count"] == 4
    assert "tests" not in rows[0]
    assert "target_reference" not in rows[0]


def test_avatar_rows_use_src_uid_cluster(tmp_path: Path) -> None:
    source = "print(1)"
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "common_core": {
                "directions": {
                    "Python-to-Java": {
                        "tasks": [
                            {
                                "src_uid": "uid-1",
                                "source_language": "Python",
                                "target_language": "Java",
                                "source_code": source,
                                "source_sha256": _hash(source),
                                "tests": ["hidden"],
                                "test_count": 1,
                            }
                        ]
                    }
                }
            }
        },
    )
    row = load_external_tasks("avatar_tc", manifest)[0]
    assert row["problem_id"] == "uid-1"
    assert row["source"] == source
    assert "tests" not in row


@pytest.mark.parametrize(
    ("benchmark", "uid_field"),
    [("transcoder_test", "problem_id"), ("avatar_tc", "src_uid")],
)
def test_full_directional_manifest_shape_is_loaded(
    tmp_path: Path, benchmark: str, uid_field: str
) -> None:
    source = "print(1)"
    task = {
        uid_field: "p-full",
        "source_language": "Python",
        "target_language": "Java",
        "source_code": source,
        "source_sha256": _hash(source),
        "test_count": 1,
    }
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {"full_directional": {"Python-to-Java": {"tasks": [task]}}},
    )
    row = load_external_tasks(benchmark, manifest, split="full_directional")[0]
    assert row["problem_id"] == "p-full"
    assert row["split"] == "full_directional"


def test_humanevalx_uses_target_declaration_not_target_solution(tmp_path: Path) -> None:
    source = "def f(x):\n    return x + 1\n"
    root = tmp_path / "humanevalx"
    for language, subdir, prompt, solution, declaration in (
        ("Python", "python", "def f(x):\n", "    return x + 1\n", "def f(x):\n"),
        (
            "Java",
            "java",
            "class Solution { int f(int x) { return x; } }",
            "",
            "class Solution { int f(int x) {",
        ),
    ):
        row = {
            "task_id": f"{language}/1",
            "prompt": prompt,
            "canonical_solution": solution,
            "declaration": declaration,
        }
        path = root / "data" / subdir / "data" / "humaneval.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "common_core": {
                "Python-to-Java": {
                    "tasks": [
                        {
                            "problem_id": "1",
                            "source_language": "Python",
                            "target_language": "Java",
                            "source_code_sha256": _hash(source),
                            "target_reference_sha256": "target digest",
                            "target_test_sha256": "test digest",
                            "test_count": 1,
                        }
                    ]
                }
            }
        },
    )
    task = load_external_tasks("humanevalx", manifest, humanevalx_root=root)[0]
    assert task["source"] == source
    assert task["target_interface"] == "class Solution { int f(int x) {"
    assert "return x;" not in task["target_interface"]


def test_xcodeeval_recovers_selected_source_by_uid(tmp_path: Path) -> None:
    source = "print(7)\n"
    root = tmp_path / "xcode"
    path = root / "code_translation" / "test" / "Python.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "lang_cluster": "Python",
                "code_uid": "code-1",
                "source_code": source,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "directions": {
                "Python-to-Java": {
                    "tasks": [
                        {
                            "src_uid": "problem-1",
                            "source_language": "Python",
                            "target_language": "Java",
                            "source_code_uid": "code-1",
                            "source_sha256": _hash(source),
                            "target_reference_sha256": "ref",
                            "test_count": 2,
                        }
                    ]
                }
            }
        },
    )
    row = load_external_tasks("xcodeeval", manifest, xcode_root=root)[0]
    assert row["source"] == source
    assert row["problem_id"] == "problem-1"
    assert "reference" not in row


def test_source_hash_mismatch_is_fatal(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_json(
        manifest,
        {
            "common_core": {
                "directions": {
                    "Python-to-Java": {
                        "tasks": [
                            {
                                "problem_id": "p1",
                                "source_language": "Python",
                                "target_language": "Java",
                                "source_code": "print(1)",
                                "source_sha256": "wrong",
                            }
                        ]
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_external_tasks("transcoder_test", manifest)


class _WordOffsetTokenizer:
    def __call__(self, text: str, **kwargs):
        import re

        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


def test_structure_builder_aligns_source_and_uses_only_interface(tmp_path: Path) -> None:
    source = "def f(x):\n    return x + 1\n"
    record = _structure_row(
        {
            "benchmark": "fixture",
            "split": "common_core",
            "direction": "Python-to-Java",
            "task_id": "Python-to-Java/p1",
            "problem_id": "p1",
            "source_language": "Python",
            "target_language": "Java",
            "source": source,
            "source_sha256": _hash(source),
            "target_interface": "static int f(int x);",
            "target_reference_sha256": "reference hash only",
            "target_test_sha256": "test hash only",
            "test_count": 4,
        },
        _WordOffsetTokenizer(),
        {},
    )
    assert record["target"] == ""
    assert record["structure"]["availability"]["ast"] is True
    assert record["structure"]["source_token_count"] > 0
    assert record["structure"]["mapped_source_token_count"] > 0
    assert "tests" not in record
    assert "target_reference" not in record
