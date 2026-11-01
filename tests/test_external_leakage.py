"""Tests for cross-benchmark XLCoST leakage filtering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.preprocess.audit_external_leakage import (
    _convert_synthetic_layout,
    _write_synthetic,
)
from scripts.preprocess.audit_xlcost_leakage import audit


def test_external_exact_match_drops_whole_training_problem(tmp_path: Path) -> None:
    """A match in one language excludes the complete paired training problem."""
    for split in ("train", "validation", "test"):
        (tmp_path / split).mkdir()
    python_code = "def f(x):\n    return x + 1\n"
    java_code = "class F { int f(int x) { return x + 1; } }\n"
    external_java = "class External { int g(int x) { return x + 1; } }\n"

    def record(problem: str, language: str, code: str, source: str) -> dict[str, str]:
        return {
            "problem_id": problem,
            "language": language,
            "code": code,
            "sha256": hashlib.sha256(code.encode()).hexdigest(),
            "source": source,
        }

    _write_synthetic(
        tmp_path / "train" / "xlcost.jsonl",
        [
            record("problem-1", "Python", python_code, "XLCoST"),
            record("problem-1", "Java", java_code, "XLCoST"),
        ],
    )
    _write_synthetic(
        tmp_path / "validation" / "external.jsonl",
        [
            record("external-1", "Java", java_code, "HumanEval-X"),
            record("external-2", "Java", external_java, "TransCoder-Test"),
        ],
    )
    _write_synthetic(tmp_path / "test" / "empty.jsonl", [])
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    _convert_synthetic_layout(tmp_path)

    result = audit(tmp_path, tmp_path / "filtered", tmp_path / "report.json")

    assert result["dropped_training_problem_ids"] == ["problem-1"]
    kept = (tmp_path / "filtered" / "train" / "xlcost.jsonl").read_text(encoding="utf-8")
    assert kept == ""
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["match_counts_by_rule"]["exact"] == 1
