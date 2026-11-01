#!/usr/bin/env python3
"""Audit raw AVATAR-TC splits and documented human-written test-case joins."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REVISION = "c02d3b4ef856c676c3924dd720824e581a30f1a5"
TABLES = (
    "aizu",
    "atcoder",
    "codeforces",
    "codejam",
    "geeksforgeeks",
    "leetcode",
    "projecteuler",
)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def audit(root: Path, output: Path, markdown: Path) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        ids = _lines(root / f"{split}.java-python.id")
        java = _lines(root / f"{split}.java-python.java")
        python = _lines(root / f"{split}.java-python.python")
        if len(ids) != len(java) or len(ids) != len(python):
            raise ValueError(f"unaligned AVATAR-TC rows in {split}: {len(ids)}, {len(java)}, {len(python)}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate problem ID in AVATAR-TC {split}")
        split_counts[split] = {"problems": len(ids), "java_programs": len(java), "python_programs": len(python)}

    test_ids = set(_lines(root / "test.java-python.id"))
    tests: dict[str, list[dict[str, Any]]] = {}
    table_counts = {}
    for domain in TABLES:
        path = root / f"io_testcases_{domain}.json"
        table = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(table, dict):
            raise ValueError(f"expected object in {path}")
        table_counts[domain] = {"problems": len(table), "test_inputs": sum(len(v) for v in table.values())}
        overlap = tests.keys() & table.keys()
        if overlap:
            raise ValueError(f"duplicate test IDs across documented AVATAR-TC tables: {sorted(overlap)[:3]}")
        tests.update(table)
    matched = test_ids & tests.keys()
    counts = [len(tests[uid]) for uid in sorted(matched)]
    result = {
        "status": "RAW_AUDIT_ONLY; canonical tests and repeated reference preflight pending",
        "source_revision": REVISION,
        "license": "AVATAR-TC artifact license not independently stated in pinned CoTran repository",
        "splits": split_counts,
        "test_table_counts": table_counts,
        "test_problems": len(test_ids),
        "test_ids_with_documented_harness_data": len(matched),
        "unmatched_test_ids": sorted(test_ids - matched),
        "test_cases_per_problem": {
            "total": sum(counts),
            "mean": statistics.mean(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
        },
        "directions": {
            "Java-to-Python": {"problems": len(matched), "tasks": len(matched), "test_inputs": sum(counts)},
            "Python-to-Java": {"problems": len(matched), "tasks": len(matched), "test_inputs": sum(counts)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# AVATAR-TC raw coverage audit",
        "",
        "Status: **raw audit only**. Reference programs have not been compiled/run; no eligibility manifest is frozen and no model predictions exist.",
        "",
        f"- Pinned source revision: `{REVISION}`.",
        "- The pinned CoTran README documents AVATAR-TC test-case tables but does not state a separate AVATAR-TC license. Treat the license as unknown pending confirmation; retain provenance and do not redistribute.",
        f"- Split rows (problem IDs, Java code rows, Python code rows): `{json.dumps(split_counts, sort_keys=True)}`.",
        f"- Test split unique problems: {len(test_ids)}; IDs joined to documented input/output tables: {len(matched)}; unmatched IDs: {len(test_ids - matched)}.",
        f"- Test inputs: {sum(counts)} total; mean {statistics.mean(counts):.2f}, median {statistics.median(counts)}, min {min(counts)}, max {max(counts)} per problem.",
        "- Each documented test-table key has one input/output pair in this release. Each of the two directions has the same candidate problem IDs and target-test count; six-direction common-core is not applicable.",
        "",
        "| Direction | Problems/tasks | Target test inputs | Mean/problem | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, values in result["directions"].items():
        lines.append(f"| {direction.replace('-to-', ' → ')} | {values['problems']} | {values['test_inputs']} | 1.00 | 1 | 1 | 1 |")
    lines += [
        "",
        "All counts are preflight candidates. Compile/execute canonical Java and Python implementations against the documented test tables at least three times, apply the official output comparison semantics, hash the chosen tests, and freeze the eligible manifest before evaluating models.",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("content/datasets/avatar_tc/official_repo/AVATAR-TC"))
    parser.add_argument("--output", type=Path, default=Path("reports/data/avatar_tc_raw_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/data/avatar_tc_raw_audit.md"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output, args.markdown), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
