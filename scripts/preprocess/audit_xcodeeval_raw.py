#!/usr/bin/env python3
"""Audit raw pinned xCodeEval test-shard coverage without model evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

LANGUAGES = ("Python", "Java", "C++")
REVISION = "aa09386604075a3e80bcdb315dd68bcc117dedc0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def describe_tests(problem_ids: set[str], test_db: dict[str, list[dict[str, Any]]]) -> str:
    counts = [len(test_db[problem_id]) for problem_id in sorted(problem_ids)]
    return (
        f"{sum(counts)}; mean {statistics.mean(counts):.2f}; "
        f"median {statistics.median(counts):g}; min {min(counts)}; max {max(counts)}"
    )


def audit(dataset_root: Path, output: Path) -> None:
    manifest_path = dataset_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("repo_id") != "NTU-NLP-sg/xCodeEval":
        raise ValueError("Unexpected xCodeEval artifact manifest repo_id")
    if manifest.get("revision") != REVISION or manifest.get("status") != "COMPLETE":
        raise ValueError("xCodeEval files must come from the pinned complete revision")

    rows = {
        language: read_jsonl(dataset_root / f"code_translation/test/{language}.jsonl")
        for language in LANGUAGES
    }
    test_db = json.loads((dataset_root / "unittest_db.json").read_text(encoding="utf-8"))
    if not all(isinstance(value, list) for value in test_db.values()):
        raise ValueError("Expected unit-test database values to be lists")

    ids = {language: {str(row["src_uid"]) for row in language_rows} for language, language_rows in rows.items()}
    common = set.intersection(*(ids[language] for language in LANGUAGES))
    union = set.union(*(ids[language] for language in LANGUAGES))
    if not union.issubset(test_db):
        missing = len(union.difference(test_db))
        raise ValueError(f"{missing} raw source problems have no hidden-test DB record")

    lines = [
        "# xCodeEval raw test split audit",
        "",
        "Status: **raw source/test-database audit only; not an eligibility manifest.** Canonical parse/compile/test preflight, repeatability screening, and deterministic one-source selection remain required before freezing evaluation tasks.",
        "",
        f"Dataset revision: `{REVISION}`  ",
        "Dataset: `NTU-NLP-sg/xCodeEval`, `code_translation/test`  ",
        "Training shards were not downloaded.",
        "",
        "## Raw rows and source IDs",
        "",
        "| Language | Raw accepted-status rows | Unique src_uid | Problems with multiple source rows | Distinct-problem test inputs (sum; mean; median; min; max) |",
        "|---|---:|---:|---:|---:|",
    ]
    for language in LANGUAGES:
        language_rows = rows[language]
        language_ids = ids[language]
        if any(row.get("exec_outcome") != "PASSED" for row in language_rows):
            raise ValueError(f"Non-PASSED exec_outcome exists in {language} raw test shard")
        multiplicities = Counter(str(row["src_uid"]) for row in language_rows)
        multi = sum(count > 1 for count in multiplicities.values())
        lines.append(
            f"| {language} | {len(language_rows)} | {len(language_ids)} | {multi} | "
            f"{describe_tests(language_ids, test_db)} |"
        )

    lines.extend(
        [
            "",
            "## Cross-language candidate sets",
            "",
            f"- Raw union: {len(union)} distinct problem IDs; `{describe_tests(union, test_db)}` hidden tests.",
            f"- Raw three-language intersection: {len(common)} distinct IDs; `{describe_tests(common, test_db)}` hidden tests.",
            f"- Common-core directional tasks before eligibility filtering: {len(common) * 6:,} (six directions per problem).",
            f"- Common-core target-test executions before eligibility filtering: {sum(len(test_db[problem_id]) for problem_id in common) * 6:,} across six translations per problem.",
            "- Pairwise intersections are reported as raw candidate counts below; they are not frozen eligibility counts.",
            "",
            "| Direction | Shared problem IDs | Translation tasks | Distinct target test inputs | Mean/problem | Median | Min | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            pair_ids = ids[source] & ids[target]
            tests = [len(test_db[problem_id]) for problem_id in sorted(pair_ids)]
            lines.append(
                f"| {source} → {target} | {len(pair_ids)} | {len(pair_ids)} | {sum(tests)} | "
                f"{statistics.mean(tests):.2f} | {statistics.median(tests):g} | "
                f"{min(tests)} | {max(tests)} |"
            )
    lines.extend(
        [
            "",
            "`exec_outcome=PASSED` is the upstream recorded outcome, not a substitute for our pinned-runtime repeated canonical test execution. The mean/median/min/max values above count unit-test inputs in the official `unittest_db.json`; they do not yet account for tasks excluded after preflight.",
            "",
            "Raw project file hashes are stored in `artifact_manifest.json` beside the downloaded data.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("content/datasets/xcodeeval") / REVISION,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data/xcodeeval_raw_audit.md")
    )
    args = parser.parse_args()
    audit(args.dataset_root, args.output)


if __name__ == "__main__":
    main()
