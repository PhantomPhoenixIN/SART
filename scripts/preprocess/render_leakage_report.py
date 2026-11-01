#!/usr/bin/env python3
# ruff: noqa: E501
"""Render the combined XLCoST internal and external leakage audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def render(internal_path: Path, external_path: Path, output: Path) -> None:
    internal: dict[str, Any] = json.loads(internal_path.read_text(encoding="utf-8"))
    external: dict[str, Any] = json.loads(external_path.read_text(encoding="utf-8"))
    directional_rows: dict[str, dict[str, int]] = {}
    for relative, counts in external["filtered_structural_full"]["train"].items():
        direction = Path(relative).parts[1]
        totals = directional_rows.setdefault(direction, {"kept": 0, "removed": 0})
        totals["kept"] += counts["kept"]
        totals["removed"] += counts["removed"]
    lines = [
        "# Data leakage audit",
        "",
        "Date: 2026-10-02",
        "Status: **Internal XLCoST split audit and external benchmark source-code audit complete.**",
        "No model predictions or external benchmark scores were inspected.",
        "",
        "## Frozen fingerprint rules",
        "",
        "| Level | Rule |",
        "|---|---|",
        "| Exact | Normalize CRLF/CR to LF; SHA256 remaining UTF-8 source unchanged. |",
        "| Lexical | Tree-sitter leaf-token sequence, comments excluded and literals preserved. |",
        "| Identifier normalized | Deterministically rename local-like identifiers by first occurrence. |",
        "| Structural | Hash Tree-sitter AST node-type preorder with comments and leaf values removed. |",
        "| Near duplicate | Token 5-gram MinHash (256 permutations, 64×4 bands), estimated Jaccard ≥0.80 and exact Jaccard ≥0.85; minimum 40 tokens. |",
        "",
        "Any same-language match drops the entire XLCoST training `problem_id` across all languages and directions. Validation, test, and external tasks are retained unchanged. The near-duplicate implementation first generates candidates with MinHash bands, then applies the frozen exact 5-gram threshold.",
        "",
        "## XLCoST internal split protection",
        "",
        f"- Parsed implementations: `{json.dumps(internal['unique_program_counts'], sort_keys=True)}`.",
        f"- Training problem IDs removed against XLCoST validation/test: **{internal['dropped_training_problem_count']}**.",
        f"- Match records by rule: `{json.dumps(internal['match_counts_by_rule'], sort_keys=True)}`.",
        "- Detailed IDs, similarities, protected split, and code hashes: `reports/data/xlcost_leakage.json`.",
        "",
        "## External benchmark source-code protection",
        "",
        f"- Parse-valid external programs fingerprinted: `{json.dumps({k: v for k, v in external['external_parsed_program_counts'].items() if k != 'parser_rejections'}, sort_keys=True)}`.",
        f"- Parser rejections retained as coverage limitations (benchmark/language): `{json.dumps(external['external_parsed_program_counts']['parser_rejections'], sort_keys=True)}`.",
        f"- XLCoST training problem IDs removed: **{external['dropped_training_problem_count']}**.",
        f"- Match records by rule: `{json.dumps(external['match_counts_by_rule'], sort_keys=True)}`.",
        "- Sources: frozen xCodeEval candidate pool, HumanEval-X prompt+canonical solution, TransCoder-Test test functions (official Python detokenizer), and AVATAR-TC test split source programs.",
        "- Only training rows were filtered. The original data and running pilot sample remain untouched; a new structure-enriched corpus is at `content/datasets/xlcost/structural_external_clean/`.",
        "- Output shard SHA256/row counts and source-manifest SHA256s are in `reports/data/external_leakage_report.json` and `content/datasets/xlcost/structural_external_clean/manifest.json`.",
        "- External benchmark task eligibility is frozen separately after canonical reference execution. This audit does not remove external tasks based on overlap and does not inspect model predictions.",
        "",
        "### Filtered training rows by direction",
        "",
        "| Direction | Kept rows | Removed rows |",
        "|---|---:|---:|",
        *[
            f"| {direction.replace('-to-', ' → ')} | {values['kept']} | {values['removed']} |"
            for direction, values in sorted(directional_rows.items())
        ],
        "",
        "## Limitations",
        "",
        "Fingerprint coverage is limited to programs accepted by the pinned Tree-sitter parsers and deterministic benchmark reconstruction. Rejected source records are counted above and remain in the benchmark; they are not evidence of no leakage. Exact and lexical rules are robust to parser-independent source overlap only for the representations actually reconstructed. Structural similarity is a conservative node-shape fingerprint and may match unrelated code or miss structurally equivalent programs. Pretraining contamination cannot be ruled out.",
        "",
        "## Provenance",
        "",
        f"- External input artifact manifest hashes: `{json.dumps(external['input_manifests'], sort_keys=True)}`.",
        f"- Training output manifest status: `{json.loads(Path('content/datasets/xlcost/structural_external_clean/manifest.json').read_text())['status']}`.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, default=Path("reports/data/xlcost_leakage.json"))
    parser.add_argument(
        "--external", type=Path, default=Path("reports/data/external_leakage_report.json")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/data/leakage_report.md"))
    args = parser.parse_args()
    render(args.internal, args.external, args.output)


if __name__ == "__main__":
    main()
