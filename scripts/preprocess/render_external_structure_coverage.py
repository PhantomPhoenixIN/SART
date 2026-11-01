#!/usr/bin/env python3
"""Audit completed external structure caches and render coverage by language."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BENCHMARKS = ("xcodeeval", "humanevalx", "transcoder_test", "avatar_tc")
SPLITS = ("common_core", "full_directional")
FORBIDDEN_KEYS = {"tests", "test", "test_cases", "canonical_solution", "target_solution"}


def _read_cache(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Incomplete structure cache: {root}")
    rows = []
    for path in sorted(root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if FORBIDDEN_KEYS.intersection(row):
                    raise ValueError(f"Test/reference content leaked into {path}")
                if row.get("target") != "":
                    raise ValueError(f"Unexpected target text in inference row: {path}")
                rows.append(row)
    return rows, manifest


def render(cache_root: Path, report_path: Path) -> str:
    lines = [
        "# External source-structure cache coverage",
        "",
        "Generated from the checksum-verified task caches; no model scores are read.",
        (
            "Model-facing rows contain source text and allowed target interface only. "
            "Hidden tests and target implementations are absent."
        ),
        "",
        (
            "| Benchmark | Set | Tasks | Problems | Directions | Source tokens | Mapped | "
            "Coverage | AST available | CFG available | DFG available |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for split in SPLITS:
            root = cache_root / benchmark / split
            rows, manifest = _read_cache(root)
            problems = {row["problem_id"] for row in rows}
            directions = {row["direction"] for row in rows}
            source_tokens = sum(row["structure"]["source_token_count"] for row in rows)
            mapped_tokens = sum(row["structure"]["mapped_source_token_count"] for row in rows)
            ast_count = sum(row["structure"]["availability"]["ast"] for row in rows)
            cfg_count = sum(row["structure"]["availability"]["cfg"] for row in rows)
            dfg_count = sum(row["structure"]["availability"]["dfg"] for row in rows)
            coverage = mapped_tokens / source_tokens if source_tokens else 0.0
            lines.append(
                f"| {benchmark} | {split} | {len(rows)} | {len(problems)} | {len(directions)} | "
                f"{source_tokens} | {mapped_tokens} | {coverage:.4%} | {ast_count}/{len(rows)} | "
                f"{cfg_count}/{len(rows)} | {dfg_count}/{len(rows)} |"
            )
            by_language: dict[str, dict[str, int]] = defaultdict(
                lambda: {"tasks": 0, "ast": 0, "tokens": 0, "mapped": 0, "fallback": 0}
            )
            for row in rows:
                lang = row["source_language"]
                counts = by_language[lang]
                counts["tasks"] += 1
                counts["ast"] += bool(row["structure"]["availability"]["ast"])
                counts["tokens"] += row["structure"]["source_token_count"]
                counts["mapped"] += row["structure"]["mapped_source_token_count"]
                counts["fallback"] += not row["structure"]["availability"]["ast"]
            lines.extend(
                [
                    "",
                    f"### {benchmark}: {split}",
                    "",
                    "| Source language | Tasks | AST parsed | Fallbacks | Token mapping |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for lang, counts in sorted(by_language.items()):
                language_coverage = counts["mapped"] / counts["tokens"] if counts["tokens"] else 0.0
                lines.append(
                    f"| {lang} | {counts['tasks']} | {counts['ast']} | {counts['fallback']} | "
                    f"{language_coverage:.4%} ({counts['mapped']}/{counts['tokens']}) |"
                )
            lines.append("")
            if "configuration_hash" in manifest:
                provenance = (
                    f"Cache config SHA256: `{manifest['configuration_hash']}`; tokenizer SHA256: "
                    f"`{manifest.get('tokenizer_sha256', 'not recorded')}`."
                )
            else:
                provenance = (
                    f"AST manifest SHA256: `{manifest.get('ast_manifest_sha256', 'unknown')}`; "
                    f"Joern manifest SHA256: `{manifest.get('joern_manifest_sha256', 'unknown')}`."
                )
            lines.append(provenance)
            if manifest.get("failures"):
                lines.append(f"Recorded preprocessing failures: {len(manifest['failures'])}.")
            lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/structural/external"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/external_structure_coverage.md")
    )
    args = parser.parse_args()
    print(render(args.cache_root, args.report))


if __name__ == "__main__":
    main()
