#!/usr/bin/env python3
"""Audit AST region counts, useful token alignment, and deterministic examples."""

from __future__ import annotations

import argparse
import html
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def audit(structure_root: Path, report_path: Path, sample_count: int = 20) -> dict[str, Any]:
    """Summarize every structural shard and write deterministic inspection files."""
    region_counts: dict[str, Counter[str]] = defaultdict(Counter)
    availability: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: Counter[str] = Counter()
    semantic_token_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(structure_root.glob("*/*/*.jsonl")):
        split = path.parts[-3]
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            language = row["source_language"]
            structure = row["structure"]
            record_counts[split] += 1
            for region in structure["regions"]:
                region_counts[language][region["region_type"]] += 1
            for relation, enabled in structure["availability"].items():
                availability[language][f"{relation}_{'available' if enabled else 'missing'}"] += 1
            useful_ids = {
                region["region_id"]
                for region in structure["regions"]
                if region["region_type"] != "PROGRAM"
            }
            for token_index, region_id in structure["source_token_to_region"]:
                token_counts[language] += 1
                if region_id in useful_ids:
                    semantic_token_counts[language] += 1
            if split == "train" and len(samples[language]) < sample_count:
                # Deterministic order by sorted direction/shard/JSONL source order.
                samples[language].append(
                    {
                        "problem_id": row["problem_id"],
                        "source_language": language,
                        "target_language": row["target_language"],
                        "source": row["source"],
                        "regions": structure["regions"],
                        "edges": structure["edges"],
                        "availability": structure["availability"],
                    }
                )

    report = {
        "records_by_split": dict(record_counts),
        "region_counts_by_language": {
            language: dict(sorted(counts.items()))
            for language, counts in sorted(region_counts.items())
        },
        "availability_by_language": {
            language: dict(sorted(counts.items()))
            for language, counts in sorted(availability.items())
        },
        "token_mapping_by_language": {
            language: {
                "source_tokens": token_counts[language],
                "mapped_to_non_program_semantic_region": semantic_token_counts[language],
                "useful_mapping_coverage": (
                    semantic_token_counts[language] / token_counts[language]
                    if token_counts[language]
                    else 0.0
                ),
            }
            for language in sorted(token_counts)
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    html_root = report_path.parent.parent / "regions"
    for language, examples in samples.items():
        slug = {"Python": "python", "Java": "java", "C++": "cpp"}[language]
        output_dir = html_root / f"{slug}_examples"
        for index, example in enumerate(examples):
            _atomic_text(
                output_dir / f"example_{index:03d}.json",
                json.dumps(example, indent=2, ensure_ascii=False) + "\n",
            )
        cards = []
        for index, example in enumerate(examples):
            cards.append(
                "<section><h2>"
                + html.escape(
                    f"{language} → {example['target_language']} · {example['problem_id']}"
                )
                + "</h2><pre>"
                + html.escape(example["source"])
                + "</pre><pre>"
                + html.escape(json.dumps(example["regions"], indent=2))
                + "</pre><pre>"
                + html.escape(json.dumps(example["edges"], indent=2))
                + "</pre></section>"
            )
        page = (
            "<!doctype html><meta charset='utf-8'><title>SART regions "
            + slug
            + "</title><style>body{font-family:sans-serif;max-width:1200px;margin:auto}"
            "pre{white-space:pre-wrap;background:#f4f4f4;padding:1em}section{border-bottom:1px solid #aaa}"
            "</style><h1>SART region inspection: "
            + html.escape(language)
            + "</h1>"
            + "\n".join(cards)
        )
        _atomic_text(output_dir / "index.html", page)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-root", type=Path, default=Path("content/datasets/xlcost/structural_ast_qwen")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/ast_structure_audit.json")
    )
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(audit(args.structure_root, args.report, args.samples), indent=2))


if __name__ == "__main__":
    main()
