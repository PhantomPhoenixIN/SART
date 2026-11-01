#!/usr/bin/env python3
"""Report unique-source CFG/DFG extraction and mapping coverage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def audit(structural_root: Path, output_json: Path, output_markdown: Path) -> dict[str, Any]:
    """Aggregate Joern availability without counting paired directions as new programs."""
    all_records: dict[tuple[str, str], dict[str, Any]] = {}
    row_counts: dict[str, int] = defaultdict(int)
    direction_counts: dict[str, int] = defaultdict(int)
    for path in sorted(structural_root.glob("*/*/*.jsonl")):
        split = path.relative_to(structural_root).parts[0]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                language = row["source_language"]
                key = (language, row["source_sha256"])
                structure = row["structure"]
                row_counts[f"{split}/{language}"] += 1
                direction = f"{split}/{row['source_language']}-to-{row['target_language']}"
                direction_counts[direction] += 1
                item = all_records.setdefault(
                    key,
                    {
                        "language": language,
                        "source_sha256": row["source_sha256"],
                        "joern": structure.get("joern", {}),
                        "availability": structure.get("availability", {}),
                        "cfg_edges": sum(
                            edge["edge_type"].startswith("CFG_")
                            for edge in structure.get("edges", [])
                        ),
                        "dfg_edges": sum(
                            edge["edge_type"].startswith("DFG_")
                            for edge in structure.get("edges", [])
                        ),
                        "split": split,
                    },
                )
                if item["availability"] != structure.get("availability", {}):
                    raise ValueError(f"Structure availability differs across paired copies: {key}")
    by_language = {}
    for language in ("Python", "Java", "C++"):
        rows = [row for row in all_records.values() if row["language"] == language]
        total_cpg = sum(row["joern"].get("cpg_nodes", 0) for row in rows)
        total_mapped = sum(row["joern"].get("mapped_cpg_nodes", 0) for row in rows)
        by_language[language] = {
            "unique_source_programs": len(rows),
            "joern_available": sum(bool(row["joern"].get("available")) for row in rows),
            "cfg_available": sum(bool(row["availability"].get("cfg")) for row in rows),
            "dfg_available": sum(bool(row["availability"].get("dfg")) for row in rows),
            "mapped_cpg_nodes": total_mapped,
            "cpg_nodes": total_cpg,
            "node_mapping_coverage": total_mapped / total_cpg if total_cpg else 0.0,
            "cfg_region_edges": sum(row["cfg_edges"] for row in rows),
            "dfg_region_edges": sum(row["dfg_edges"] for row in rows),
        }
    report = {
        "status": "COMPLETE",
        "unique_source_count": len(all_records),
        "directional_row_counts": dict(sorted(row_counts.items())),
        "directional_task_counts": dict(sorted(direction_counts.items())),
        "languages": by_language,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Joern structural extraction coverage",
        "",
        "Counts are by unique source implementation (language + SHA256), not repeated translation "
        "directions.",
        "Missing CFG/DFG are masked; no relations are imputed. CPG-node mapping coverage is pooled "
        "over nodes with offsets.",
        "",
        f"Unique source implementations: {len(all_records):,}",
        "",
        "| Language | Unique sources | Joern available | CFG available | DFG available | "
        "CPG node mapping | CFG region edges | DFG region edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for language, stats in by_language.items():
        lines.append(
            f"| {language} | {stats['unique_source_programs']:,} | {stats['joern_available']:,} "
            f"| {stats['cfg_available']:,} | {stats['dfg_available']:,} "
            f"| {stats['node_mapping_coverage']:.2%} "
            f"({stats['mapped_cpg_nodes']:,}/{stats['cpg_nodes']:,}) "
            f"| {stats['cfg_region_edges']:,} | {stats['dfg_region_edges']:,} |"
        )
    lines.extend(["", "## Directional row counts", ""])
    for label, count in sorted(direction_counts.items()):
        lines.append(f"- `{label}`: {count:,}")
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-root", type=Path, default=Path("content/datasets/xlcost/structural_full")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/data/joern_coverage.json")
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=Path("reports/data/joern_coverage.md")
    )
    args = parser.parse_args()
    result = audit(**vars(args))
    print(
        json.dumps(
            {"unique_sources": result["unique_source_count"], "languages": result["languages"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
