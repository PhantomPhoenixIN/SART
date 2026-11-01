#!/usr/bin/env python3
"""Freeze source-region to target-reference structural mismatch for H4, pre-scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.freeze_structural_difficulty import sha256_bytes
from src.regions.extract import extract_region_graph


def _fallback_region(source: str) -> list[dict[str, Any]]:
    return [{
        "region_id": 0, "region_type": "PROGRAM", "raw_node_type": "PARSER_FAILURE_FALLBACK",
        "start_byte": 0, "end_byte": len(source.encode("utf-8")), "parent_id": None,
        "ast_depth": 0, "child_count": 0,
    }]


def _regions(source: str, language: str) -> tuple[list[dict[str, Any]], bool]:
    try:
        graph = extract_region_graph(source, language)
        return [
            {
                "region_id": r.region_id, "region_type": r.region_type,
                "raw_node_type": r.raw_node_type, "start_byte": r.start_byte,
                "end_byte": r.end_byte, "parent_id": r.parent_id,
                "ast_depth": r.ast_depth, "child_count": r.child_count,
            }
            for r in graph.regions
        ], True
    except (ValueError, UnicodeError):
        return _fallback_region(source), False


def _region_cost(source: dict[str, Any], target: dict[str, Any]) -> float:
    type_cost = float(source["region_type"] != target["region_type"])
    depth_cost = min(1.0, abs(math.log1p(source["ast_depth"]) - math.log1p(target["ast_depth"])) / math.log(33))
    child_cost = min(1.0, abs(math.log1p(source["child_count"]) - math.log1p(target["child_count"])) / math.log(17))
    source_bytes = max(0, source["end_byte"] - source["start_byte"])
    target_bytes = max(0, target["end_byte"] - target["start_byte"])
    length_cost = min(1.0, abs(math.log1p(source_bytes) - math.log1p(target_bytes)) / math.log(2049))
    return (type_cost + depth_cost + child_cost + length_cost) / 4.0


def _match_regions(source: list[dict[str, Any]], target: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimum-cost one-to-one match with unit-cost dummy nodes for unmatched regions."""
    n, m = len(source), len(target)
    size = n + m
    costs = [[0.0] * size for _ in range(size)]
    for i, source_region in enumerate(source):
        for j, target_region in enumerate(target):
            costs[i][j] = _region_cost(source_region, target_region)
        for j in range(m, size):
            costs[i][j] = 1.0
    for i in range(n, size):
        for j in range(m):
            costs[i][j] = 1.0
    rows, cols = linear_sum_assignment(costs)
    mapping = {int(i): int(j) for i, j in zip(rows, cols, strict=True) if i < n and j < m}
    result = []
    for i, region in enumerate(source):
        j = mapping.get(i)
        matched = target[j] if j is not None else None
        result.append({
            "source_region_id": region["region_id"],
            "source_features": region,
            "target_region_id": matched["region_id"] if matched else None,
            "target_features": matched,
            "unmatched_target_region": matched is None,
            "structural_difficulty": _region_cost(region, matched) if matched else 1.0,
        })
    return result


def freeze(manifest_path: Path, selection_path: Path, output_path: Path) -> dict[str, Any]:
    manifest_bytes, selection_bytes = manifest_path.read_bytes(), selection_path.read_bytes()
    manifest, selection = json.loads(manifest_bytes), json.loads(selection_bytes)
    if manifest.get("status") != "FROZEN_PREMODEL":
        raise ValueError("H4 region difficulty requires the frozen pre-model task manifest")
    if manifest.get("selection_manifest_sha256") != sha256_bytes(selection_bytes):
        raise ValueError("Reference selection does not match frozen task manifest")
    records = []
    for direction, direction_data in sorted(manifest["directions"].items()):
        for task in direction_data["tasks"]:
            problem = str(task["src_uid"])
            source_language, target_language = task["source_language"], task["target_language"]
            src = next((row for row in selection["candidate_pool"][source_language][problem]
                        if row["code_uid"] == task["source_code_uid"]), None)
            tgt = next((row for row in selection["candidate_pool"][target_language][problem]
                        if row["code_uid"] == task["target_reference_code_uid"]), None)
            if src is None or tgt is None:
                raise ValueError(f"Frozen source/reference missing for {direction}/{problem}")
            source_code, target_code = src["source_code"], tgt["source_code"]
            source_hash, target_hash = sha256_bytes(source_code.encode()), sha256_bytes(target_code.encode())
            if source_hash != task["source_sha256"] or target_hash != task["target_reference_sha256"]:
                raise ValueError(f"Frozen code digest mismatch for {direction}/{problem}")
            source_regions, source_parse = _regions(source_code, source_language)
            target_regions, target_parse = _regions(target_code, target_language)
            region_pairs = _match_regions(source_regions, target_regions) if target_parse else []
            records.append({
                "task_id": f"{direction}/{problem}", "problem_id": problem,
                "direction": direction, "source_language": source_language,
                "target_language": target_language, "source_sha256": source_hash,
                "target_reference_sha256": target_hash,
                "source_region_parse_valid": source_parse,
                "target_region_parse_valid": target_parse,
                "source_region_count": len(source_regions),
                "target_region_count": len(target_regions),
                "regions": region_pairs,
            })
    paired_regions = sum(len(row["regions"]) for row in records)
    result = {
        "schema_version": 1, "status": "FROZEN_PREMODEL",
        "benchmark": "xcodeeval", "split": "common_core",
        "difficulty_definition": (
            "One-to-one minimum-cost source/reference region assignment. Pair cost is the equal-weight "
            "mean of normalized semantic region-type mismatch, log AST-depth difference, log child-count "
            "difference, and log byte-span-length difference. Unmatched source regions receive cost 1."
        ),
        "matching_limitations": (
            "Region correspondence is a structural proxy, not a semantic translation alignment. "
            "It is frozen before model outputs and is used only for retrospective association."
        ),
        "task_manifest_sha256": sha256_bytes(manifest_bytes),
        "reference_selection_manifest_sha256": sha256_bytes(selection_bytes),
        "task_count": len(records), "paired_region_count": paired_regions,
        "source_parse_valid_count": sum(r["source_region_parse_valid"] for r in records),
        "target_parse_valid_count": sum(r["target_region_parse_valid"] for r in records),
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(
        "# Frozen xCodeEval Common-Core per-region structural difficulty\n\n"
        "Status: `FROZEN_PREMODEL`; no prediction or execution result is read.\n\n"
        f"Tasks: {len(records)}; source parse valid: {result['source_parse_valid_count']}; "
        f"target parse valid: {result['target_parse_valid_count']}; matched/source regions: {paired_regions}.\n\n"
        f"Definition: {result['difficulty_definition']}\n\n"
        f"Limitation: {result['matching_limitations']}\n\n"
        f"Task-manifest SHA256: `{result['task_manifest_sha256']}`\n\n"
        f"Reference-selection SHA256: `{result['reference_selection_manifest_sha256']}`\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/xcodeeval_common_core_manifest.json"))
    parser.add_argument("--selection", type=Path, default=Path("data/manifests/xcodeeval_reference_selection.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/analysis/xcodeeval_common_core_region_difficulty.json"))
    args = parser.parse_args()
    result = freeze(args.manifest, args.selection, args.output)
    print(json.dumps({k: result[k] for k in ("status", "task_count", "paired_region_count", "source_parse_valid_count", "target_parse_valid_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
