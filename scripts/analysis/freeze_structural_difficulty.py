#!/usr/bin/env python3
"""Freeze reference-derived xCodeEval structural-difficulty features before scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import braycurtis, jensenshannon
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parsing.tree_sitter_adapter import parse_tree
from src.regions.extract import extract_region_graph

LANGUAGE_SUFFIX = {"Python": "py", "Java": "java", "C++": "cpp"}
IGNORED_AST_NODES = {
    "comment", "identifier", "field_identifier", "type_identifier", "namespace_identifier",
    "string", "string_literal", "character_literal", "integer", "integer_literal",
    "float", "float_literal", "true", "false", "null", "nullptr",
}
CALL_NODES = {"call", "call_expression", "method_invocation"}
OPERATOR_GROUPS = {
    "arithmetic": {"+", "-", "*", "/", "%", "**", "//", "<<", ">>"},
    "comparison": {"==", "!=", "<", "<=", ">", ">="},
    "boolean": {"and", "or", "not", "&&", "||", "!"},
    "assignment": {"=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>="},
}
FEATURES = ["ast_node_distance", "region_type_divergence", "cfg_feature_distance",
            "dfg_feature_distance", "api_divergence", "operator_divergence"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def _parseable_source(code: str, language: str, runtime: str | None) -> str:
    """Use pinned Tree-sitter directly, with deterministic Python-2-to-3 parsing fallback."""
    try:
        tree = parse_tree(code, language)
        if not tree.root_node.has_error:
            return code
    except Exception:
        if language != "Python":
            raise
    if language == "C++":
        # Keep difficult-but-compiler-accepted C++ for lexical and cached Joern features;
        # the AST/region dimensions will be explicitly masked when Tree-sitter recovers.
        return code
    if language != "Python" or (runtime and "2" not in runtime):
        raise ValueError(f"Reference is not parseable as {language} ({runtime})")
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package

    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    converted = str(tool.refactor_string(code if code.endswith("\n") else code + "\n", "reference.py"))
    tree = parse_tree(converted, language)
    if tree.root_node.has_error:
        raise ValueError("Python-2 conversion did not produce a parseable reference")
    return converted


def _distribution_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([left[key] for key in keys], dtype=np.float64)
    b = np.asarray([right[key] for key in keys], dtype=np.float64)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    return float(braycurtis(a, b))


def _normalized(counter: Counter[str]) -> Counter[str]:
    total = float(sum(counter.values()))
    if total <= 0:
        return Counter()
    return Counter({key: value / total for key, value in counter.items()})


def _js_distance(left: Counter[str], right: Counter[str]) -> float:
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    a = np.asarray([left[key] for key in keys], dtype=np.float64)
    b = np.asarray([right[key] for key in keys], dtype=np.float64)
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return 1.0
    a = a / a.sum()
    b = b / b.sum()
    a /= a.sum()
    b /= b.sum()
    return float(jensenshannon(a, b, base=2.0))


def _python2_runtime(runtime: str | None) -> bool:
    return bool(runtime and "2" in runtime)


def _program_features(code: str, language: str, runtime: str | None) -> dict[str, Any]:
    parsed = _parseable_source(code, language, runtime)
    tree = parse_tree(parsed, language)
    syntax_valid = not tree.root_node.has_error
    source_bytes = parsed.encode("utf-8")
    ast_types: Counter[str] = Counter()
    calls: set[str] = set()
    operators: Counter[str] = Counter()
    stack = [tree.root_node]
    operator_texts = {operator for values in OPERATOR_GROUPS.values() for operator in values}
    while stack:
        node = stack.pop()
        if node.is_named and node.type not in IGNORED_AST_NODES:
            ast_types[node.type] += 1
        if node.type in CALL_NODES:
            callee = node.child_by_field_name("function") or node.child_by_field_name("name")
            if callee is not None:
                spelling = source_bytes[callee.start_byte:callee.end_byte].decode("utf-8", "replace")
                names = re.findall(r"[A-Za-z_$][\w$]*", spelling)
                if names:
                    calls.add(".".join(names))
        for child in node.children:
            if child.child_count == 0:
                spelling = source_bytes[child.start_byte:child.end_byte].decode("utf-8", "replace")
                if spelling in operator_texts:
                    operators[spelling] += 1
        stack.extend(node.children)
    regions = None
    if syntax_valid:
        graph = extract_region_graph(parsed, language)
        regions = Counter(region.region_type for region in graph.regions)
    if not syntax_valid and language == "C++":
        # Syntax recovery can still expose call/operator leaves, but do not treat its
        # partial tree as a complete AST; lexical fallback preserves those dimensions.
        call_pattern = re.compile(r"([A-Za-z_$][\w$]*(?:(?:::|\.|->)[A-Za-z_$][\w$]*)*)\s*\(")
        control = {"if", "for", "while", "switch", "catch", "sizeof", "decltype", "return"}
        calls.update(name for name in call_pattern.findall(parsed) if name not in control)
        for operator in operator_texts:
            operators[operator] = max(operators[operator], len(re.findall(re.escape(operator), parsed)))
    return {"ast": _normalized(ast_types) if syntax_valid else None,
            "regions": regions, "calls": calls,
            "operators": _normalized(operators), "parsed_source": parsed,
            "python2_normalized": _python2_runtime(runtime) and parsed != code,
            "syntax_valid": syntax_valid}


def _edge_feature(row: dict[str, Any] | None, family: str) -> Counter[str] | None:
    if row is None or not row.get(f"{family}_available", False):
        return None
    edges = row.get(f"{family}_edges", [])
    node_count = max(1, int(row.get("cpg_nodes", 0)))
    counts = Counter(str(edge["edge_type"]) for edge in edges)
    # Normalize edge counts by node count so cross-language graph size does not dominate.
    return Counter({key: value / node_count for key, value in counts.items()})


def _jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def _load_edges(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                records[str(row["source_sha256"])] = row
    return records


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Frozen xCodeEval Common-Core structural difficulty",
        "",
        "Status: `FROZEN_PREMODEL`. Features are computed only from frozen selected source programs,",
        "accepted canonical target references, and the frozen Joern cache. No model predictions,",
        "outcomes, or scores are read. Quartiles must not be recomputed after outcomes are observed.",
        "",
        f"Tasks: {result['task_count']} across six directions and {len({row['problem_id'] for row in result['records']})} common problems.",
        "",
        "## Feature definition",
        "",
        "The composite is the equal-weight mean of within-direction percentile ranks over available",
        "dimensions (at least four of six per task): named AST node-type distribution distance,",
        "semantic-region type Jensen-Shannon distance, CFG edge-type density distance, DFG edge-type",
        "density distance, call-name Jaccard divergence, and operator-histogram distance. AST node",
        "distance is a normalized feature-profile distance, not ordered tree-edit distance. If either",
        "program fails the pinned Tree-sitter parse, AST and region dimensions are masked; available",
        "CFG/DFG/API/operator features remain, and parse status is retained per side. Python 2 sources",
        "are normalized with lib2to3 only when required for parsing, with that operation recorded.",
        "",
        "## Feature coverage",
        "",
        "| Feature | Tasks with value | Total |",
        "|---|---:|---:|",
    ]
    for name, count in result["feature_coverage"].items():
        lines.append(f"| {name} | {count} | {result['task_count']} |")
    lines.extend([
        "",
        "Every task received a frozen quartile label. CFG and DFG feature coverage is required on",
        "both source and reference; where those are available, parser failures leave four other",
        "dimensions available. Coverage flags and exact task-level features are in the JSON artifact.",
        "",
        "## Direction-specific frozen thresholds",
        "",
        "| Direction | Q1/Q2 edge | Q2/Q3 edge | Q3/Q4 edge | Q1 | Q2 | Q3 | Q4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for direction in sorted(result["quartile_edges_by_direction"]):
        q1, q2, q3 = result["quartile_edges_by_direction"][direction]
        counts = result["quartile_counts_by_direction"][direction]
        lines.append(
            f"| {direction} | {q1:.6f} | {q2:.6f} | {q3:.6f} | "
            f"{counts['Q1_easiest']} | {counts['Q2']} | {counts['Q3']} | {counts['Q4_hardest']} |"
        )
    lines.extend([
        "",
        "Quartiles are ranked independently within each direction so language-pair baselines do not",
        "set a shared threshold. The primary H3 comparison should join outcomes by problem ID and",
        "direction using these frozen labels; keep any missing outcome cells explicit.",
        "",
        "## Provenance hashes",
        "",
        f"- xCodeEval Common-Core task manifest: `{result['task_manifest_sha256']}`",
        f"- xCodeEval reference-selection manifest: `{result['reference_selection_manifest_sha256']}`",
        f"- External Joern edge manifest: `{result['joern_manifest_sha256']}`",
        "",
        "Machine-readable task-level features and source/reference hashes: `xcodeeval_common_core_structural_difficulty.json`.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def freeze(
    manifest_path: Path,
    selection_path: Path,
    edge_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest_bytes = manifest_path.read_bytes()
    selection_bytes = selection_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    selection = json.loads(selection_bytes)
    selection_sha256 = sha256_bytes(selection_bytes)
    if manifest.get("status") != "FROZEN_PREMODEL":
        raise ValueError("Difficulty features require the frozen pre-model xCodeEval task manifest")
    if manifest.get("selection_manifest_sha256") != selection_sha256:
        raise ValueError("Reference-selection file does not match the frozen task manifest hash")
    edge_manifest_path = edge_root / "manifest.json"
    edge_manifest = json.loads(edge_manifest_path.read_text(encoding="utf-8"))
    edge_index = _load_edges(edge_root)
    feature_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    edge_cache: dict[str, dict[str, Any] | None] = {}
    records: list[dict[str, Any]] = []

    def cached_features(code: str, language: str, runtime: str | None) -> tuple[str, dict[str, Any]]:
        digest = sha256_bytes(code.encode("utf-8"))
        key = (language, digest, runtime or "")
        if key not in feature_cache:
            feature_cache[key] = _program_features(code, language, runtime)
        return digest, feature_cache[key]

    for direction, direction_data in sorted(manifest["directions"].items()):
        for task in direction_data["tasks"]:
            problem = str(task["src_uid"])
            src_language = str(task["source_language"])
            tgt_language = str(task["target_language"])
            src_candidates = selection["candidate_pool"][src_language][problem]
            src_row = next((row for row in src_candidates
                            if row["code_uid"] == task["source_code_uid"]), None)
            tgt_candidates = selection["candidate_pool"][tgt_language][problem]
            if src_row is None:
                raise ValueError(f"Frozen source selection missing: {direction}/{problem}")
            tgt_row = next((row for row in tgt_candidates
                            if row["code_uid"] == task["target_reference_code_uid"]), None)
            if tgt_row is None:
                raise ValueError(f"Frozen target reference missing: {direction}/{problem}")
            src_hash, src = cached_features(src_row["source_code"], src_language,
                                            str(src_row.get("runtime_label", "")))
            tgt_hash, tgt = cached_features(tgt_row["source_code"], tgt_language,
                                            str(tgt_row.get("runtime_label", "")))
            if src_hash != task["source_sha256"] or tgt_hash != task["target_reference_sha256"]:
                raise ValueError(f"Frozen source/reference digest mismatch: {direction}/{problem}")
            left_edge = edge_cache.setdefault(src_hash, edge_index.get(src_hash))
            right_edge = edge_cache.setdefault(tgt_hash, edge_index.get(tgt_hash))
            cfg_left, cfg_right = _edge_feature(left_edge, "cfg"), _edge_feature(right_edge, "cfg")
            dfg_left, dfg_right = _edge_feature(left_edge, "dfg"), _edge_feature(right_edge, "dfg")
            cfg_distance = _distribution_distance(cfg_left, cfg_right) if cfg_left is not None and cfg_right is not None else None
            dfg_distance = _distribution_distance(dfg_left, dfg_right) if dfg_left is not None and dfg_right is not None else None
            components = {
                "ast_node_distance": (
                    _distribution_distance(src["ast"], tgt["ast"])
                    if src["ast"] is not None and tgt["ast"] is not None else None
                ),
                "region_type_divergence": (
                    _js_distance(src["regions"], tgt["regions"])
                    if src["regions"] is not None and tgt["regions"] is not None else None
                ),
                "cfg_feature_distance": cfg_distance,
                "dfg_feature_distance": dfg_distance,
                "api_divergence": _jaccard_distance(src["calls"], tgt["calls"]),
                "operator_divergence": _distribution_distance(src["operators"], tgt["operators"]),
            }
            records.append({
                "task_id": f"{direction}/{problem}", "problem_id": problem,
                "direction": direction, "source_language": src_language,
                "target_language": tgt_language, "source_sha256": src_hash,
                "target_reference_sha256": tgt_hash, "features": components,
                "availability": {
                    "source_cfg": bool(left_edge and left_edge.get("cfg_available")),
                    "target_cfg": bool(right_edge and right_edge.get("cfg_available")),
                    "source_dfg": bool(left_edge and left_edge.get("dfg_available")),
                    "target_dfg": bool(right_edge and right_edge.get("dfg_available")),
                    "source_python2_normalized": bool(src["python2_normalized"]),
                    "target_python2_normalized": bool(tgt["python2_normalized"]),
                    "source_ast_parse_valid": bool(src["syntax_valid"]),
                    "target_ast_parse_valid": bool(tgt["syntax_valid"]),
                },
            })

    # Convert every feature to within-direction percentile ranks, then average available ranks.
    per_direction: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        per_direction.setdefault(row["direction"], []).append(row)
    quartile_edges: dict[str, list[float]] = {}
    quartile_counts: dict[str, dict[str, int]] = {}
    for direction, rows in per_direction.items():
        rank_columns: dict[str, dict[str, float]] = {}
        for feature in FEATURES:
            observed = [(index, row["features"][feature]) for index, row in enumerate(rows)
                        if row["features"][feature] is not None]
            if not observed:
                continue
            values = np.asarray([value for _, value in observed], dtype=np.float64)
            ranks = rankdata(values, method="average")
            denominator = max(1, len(values) - 1)
            rank_columns[feature] = {
                rows[index]["task_id"]: float((rank - 1.0) / denominator)
                for (index, _), rank in zip(observed, ranks, strict=True)
            }
        scores = []
        for row in rows:
            ranks = [column[row["task_id"]] for column in rank_columns.values()
                     if row["task_id"] in column]
            if len(ranks) < 4:
                raise ValueError(f"Too few available structural dimensions for {row['task_id']}")
            score = float(np.mean(ranks))
            row["difficulty_component_percentiles"] = {
                feature: column[row["task_id"]]
                for feature, column in rank_columns.items() if row["task_id"] in column
            }
            row["structural_difficulty"] = score
            scores.append(score)
        edges = [float(value) for value in np.quantile(scores, [0.25, 0.5, 0.75])]
        quartile_edges[direction] = edges
        counts = {"Q1_easiest": 0, "Q2": 0, "Q3": 0, "Q4_hardest": 0}
        labels = list(counts)
        for row, score in zip(rows, scores, strict=True):
            quartile = int(np.searchsorted(edges, score, side="right"))
            row["difficulty_quartile"] = labels[quartile]
            counts[labels[quartile]] += 1
        quartile_counts[direction] = counts

    record = {
        "schema_version": 1, "status": "FROZEN_PREMODEL",
        "benchmark": "xcodeeval", "split": "common_core",
        "task_manifest_sha256": sha256_bytes(manifest_bytes),
        "reference_selection_manifest_sha256": sha256_bytes(selection_bytes),
        "joern_manifest_sha256": sha256_bytes(edge_manifest_path.read_bytes()),
        "difficulty_definition": {
            "scope": "reference-derived source-to-target structure; never model input",
            "ast_node_distance": "Bray-Curtis distance between normalized named Tree-sitter AST-node histograms; this is a structural profile distance, not ordered tree-edit distance",
            "region_type_divergence": "base-2 Jensen-Shannon distance between normalized language-normalized region-type histograms",
            "cfg_feature_distance": "Bray-Curtis distance between CFG edge-type counts normalized by CPG node count; null if either side lacks CFG",
            "dfg_feature_distance": "Bray-Curtis distance between DFG edge-type counts normalized by CPG node count; null if either side lacks DFG",
            "api_divergence": "Jaccard distance between extracted call-name sets",
            "operator_divergence": "Bray-Curtis distance between recognized operator histograms",
            "composite": "Equal-weight mean of available within-direction percentile ranks; require at least four of six dimensions",
            "quartiles": "Per-direction empirical 25th/50th/75th percentile edges frozen before model outcomes",
            "python2": "If native Tree-sitter parse fails and frozen runtime indicates Python 2, parse deterministic lib2to3-converted syntax; retain normalization flags",
        },
        "quartile_edges_by_direction": quartile_edges,
        "quartile_counts_by_direction": quartile_counts,
        "feature_coverage": {
            feature: sum(row["features"][feature] is not None for row in records)
            for feature in FEATURES
        },
        "task_count": len(records), "records": records,
    }
    _atomic_json(output_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/xcodeeval_common_core_manifest.json"))
    parser.add_argument("--reference-selection", type=Path, default=Path("data/manifests/xcodeeval_reference_selection.json"))
    parser.add_argument("--joern-root", type=Path, default=Path("content/datasets/external_joern_edges"))
    parser.add_argument("--output", type=Path, default=Path("reports/analysis/xcodeeval_common_core_structural_difficulty.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/analysis/xcodeeval_common_core_structural_difficulty.md"))
    args = parser.parse_args()
    result = freeze(args.manifest, args.reference_selection, args.joern_root, args.output)
    _write_summary(args.report, result)
    print(json.dumps({"status": result["status"], "task_count": result["task_count"],
                      "feature_coverage": result["feature_coverage"],
                      "quartile_counts_by_direction": result["quartile_counts_by_direction"]}, indent=2))


if __name__ == "__main__":
    main()
