"""Map Joern CPG control/data-flow edges onto Tree-sitter semantic regions."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _csv_rows(directory: Path, stem: str) -> list[dict[str, str]]:
    header_path = directory / f"{stem}_header.csv"
    data_path = directory / f"{stem}_data.csv"
    if not header_path.is_file() or not data_path.is_file():
        return []
    with header_path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    with data_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, fieldnames=header))
    return [
        {
            (key.rsplit(":", 1)[0] if ":" in key and not key.startswith(":") else key): value
            for key, value in row.items()
        }
        for row in rows
    ]


def _integer(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _joern_offset_to_byte(offset: int, source: str | None, adjustment: int = 0) -> int:
    """Convert Joern's character offsets to Tree-sitter UTF-8 byte offsets."""
    offset += adjustment
    if source is None or offset < 0:
        return offset
    if offset == len(source) + 1:
        return len(source.encode("utf-8"))
    if offset > len(source):
        return offset
    return len(source[:offset].encode("utf-8"))


def map_joern_edges_by_file(
    export_directory: Path,
    file_regions: dict[str, list[dict[str, Any]]],
    file_sources: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map one multi-file Joern export onto each source file's semantic regions."""
    file_names = {
        row[":ID"]: Path(row.get("NAME", "")).name
        for row in _csv_rows(export_directory, "nodes_FILE")
        if row.get(":ID") and row.get("NAME")
    }
    node_files = dict(file_names)
    for row in _csv_rows(export_directory, "nodes_METHOD"):
        if row.get(":ID") and row.get("FILENAME"):
            node_files[row[":ID"]] = Path(row["FILENAME"]).name
    containment = _csv_rows(export_directory, "edges_CONTAINS")
    children: dict[str, list[str]] = defaultdict(list)
    for row in containment:
        children[row.get(":START_ID", "")].append(row.get(":END_ID", ""))
    queue = deque(node_files)
    while queue:
        parent = queue.popleft()
        for child in children.get(parent, ()):
            if child not in node_files:
                node_files[child] = node_files[parent]
                queue.append(child)
    wanted_names = {Path(name).name: regions for name, regions in file_regions.items()}
    seen_names = set(file_names.values()) & set(wanted_names)
    node_spans: dict[str, tuple[str, int]] = {}
    total_nodes = {name: 0 for name in wanted_names}
    mapped_nodes = {name: 0 for name in wanted_names}
    for path in sorted(export_directory.glob("nodes_*_data.csv")):
        label = path.name.removeprefix("nodes_").removesuffix("_data.csv")
        for row in _csv_rows(export_directory, f"nodes_{label}"):
            node_id = row.get(":ID")
            filename_value = row.get("FILENAME", "")
            filename = (
                Path(filename_value).name
                if filename_value not in {"", "<empty>", "<unknown>", "N/A"}
                else node_files.get(node_id, "")
            )
            regions = wanted_names.get(filename)
            if regions is None:
                continue
            start = _integer(row.get("OFFSET"))
            end = _integer(row.get("OFFSET_END"))
            if node_id is None or start is None or end is None or end < start:
                continue
            total_nodes[filename] += 1
            source_text = (file_sources or {}).get(filename)
            spans = [(start, end)]
            if source_text is not None and not source_text.isascii():
                spans.extend(
                    (
                        _joern_offset_to_byte(start, source_text, start_adjustment),
                        _joern_offset_to_byte(end, source_text, end_adjustment),
                    )
                    for start_adjustment in (-1, 0)
                    for end_adjustment in (-2, -1, 0)
                )
            candidates = [
                (
                    int(region["end_byte"]) - int(region["start_byte"]),
                    span_index,
                    int(region["region_id"]),
                )
                for span_index, (candidate_start, candidate_end) in enumerate(spans)
                if candidate_end > candidate_start
                for region in regions
                if int(region["start_byte"]) <= candidate_start
                and candidate_end <= int(region["end_byte"])
                and region.get("region_type") != "PROGRAM"
            ]
            if candidates:
                node_spans[node_id] = filename, min(candidates)[2]
                mapped_nodes[filename] += 1

    relation_files = {
        "CFG_NEXT": export_directory / "edges_CFG_header.csv",
        "DFG_DEF_USE": export_directory / "edges_REACHING_DEF_header.csv",
    }
    region_edges: dict[str, dict[str, set[tuple[int, int]]]] = {
        relation: {name: set() for name in wanted_names} for relation in relation_files
    }
    for joern_type, sart_type in (("CFG", "CFG_NEXT"), ("REACHING_DEF", "DFG_DEF_USE")):
        for row in _csv_rows(export_directory, f"edges_{joern_type}"):
            source = node_spans.get(row.get(":START_ID", ""))
            target = node_spans.get(row.get(":END_ID", ""))
            if (
                source is not None
                and target is not None
                and source[0] == target[0]
                and source[1] != target[1]
            ):
                region_edges[sart_type][source[0]].add((source[1], target[1]))

    results = {}
    for name in wanted_names:
        if name not in seen_names:
            results[name] = {
                "available": False,
                "reason": "joern_file_node_missing",
                "cfg_edges": [],
                "dfg_edges": [],
                "cpg_nodes": 0,
                "mapped_cpg_nodes": 0,
            }
            continue
        if total_nodes[name] == 0:
            results[name] = {
                "available": False,
                "reason": "no_spanned_cpg_nodes",
                "cfg_edges": [],
                "dfg_edges": [],
                "cpg_nodes": 0,
                "mapped_cpg_nodes": 0,
            }
            continue
        results[name] = {
            "available": True,
            "reason": None,
            "cfg_available": relation_files["CFG_NEXT"].is_file(),
            "dfg_available": relation_files["DFG_DEF_USE"].is_file(),
            "cfg_edges": [
                {"source": source, "target": target, "edge_type": "CFG_NEXT"}
                for source, target in sorted(region_edges["CFG_NEXT"][name])
            ],
            "dfg_edges": [
                {"source": source, "target": target, "edge_type": "DFG_DEF_USE"}
                for source, target in sorted(region_edges["DFG_DEF_USE"][name])
            ],
            "cpg_nodes": total_nodes[name],
            "mapped_cpg_nodes": mapped_nodes[name],
            "node_mapping_coverage": mapped_nodes[name] / total_nodes[name]
            if total_nodes[name]
            else 0.0,
        }
    return results


def map_joern_edges(
    export_directory: Path,
    source_filename: str,
    regions: list[dict[str, Any]],
    source: str | None = None,
) -> dict[str, Any]:
    """Project one file's Joern CFG/REACHING_DEF links onto semantic regions."""
    name = Path(source_filename).name
    sources = {name: source} if source is not None else None
    return map_joern_edges_by_file(export_directory, {name: regions}, sources)[name]
