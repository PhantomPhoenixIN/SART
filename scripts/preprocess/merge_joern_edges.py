#!/usr/bin/env python3
"""Create full AST+CFG+DFG structures without overwriting verified AST artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return sha256(path)


def read_edge_index(edges_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(edges_root.glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["source_language"], row["source_sha256"])
            if key in result:
                raise ValueError(f"Duplicate Joern edge record: {key}")
            result[key] = row
    return result


def merge_row(row: dict[str, Any], index: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Attach typed Joern links and update graph availability/degree features."""
    key = (row["source_language"], row["source_sha256"])
    joern = index.get(key, {})
    structure = dict(row["structure"])
    regions = [dict(region) for region in structure["regions"]]
    edges = list(structure.get("edges", []))
    cfg_edges = joern.get("cfg_edges", []) if joern.get("available") else []
    dfg_edges = joern.get("dfg_edges", []) if joern.get("available") else []
    edges.extend(cfg_edges)
    edges.extend(dfg_edges)
    for region in regions:
        region["cfg_degree"] = 0
        region["dfg_degree"] = 0
    for edge in cfg_edges:
        regions[edge["source"]]["cfg_degree"] += 1
        regions[edge["target"]]["cfg_degree"] += 1
    for edge in dfg_edges:
        regions[edge["source"]]["dfg_degree"] += 1
        regions[edge["target"]]["dfg_degree"] += 1
    structure["regions"] = regions
    structure["edges"] = edges
    structure["availability"] = {
        **structure.get("availability", {}),
        "cfg": bool(joern.get("available") and joern.get("cfg_available")),
        "dfg": bool(joern.get("available") and joern.get("dfg_available")),
    }
    structure["joern"] = {
        "available": bool(joern.get("available", False)),
        "reason": joern.get("reason", "joern_record_missing"),
        "cpg_nodes": int(joern.get("cpg_nodes", 0)),
        "mapped_cpg_nodes": int(joern.get("mapped_cpg_nodes", 0)),
        "node_mapping_coverage": float(joern.get("node_mapping_coverage", 0.0)),
    }
    return {**row, "structure": structure}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def merge(
    ast_root: Path,
    edges_root: Path,
    train_sample_manifest: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write merged all-split shards and a structurally enriched train sample."""
    ast_manifest = json.loads((ast_root / "manifest.json").read_text())
    edge_manifest = json.loads((edges_root / "manifest.json").read_text())
    if ast_manifest.get("status") != "COMPLETE" or edge_manifest.get("status") not in {
        "COMPLETE",
        "COMPLETE_WITH_FAILURES",
    }:
        raise ValueError("Both AST and Joern manifests must be complete before graph merge")
    index = read_edge_index(edges_root)
    output_root.mkdir(parents=True, exist_ok=True)
    completed = {}
    for source in sorted(ast_root.glob("*/*/*.jsonl")):
        relative = source.relative_to(ast_root)
        destination = output_root / relative
        input_hash = sha256(source)
        rows = [merge_row(row, index) for row in _read_jsonl(source)]
        output_hash = atomic_jsonl(destination, rows)
        completed[str(relative)] = {
            "path": str(destination),
            "records": len(rows),
            "input_sha256": input_hash,
            "bytes": destination.stat().st_size,
            "sha256": output_hash,
        }

    sample_manifest = json.loads(train_sample_manifest.read_text())
    sample_files = {}
    sample_root = output_root / "train_sample"
    for direction in sample_manifest["direction_order"]:
        source = Path(sample_manifest["files"][direction]["path"])
        rows = [merge_row(row, index) for row in _read_jsonl(source)]
        destination = sample_root / f"{direction}.jsonl"
        output_hash = atomic_jsonl(destination, rows)
        sample_files[direction] = {
            **sample_manifest["files"][direction],
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": output_hash,
        }
    output_sample_manifest = {
        **sample_manifest,
        "status": "FROZEN_WITH_JOERN_GRAPH",
        "files": sample_files,
        "ast_manifest_sha256": sha256(ast_root / "manifest.json"),
        "joern_manifest_sha256": sha256(edges_root / "manifest.json"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(sample_root / "manifest.json", output_sample_manifest)
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "ast_manifest_sha256": sha256(ast_root / "manifest.json"),
        "joern_manifest_sha256": sha256(edges_root / "manifest.json"),
        "source_unique_count": len(index),
        "completed_shards": completed,
        "train_sample_manifest": str(sample_root / "manifest.json"),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ast-root", type=Path, default=Path("content/datasets/xlcost/structural_ast_qwen")
    )
    parser.add_argument(
        "--edges-root", type=Path, default=Path("content/datasets/xlcost/joern_edges")
    )
    parser.add_argument(
        "--train-sample-manifest",
        type=Path,
        default=Path("content/datasets/xlcost/train_sample_manifest.json"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("content/datasets/xlcost/structural_full")
    )
    args = parser.parse_args()
    manifest = merge(**vars(args))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "shards": len(manifest["completed_shards"]),
                "unique_sources": manifest["source_unique_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
