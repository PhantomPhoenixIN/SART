#!/usr/bin/env python3
"""Extract unique external-source CFG/DFG edges and merge them into task caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts.preprocess.extract_joern_edges import run_extraction
from scripts.preprocess.merge_joern_edges import merge_row, read_edge_index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256(temporary)
        os.replace(temporary, path)
        return digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_union_manifest(ast_root: Path) -> dict[str, Any]:
    inputs = {}
    for benchmark in ("xcodeeval", "humanevalx", "transcoder_test", "avatar_tc"):
        for split in ("common_core", "full_directional"):
            path = ast_root / benchmark / split / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "COMPLETE":
                raise ValueError(f"External AST cache is incomplete: {path}")
            inputs[str(path.relative_to(ast_root))] = _sha256(path)
    payload = {"schema_version": 1, "status": "COMPLETE", "input_manifests": inputs}
    _write_json(ast_root / "manifest.json", payload)
    return payload


def _merge_all(ast_root: Path, edge_root: Path, output_root: Path) -> dict[str, Any]:
    index = read_edge_index(edge_root)
    manifests: dict[str, Any] = {}
    for benchmark in ("xcodeeval", "humanevalx", "transcoder_test", "avatar_tc"):
        per_benchmark: dict[str, Any] = {"status": "COMPLETE", "splits": {}}
        for split in ("common_core", "full_directional"):
            input_root = ast_root / benchmark / split
            output_split = output_root / benchmark / split
            files = {}
            totals = {"tasks": 0, "cfg_edges": 0, "dfg_edges": 0, "joern_available": 0}
            for source in sorted(input_root.rglob("*.jsonl")):
                relative = source.relative_to(input_root)
                target = output_split / relative
                with source.open(encoding="utf-8") as stream:
                    rows = [merge_row(json.loads(line), index) for line in stream]
                digest = _write_jsonl(target, rows)
                files[str(relative)] = {
                    "path": str(target),
                    "records": len(rows),
                    "input_sha256": _sha256(source),
                    "sha256": digest,
                }
                totals["tasks"] += len(rows)
                totals["cfg_edges"] += sum(
                    edge["edge_type"].startswith("CFG_")
                    for row in rows
                    for edge in row["structure"]["edges"]
                )
                totals["dfg_edges"] += sum(
                    edge["edge_type"].startswith("DFG_")
                    for row in rows
                    for edge in row["structure"]["edges"]
                )
                totals["joern_available"] += sum(
                    row["structure"].get("joern", {}).get("available", False) for row in rows
                )
            split_record = {
                "status": "COMPLETE",
                "ast_manifest_sha256": _sha256(input_root / "manifest.json"),
                "joern_manifest_sha256": _sha256(edge_root / "manifest.json"),
                "files": files,
                "totals": totals,
            }
            _write_json(output_split / "manifest.json", split_record)
            per_benchmark["splits"][split] = split_record
        _write_json(output_root / benchmark / "manifest.json", per_benchmark)
        manifests[benchmark] = per_benchmark
    _write_json(output_root / "manifest.json", {"status": "COMPLETE", "benchmarks": manifests})
    return manifests


def run(
    ast_root: Path,
    source_root: Path,
    edge_root: Path,
    output_root: Path,
    joern_root: Path,
    java_home: Path,
    workers: int = 8,
    shard_size: int = 64,
) -> dict[str, Any]:
    """Run the resumable official Joern extractor, then mask-merge available edges."""
    _prepare_union_manifest(ast_root)
    edges = run_extraction(
        ast_root=ast_root,
        source_root=source_root,
        output_root=edge_root,
        joern_root=joern_root,
        java_home=java_home,
        workers=workers,
        shard_size=shard_size,
        log_root=Path("logs/joern_external"),
    )
    merged = _merge_all(ast_root, edge_root, output_root)
    return {
        "edge_status": edges["status"],
        "edge_failures": len(edges["failures"]),
        "merged": merged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ast-root", type=Path, default=Path("data/structural/external"))
    parser.add_argument(
        "--source-root", type=Path, default=Path("content/datasets/external_joern_sources")
    )
    parser.add_argument(
        "--edge-root", type=Path, default=Path("content/datasets/external_joern_edges")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/structural/external_full")
    )
    parser.add_argument(
        "--joern-root", type=Path, default=Path("content/tools/joern/v4.0.644/joern-cli")
    )
    parser.add_argument(
        "--java-home", type=Path, default=Path("content/tools/temurin/21.0.12.1+1/runtime")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()
    result = run(**vars(args))
    print(
        json.dumps(
            {
                "edge_status": result["edge_status"],
                "edge_failures": result["edge_failures"],
                "merged_benchmarks": sorted(result["merged"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
