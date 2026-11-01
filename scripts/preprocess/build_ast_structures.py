#!/usr/bin/env python3
"""Build resume-safe Qwen token-aligned AST region shards for final XLCoST data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.data.translation_protocol import serialize_prompt
from src.regions.alignment import align_prompt_offsets
from src.regions.extract import extract_region_graph


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return _sha256(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _valid_completed_shard(path: Path, metadata: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == metadata.get("bytes")
        and _sha256(path) == metadata.get("sha256")
    )


def _source_span(prompt: str, source_language: str, source: str) -> tuple[int, int]:
    marker = f"### SOURCE PROGRAM ({source_language})\n"
    start = prompt.index(marker) + len(marker)
    return start, start + len(source)


def build(input_root: Path, output_root: Path, tokenizer_path: Path, shard_size: int) -> dict:
    """Build or resume deterministic structural shards with atomic output writes."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_hash = _sha256(tokenizer_path / "tokenizer.json")
    config = {
        "schema_version": 1,
        "input_manifest_sha256": _sha256(input_root / "manifest.json"),
        "tokenizer_json_sha256": tokenizer_hash,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "shard_size": shard_size,
        "structure": "Tree-sitter semantic regions with AST_PARENT/AST_CHILD only",
        "token_alignment": "smallest enclosing semantic region from prompt offset mappings",
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output_root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if previous and previous.get("configuration_hash") != config_hash:
        raise ValueError("Existing structural manifest has a different configuration")
    completed: dict[str, Any] = previous.get("completed_shards", {})
    failures: list[dict[str, Any]] = previous.get("failures", [])
    graph_cache: dict[tuple[str, str], Any] = {}
    total_records = 0
    total_source_tokens = 0
    total_mapped_tokens = 0

    for split in ("train", "validation", "test"):
        for input_path in sorted((input_root / split).glob("*.jsonl")):
            rows = [json.loads(line) for line in input_path.open(encoding="utf-8")]
            relative_input = f"{split}/{input_path.name}"
            for shard_index, start in enumerate(range(0, len(rows), shard_size)):
                shard_key = f"{relative_input}:{shard_index:05d}"
                output_path = output_root / split / input_path.stem / f"shard_{shard_index:05d}.jsonl"
                if shard_key in completed and _valid_completed_shard(
                    output_path, completed[shard_key]
                ):
                    total_records += completed[shard_key]["records"]
                    total_source_tokens += completed[shard_key]["source_tokens"]
                    total_mapped_tokens += completed[shard_key]["mapped_source_tokens"]
                    continue
                output_rows = []
                shard_source_tokens = 0
                shard_mapped_tokens = 0
                for row in rows[start : start + shard_size]:
                    try:
                        cache_key = (row["source_language"], row["source_sha256"])
                        graph = graph_cache.get(cache_key)
                        if graph is None:
                            graph = extract_region_graph(row["source"], row["source_language"])
                            graph_cache[cache_key] = graph
                        prompt = serialize_prompt(
                            row["source_language"], row["target_language"], row["source"]
                        )
                        encoded = tokenizer(
                            prompt,
                            add_special_tokens=False,
                            return_offsets_mapping=True,
                        )
                        mapping = align_prompt_offsets(
                            row["source"],
                            encoded["offset_mapping"],
                            _source_span(prompt, row["source_language"], row["source"]),
                            graph.regions,
                        )
                        mapped = sum(region_id is not None for _, region_id in mapping)
                        shard_source_tokens += len(mapping)
                        shard_mapped_tokens += mapped
                        output_rows.append(
                            {
                                **row,
                                "structure": {
                                    "regions": [asdict(region) for region in graph.regions],
                                    "edges": [asdict(edge) for edge in graph.edges],
                                    "availability": {
                                        "ast": graph.ast_available,
                                        "cfg": graph.cfg_available,
                                        "dfg": graph.dfg_available,
                                    },
                                    "prompt_token_count": len(encoded["input_ids"]),
                                    "source_token_to_region": mapping,
                                    "source_token_count": len(mapping),
                                    "mapped_source_token_count": mapped,
                                },
                            }
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "split": split,
                                "direction": input_path.stem,
                                "problem_id": row.get("problem_id"),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        raise
                digest = _atomic_jsonl(output_path, output_rows)
                completed[shard_key] = {
                    "path": str(output_path),
                    "records": len(output_rows),
                    "bytes": output_path.stat().st_size,
                    "sha256": digest,
                    "source_tokens": shard_source_tokens,
                    "mapped_source_tokens": shard_mapped_tokens,
                }
                total_records += len(output_rows)
                total_source_tokens += shard_source_tokens
                total_mapped_tokens += shard_mapped_tokens
                _atomic_json(
                    manifest_path,
                    {
                        **config,
                        "configuration_hash": config_hash,
                        "status": "RUNNING",
                        "completed_shards": completed,
                        "failures": failures,
                    },
                )
                print(f"completed {shard_key}", flush=True)

    manifest = {
        **config,
        "configuration_hash": config_hash,
        "status": "COMPLETE",
        "completed_shards": completed,
        "failures": failures,
        "records": total_records,
        "source_tokens": total_source_tokens,
        "mapped_source_tokens": total_mapped_tokens,
        "mapping_coverage": total_mapped_tokens / total_source_tokens,
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("content/datasets/xlcost/final_program_level"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("content/datasets/xlcost/structural_ast_qwen"),
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()
    manifest = build(args.input_root, args.output_root, args.tokenizer, args.shard_size)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": manifest["records"],
                "mapping_coverage": manifest["mapping_coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
