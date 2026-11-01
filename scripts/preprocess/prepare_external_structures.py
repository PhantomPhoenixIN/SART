#!/usr/bin/env python3
"""Build resumable, source-only region/token caches for frozen external tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.data.translation_protocol import serialize_prompt
from src.evaluation.external_tasks import load_external_tasks
from src.regions.alignment import align_prompt_offsets
from src.regions.extract import Region, extract_region_graph


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256_file(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, path)
        return size, digest
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _source_prompt_span(prompt: str, source_language: str, source: str) -> tuple[int, int]:
    marker = f"### SOURCE PROGRAM ({source_language})\n"
    start = prompt.index(marker) + len(marker)
    return start, start + len(source)


def _fallback_program_region(source: str) -> tuple[list[dict[str, Any]], bool]:
    """Keep parser failures routable through one explicit, non-semantic container."""
    source_bytes = source.encode("utf-8")
    region = Region(
        region_id=0,
        region_type="PROGRAM",
        raw_node_type="PARSER_FAILURE_FALLBACK",
        start_byte=0,
        end_byte=len(source_bytes),
        start_point=(0, 0),
        end_point=(source.count("\n"), len(source.rsplit("\n", 1)[-1])),
        parent_id=None,
        ast_depth=0,
        child_count=0,
    )
    return [asdict(region)], False


def _structure_row(
    task: dict[str, Any],
    tokenizer,
    graph_cache: dict[tuple[str, str], tuple[list, list, bool, str]],
) -> dict[str, Any]:
    source = task["source"]
    language = task["source_language"]
    prompt = serialize_prompt(
        language, task["target_language"], source, task.get("target_interface", "")
    )
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    cache_key = (language, task["source_sha256"])
    if cache_key not in graph_cache:
        try:
            graph = extract_region_graph(source, language)
            graph_cache[cache_key] = (
                [asdict(region) for region in graph.regions],
                [asdict(edge) for edge in graph.edges],
                True,
                "",
            )
        except (ValueError, UnicodeError) as exc:
            regions, ast_available = _fallback_program_region(source)
            parse_error = f"{type(exc).__name__}: {exc}"
            graph_cache[cache_key] = (regions, [], ast_available, parse_error)
    regions, edges, ast_available, parse_error = graph_cache[cache_key]
    region_objects = tuple(Region(**region) for region in regions)
    token_mapping = align_prompt_offsets(
        source,
        encoded["offset_mapping"],
        _source_prompt_span(prompt, language, source),
        region_objects,
    )
    mapping = [[token_index, region_id] for token_index, region_id in token_mapping]
    mapped = sum(region_id is not None for _, region_id in token_mapping)
    return {
        **task,
        "target": "",
        "structure": {
            "regions": regions,
            "edges": edges,
            "availability": {"ast": ast_available, "cfg": False, "dfg": False},
            "prompt_token_count": len(encoded["input_ids"]),
            "source_token_to_region": mapping,
            "source_token_count": len(mapping),
            "mapped_source_token_count": mapped,
            "parser_error": parse_error,
        },
    }


def _valid_shard(path: Path, metadata: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == metadata.get("bytes")
        and _sha256_file(path) == metadata.get("sha256")
    )


def prepare(
    benchmark: str,
    manifest_path: Path,
    output_root: Path,
    tokenizer_path: Path,
    split: str = "common_core",
    xcode_root: Path | None = None,
    humanevalx_root: Path | None = None,
    shard_size: int = 128,
    force_reset: bool = False,
) -> dict[str, Any]:
    """Prepare/resume one frozen benchmark manifest without exposing target labels."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_file = tokenizer_path / "tokenizer.json"
    config = {
        "schema_version": 1,
        "benchmark": benchmark,
        "split": split,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_sha256": _sha256_file(tokenizer_file),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "shard_size": shard_size,
        "parser": "Tree-sitter semantic AST regions; parser failure uses PROGRAM fallback",
        "prompt_protocol": "translation_protocol v1",
        "labels_or_tests_in_model_input": False,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path_out = output_root / "manifest.json"
    previous = json.loads(manifest_path_out.read_text()) if manifest_path_out.exists() else {}
    if previous and previous.get("configuration_hash") != config_hash:
        if not force_reset:
            raise ValueError("Existing structure cache has a different configuration hash")
        previous = {}
    completed = previous.get("completed_shards", {})
    tasks = load_external_tasks(
        benchmark,
        manifest_path,
        split=split,
        xcode_root=xcode_root,
        humanevalx_root=humanevalx_root,
    )
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_direction[task["direction"]].append(task)
    graph_cache: dict[tuple[str, str], tuple[list, list, bool, str]] = {}
    totals = {"tasks": 0, "source_tokens": 0, "mapped_source_tokens": 0, "ast_available": 0}
    failures = list(previous.get("failures", []))
    for direction, direction_tasks in sorted(by_direction.items()):
        for shard_id, start in enumerate(range(0, len(direction_tasks), shard_size)):
            key = f"{direction}:{shard_id:05d}"
            target = output_root / direction / f"shard_{shard_id:05d}.jsonl"
            old = completed.get(key)
            if old and _valid_shard(target, old):
                for field in totals:
                    totals[field] += int(old.get(field, 0))
                continue
            records = []
            for task in direction_tasks[start : start + shard_size]:
                try:
                    records.append(_structure_row(task, tokenizer, graph_cache))
                except Exception as exc:
                    failures.append(
                        {
                            "task_id": task["task_id"],
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    raise
            size, digest = _atomic_jsonl(target, records)
            shard_totals = {
                "tasks": len(records),
                "source_tokens": sum(r["structure"]["source_token_count"] for r in records),
                "mapped_source_tokens": sum(
                    r["structure"]["mapped_source_token_count"] for r in records
                ),
                "ast_available": sum(r["structure"]["availability"]["ast"] for r in records),
            }
            completed[key] = {"path": str(target), "bytes": size, "sha256": digest, **shard_totals}
            for field in totals:
                totals[field] += int(shard_totals[field])
            _atomic_json(
                manifest_path_out,
                {
                    **config,
                    "configuration_hash": config_hash,
                    "status": "RUNNING",
                    "completed_shards": completed,
                    "failures": failures,
                    "totals": totals,
                },
            )
    final = {
        **config,
        "configuration_hash": config_hash,
        "status": "COMPLETE",
        "completed_shards": completed,
        "failures": failures,
        "totals": totals,
        "ast_coverage": totals["ast_available"] / totals["tasks"] if totals["tasks"] else 0.0,
        "token_mapping_coverage": (
            totals["mapped_source_tokens"] / totals["source_tokens"]
            if totals["source_tokens"]
            else 0.0
        ),
    }
    _atomic_json(manifest_path_out, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["xcodeeval", "humanevalx", "transcoder_test", "avatar_tc"],
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["common_core", "full_directional"], default="common_core"
    )
    parser.add_argument("--xcode-root", type=Path)
    parser.add_argument("--humanevalx-root", type=Path)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--force-reset", action="store_true")
    args = parser.parse_args()
    result = prepare(
        benchmark=args.benchmark,
        manifest_path=args.manifest,
        output_root=args.output_root,
        tokenizer_path=args.tokenizer,
        split=args.split,
        xcode_root=args.xcode_root,
        humanevalx_root=args.humanevalx_root,
        shard_size=args.shard_size,
        force_reset=args.force_reset,
    )
    summary = result["totals"] | {"status": result["status"], "output": str(args.output_root)}
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
