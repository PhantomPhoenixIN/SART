#!/usr/bin/env python3
"""Create a deterministic equal-count six-direction sample from clean XLCoST train."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.data.translation_protocol import serialize_prompt, serialize_target


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def freeze(
    structure_root: Path,
    output_path: Path,
    tokenizer_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze one shared set of equal-count source/target tasks per direction."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((structure_root / "train").glob("*/*.jsonl")):
        direction = path.parent.name
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            candidates[direction][row["problem_id"]] = row
    if len(candidates) != 6:
        raise ValueError(f"Expected six translation directions; got {sorted(candidates)}")
    common_count = min(len(rows) for rows in candidates.values())
    selected: dict[str, list[dict[str, Any]]] = {}
    direction_stats: dict[str, Any] = {}
    for direction in sorted(candidates):
        rows = sorted(
            candidates[direction].values(),
            key=lambda row: hashlib.sha256(
                f"{seed}|{direction}|{row['problem_id']}".encode()
            ).hexdigest(),
        )[:common_count]
        rows.sort(key=lambda row: (row["problem_id"], row["source_sha256"]))
        prompt_tokens = 0
        target_tokens = 0
        for row in rows:
            prompt = serialize_prompt(
                row["source_language"], row["target_language"], row["source"]
            )
            prompt_tokens += len(tokenizer.encode(prompt, add_special_tokens=False))
            target_tokens += len(
                tokenizer.encode(serialize_target(row["target"]), add_special_tokens=False)
            ) + 1
        selected[direction] = rows
        direction_stats[direction] = {
            "available": len(candidates[direction]),
            "selected": len(rows),
            "prompt_tokens": prompt_tokens,
            "target_tokens_including_eos": target_tokens,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_shards: dict[str, dict[str, Any]] = {}
    for direction, rows in selected.items():
        target = output_path.parent / f"{output_path.stem}_{direction}.jsonl"
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        output_shards[direction] = {
            "path": str(target),
            "records": len(rows),
            "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "problem_ids": [row["problem_id"] for row in rows],
        }
    manifest = {
        "schema_version": 1,
        "status": "FROZEN",
        "seed": seed,
        "selection": "equal-count K=1 examples per direction, selected by SHA256(seed|direction|problem_id)",
        "direction_order": sorted(selected),
        "examples_per_direction": common_count,
        "total_examples_per_epoch": common_count * 6,
        "direction_stats": direction_stats,
        "files": output_shards,
        "source_structure_manifest_sha256": hashlib.sha256(
            (structure_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "tokenizer_json_sha256": hashlib.sha256(
            (tokenizer_path / "tokenizer.json").read_bytes()
        ).hexdigest(),
    }
    _atomic_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure-root", type=Path, default=Path("content/datasets/xlcost/structural_ast_qwen")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("content/datasets/xlcost/train_sample_manifest.json")
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = freeze(args.structure_root, args.output, args.tokenizer, args.seed)
    print(json.dumps({k: manifest[k] for k in ("examples_per_direction", "total_examples_per_epoch", "direction_stats")}, indent=2))


if __name__ == "__main__":
    main()
