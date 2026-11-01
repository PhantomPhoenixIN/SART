#!/usr/bin/env python3
"""Audit Qwen token lengths and resolve the frozen training/generation limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from src.data.translation_protocol import serialize_prompt, serialize_target


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _summary(values: list[int]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.int64)
    return {
        "count": int(data.size),
        "min": int(data.min()),
        "median": float(np.median(data)),
        "mean": float(data.mean()),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "p99_5": float(np.percentile(data, 99.5)),
        "max": int(data.max()),
    }


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


def audit(data_root: Path, tokenizer_path: Path, output: Path) -> dict[str, Any]:
    """Measure prompt/target lengths for clean train and in-domain validation records."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    by_direction: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"prompt": [], "target": [], "combined": []}
    )
    by_split: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"prompt": [], "target": [], "combined": []}
    )
    selection_targets: list[int] = []
    selection_combined: list[int] = []
    for split in ("train", "validation", "test"):
        for path in sorted((data_root / split).glob("*.jsonl")):
            direction = path.stem
            for number, row in enumerate(_rows(path), start=1):
                prompt = serialize_prompt(
                    row["source_language"], row["target_language"], row["source"]
                )
                target = serialize_target(row["target"])
                prompt_count = len(tokenizer.encode(prompt, add_special_tokens=False))
                target_count = len(tokenizer.encode(target, add_special_tokens=False)) + 1
                combined_count = prompt_count + target_count
                for collection in (by_direction[direction], by_split[split]):
                    collection["prompt"].append(prompt_count)
                    collection["target"].append(target_count)
                    collection["combined"].append(combined_count)
                if split in {"train", "validation"}:
                    selection_targets.append(target_count)
                    selection_combined.append(combined_count)
                if number % 2000 == 0:
                    print(f"tokenized split={split} direction={direction} rows={number}", flush=True)

    target_p995 = float(np.percentile(np.asarray(selection_targets), 99.5))
    max_new_tokens = int(math.ceil(target_p995 / 256.0) * 256)
    context_candidates = (2048, 4096)
    exclusions = {
        str(limit): {
            "count": sum(value > limit for value in selection_combined),
            "rate": sum(value > limit for value in selection_combined)
            / len(selection_combined),
        }
        for limit in context_candidates
    }
    resolved_context = next(
        (limit for limit in context_candidates if exclusions[str(limit)]["rate"] <= 0.005),
        4096,
    )
    report = {
        "schema_version": 1,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_files_sha256": {
            name: hashlib.sha256((tokenizer_path / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
        },
        "protocol": "docs/TRANSLATION_PROTOCOL.md version 1",
        "target_length_includes_eos": True,
        "by_split": {
            split: {kind: _summary(values) for kind, values in kinds.items()}
            for split, kinds in by_split.items()
        },
        "by_direction": {
            direction: {kind: _summary(values) for kind, values in kinds.items()}
            for direction, kinds in sorted(by_direction.items())
        },
        "train_validation_combined_context_exclusions": exclusions,
        "resolved_limit_exclusions_by_split": {
            split: {
                "count": sum(value > resolved_context for value in kinds["combined"]),
                "rate": sum(value > resolved_context for value in kinds["combined"])
                / len(kinds["combined"]),
            }
            for split, kinds in by_split.items()
        },
        "resolved_limit_exclusions_by_direction": {
            direction: {
                "count": sum(value > resolved_context for value in kinds["combined"]),
                "rate": sum(value > resolved_context for value in kinds["combined"])
                / len(kinds["combined"]),
            }
            for direction, kinds in sorted(by_direction.items())
        },
        "resolved": {
            "target_p99_5": target_p995,
            "max_new_tokens": max_new_tokens,
            "max_sequence_length": resolved_context,
            "over_context_rule": "exclude examples whose prompt plus target and EOS exceed max_sequence_length",
        },
    }
    _atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("content/datasets/xlcost/clean_program_level"),
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data/qwen_token_length_audit.json")
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.data_root, args.tokenizer, args.output)["resolved"], indent=2))


if __name__ == "__main__":
    main()
