#!/usr/bin/env python3
"""Freeze a balanced, outcome-blind XLCoST validation sample for efficiency profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(validation_root: Path, output: Path, per_direction: int = 4) -> dict[str, Any]:
    manifest_path = validation_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validation manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Efficiency profiling sample must use the complete leakage-audited XLCoST split")
    validation_files = manifest.get("files", {}).get("validation", {})
    if not validation_files or any(item.get("removed", 0) != 0 for item in validation_files.values()):
        raise ValueError("Validation examples must be untouched by external leakage filtering")
    for relative, item in validation_files.items():
        path = validation_root / relative
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError(f"Frozen validation shard digest mismatch: {path}")
    rows: list[dict[str, Any]] = []
    validation_root = validation_root / "validation"
    for path in sorted(validation_root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                direction = f"{row['source_language']}-to-{row['target_language']}"
                key = f"SART-efficiency-profile-v1|{direction}|{row['problem_id']}"
                stable_hash = hashlib.sha256(key.encode()).hexdigest()
                rows.append({
                    "dataset_index": len(rows), "problem_id": str(row["problem_id"]),
                    "direction": direction, "source_sha256": row.get("source_sha256"),
                    "selection_sha256": stable_hash,
                })
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[row["direction"]].append(row)
    if len(by_direction) != 6 or any(len(items) < per_direction for items in by_direction.values()):
        raise ValueError("Frozen validation set must have six directions with enough examples")
    selected = []
    for direction, items in sorted(by_direction.items()):
        selected.extend(sorted(items, key=lambda item: (item["selection_sha256"], item["problem_id"]))[:per_direction])
    record = {
        "schema_version": 1,
        "status": "FROZEN_PREMODEL",
        "purpose": "in-domain efficiency profiling only; no external outcomes or model predictions read",
        "validation_manifest_sha256": sha256(manifest_path),
        "validation_source_manifest_sha256": manifest["source_manifest_sha256"],
        "validation_shard_sha256": {
            f"validation/{name}": item["sha256"]
            for name, item in sorted(validation_files.items())
        },
        "validation_example_count": len(rows),
        "per_direction": per_direction,
        "direction_count": 6,
        "sample_count": len(selected),
        "selection": "lowest SHA256(SART-efficiency-profile-v1|direction|problem_id) within direction",
        "samples": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen profile sample: {output}")
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path("content/datasets/xlcost/structural_external_clean"))
    parser.add_argument("--per-direction", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("configs/evaluation/efficiency_profile_sample.json"))
    args = parser.parse_args()
    result = freeze(args.validation_root, args.output, per_direction=args.per_direction)
    print(json.dumps({k: result[k] for k in ("status", "sample_count", "per_direction", "validation_manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
