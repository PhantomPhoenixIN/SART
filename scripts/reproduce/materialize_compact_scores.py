#!/usr/bin/env python3
"""Rebuild the score-row tree used by the published analysis scripts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.compact_root / "score_export_index.json").read_text())
    for benchmark, details in sorted(index["benchmarks"].items()):
        source = args.compact_root / details["file"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != details["sha256"]:
            raise ValueError(f"Checksum mismatch: {source}")
        count = 0
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                revision = Path(row["backbone"]).name
                task_hash = hashlib.sha256(str(row["task_id"]).encode()).hexdigest()[:24]
                target = (args.output / revision / row["experiment_id"] / benchmark /
                          row["split"] / row["direction"] / f"{task_hash}.json")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"Duplicate task outcome: {target}")
                target.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
                count += 1
        if count != details["tasks"]:
            raise ValueError(f"{benchmark}: {count} task rows, expected {details['tasks']}")
        print(f"{benchmark}: restored {count} score rows")


if __name__ == "__main__":
    main()
