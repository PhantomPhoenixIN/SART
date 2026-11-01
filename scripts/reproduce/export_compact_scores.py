#!/usr/bin/env python3
"""Export compact task outcomes from complete Protocol v3 score runs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS = (
    "architecture", "backbone", "benchmark", "direction", "experiment_id",
    "seed", "split", "task_id", "problem_id", "pass_at_1", "status",
    "execution_status", "compile_status", "runtime_status", "flaky",
    "tests_passed", "tests_total", "latency_seconds", "repeats",
    "source_sha256", "target_reference_sha256", "target_test_sha256",
    "generated_code_sha256", "prediction_record_sha256",
    "eligibility_manifest_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifests = sorted(root.glob("*/*/*/*/score_manifest.json"))
    if not manifests:
        raise ValueError(f"No score manifests found under {root}")
    by_benchmark: dict[str, list[Path]] = {}
    pending = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") == "COMPLETE":
            by_benchmark.setdefault(str(manifest["benchmark"]), []).append(path)
        else:
            pending.append(str(path.relative_to(root)))

    index = {"schema_version": 1, "source": "Protocol v3 per-task score JSON",
             "export_fields": list(FIELDS), "benchmarks": {}, "pending_runs": pending}
    for benchmark, paths in sorted(by_benchmark.items()):
        target = output / f"{benchmark}_score_rows.jsonl.gz"
        counts = Counter()
        entries = []
        with target.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as zipped:
                for manifest_path in paths:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    count = 0
                    for task_path in sorted(manifest_path.parent.glob("*/*.json")):
                        row = json.loads(task_path.read_text(encoding="utf-8"))
                        if "task_id" not in row:
                            continue
                        record = {field: row.get(field) for field in FIELDS}
                        zipped.write((json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode())
                        count += 1
                    expected = int(manifest["tasks_total"])
                    if count != expected:
                        raise ValueError(f"{manifest_path}: {count} tasks, expected {expected}")
                    counts["tasks"] += count
                    counts["runs"] += 1
                    entries.append({"manifest": str(manifest_path.relative_to(root)),
                                    "manifest_sha256": sha256(manifest_path), "tasks": count})
                    if counts["runs"] % 25 == 0:
                        print(f"{benchmark}: {counts['runs']} runs, {counts['tasks']} tasks", flush=True)
        index["benchmarks"][benchmark] = {
            "runs": counts["runs"], "tasks": counts["tasks"],
            "file": target.name, "sha256": sha256(target), "manifests": entries,
        }
    (output / "score_export_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.results_root, args.output)
    print({key: {"runs": value["runs"], "tasks": value["tasks"]}
           for key, value in summary["benchmarks"].items()})


if __name__ == "__main__":
    main()
