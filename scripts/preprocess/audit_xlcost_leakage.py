#!/usr/bin/env python3
"""Audit XLCoST cross-split leakage and create a problem-filtered training corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.leakage import (
    exact_jaccard,
    fingerprints,
    minhash_signature,
    token_ngram_hashes,
)

SPLITS = ("train", "validation", "test")
PROTECTED_SPLITS = {"validation", "test"}
FINGERPRINT_FIELDS = ("exact", "lexical", "identifier_normalized", "ast")


@dataclass(frozen=True)
class Program:
    """One unique language implementation observed in a parsed split."""

    split: str
    problem_id: str
    language: str
    code_sha256: str
    source: str


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _collect_programs(root: Path) -> dict[str, list[Program]]:
    by_split: dict[str, dict[tuple[str, str, str], Program]] = {
        split: {} for split in SPLITS
    }
    for split in SPLITS:
        for path in sorted((root / split).glob("*.jsonl")):
            for row in _rows(path):
                for side in ("source", "target"):
                    language = row[f"{side}_language"]
                    source = row[side]
                    digest = row[f"{side}_sha256"]
                    key = (row["problem_id"], language, digest)
                    by_split[split].setdefault(
                        key,
                        Program(split, row["problem_id"], language, digest, source),
                    )
    return {split: list(values.values()) for split, values in by_split.items()}


def _atomic_json(path: Path, value: Any) -> str:
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_filtered_jsonl(source: Path, target: Path, dropped: set[str]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    kept = 0
    removed = 0
    try:
        with source.open(encoding="utf-8") as incoming, os.fdopen(
            fd, "w", encoding="utf-8"
        ) as outgoing:
            for line in incoming:
                row = json.loads(line)
                if row["problem_id"] in dropped:
                    removed += 1
                    continue
                outgoing.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                kept += 1
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {
        "kept": kept,
        "removed": removed,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def audit(parsed_root: Path, output_root: Path, report_path: Path) -> dict[str, Any]:
    """Run frozen leakage checks and materialize the filtered training split."""
    programs = _collect_programs(parsed_root)
    computed: dict[tuple[str, str, str, str], Any] = {}
    indexes: dict[str, dict[str, dict[str, list[Program]]]] = {
        field: defaultdict(lambda: defaultdict(list)) for field in FINGERPRINT_FIELDS
    }
    lexical_data: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    for split in SPLITS:
        for number, program in enumerate(programs[split], start=1):
            fp = fingerprints(program.source, program.language)
            key = (split, program.problem_id, program.language, program.code_sha256)
            computed[key] = fp
            if split in PROTECTED_SPLITS:
                for field in FINGERPRINT_FIELDS:
                    indexes[field][program.language][getattr(fp, field)].append(program)
                if len(fp.lexical_tokens) >= 40:
                    ngrams = token_ngram_hashes(fp.lexical_tokens, 5)
                    lexical_data[key] = (ngrams, minhash_signature(ngrams, 256, 42))
            if number % 1000 == 0:
                print(f"fingerprints split={split} completed={number}", flush=True)

    matches: list[dict[str, Any]] = []
    dropped: set[str] = set()
    matched_pairs: set[tuple[str, str, str, str, str]] = set()
    for number, train in enumerate(programs["train"], start=1):
        key = ("train", train.problem_id, train.language, train.code_sha256)
        fp = computed[key]
        for field in FINGERPRINT_FIELDS:
            for protected in indexes[field][train.language].get(getattr(fp, field), []):
                pair_key = (
                    train.problem_id,
                    protected.problem_id,
                    train.language,
                    protected.split,
                    field,
                )
                if pair_key in matched_pairs:
                    continue
                matched_pairs.add(pair_key)
                dropped.add(train.problem_id)
                matches.append(
                    {
                        "rule": field,
                        "language": train.language,
                        "train_problem_id": train.problem_id,
                        "protected_problem_id": protected.problem_id,
                        "protected_split": protected.split,
                        "train_code_sha256": train.code_sha256,
                        "protected_code_sha256": protected.code_sha256,
                    }
                )
        if number % 1000 == 0:
            print(f"exact_checks completed={number}", flush=True)

    # Use 64 bands of four MinHash values to generate candidates, then enforce both the
    # predefined 0.80 estimated and 0.85 exact Jaccard thresholds.
    band_index: dict[str, dict[tuple[int, bytes], list[Program]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split in PROTECTED_SPLITS:
        for program in programs[split]:
            key = (split, program.problem_id, program.language, program.code_sha256)
            if key not in lexical_data:
                continue
            signature = lexical_data[key][1]
            for band in range(64):
                band_index[program.language][
                    (band, signature[band * 4 : (band + 1) * 4].tobytes())
                ].append(program)

    for number, train in enumerate(programs["train"], start=1):
        fp = computed[("train", train.problem_id, train.language, train.code_sha256)]
        if len(fp.lexical_tokens) < 40:
            continue
        train_ngrams = token_ngram_hashes(fp.lexical_tokens, 5)
        train_signature = minhash_signature(train_ngrams, 256, 42)
        candidates: dict[tuple[str, str, str], Program] = {}
        for band in range(64):
            band_key = (band, train_signature[band * 4 : (band + 1) * 4].tobytes())
            for protected in band_index[train.language].get(band_key, []):
                candidates[(protected.split, protected.problem_id, protected.code_sha256)] = protected
        for protected in candidates.values():
            protected_key = (
                protected.split,
                protected.problem_id,
                protected.language,
                protected.code_sha256,
            )
            protected_ngrams, protected_signature = lexical_data[protected_key]
            estimated = float(np.mean(train_signature == protected_signature))
            if estimated < 0.80:
                continue
            similarity = exact_jaccard(train_ngrams, protected_ngrams)
            if similarity < 0.85:
                continue
            pair_key = (
                train.problem_id,
                protected.problem_id,
                train.language,
                protected.split,
                "near_duplicate",
            )
            if pair_key in matched_pairs:
                continue
            matched_pairs.add(pair_key)
            dropped.add(train.problem_id)
            matches.append(
                {
                    "rule": "near_duplicate",
                    "language": train.language,
                    "train_problem_id": train.problem_id,
                    "protected_problem_id": protected.problem_id,
                    "protected_split": protected.split,
                    "estimated_jaccard": estimated,
                    "exact_jaccard": similarity,
                    "train_code_sha256": train.code_sha256,
                    "protected_code_sha256": protected.code_sha256,
                }
            )
        if number % 500 == 0:
            print(f"near_duplicate_checks completed={number}", flush=True)

    output_files: dict[str, dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        output_files[split] = {}
        for source in sorted((parsed_root / split).glob("*.jsonl")):
            target = output_root / split / source.name
            output_files[split][source.name] = _atomic_filtered_jsonl(
                source, target, dropped if split == "train" else set()
            )

    counts_by_rule: dict[str, int] = defaultdict(int)
    for match in matches:
        counts_by_rule[match["rule"]] += 1
    report = {
        "schema_version": 1,
        "configuration": {
            "exact_normalization": "line endings only",
            "lexical": "Tree-sitter leaves excluding comments",
            "identifier_normalized": "first-occurrence local-like identifier renaming",
            "structural": "Tree-sitter node-type preorder, values/comments removed",
            "near_duplicate": {
                "tokens": "lexical token 5-grams",
                "min_tokens": 40,
                "minhash_permutations": 256,
                "candidate_estimated_jaccard": 0.80,
                "required_exact_jaccard": 0.85,
                "lsh_bands": 64,
                "rows_per_band": 4,
            },
            "exclusion": "drop complete XLCoST training problem on any protected match",
        },
        "unique_program_counts": {split: len(rows) for split, rows in programs.items()},
        "dropped_training_problem_count": len(dropped),
        "dropped_training_problem_ids": sorted(dropped),
        "match_counts_by_rule": dict(sorted(counts_by_rule.items())),
        "matches": sorted(
            matches,
            key=lambda item: (
                item["train_problem_id"],
                item["protected_split"],
                item["protected_problem_id"],
                item["language"],
                item["rule"],
            ),
        ),
        "output_files": output_files,
    }
    report["source_manifest_sha256"] = hashlib.sha256(
        (parsed_root / "manifest.json").read_bytes()
    ).hexdigest()
    _atomic_json(report_path, report)
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "leakage_report": str(report_path),
        "leakage_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "dropped_training_problem_count": len(dropped),
        "output_files": output_files,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=Path("content/datasets/xlcost/parsed_program_level"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("content/datasets/xlcost/clean_program_level"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data/xlcost_leakage.json"),
    )
    args = parser.parse_args()
    report = audit(args.parsed_root, args.output_root, args.report)
    print(
        json.dumps(
            {
                "dropped_training_problem_count": report["dropped_training_problem_count"],
                "match_counts_by_rule": report["match_counts_by_rule"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
