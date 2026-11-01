#!/usr/bin/env python3
"""Audit XLCoST program-level pair files and grouped split boundaries."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.xlcost_format import detokenize_xlcost

PAIRS = {"Java-Python": ("Java", "Python"), "Java-C++": ("Java", "C++"), "C++-Python": ("C++", "Python")}
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}
EXTENSIONS = {"Python": ".py", "Java": ".java", "C++": ".cpp"}


def read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as stream:
        return [line.rstrip("\r\n") for line in stream]


def quantiles(values: list[int]) -> str:
    return (
        f"mean {statistics.mean(values):.1f}, median {statistics.median(values):g}, "
        f"min {min(values)}, max {max(values)}"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit(root: Path, output: Path) -> None:
    manifest_path = root / "extraction_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or len(manifest.get("files", {})) != 36:
        raise ValueError("XLCoST program-level extraction is incomplete")

    split_ids: dict[str, dict[str, set[str]]] = {
        pair: {} for pair in PAIRS
    }
    global_id_splits: dict[str, set[str]] = defaultdict(set)
    pair_lengths: dict[str, dict[str, list[int]]] = defaultdict(dict)
    exact_hashes: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    parse_stats: dict[str, Counter[str]] = defaultdict(Counter)
    map_duplicates: dict[str, int] = {}
    row_counts: dict[str, int] = {}

    for pair, languages in PAIRS.items():
        for split, split_label in SPLIT_NAMES.items():
            maps: dict[str, list[str]] = {}
            codes: dict[str, list[str]] = {}
            base = root / pair
            for language in languages:
                map_path = base / f"{split}-{language}-map.jsonl"
                code_path = base / f"{split}-{pair}-tok{EXTENSIONS[language]}"
                maps[language] = read_lines(map_path)
                codes[language] = read_lines(code_path)
                if len(maps[language]) != len(codes[language]):
                    raise ValueError(
                        f"Map/code count mismatch: {map_path} ({len(maps[language])}) vs "
                        f"{code_path} ({len(codes[language])})"
                    )
                duplicate_count = len(maps[language]) - len(set(maps[language]))
                map_duplicates[f"{pair}/{split}/{language}"] = duplicate_count
                row_counts[f"{pair}/{split}/{language}"] = len(maps[language])
                canonical_lengths = [len(code.split()) for code in codes[language]]
                pair_lengths[f"{pair}/{split}"][language] = canonical_lengths
                for problem_id, code in zip(maps[language], codes[language], strict=True):
                    global_id_splits[problem_id].add(split_label)
                    exact_hashes[(language, split_label)][problem_id].add(
                        sha256_text(" ".join(code.split()))
                    )
                    if language == "Python":
                        try:
                            ast.parse(detokenize_xlcost(code, language))
                            parse_stats[f"{pair}/{split_label}/{language}"]["parsed"] += 1
                        except SyntaxError:
                            parse_stats[f"{pair}/{split_label}/{language}"]["parse_error"] += 1
            if maps[languages[0]] != maps[languages[1]]:
                raise ValueError(f"Parallel map order/IDs differ: {pair}/{split}")
            split_ids[pair][split_label] = set(maps[languages[0]])

    within_split_overlap = {
        pair: {
            f"{left}/{right}": len(split_ids[pair][left] & split_ids[pair][right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
        for pair in PAIRS
    }
    split_conflicts = sum(len(labels) > 1 for labels in global_id_splits.values())
    per_language_exact_dupes: dict[str, dict[str, int]] = {}
    exact_leakage: dict[str, dict[str, int]] = {}
    for language in ("Python", "Java", "C++"):
        by_split: dict[str, Counter[str]] = {}
        for split_label in ("train", "validation", "test"):
            # A problem occurs in two parallel-pair folders. Count the same
            # (problem ID, code hash) only once; a distinct ID is a duplicate.
            hashes = [
                digest
                for hashes_for_problem in exact_hashes[(language, split_label)].values()
                for digest in hashes_for_problem
            ]
            by_split[split_label] = Counter(hashes)
        per_language_exact_dupes[language] = {
            split: sum(count - 1 for count in counts.values() if count > 1)
            for split, counts in by_split.items()
        }
        exact_hash_sets = {split: set(counts) for split, counts in by_split.items()}
        exact_leakage[language] = {
            "train_validation": len(exact_hash_sets["train"] & exact_hash_sets["validation"]),
            "train_test": len(exact_hash_sets["train"] & exact_hash_sets["test"]),
            "validation_test": len(exact_hash_sets["validation"] & exact_hash_sets["test"]),
        }

    lines = [
        "# XLCoST program-level raw data audit",
        "",
        "Status: **raw structure/count audit; token detokenization, parsing, leakage normalization, and clean grouped split construction remain open.** No external predictions were inspected.",
        "",
        f"Source archive SHA256: `{manifest['source_archive_sha256']}`  ",
        f"Extracted {len(manifest['files'])} files from `generation/pair_data_tok_full` only.",
        "",
        "## Pair counts and length distributions",
        "",
        "Each undirected parallel pair provides its two directed translation tasks. Length counts below are whitespace-separated pretokenized XLCoST tokens; they are not the selected model tokenizer's subword lengths.",
        "",
        "| Parallel pair | Split | Programs | First-language tokens | Second-language tokens |",
        "|---|---|---:|---|---|",
    ]
    for pair, languages in PAIRS.items():
        for split in ("train", "validation", "test"):
            n = row_counts[f"{pair}/{('val' if split == 'validation' else split)}/{languages[0]}"]
            count_key = "val" if split == "validation" else split
            values_a = pair_lengths[f"{pair}/{count_key}"][languages[0]]
            values_b = pair_lengths[f"{pair}/{count_key}"][languages[1]]
            lines.append(
                f"| {languages[0]} ↔ {languages[1]} | {split} | {n} | "
                f"{quantiles(values_a)} | {quantiles(values_b)} |"
            )

    lines.extend(
        [
            "",
            "## Problem-level split audit",
            "",
            f"- Distinct program IDs across all pair folders/splits: {len(global_id_splits)}.",
            f"- IDs assigned to more than one official split across languages/pairs: {split_conflicts}.",
            f"- Pair-folder train/validation/test overlaps: `{json.dumps(within_split_overlap, sort_keys=True)}`.",
            "- A nonzero problem-level overlap requires creating and freezing a grouped clean split before training; never preserve cross-language sibling programs across partitions.",
            "",
            "## Exact duplicate audit (pretokens only)",
            "",
            f"- Exact whitespace-canonical duplicates within partitions, counted over selected source files and pair folders: `{json.dumps(per_language_exact_dupes, sort_keys=True)}`.",
            f"- Exact hash intersections across partitions: `{json.dumps(exact_leakage, sort_keys=True)}`.",
            "- These are preliminary exact checks. Comment-stripped, identifier-normalized, AST, and near-duplicate checks still need implementation after detokenization/parser validation.",
            "",
            "## Directional counts",
            "",
            "| Direction | Train | Validation | Test |",
            "|---|---:|---:|---:|",
        ]
    )
    for pair, languages in PAIRS.items():
        for source, target in (languages, languages[::-1]):
            counts = [
                len(split_ids[pair][split])
                for split in ("train", "validation", "test")
            ]
            lines.append(f"| {source} → {target} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines.extend(
        [
            "",
            "## Parsing and representation",
            "",
            "The official program-level files are whitespace-tokenized text; Python rows include `NEW_LINE`, `INDENT`, and `DEDENT` markers. The current deterministic formatting adapter restores these markers and common lexical operators. Python 3 AST parse coverage is:",
            "",
            "| Parallel pair | Split | Parse-valid Python programs | Parse errors | Parse rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for pair in ("Java-Python", "C++-Python"):
        for split in ("train", "validation", "test"):
            stats = parse_stats[f"{pair}/{split}/Python"]
            total = stats["parsed"] + stats["parse_error"]
            lines.append(
                f"| {pair} | {split} | {stats['parsed']} | {stats['parse_error']} | "
                f"{stats['parsed'] / total:.2%} |"
            )
    lines.extend(
        [
            "",
            "A syntax-only audit with pinned Tree-sitter grammars for Java and C++ remains outstanding. Python 2-era print syntax, malformed formatting, and some indentation defects account for remaining Python parse errors; invalid pairs will be explicitly counted and excluded from training under one predeclared rule applied identically to all architectures, rather than silently repaired.",
            "",
            "The archive contains no explicit data license/terms file; the dataset license remains unverified and the archive must not be redistributed.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("content/datasets/xlcost/program_level"))
    parser.add_argument("--output", type=Path, default=Path("reports/data/xlcost_audit.md"))
    args = parser.parse_args()
    audit(args.root, args.output)


if __name__ == "__main__":
    main()
