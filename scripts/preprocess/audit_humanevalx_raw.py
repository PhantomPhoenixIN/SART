#!/usr/bin/env python3
"""Audit raw HumanEval-X task alignment and embedded test counts."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path
from typing import Any

FILES = {"Python": "python", "Java": "java", "C++": "cpp"}


def _top_level_call_items(text: str, call_pattern: str) -> int:
    match = re.search(call_pattern, text)
    if not match:
        raise ValueError(f"test source has no call matching {call_pattern}")
    start = match.end()
    depth = 1
    quote: str | None = None
    escaped = False
    commas = 0
    has_content = False
    for char in text[start:]:
        if quote:
            has_content = True
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            has_content = True
        elif char in "([{":
            depth += 1
            has_content = True
        elif char in ")]}" :
            depth -= 1
            if depth == 0:
                return commas + int(has_content)
            has_content = True
        elif char == "," and depth == 1:
            commas += 1
            has_content = False
        elif not char.isspace():
            has_content = True
    raise ValueError("unclosed test call")


def _test_count(test: str, language: str) -> int:
    if language == "Python":
        tree = ast.parse(test)
        def count(node: ast.AST, multiplier: int = 1) -> int:
            total = int(isinstance(node, ast.Assert)) * multiplier
            for child in ast.iter_child_nodes(node):
                child_multiplier = multiplier
                if isinstance(node, (ast.For, ast.AsyncFor)) and child in node.body:
                    iterator = node.iter
                    if isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name) and iterator.func.id == "range":
                        try:
                            child_multiplier *= len(range(*[ast.literal_eval(arg) for arg in iterator.args]))
                        except (ValueError, TypeError):
                            pass
                total += count(child, child_multiplier)
            return total
        return count(tree)
    if language == "Java":
        if re.search(r"Arrays\.asList\s*\(", test):
            return _top_level_call_items(test, r"Arrays\.asList\s*\(")
        loop = re.search(r"for\s*\([^;]*;[^;<]*<\s*(\d+)\s*;[^)]*\)\s*\{", test)
        if loop:
            return int(loop.group(1))
        raise ValueError("unsupported Java harness test-count pattern")
    direct = len(re.findall(r"\bassert\s*\(", test))
    loop = re.search(r"for\s*\([^;]*;[^;<]*<\s*(\d+)\s*;[^)]*\)\s*\{", test)
    return int(loop.group(1)) if loop and direct else direct


def audit(root: Path, output: Path, markdown: Path) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for language, folder in FILES.items():
        path = root / folder / "data/humaneval.jsonl"
        lang_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        indexed: dict[str, dict[str, Any]] = {}
        for row in lang_rows:
            uid = row["task_id"].split("/", 1)[-1]
            if uid in indexed:
                raise ValueError(f"duplicate task ID {row['task_id']} in {path}")
            indexed[uid] = row
        rows[language] = indexed

    common = set.intersection(*(set(v) for v in rows.values()))
    if any(set(v) != common for v in rows.values()):
        raise ValueError("HumanEval-X task IDs are not aligned across three languages")
    test_counts = {
        language: {uid: _test_count(row["test"], language) for uid, row in lang_rows.items()}
        for language, lang_rows in rows.items()
    }
    directions = {}
    for source in FILES:
        for target in FILES:
            if source == target:
                continue
            values = [test_counts[target][uid] for uid in sorted(common)]
            directions[f"{source}-to-{target}"] = {
                "problems": len(common),
                "translation_tasks": len(common),
                "total_target_test_assertions": sum(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    per_problem = [sum(test_counts[lang][uid] for lang in FILES) for uid in common]
    result = {
        "status": "RAW_AUDIT_ONLY; canonical execution/repeatability preflight pending",
        "dataset_revision": "62c78627f3072a1454fa0cb0184737cafe5e4198",
        "languages": list(FILES),
        "unique_problems": len(common),
        "aligned_task_ids": sorted(common),
        "tasks_by_language": {lang: len(data) for lang, data in rows.items()},
        "common_task_count": len(common),
        "common_directional_task_count": len(common) * 6,
        "unique_test_assertions_across_three_target_harnesses": sum(per_problem),
        "total_test_executions_across_six_translation_directions": 2 * sum(per_problem),
        "test_assertions_per_problem_across_three_languages": {
            "mean": statistics.mean(per_problem),
            "median": statistics.median(per_problem),
            "min": min(per_problem),
            "max": max(per_problem),
        },
        "by_direction": directions,
        "test_count_semantics": {
            "Python": "assert statements in the official embedded check function",
            "Java": "top-level entries in the official Arrays.asList expected-result vector",
            "C++": "assert macro calls in the official test body",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# HumanEval-X raw functional coverage audit",
        "",
        "Status: **raw audit only**. Canonical solutions have not yet been executed repeatedly; no eligible manifest is frozen and no model predictions exist.",
        "",
        f"- Dataset revision: `{result['dataset_revision']}`.",
        f"- Unique aligned problems: {len(common)}.",
        f"- Translation tasks: {len(common) * 6} across six directions.",
        f"- Distinct test assertions across the three language harnesses: {sum(per_problem)}.",
        f"- Target-harness test executions across all six directions: {2 * sum(per_problem)}.",
        "",
        "| Direction | Problems/tasks | Total target test inputs | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction, item in directions.items():
        lines.append(
            f"| {direction.replace('-to-', ' → ')} | {item['problems']} | "
            f"{item['total_target_test_assertions']} | {item['mean']:.2f} | "
            f"{item['median']} | {item['min']} | {item['max']} |"
        )
    lines += [
        "",
        "Test inputs are counted from each language's official executable harness; Python assertions, Java expected-result vector entries and C++ assertion calls are the respective test-case units. The pinned CodeGeeX evaluator and runtime must be used or matched. Repeat canonical execution at least three times, freeze exclusions and test hashes before any model inference, and cluster comparisons by underlying task ID.",
    ]
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("content/datasets/humanevalx/62c78627f3072a1454fa0cb0184737cafe5e4198/data"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/data/humanevalx_raw_audit.json"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/data/humanevalx_raw_audit.md"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.output, args.markdown), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
