#!/usr/bin/env python3
"""Audit TransCoder-Test function and reference-harness coverage (preflight only)."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

LANGS = {"python": ("Python", ".py"), "java": ("Java", ".java"), "cpp": ("C++", ".cpp")}


def _ids(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " | " not in line:
            raise ValueError(f"Unexpected TransCoder row in {path}")
        uid, code = line.split(" | ", 1)
        if uid in result:
            raise ValueError(f"Duplicate function ID {uid} in {path}")
        result[uid] = code
    return result


def _delimited_item_count(text: str, start: int, opening: str) -> int:
    """Count top-level values in a delimited test-vector initializer."""
    closing = {"{": "}", "(": ")"}[opening]
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    commas = 0
    content = False
    for char in text[start:]:
        if quote:
            content = True
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            content = True
        elif char in pairs:
            stack.append(pairs[char])
            content = True
        elif char in "})]":
            if not stack:
                if char != closing:
                    raise ValueError("Mismatched test-vector initializer")
                return commas + int(content)
            if stack.pop() != char:
                raise ValueError("Mismatched nested test-vector initializer")
            if not stack:
                if char != closing:
                    raise ValueError("Mismatched test-vector initializer")
                return commas + int(content)
            content = True
        elif char == "," and len(stack) == 1:
            commas += 1
        elif not char.isspace() and char != ",":
            content = True
    raise ValueError("Unclosed test-vector initializer")


def _case_count(script: Path, language: str) -> int:
    text = script.read_text(encoding="utf-8")
    if language == "python":
        match = re.search(r"\bparam\s*=\s*\[", text)
        if not match:
            raise ValueError(f"No Python param list in {script}")
        start = match.end()
        stack = ["]"]
        quote: str | None = None
        escaped = False
        end = start
        for _end, char in enumerate(text[start:], start):
            end = _end
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
            elif char in "[{(":
                stack.append({"[": "]", "{": "}", "(": ")"}[char])
            elif char in "]})":
                if not stack or char != stack.pop():
                    raise ValueError(f"Unbalanced Python parameter literal in {script}")
                if not stack:
                    break
        else:
            raise ValueError(f"Unclosed Python parameter list in {script}")
        body = text[start:end]
        depth = 0
        quote = None
        escaped = False
        items = 1 if body.strip() else 0
        for char in body:
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
            elif char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
            elif char == "," and depth == 0:
                items += 1
        return items
    counts: dict[str, int] = defaultdict(int)
    if language == "java":
        for name in re.findall(r"\b(param\d+)\.add\(", text):
            counts[name] += 1
        if not counts:
            for match in re.finditer(r"\b(param\d+)\s*=\s*Arrays\.asList\s*(\()", text):
                counts[match.group(1)] = _delimited_item_count(text, match.end() - 1, "(")
    else:
        for match in re.finditer(r"\bvector\s*<[^;]+?>\s*(param\d+)\s*\{([^{}]*)\}", text):
            values = match.group(2).strip()
            counts[match.group(1)] = 0 if not values else len(values.split(","))
        if not counts:
            for match in re.finditer(r"\b(param\d+)\s*(\{)", text):
                counts[match.group(1)] = _delimited_item_count(text, match.end() - 1, "{")
    if not counts:
        raise ValueError(f"No parameter vectors found in {script}")
    if len(set(counts.values())) != 1:
        raise ValueError(f"Unequal test-vector lengths in {script}: {dict(counts)}")
    return next(iter(counts.values()))


def audit(data_root: Path, scripts_root: Path, output: Path) -> dict[str, Any]:
    raw = (
        data_root
        / "1c21887dc55a88ea6535e669952d419256fbc24a/raw/TransCoder_tokenized_test_set_functions"
    )
    by_lang = {
        lang: _ids(raw / f"test.{short}.shuf.test.tok") for short, (lang, _) in LANGS.items()
    }
    paths: dict[str, dict[str, Path]] = {}
    counts: dict[str, dict[str, int]] = {}
    syntax_errors: dict[str, list[dict[str, str]]] = {}
    count_errors: dict[str, list[dict[str, str]]] = {}
    for short, (lang, extension) in LANGS.items():
        folder = scripts_root / short
        paths[lang] = {uid: folder / f"{uid}{extension}" for uid in by_lang[lang]}
        syntax_errors[lang] = []
        count_errors[lang] = []
        if short == "python":
            for uid, path in paths[lang].items():
                if path.is_file():
                    try:
                        ast.parse(path.read_text(encoding="utf-8"))
                    except SyntaxError as exc:
                        syntax_errors[lang].append(
                            {"problem_id": uid, "error": f"{exc.msg} at line {exc.lineno}"}
                        )
        counts[lang] = {}
        for uid, path in paths[lang].items():
            if not path.is_file():
                continue
            try:
                counts[lang][uid] = _case_count(path, short)
            except ValueError as exc:
                count_errors[lang].append({"problem_id": uid, "error": str(exc)})

    intersections = set.intersection(*(set(rows) for rows in by_lang.values()))
    invalid_python_harnesses = {item["problem_id"] for item in syntax_errors["Python"]}
    candidates = {
        lang: {
            uid
            for uid in intersections
            if uid in counts[lang]
            and not (lang == "Python" and uid in invalid_python_harnesses)
        }
        for lang in by_lang
    }
    common = set.intersection(*candidates.values())
    directions = {}
    for source in by_lang:
        for target in by_lang:
            if source == target:
                continue
            task_ids = sorted(candidates[target] & set(by_lang[source]))
            test_counts = [counts[target][uid] for uid in task_ids]
            directions[f"{source}-to-{target}"] = {
                "candidate_problem_ids": len(task_ids),
                "target_harness_tests": sum(test_counts),
                "mean_tests_per_problem": statistics.mean(test_counts) if test_counts else None,
                "median_tests_per_problem": statistics.median(test_counts) if test_counts else None,
                "min_tests_per_problem": min(test_counts) if test_counts else None,
                "max_tests_per_problem": max(test_counts) if test_counts else None,
            }
    common_tests = [sum(counts[lang][uid] for lang in by_lang) for uid in sorted(common)]
    report = {
        "status": (
            "RAW_CANDIDATE_AUDIT_ONLY; canonical reference compilation/execution "
            "and repeated-test screen pending"
        ),
        "source_revision": "1c21887dc55a88ea6535e669952d419256fbc24a",
        "raw_rows_by_language": {lang: len(rows) for lang, rows in by_lang.items()},
        "raw_ids_shared_by_all_languages": len(intersections),
        "harness_coverage_by_language": {
            lang: {
                "available_harnesses": len(counts[lang]),
                "missing_harnesses": len(by_lang[lang]) - len(counts[lang]),
            }
            for lang in by_lang
        },
        "python_harness_syntax_errors": syntax_errors["Python"],
        "harness_test_count_errors": count_errors,
        "common_core_candidate_problems": len(common),
        "common_core_candidate_tasks": len(common) * 6,
        "common_core_total_language_harness_tests": sum(common_tests),
        "common_core_total_translation_test_executions": 2 * sum(common_tests),
        "common_core_tests_per_problem_across_three_language_harnesses": {
            "mean": statistics.mean(common_tests) if common_tests else None,
            "median": statistics.median(common_tests) if common_tests else None,
            "min": min(common_tests) if common_tests else None,
            "max": max(common_tests) if common_tests else None,
        },
        "full_directional_candidates": directions,
        "problem_ids_common_core": sorted(common),
        "problem_ids_by_direction": {
            key: sorted(candidates[target] & set(by_lang[source]))
            for key, (source, target) in {
                f"{source}-to-{target}": (source, target)
                for source in by_lang
                for target in by_lang
                if source != target
            }.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("content/datasets/transcoder-test"))
    parser.add_argument(
        "--scripts-root",
        type=Path,
        default=Path(
            "content/datasets/transcoder-test/official_repo/data/evaluation/"
            "geeks_for_geeks_successful_test_scripts"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/data/transcoder_test_raw_audit.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.data_root, args.scripts_root, args.output), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
