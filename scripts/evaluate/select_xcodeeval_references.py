#!/usr/bin/env python3
"""Freeze deterministic xCodeEval source/reference choices before any model scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from src.evaluation.namespace_sandbox import SandboxLimits, run_isolated
from src.parsing.tree_sitter_adapter import parse_program

LANGUAGES = ("Python", "Java", "C++")
TOKENIZER = "content/models/qwen2.5-coder-7b/0396a76181e127dfc13e5c5ec48a8cee09938b02"
GCC12 = Path("/apps/apps/gcc/gcc-12.3.0/gc")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def python2_parse_success(source: str) -> tuple[bool, str]:
    """Use CPython's bundled Python-2-compatible grammar without rewriting source."""
    try:
        from lib2to3 import pygram, pytree
        from lib2to3.pgen2 import driver

        parser = driver.Driver(pygram.python_grammar, convert=pytree.convert)
        parser.parse_string(source if source.endswith("\n") else source + "\n")
        return True, ""
    except Exception as exc:  # A parser rejection is recorded as a candidate exclusion.
        return False, f"{type(exc).__name__}: {exc}"


def _python_compile_success(source: str, python2: bool) -> tuple[bool, str]:
    runtime = (
        Path("content/tools/python2.7.18/stage/opt/python2")
        if python2
        else Path("env")
    )
    if not runtime.is_dir():
        return False, f"private Python {'2' if python2 else '3'} runtime unavailable"
    interpreter = "/opt/python2/bin/python2.7" if python2 else "/opt/venv/bin/python"
    target = "/opt/python2" if python2 else "/opt/venv"
    with tempfile.TemporaryDirectory(prefix="sart-python-parse-") as temp:
        result = run_isolated(
            [
                interpreter,
                "-c",
                "import sys; compile(sys.stdin.read(), '<xcodeeval-candidate>', 'exec')",
            ],
            assets=Path(temp),
            runtime_mounts={runtime: target},
            stdin=source.encode("utf-8"),
            limits=SandboxLimits(wall_seconds=5, cpu_seconds=3),
        )
    message = result.stderr.decode("utf-8", errors="replace")[:500]
    return result.returncode == 0, message


def _sanitize_cpp(source: str) -> str:
    """Apply the pinned ExecEval C/C++ compatibility sanitizer before syntax checks."""
    repo = Path("content/tools/execeval/official_repo/execution_engine").resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from settings import sanitize_c_cpp

    return sanitize_c_cpp(source)


def _cpp_compiler_parse_success(source: str, runtime_label: str) -> tuple[bool, str]:
    source = _sanitize_cpp(source)
    standard_match = re.search(r"C\+\+(0x|11|14|17|20)", runtime_label)
    standard = standard_match.group(1) if standard_match else None
    standard = {"0x": "c++0x"}.get(standard, f"c++{standard}" if standard else None)
    compiler = GCC12 / "bin/g++"
    if not compiler.is_file():
        return False, "cluster GCC 12.3 module unavailable"
    command = [str(compiler), "-fsyntax-only", "-w", "-DONLINE_JUDGE"]
    if standard:
        command.append(f"-std={standard}")
    command.extend(("-include", "math.h", "-include", "limits.h", "/assets/candidate.cpp"))
    with tempfile.TemporaryDirectory(prefix="sart-cpp-parse-") as temp:
        assets = Path(temp)
        (assets / "candidate.cpp").write_text(source, encoding="utf-8")
        result = run_isolated(
            command,
            assets=assets,
            environment={
                "PATH": f"{GCC12 / 'bin'}:/usr/bin:/bin",
                "LD_LIBRARY_PATH": f"{GCC12 / 'lib64'}:{GCC12 / 'lib'}",
            },
            limits=SandboxLimits(wall_seconds=10, cpu_seconds=8),
            seccomp=False,
        )
    message = result.stderr.decode("utf-8", errors="replace")[:500]
    return result.returncode == 0, message


def parse_candidate(source: str, language: str, runtime_label: str) -> tuple[bool, str, str]:
    if language == "Python":
        python2 = "2" in runtime_label
        if not python2:
            tree_result = parse_program(source, language)
            if tree_result.valid:
                return True, "tree_sitter", ""
            valid, diagnostic = _python_compile_success(source, python2=False)
            return (
                valid,
                "cpython3-compile-fallback" if valid else "rejected",
                diagnostic or f"tree_sitter_errors={tree_result.error_count}",
            )
        valid, diagnostic = python2_parse_success(source)
        if valid:
            return True, "lib2to3-python2", ""
        valid, compiler_diagnostic = _python_compile_success(source, python2=True)
        return (
            valid,
            "cpython2-compile-fallback" if valid else "rejected",
            compiler_diagnostic or diagnostic,
        )
    result = parse_program(source, language)
    if result.valid:
        return True, "tree_sitter", ""
    if language == "C++":
        valid, diagnostic = _cpp_compiler_parse_success(source, runtime_label)
        return (
            valid,
            "execEval-compiler-parse-fallback" if valid else "rejected",
            diagnostic or f"tree_sitter_errors={result.error_count}",
        )
    return False, "rejected", f"tree_sitter_errors={result.error_count}"


def load_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for language in LANGUAGES:
        path = root / "code_translation/test" / f"{language}.jsonl"
        result[language] = [json.loads(line) for line in path.open(encoding="utf-8")]
    return result


def select_references(root: Path, tokenizer_path: Path) -> dict[str, Any]:
    """Select one parse-valid accepted implementation per problem/language."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    all_rows = load_rows(root)
    test_db = json.loads((root / "unittest_db.json").read_text(encoding="utf-8"))
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_pool: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audit = {}
    for language, rows in all_rows.items():
        by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
        reasons: dict[str, int] = defaultdict(int)
        parser_paths: dict[str, int] = defaultdict(int)
        rejected_examples: list[dict[str, str]] = []
        accepted = 0
        parsed = 0
        for row in rows:
            if row.get("exec_outcome") != "PASSED":
                reasons["not_upstream_accepted"] += 1
                continue
            accepted += 1
            source = row["source_code"]
            valid, parse_method, parse_error = parse_candidate(source, language, row["lang"])
            if not valid:
                reasons["parser_rejected"] += 1
                if len(rejected_examples) < 20:
                    rejected_examples.append(
                        {
                            "src_uid": row["src_uid"],
                            "code_uid": row["code_uid"],
                            "reason": parse_error,
                        }
                    )
                continue
            parsed += 1
            parser_paths[parse_method] += 1
            source_tokens = len(tokenizer.encode(source, add_special_tokens=False))
            candidate = {
                "src_uid": row["src_uid"],
                "code_uid": row["code_uid"],
                "language": language,
                "runtime_label": row["lang"],
                "source_code": source,
                "source_sha256": sha256_text(source),
                "token_count": source_tokens,
                "parse_method": parse_method,
            }
            by_problem[row["src_uid"]].append(candidate)
        selected[language] = {}
        candidate_pool[language] = {}
        for src_uid, candidates in by_problem.items():
            candidates.sort(key=lambda item: (item["token_count"], item["code_uid"]))
            median = (len(candidates) - 1) / 2
            median_length = candidates[int(median)]["token_count"]
            selected_candidate = min(
                candidates,
                key=lambda item: (abs(item["token_count"] - median_length), item["code_uid"]),
            )
            tests = test_db.get(src_uid)
            if not tests:
                reasons["missing_hidden_tests"] += 1
                continue
            candidate_pool[language][src_uid] = sorted(
                candidates,
                key=lambda item: (
                    abs(item["token_count"] - median_length),
                    item["code_uid"],
                ),
            )
            selected[language][src_uid] = {
                **selected_candidate,
                "candidate_count": len(candidates),
                "median_token_count": median_length,
                "test_count": len(tests),
                "raw_test_sha256": hashlib.sha256(
                    json.dumps(tests, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "tests": tests,
            }
        audit[language] = {
            "raw_rows": len(rows),
            "accepted_rows": accepted,
            "parse_valid_rows": parsed,
            "selected_problem_count": len(selected[language]),
            "parser_methods": dict(sorted(parser_paths.items())),
            "selection_failures": dict(sorted(reasons.items())),
            "parser_rejection_examples": rejected_examples,
        }

    ids = {language: set(rows) for language, rows in selected.items()}
    intersections = {
        "common_core_src_uids": sorted(set.intersection(*(ids[lang] for lang in LANGUAGES))),
        "pairwise_src_uids": {},
    }
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            intersections["pairwise_src_uids"][f"{source}-to-{target}"] = sorted(
                ids[source] & ids[target]
            )
    source_hashes = {}
    for path in [
        root / "code_translation/test/Python.jsonl",
        root / "code_translation/test/Java.jsonl",
        root / "code_translation/test/C++.jsonl",
        root / "unittest_db.json",
    ]:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "status": "PREMODEL_REFERENCE_SELECTION; canonical execution pending",
        "dataset_revision": "aa09386604075a3e80bcdb315dd68bcc117dedc0",
        "selection_rule": (
            "upstream exec_outcome PASSED; language parser valid; closest to lower median "
            "Qwen2.5-Coder-7B source token count; tie by ascending stable code_uid"
        ),
        "tokenizer": TOKENIZER,
        "runtime_labels": {
            lang: next(iter(selected[lang].values()), {}).get("runtime_label") for lang in LANGUAGES
        },
        "language_audit": audit,
        "source_artifact_sha256": source_hashes,
        "intersections": intersections,
        "selected": selected,
        "candidate_pool": candidate_pool,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("content/datasets/xcodeeval/aa09386604075a3e80bcdb315dd68bcc117dedc0"),
    )
    parser.add_argument("--tokenizer", type=Path, default=Path(TOKENIZER))
    parser.add_argument(
        "--output", type=Path, default=Path("data/manifests/xcodeeval_reference_selection.json")
    )
    args = parser.parse_args()
    manifest = select_references(args.dataset_root, args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "audit": manifest["language_audit"],
                "common_core_raw_selected": len(manifest["intersections"]["common_core_src_uids"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
