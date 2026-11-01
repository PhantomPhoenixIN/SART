#!/usr/bin/env python3
"""Remove XLCoST training problems overlapping frozen external evaluation sources.

This adapter builds a temporary, normalized three-split corpus and delegates matching
to the project's frozen exact/lexical/identifier/AST/MinHash leakage implementation.
External examples are protected; only complete XLCoST training problem IDs are removed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
import tokenize
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess.audit_xlcost_leakage import audit  # noqa: E402
from src.parsing.tree_sitter_adapter import parse_tree  # noqa: E402


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _atomic_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_python_detokenizer(root: Path):
    """Load the frozen upstream TransCoder Python detokenizer without clang imports."""
    source = (
        root / "content/datasets/transcoder-test/official_repo/preprocessing/src/code_tokenizer.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "detokenize_python"
    )
    namespace = {"tokenize": tokenize, "BytesIO": BytesIO}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "official-detokenizer", "exec"),
        namespace,
    )
    return namespace["detokenize_python"]


def _put(out: list[dict[str, Any]], uid: str, language: str, code: str, source: str) -> bool:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    try:
        root = parse_tree(code, language).root_node
        if root.has_error:
            return False
    except Exception:
        return False
    digest = hashlib.sha256(code.encode()).hexdigest()
    out.append(
        {"problem_id": uid, "language": language, "code": code, "sha256": digest, "source": source}
    )
    return True


def _external_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    programs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    selection = json.loads((root / "data/manifests/xcodeeval_reference_selection.json").read_text())
    for lang in ("Python", "Java", "C++"):
        candidates = selection["candidate_pool"][lang]
        before = len(programs)
        failed = 0
        for uid, candidate_list in candidates.items():
            for candidate in candidate_list:
                failed += not _put(
                    programs,
                    f"xcodeeval:{uid}:{candidate['code_uid']}",
                    lang,
                    candidate["source_code"],
                    "xCodeEval",
                )
        counts[f"xCodeEval/{lang}"] = len(programs) - before
        rejected[f"xCodeEval/{lang}"] = failed

    hroot = root / "content/datasets/humanevalx/62c78627f3072a1454fa0cb0184737cafe5e4198/data"
    for lang, folder in (("Python", "python"), ("Java", "java"), ("C++", "cpp")):
        before = len(programs)
        failed = 0
        for row in _read_jsonl(hroot / folder / "data/humaneval.jsonl"):
            # The executable canonical implementation is the complete prompt plus solution.
            failed += not _put(
                programs,
                f"humanevalx:{row['task_id']}",
                lang,
                row["prompt"] + row["canonical_solution"],
                "HumanEval-X",
            )
        counts[f"HumanEval-X/{lang}"] = len(programs) - before
        rejected[f"HumanEval-X/{lang}"] = failed

    transcoder = root / (
        "content/datasets/transcoder-test/1c21887dc55a88ea6535e669952d419256fbc24a/raw/"
        "TransCoder_tokenized_test_set_functions"
    )
    detokenize_python = _official_python_detokenizer(root)
    for file_lang, lang in (("python", "Python"), ("java", "Java"), ("cpp", "C++")):
        before = len(programs)
        failed = 0
        path = transcoder / f"test.{file_lang}.shuf.test.tok"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if " | " not in line:
                    continue
                uid, code = line.split(" | ", 1)
                if lang == "Python":
                    code = detokenize_python(code)
                # Tokenized functions preserve semantic token strings; spaced reconstruction is
                # conservative and often parseable. Parser failures are counted and skipped.
                failed += not _put(programs, f"transcoder:{uid}", lang, code, "TransCoder-Test")
        counts[f"TransCoder-Test/{lang}"] = len(programs) - before
        rejected[f"TransCoder-Test/{lang}"] = failed

    avatar = root / "content/datasets/avatar_tc/official_repo/AVATAR-TC"
    ids = (avatar / "test.java-python.id").read_text(encoding="utf-8").splitlines()
    for lang, suffix in (("Java", "java"), ("Python", "python")):
        before = len(programs)
        failed = 0
        lines = (avatar / f"test.java-python.{suffix}").read_text(encoding="utf-8").splitlines()
        if len(lines) != len(ids):
            raise ValueError("AVATAR-TC test code and IDs are not aligned")
        for uid, code in zip(ids, lines, strict=True):
            if lang == "Python":
                code = (
                    code.replace(" NEW_LINE", "\n")
                    .replace(" INDENT", "\n    ")
                    .replace(" DEDENT", "")
                )
            else:
                code = code.replace(" NEW_LINE", "\n")
            failed += not _put(programs, f"avatar:{uid}", lang, code, "AVATAR-TC")
        counts[f"AVATAR-TC/{lang}"] = len(programs) - before
        rejected[f"AVATAR-TC/{lang}"] = failed
    counts["parser_rejections"] = rejected
    return programs, counts


def run(root: Path, full_root: Path, output_root: Path, report_path: Path) -> dict[str, Any]:
    external, parsed_counts = _external_sources(root)
    with tempfile.TemporaryDirectory(prefix="sart-external-leak-") as tmp_name:
        tmp = Path(tmp_name)
        for split in ("train", "validation", "test"):
            (tmp / split).mkdir()
        train_path = tmp / "train" / "xlcost.jsonl"
        unique: dict[tuple[str, str, str], dict[str, Any]] = {}
        for shard in sorted((full_root / "train").glob("*/*.jsonl")):
            for row in _read_jsonl(shard):
                for side in ("source", "target"):
                    lang, code = row[f"{side}_language"], row[side]
                    digest = hashlib.sha256(code.encode()).hexdigest()
                    unique.setdefault(
                        (row["problem_id"], lang, digest),
                        {
                            "problem_id": row["problem_id"],
                            "language": lang,
                            "code": code,
                            "sha256": digest,
                            "source": "XLCoST",
                        },
                    )
        _write_synthetic(train_path, unique.values())
        _write_synthetic(tmp / "validation" / "external.jsonl", external)
        _write_synthetic(tmp / "test" / "empty.jsonl", [])
        (tmp / "manifest.json").write_text("{}\n", encoding="utf-8")
        # Adapter layout expected by the shared audit: one code placed in both sides.
        _convert_synthetic_layout(tmp)
        matching_report = tmp / "match_report.json"
        filtered = tmp / "filtered"
        result = audit(tmp, filtered, matching_report)
        dropped = set(result["dropped_training_problem_ids"])
        output_files: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            output_files[split] = {}
            for source in sorted((full_root / split).glob("*/*.jsonl")):
                target = output_root / split / source.relative_to(full_root / split)
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_suffix(target.suffix + ".tmp")
                kept = removed = 0
                with (
                    source.open(encoding="utf-8") as incoming,
                    temp.open("w", encoding="utf-8") as outgoing,
                ):
                    for line in incoming:
                        row = json.loads(line)
                        if split == "train" and row["problem_id"] in dropped:
                            removed += 1
                        else:
                            outgoing.write(line)
                            kept += 1
                    outgoing.flush()
                temp.replace(target)
                output_files[split][str(target.relative_to(output_root))] = {
                    "kept": kept,
                    "removed": removed,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
        report = {
            "status": "COMPLETE",
            "external_parsed_program_counts": parsed_counts,
            "external_sources": ["xCodeEval", "TransCoder-Test", "HumanEval-X", "AVATAR-TC"],
            "fingerprint_rules": result["configuration"],
            "dropped_training_problem_count": len(dropped),
            "dropped_training_problem_ids": sorted(dropped),
            "match_counts_by_rule": result["match_counts_by_rule"],
            "matches": result["matches"],
            "filtered_structural_full": output_files,
            "input_manifests": _input_manifest_hashes(root, full_root),
            "note": (
                "Parse-failing external records remain in the raw benchmarks but were omitted "
                "from source fingerprinting; parser rejections are counted above."
            ),
        }
        report_hash = _atomic_json(report_path, report)
        _atomic_json(
            output_root / "manifest.json",
            {
                "schema_version": 1,
                "status": "COMPLETE",
                "source_manifest_sha256": hashlib.sha256(
                    (full_root / "manifest.json").read_bytes()
                ).hexdigest(),
                "external_leakage_report": str(report_path),
                "external_leakage_report_sha256": report_hash,
                "dropped_training_problem_count": len(dropped),
                "dropped_training_problem_ids": sorted(dropped),
                "files": output_files,
            },
        )
        return report


def _input_manifest_hashes(root: Path, full_root: Path) -> dict[str, str]:
    paths = {
        "XLCoST_structural_full": full_root / "manifest.json",
        "xCodeEval": root
        / (
            "content/datasets/xcodeeval/aa09386604075a3e80bcdb315dd68bcc117dedc0/"
            "artifact_manifest.json"
        ),
        "xCodeEval_selection": root / "data/manifests/xcodeeval_reference_selection.json",
        "HumanEval-X": root
        / (
            "content/datasets/humanevalx/62c78627f3072a1454fa0cb0184737cafe5e4198/"
            "artifact_manifest.json"
        ),
        "TransCoder-Test": root
        / (
            "content/datasets/transcoder-test/1c21887dc55a88ea6535e669952d419256fbc24a/"
            "artifact_manifest.json"
        ),
        "AVATAR-TC": root / "content/datasets/avatar_tc/artifact_manifest.json",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if path.exists()
    }


def finalize_existing(report_path: Path, output_root: Path, root: Path, full_root: Path) -> None:
    """Validate completed shards and add provenance/manifest after a successful audit."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for split_files in report["filtered_structural_full"].values():
        for relative, metadata in split_files.items():
            path = output_root / relative
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
            ):
                raise ValueError(f"Filtered shard failed integrity validation: {path}")
    report["input_manifests"] = _input_manifest_hashes(root, full_root)
    report["xlcost_source_manifest_sha256"] = hashlib.sha256(
        (full_root / "manifest.json").read_bytes()
    ).hexdigest()
    report_hash = _atomic_json(report_path, report)
    _atomic_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "source_manifest_sha256": report["xlcost_source_manifest_sha256"],
            "external_leakage_report": str(report_path),
            "external_leakage_report_sha256": report_hash,
            "dropped_training_problem_count": report["dropped_training_problem_count"],
            "dropped_training_problem_ids": report["dropped_training_problem_ids"],
            "files": report["filtered_structural_full"],
        },
    )


def _write_synthetic(path: Path, records: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _convert_synthetic_layout(root: Path) -> None:
    """Convert normalized single-code rows to the shared audit's source/target schema."""
    for split in ("train", "validation", "test"):
        for path in (root / split).glob("*.jsonl"):
            temp = path.with_suffix(".converted")
            with path.open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
                for line in source:
                    row = json.loads(line)
                    lang = row["language"]
                    target.write(
                        json.dumps(
                            {
                                "problem_id": row["problem_id"],
                                "source_language": lang,
                                "target_language": lang,
                                "source": row["code"],
                                "target": row["code"],
                                "source_sha256": row["sha256"],
                                "target_sha256": row["sha256"],
                            }
                        )
                        + "\n"
                    )
            temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--full-root", type=Path, default=Path("content/datasets/xlcost/structural_full")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("content/datasets/xlcost/structural_external_clean"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/data/external_leakage_report.json")
    )
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing(args.report, args.output_root, args.root, args.full_root)
        print(f"validated and finalized {args.report}")
        return
    print(json.dumps(run(args.root, args.full_root, args.output_root, args.report), indent=2))


if __name__ == "__main__":
    main()
