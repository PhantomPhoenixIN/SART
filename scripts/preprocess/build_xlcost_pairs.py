#!/usr/bin/env python3
"""Create syntax-filtered, aligned XLCoST direction records and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.data.xlcost_format import detokenize_xlcost
from src.parsing.tree_sitter_adapter import parse_program

PAIRS = {
    "Java-Python": ("Java", "Python"),
    "Java-C++": ("Java", "C++"),
    "C++-Python": ("C++", "Python"),
}
EXT = {"Python": ".py", "Java": ".java", "C++": ".cpp"}
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def _read(path: Path) -> list[str]:
    return [line.rstrip("\r\n") for line in path.open(encoding="utf-8")]


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def build(
    source_root: Path,
    output_root: Path,
    parser_fn: Callable[[str, str], Any] = parse_program,
) -> dict[str, Any]:
    source_manifest = json.loads((source_root / "extraction_manifest.json").read_text())
    if source_manifest.get("status") != "COMPLETE":
        raise ValueError("Refusing incomplete XLCoST extraction")
    result: dict[str, Any] = {
        "schema_version": 1,
        "source_archive_sha256": source_manifest["source_archive_sha256"],
        "parser": "Tree-sitter pinned language grammars; any syntax error rejects aligned pair",
        "pairing": "official aligned row only; at most one pair per problem and direction",
        "splits": {},
    }
    for split, folder_split in SPLITS.items():
        directional: dict[str, list[dict[str, Any]]] = {}
        counts: dict[str, dict[str, int]] = {}
        for pair_name, languages in PAIRS.items():
            folder = source_root / pair_name
            ids_a = _read(folder / f"{folder_split}-{languages[0]}-map.jsonl")
            ids_b = _read(folder / f"{folder_split}-{languages[1]}-map.jsonl")
            codes = {
                lang: _read(folder / f"{folder_split}-{pair_name}-tok{EXT[lang]}")
                for lang in languages
            }
            if ids_a != ids_b or any(len(ids_a) != len(codes[lang]) for lang in languages):
                raise ValueError(f"Unaligned XLCoST rows in {pair_name}/{split}")
            directional[f"{languages[0]}-to-{languages[1]}"] = []
            directional[f"{languages[1]}-to-{languages[0]}"] = []
            stats = {"aligned": len(ids_a), "eligible": 0, "excluded_parse": 0}
            exclusions: list[dict[str, Any]] = []
            for index, problem_id in enumerate(ids_a):
                programs = {
                    lang: detokenize_xlcost(codes[lang][index], lang) for lang in languages
                }
                parses = {lang: parser_fn(text, lang) for lang, text in programs.items()}
                if not all(parsed.valid for parsed in parses.values()):
                    stats["excluded_parse"] += 1
                    exclusions.append(
                        {
                            "problem_id": problem_id,
                            "reason": "aligned_pair_parse_failure",
                            "languages": {
                                lang: {
                                    "valid": parsed.valid,
                                    "error_count": parsed.error_count,
                                    "root_type": parsed.root_type,
                                }
                                for lang, parsed in parses.items()
                            },
                        }
                    )
                    continue
                stats["eligible"] += 1
                for source, target in (languages, languages[::-1]):
                    directional[f"{source}-to-{target}"].append(
                        {
                            "problem_id": problem_id,
                            "source_language": source,
                            "target_language": target,
                            "source": programs[source],
                            "target": programs[target],
                            "source_sha256": hashlib.sha256(
                                programs[source].encode("utf-8")
                            ).hexdigest(),
                            "target_sha256": hashlib.sha256(
                                programs[target].encode("utf-8")
                            ).hexdigest(),
                        }
                    )
            counts[pair_name] = {**stats, "exclusions": exclusions}
        split_hashes = {}
        for direction, rows in directional.items():
            split_hashes[direction] = {
                "records": len(rows),
                "sha256": _atomic_jsonl(output_root / split / f"{direction}.jsonl", rows),
            }
        result["splits"][split] = {"pairs": counts, "directions": split_hashes}
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("content/datasets/xlcost/program_level"))
    parser.add_argument("--output", type=Path, default=Path("content/datasets/xlcost/parsed_program_level"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
