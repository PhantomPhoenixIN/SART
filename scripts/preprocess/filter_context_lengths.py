#!/usr/bin/env python3
"""Apply the frozen tokenizer context-length rule to every XLCoST split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from src.data.translation_protocol import serialize_prompt, serialize_target


def _atomic_filter(source: Path, target: Path, tokenizer, limit: int) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    kept = 0
    exclusions = []
    try:
        with source.open(encoding="utf-8") as incoming, os.fdopen(
            fd, "w", encoding="utf-8"
        ) as outgoing:
            for line in incoming:
                row = json.loads(line)
                prompt = serialize_prompt(
                    row["source_language"], row["target_language"], row["source"]
                )
                target_text = serialize_target(row["target"])
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                target_tokens = len(tokenizer.encode(target_text, add_special_tokens=False)) + 1
                combined = prompt_tokens + target_tokens
                if combined > limit:
                    exclusions.append(
                        {
                            "problem_id": row["problem_id"],
                            "prompt_tokens": prompt_tokens,
                            "target_tokens_including_eos": target_tokens,
                            "combined_tokens": combined,
                        }
                    )
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
        "input": kept + len(exclusions),
        "kept": kept,
        "excluded": len(exclusions),
        "exclusions": exclusions,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("content/datasets/xlcost/clean_program_level"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("content/datasets/xlcost/final_program_level"),
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--length-audit",
        type=Path,
        default=Path("reports/data/qwen_token_length_audit.json"),
    )
    args = parser.parse_args()
    audit = json.loads(args.length_audit.read_text())
    limit = int(audit["resolved"]["max_sequence_length"])
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    files = {}
    for split in ("train", "validation", "test"):
        files[split] = {}
        for source in sorted((args.input_root / split).glob("*.jsonl")):
            print(f"filtering {source}", flush=True)
            files[split][source.name] = _atomic_filter(
                source, args.output_root / split / source.name, tokenizer, limit
            )
    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_manifest_sha256": hashlib.sha256(
            (args.input_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "length_audit_sha256": hashlib.sha256(args.length_audit.read_bytes()).hexdigest(),
        "max_sequence_length": limit,
        "max_new_tokens": int(audit["resolved"]["max_new_tokens"]),
        "files": files,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps({"max_sequence_length": limit, "files": files}, indent=2))


if __name__ == "__main__":
    main()
