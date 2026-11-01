#!/usr/bin/env python3
"""Resume-safe Joern CFG/DFG extraction from finalized Tree-sitter shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.regions.joern_edges import map_joern_edges_by_file

LANGUAGE_FILES = {"Python": ".py", "Java": ".java", "C++": ".cpp"}
FRONTENDS = {"Python": "pysrc2cpg", "Java": "javasrc2cpg", "C++": "c2cpg.sh"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return sha256(path)


def collect_sources(ast_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect unique language/source hashes and their canonical region inventory."""
    sources: dict[str, dict[str, dict[str, Any]]] = {language: {} for language in LANGUAGE_FILES}
    for path in sorted(ast_root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                language = row["source_language"]
                digest = row["source_sha256"]
                value = sources[language].setdefault(
                    digest,
                    {
                        "source": row["source"],
                        "regions": row["structure"]["regions"],
                    },
                )
                if (
                    value["source"] != row["source"]
                    or value["regions"] != row["structure"]["regions"]
                ):
                    raise ValueError(f"Inconsistent source/region record for {language}:{digest}")
    return sources


def prepare_source_files(
    sources: dict[str, dict[str, dict[str, Any]]],
    source_root: Path,
) -> dict[str, list[str]]:
    """Write content-addressed files with original source bytes and return hashes."""
    result: dict[str, list[str]] = {}
    for language, records in sources.items():
        result[language] = sorted(records)
        for digest, record in records.items():
            code = record["source"]
            actual = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if actual != digest:
                raise ValueError(f"Source checksum mismatch for {language}:{digest}")
            path = source_root / language.replace("+", "p") / f"{digest}{LANGUAGE_FILES[language]}"
            if path.is_file() and sha256(path) == digest:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(code.encode("utf-8"))
            os.replace(temporary, path)
    return result


def _valid_shard(path: Path, meta: dict[str, Any]) -> bool:
    return (
        not meta.get("has_failures", False)
        and path.is_file()
        and path.stat().st_size == meta["bytes"]
        and sha256(path) == meta["sha256"]
    )


def _run_shard(
    language: str,
    shard_index: int,
    selected: list[str],
    sources: dict[str, dict[str, dict[str, Any]]],
    source_root: Path,
    output_root: Path,
    joern_root: Path,
    java_env: dict[str, str],
    log_root: Path,
    keep_work: bool,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Process one isolated deterministic source shard (safe for worker threads)."""
    language_key = language.replace("+", "p")
    suffix = LANGUAGE_FILES[language]
    language_dir = source_root / language_key
    shard_key = f"{language}/{shard_index:05d}"
    output_path = output_root / language_key / f"shard_{shard_index:05d}.jsonl"
    shard_dir = output_root / "work" / language_key / f"shard_{shard_index:05d}"
    input_dir = shard_dir / "src"
    cpg_path = shard_dir / "cpg.bin"
    export_dir = shard_dir / "export"
    shutil.rmtree(shard_dir, ignore_errors=True)
    input_dir.mkdir(parents=True)
    file_regions: dict[str, list[dict[str, Any]]] = {}
    file_sources: dict[str, str] = {}
    hash_by_filename: dict[str, str] = {}
    for source_hash in selected:
        filename = f"{source_hash}{suffix}"
        shutil.copyfile(language_dir / filename, input_dir / filename)
        file_regions[filename] = sources[language][source_hash]["regions"]
        file_sources[filename] = sources[language][source_hash]["source"]
        hash_by_filename[filename] = source_hash
    frontend = joern_root / FRONTENDS[language]
    if language == "C++":
        frontend = joern_root / "c2cpg.sh"
    command = [
        str(frontend),
        str(input_dir),
        "--enable-file-content",
        "-o",
        str(cpg_path),
    ]
    log_path = log_root / language_key / f"shard_{shard_index:05d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                env=java_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
            export_command = [
                str(joern_root / "bin" / "joern-export"),
                "--repr",
                "all",
                "--format",
                "neo4jcsv",
                "--out",
                str(export_dir),
                str(cpg_path),
            ]
            subprocess.run(
                export_command,
                check=True,
                env=java_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        mapped = map_joern_edges_by_file(export_dir, file_regions, file_sources)
        rows = [
            {
                "source_language": language,
                "source_sha256": source_hash,
                **mapped[f"{source_hash}{suffix}"],
            }
            for source_hash in selected
        ]
    except Exception as exc:
        failure = {
            "shard": shard_key,
            "language": language,
            "source_sha256s": selected,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "log": str(log_path),
        }
        rows = [
            {
                "source_language": language,
                "source_sha256": source_hash,
                "available": False,
                "reason": "joern_shard_failed",
                "cfg_edges": [],
                "dfg_edges": [],
            }
            for source_hash in selected
        ]
    output_hash = write_jsonl(output_path, rows)
    meta = {
        "path": str(output_path),
        "records": len(rows),
        "bytes": output_path.stat().st_size,
        "sha256": output_hash,
        "log": str(log_path),
        "has_failures": failure is not None,
    }
    if not keep_work and failure is None:
        shutil.rmtree(shard_dir, ignore_errors=True)
    return shard_key, meta, failure


def run_extraction(
    ast_root: Path,
    source_root: Path,
    output_root: Path,
    joern_root: Path,
    java_home: Path,
    shard_size: int,
    keep_work: bool = False,
    workers: int = 1,
    log_root: Path = Path("logs/joern"),
) -> dict[str, Any]:
    """Run official Joern frontends and map exported CFG/DFG edges to AST regions."""
    sources = collect_sources(ast_root)
    source_keys = prepare_source_files(sources, source_root)
    config = {
        "schema_version": 1,
        "ast_manifest_sha256": sha256(ast_root / "manifest.json"),
        "source_count_by_language": {key: len(value) for key, value in source_keys.items()},
        "source_root": str(source_root.resolve()),
        "joern_root": str(joern_root.resolve()),
        "joern_release": "4.0.644",
        "joern_archive_sha256": "407358d2ccb506e4b3d8e674456fbc08a734e94add9f8d6a06136de2afae7110",
        "java_home": str(java_home.resolve()),
        "shard_size": shard_size,
        "frontend_map": FRONTENDS,
        "export": {"representation": "all", "format": "neo4jcsv"},
        "edge_mapping": "smallest enclosing non-PROGRAM Tree-sitter region by Joern source offsets",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest_path = output_root / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if previous and previous.get("configuration_hash") != config_hash:
        raise ValueError("Existing Joern manifest has a different configuration")
    completed: dict[str, Any] = previous.get("completed_shards", {})
    failures: list[dict[str, Any]] = previous.get("failures", [])
    output_root.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive")
    java_env = os.environ.copy()
    java_env.update(
        {
            "JAVA_HOME": str(java_home.resolve()),
            "PATH": f"{java_home.resolve() / 'bin'}:{os.environ['PATH']}",
            "JAVA_OPTS": os.environ.get("JAVA_OPTS", "-Xmx8g"),
        }
    )
    tasks = []
    for language, digests in source_keys.items():
        for shard_index, start in enumerate(range(0, len(digests), shard_size)):
            shard_key = f"{language}/{shard_index:05d}"
            output_path = (
                output_root / language.replace("+", "p") / f"shard_{shard_index:05d}.jsonl"
            )
            if shard_key not in completed or not _valid_shard(output_path, completed[shard_key]):
                tasks.append((language, shard_index, digests[start : start + shard_size]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _run_shard,
                language,
                shard_index,
                selected,
                sources,
                source_root,
                output_root,
                joern_root,
                java_env,
                log_root,
                keep_work,
            )
            for language, shard_index, selected in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            shard_key, metadata, failure = future.result()
            if failure is not None:
                failures.append(failure)
            completed[shard_key] = metadata
            write_json(
                manifest_path,
                {
                    **config,
                    "configuration_hash": config_hash,
                    "status": "RUNNING",
                    "completed_shards": completed,
                    "failures": failures,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"completed {shard_key} ({metadata['records']} sources)", flush=True)

    manifest = {
        **config,
        "configuration_hash": config_hash,
        "status": "COMPLETE_WITH_FAILURES" if failures else "COMPLETE",
        "completed_shards": completed,
        "failures": failures,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ast-root", type=Path, default=Path("content/datasets/xlcost/structural_ast_qwen")
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("content/datasets/xlcost/joern_sources")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("content/datasets/xlcost/joern_edges")
    )
    parser.add_argument(
        "--joern-root", type=Path, default=Path("content/tools/joern/v4.0.644/joern-cli")
    )
    parser.add_argument(
        "--java-home", type=Path, default=Path("content/tools/temurin/21.0.12.1+1/runtime")
    )
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-root", type=Path, default=Path("logs/joern"))
    args = parser.parse_args()
    result = run_extraction(**vars(args))
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_shards": len(result["completed_shards"]),
                "failures": len(result["failures"]),
                "source_count_by_language": result["source_count_by_language"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
