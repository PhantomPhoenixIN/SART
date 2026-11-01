#!/usr/bin/env python3
"""Copy one exact Hugging Face cache snapshot into the project with hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import scan_cache_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_snapshot(repo_id: str, revision: str, destination: Path) -> None:
    cache = scan_cache_dir()
    repo = next(
        (item for item in cache.repos if item.repo_type == "model" and item.repo_id == repo_id),
        None,
    )
    if repo is None:
        raise RuntimeError(f"Model {repo_id!r} is not present in the detected HF cache")
    snapshot = next(
        (item for item in repo.revisions if item.commit_hash == revision),
        None,
    )
    if snapshot is None:
        revisions = ", ".join(sorted(item.commit_hash for item in repo.revisions))
        raise RuntimeError(f"Pinned revision missing from cache. Cached revisions: {revisions}")

    # snapshot_download can leave a revision directory with only some large Xet
    # blobs completed. Treat the safetensors index, rather than the presence of
    # any single weight file, as the authority for model-weight completeness.
    cached_by_name = {item.file_name: item for item in snapshot.files}
    index_file = cached_by_name.get("model.safetensors.index.json")
    if index_file is not None:
        index = json.loads(Path(index_file.file_path).read_text(encoding="utf-8"))
        required_shards = set(index.get("weight_map", {}).values())
        missing_shards = sorted(required_shards - set(cached_by_name))
        if missing_shards:
            raise RuntimeError(
                "Pinned HF snapshot is incomplete; safetensors index references "
                f"missing shard(s): {', '.join(missing_shards)}"
            )

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "artifact_manifest.json"
    manifest: dict[str, object] = {
        "repo_id": repo_id,
        "revision": revision,
        "expected_file_count": len(snapshot.files),
        "status": "IN_PROGRESS",
        "files": {},
    }
    copied: dict[str, dict[str, object]] = {}
    for cached_file in sorted(snapshot.files, key=lambda item: item.file_name):
        source = Path(cached_file.file_path)
        target = destination / cached_file.file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise RuntimeError(f"Cached snapshot file is missing: {source}")
        digest = sha256_file(source)
        if target.exists() and target.stat().st_size == source.stat().st_size:
            if sha256_file(target) != digest:
                raise RuntimeError(f"Existing destination has wrong content: {target}")
        else:
            temporary = target.with_name(f".{target.name}.partial")
            with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if sha256_file(temporary) != digest:
                raise RuntimeError(f"Copy checksum mismatch for {target}")
            temporary.replace(target)
        copied[cached_file.file_name] = {
            "size_bytes": target.stat().st_size,
            "sha256": digest,
        }
        manifest["files"] = copied
        atomic_json(manifest_path, manifest)
        print(f"copied/verified {cached_file.file_name} ({target.stat().st_size} bytes)", flush=True)
    copied_shards = required_shards if index_file is not None else set()
    missing_copied_shards = sorted(
        name for name in copied_shards
        if name not in copied or not (destination / name).is_file()
    )
    if missing_copied_shards:
        raise RuntimeError(
            "Refusing COMPLETE artifact manifest; copied model is missing indexed "
            f"shard(s): {', '.join(missing_copied_shards)}"
        )
    manifest["status"] = "COMPLETE"
    manifest["files"] = copied
    atomic_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    copy_snapshot(args.repo_id, args.revision, args.destination)
