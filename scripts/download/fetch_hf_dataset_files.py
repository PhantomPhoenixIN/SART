#!/usr/bin/env python3
"""Fetch pinned public HF dataset files through the HF cache into SART."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def fetch_files(repo_id: str, revision: str, filenames: list[str], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "artifact_manifest.json"
    manifest: dict[str, object] = {
        "repo_id": repo_id,
        "revision": revision,
        "repo_type": "dataset",
        "status": "IN_PROGRESS",
        "files": {},
    }
    copied: dict[str, dict[str, object]] = {}
    for filename in filenames:
        cached_name = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
            )
        )
        digest = file_sha256(cached_name)
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == cached_name.stat().st_size:
            if file_sha256(target) != digest:
                raise RuntimeError(f"Existing project artifact checksum mismatch: {target}")
        else:
            temporary = target.with_name(f".{target.name}.partial")
            with cached_name.open("rb") as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if file_sha256(temporary) != digest:
                raise RuntimeError(f"Copied artifact checksum mismatch: {target}")
            temporary.replace(target)
        copied[filename] = {"size_bytes": target.stat().st_size, "sha256": digest}
        manifest["files"] = copied
        atomic_write_json(manifest_path, manifest)
        print(f"verified {filename}: {target.stat().st_size} bytes sha256={digest}", flush=True)
    manifest["status"] = "COMPLETE"
    manifest["files"] = copied
    atomic_write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--file", action="append", dest="filenames", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    fetch_files(arguments.repo_id, arguments.revision, arguments.filenames, arguments.destination)
