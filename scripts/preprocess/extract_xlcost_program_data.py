#!/usr/bin/env python3
"""Extract only the three required program-level XLCoST language pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

PAIR_DIRS = ("Java-Python", "Java-C++", "C++-Python")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def selected(name: str) -> bool:
    parts = PurePosixPath(name).parts
    if len(parts) != 5 or parts[:3] != ("XLCoST_data", "generation", "pair_data_tok_full"):
        return False
    pair, filename = parts[3], parts[4]
    return (
        pair in PAIR_DIRS
        and filename.split("-", 1)[0] in SPLITS
        and ("-map.jsonl" in filename or "-tok." in filename)
    )


def extract(archive_path: Path, output_dir: Path) -> None:
    sidecar_path = archive_path.with_suffix(archive_path.suffix + ".manifest.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not sidecar.get("archive_valid_zip") or sha256(archive_path) != sidecar.get("sha256"):
        raise ValueError("XLCoST archive failed its recorded checksum/ZIP validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.json"
    manifest: dict[str, object] = {
        "source_archive_sha256": sidecar["sha256"],
        "selected_program_level_pairs": list(PAIR_DIRS),
        "status": "IN_PROGRESS",
        "files": {},
    }
    extracted: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            (info for info in archive.infolist() if selected(info.filename)),
            key=lambda info: info.filename,
        )
        for info in members:
            relative = PurePosixPath(*PurePosixPath(info.filename).parts[3:])
            target = output_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                digest = sha256(target)
            else:
                temporary = target.with_name(f".{target.name}.partial")
                with archive.open(info) as source, temporary.open("wb") as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                if temporary.stat().st_size != info.file_size:
                    raise ValueError(f"Extracted size mismatch for {info.filename}")
                digest = sha256(temporary)
                temporary.replace(target)
            extracted[relative.as_posix()] = {"size_bytes": target.stat().st_size, "sha256": digest}
            manifest["files"] = extracted
            atomic_json(manifest_path, manifest)
            print(f"verified {relative.as_posix()} ({target.stat().st_size} bytes)", flush=True)
    expected = len(PAIR_DIRS) * len(SPLITS) * 4
    if len(extracted) != expected:
        raise ValueError(f"Expected {expected} extracted files, found {len(extracted)}")
    manifest["expected_file_count"] = expected
    manifest["status"] = "COMPLETE"
    manifest["files"] = extracted
    atomic_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("content/datasets/xlcost/raw/XLCoST_data.zip")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("content/datasets/xlcost/program_level")
    )
    args = parser.parse_args()
    extract(args.archive, args.output)


if __name__ == "__main__":
    main()
