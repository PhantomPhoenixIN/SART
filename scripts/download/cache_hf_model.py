#!/usr/bin/env python3
"""Download an exact model revision into Hugging Face's configured cache."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# HF's default 10-second read timeout is too short for large, range-resumed blobs
# on some cluster egress paths. Respect an explicit caller override.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    attempt = 0
    while True:
        attempt += 1
        try:
            snapshot = snapshot_download(
                repo_id=args.repo_id,
                revision=args.revision,
                repo_type="model",
            )
            print(f"Verified cached snapshot directory: {Path(snapshot).resolve()}", flush=True)
            return
        except Exception as exc:
            delay = min(300, 30 * (2 ** min(attempt - 1, 4)))
            print(
                f"snapshot download attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {delay}s",
                flush=True,
            )
            time.sleep(delay)


if __name__ == "__main__":
    main()
