#!/usr/bin/env python3
"""Wait for the pinned Qwen-14B cache download, then copy and stage its runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Qwen/Qwen2.5-Coder-14B"
REVISION = "f2ad5164aade432d6d56c24bb71589184d5d613d"
DESTINATION = (
    "content/models/qwen2.5-coder-14b/f2ad5164aade432d6d56c24bb71589184d5d613d"
)
DOWNLOAD_COMMAND = "scripts/download/cache_hf_model.py --repo-id Qwen/Qwen2.5-Coder-14B"
DOWNLOAD_LOG = ROOT / "logs/scaling_family/qwen14b_resume_download.log"


def process_command(pid: int) -> str:
    """Read the full command line; `ps -o args` can truncate long absolute paths."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}\n")
        stream.flush()


def stage_after_download(download_pid: int, poll_seconds: int, log_path: Path) -> None:
    while True:
        log(log_path, f"watching download pid={download_pid} revision={REVISION}")
        command = process_command(download_pid)
        while DOWNLOAD_COMMAND in command:
            time.sleep(poll_seconds)
            command = process_command(download_pid)

        from huggingface_hub.constants import HF_HUB_CACHE

        snapshot = (
            Path(HF_HUB_CACHE)
            / "models--Qwen--Qwen2.5-Coder-14B"
            / "snapshots"
            / REVISION
        )
        index_path = snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        required_shards = set(index.get("weight_map", {}).values())
        missing_shards = sorted(
            name for name in required_shards if not (snapshot / name).is_file()
        )
        if not (snapshot / "config.json").is_file() or not index_path.is_file() or missing_shards:
            log(
                log_path,
                f"download pid={download_pid} ended with incomplete snapshot; "
                f"missing shards={missing_shards or ['model index/config']}; restarting pinned download",
            )
            DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
            with DOWNLOAD_LOG.open("a", encoding="utf-8") as output:
                process = subprocess.Popen(
                    [
                        str(ROOT / "env/bin/python"),
                        str(ROOT / "scripts/download/cache_hf_model.py"),
                        "--repo-id", REPO_ID,
                        "--revision", REVISION,
                    ],
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            download_pid = process.pid
            log(log_path, f"restarted pinned Qwen14B download pid={download_pid}")
            time.sleep(poll_seconds)
            continue

        try:
            log(log_path, f"download exited; copying verified snapshot from {snapshot}")
            copy_result = subprocess.run(
                [
                    str(ROOT / "env/bin/python"),
                    str(ROOT / "scripts/download/copy_cached_hf_snapshot.py"),
                    "--repo-id", REPO_ID,
                    "--revision", REVISION,
                    "--destination", DESTINATION,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            log(log_path, copy_result.stdout.strip())
            stage_result = subprocess.run(
                [
                    str(ROOT / "env/bin/python"),
                    str(ROOT / "scripts/train/prepare_scaling_family_runs.py"),
                    "--backbone", "qwen14b",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            log(log_path, stage_result.stdout.strip())
            return
        except BaseException as exc:
            log(log_path, f"staging failed: {type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--log", type=Path, default=Path("logs/scaling_family/qwen14b_staging.log")
    )
    args = parser.parse_args()
    if args.download_pid < 1 or args.poll_seconds < 10:
        parser.error("download-pid must be positive and poll-seconds at least 10")
    stage_after_download(args.download_pid, args.poll_seconds, args.log)


if __name__ == "__main__":
    main()
