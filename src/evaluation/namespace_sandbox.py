"""Run one untrusted benchmark process in a private rootless Linux namespace."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CLONE_FLAGS = ("--user", "--map-root-user", "--mount", "--pid", "--fork", "--net", "--ipc", "--uts")
MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_REMOUNT = 32
MS_BIND = 4096
MS_REC = 16384
MS_PRIVATE = 1 << 18


@dataclass(frozen=True)
class SandboxLimits:
    """Per-task limits; memory is virtual address space (RLIMIT_AS)."""

    wall_seconds: float = 10.0
    cpu_seconds: int = 8
    address_space_bytes: int = 2 * 1024**3
    file_size_bytes: int = 64 * 1024**2
    work_tmpfs_bytes: int = 128 * 1024**2
    output_bytes: int = 1024**2
    open_files: int = 64
    extra_process_slots: int = 64


@dataclass(frozen=True)
class SandboxResult:
    """Bounded result from one isolated process."""

    returncode: int | None
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    timed_out: bool
    output_limited: bool


def _libc_mount(
    source: str | None,
    target: str,
    fstype: str | None = None,
    flags: int = 0,
    data: str | None = None,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    mount = libc.mount
    mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    ]
    mount.restype = ctypes.c_int
    result = mount(
        source.encode() if source is not None else None,
        os.fsencode(target),
        fstype.encode() if fstype is not None else None,
        flags,
        data.encode() if data is not None else None,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _bind_readonly(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    _libc_mount(str(source), str(target), flags=MS_BIND | MS_REC)
    recursive = MS_REC if source.is_dir() else 0
    _libc_mount(None, str(target), flags=MS_BIND | MS_REMOUNT | MS_RDONLY | recursive)


def _limit_resources(limits: SandboxLimits, process_limit: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (limits.address_space_bytes, limits.address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))


def _uid_task_count(uid: int) -> int:
    """Count visible tasks for this UID to set RLIMIT_NPROC just above current use."""
    count = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            status = (entry / "status").read_text().splitlines()
            real_uid = int(next(line for line in status if line.startswith("Uid:")).split()[1])
            threads = int(next(line for line in status if line.startswith("Threads:")).split()[1])
            if real_uid == uid:
                count += threads
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    return count


def _inner(config_path: Path) -> int:
    config = json.loads(config_path.read_text())
    root = Path(config["root"])
    assets = Path(config["assets"])
    _libc_mount(None, "/", flags=MS_REC | MS_PRIVATE)

    for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/apps"):
        source = Path(path)
        if source.exists():
            _bind_readonly(source, root / path.lstrip("/"))

    for source, destination in config["runtime_mounts"].items():
        _bind_readonly(Path(source), root / destination.lstrip("/"))

    for directory in ("dev", "proc", "work", "assets", "opt"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _bind_readonly(assets, root / "assets")
    _libc_mount(
        "tmpfs",
        str(root / "work"),
        "tmpfs",
        MS_NOSUID | MS_NODEV,
        f"size={config['limits']['work_tmpfs_bytes']},mode=1777",
    )
    (root / "work/tmp").mkdir()
    (root / "tmp").symlink_to("/work/tmp")
    _libc_mount("tmpfs", str(root / "dev"), "tmpfs", MS_NOSUID, "size=8m,mode=755")
    for name in ("null", "zero", "random", "urandom"):
        target = root / "dev" / name
        target.touch()
        _libc_mount(f"/dev/{name}", str(target), flags=MS_BIND)
    (root / "dev/shm").mkdir()
    _libc_mount("tmpfs", str(root / "dev/shm"), "tmpfs", MS_NOSUID | MS_NODEV, "size=16m,mode=1777")
    _libc_mount("proc", str(root / "proc"), "proc", MS_NOSUID | MS_NODEV | MS_RDONLY)

    os.chroot(root)
    os.chdir("/work")
    if not shutil.which("/usr/bin/setpriv"):
        raise RuntimeError("setpriv is required in the isolated root")
    limits = SandboxLimits(**config["limits"])
    _limit_resources(limits, config["process_limit"])
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/work/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        **config["environment"],
    }
    os.environ.clear()
    os.environ.update(environment)
    command = list(config["command"])
    if config["seccomp"]:
        command.insert(0, "/opt/sart/seccomp_exec")
    command = [
        "/usr/bin/setpriv",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
        *command,
    ]
    os.execvpe(command[0], command, environment)
    return 127


def _collect_limited(
    process: subprocess.Popen[bytes], limit: int, deadline: float, stdin: bytes
) -> tuple[bytes, bytes, bool, bool]:
    import selectors

    selector = selectors.DefaultSelector()
    assert process.stdout is not None and process.stderr is not None
    for stream in (process.stdout, process.stderr):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, "read")
    assert process.stdin is not None
    os.set_blocking(process.stdin.fileno(), False)
    input_offset = 0
    if stdin:
        selector.register(process.stdin, selectors.EVENT_WRITE, "write")
    else:
        process.stdin.close()
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    exceeded = False
    timed_out = False
    while selector.get_map() or process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        for key, _ in selector.select(min(remaining, 0.2)):
            if key.data == "write":
                try:
                    input_offset += os.write(
                        key.fileobj.fileno(), stdin[input_offset : input_offset + 65536]
                    )
                except BrokenPipeError:
                    input_offset = len(stdin)
                if input_offset >= len(stdin):
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                continue
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            current = sum(len(buffer) for buffer in buffers.values())
            room = max(0, limit - current)
            buffers[key.fileobj].extend(chunk[:room])
            if len(chunk) > room:
                exceeded = True
                break
        if exceeded:
            break
    if timed_out or exceeded:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    selector.close()
    if process.stdin and not process.stdin.closed:
        process.stdin.close()
    return bytes(buffers[process.stdout]), bytes(buffers[process.stderr]), timed_out, exceeded


def _process_group_limit(limit: SandboxLimits) -> int:
    """Return a task limit that leaves room for the evaluator and compiler workers."""
    return _uid_task_count(os.getuid()) + limit.extra_process_slots


def run_isolated(
    command: Sequence[str],
    *,
    assets: Path,
    limits: SandboxLimits | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_mounts: Mapping[Path, str] | None = None,
    stdin: bytes = b"",
    seccomp: bool = True,
) -> SandboxResult:
    """Run a command after removing host files, capabilities and network access."""
    if not command:
        raise ValueError("command must not be empty")
    limits = limits or SandboxLimits()
    assets = assets.resolve(strict=True)
    if not assets.is_dir():
        raise ValueError("assets must be a directory")
    with tempfile.TemporaryDirectory(prefix="sart-sandbox-") as temporary:
        base = Path(temporary)
        root = base / "root"
        root.mkdir()
        config_path = base / "config.json"
        mounts = {
            Path(source).resolve(): target for source, target in (runtime_mounts or {}).items()
        }
        if seccomp:
            launcher = Path(__file__).resolve().parents[2] / "content/tools/sandbox/seccomp_exec"
            if not launcher.is_file():
                raise FileNotFoundError(f"Compile the seccomp launcher first: {launcher}")
            mounts[launcher.resolve()] = "/opt/sart/seccomp_exec"
        config_path.write_text(
            json.dumps(
                {
                    "root": str(root),
                    "assets": str(assets),
                    "runtime_mounts": {str(source): target for source, target in mounts.items()},
                    "limits": vars(limits),
                    "process_limit": _process_group_limit(limits),
                    "command": list(command),
                    "environment": dict(environment or {}),
                    "seccomp": seccomp,
                }
            )
        )
        started = time.monotonic()
        argv = [
            "unshare",
            *CLONE_FLAGS,
            sys.executable,
            str(Path(__file__).resolve()),
            "--inner",
            str(config_path),
        ]
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr, timed_out, output_limited = _collect_limited(
            process, limits.output_bytes, started + limits.wall_seconds, stdin
        )
        return SandboxResult(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            wall_seconds=time.monotonic() - started,
            timed_out=timed_out,
            output_limited=output_limited,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner", type=Path)
    args = parser.parse_args()
    if args.inner is None:
        parser.error("--inner is required for the namespace entry point")
    raise SystemExit(_inner(args.inner))


if __name__ == "__main__":
    main()
