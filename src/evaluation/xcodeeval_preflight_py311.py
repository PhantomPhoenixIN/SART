"""Run canonical xCodeEval references inside the isolated ExecEval-compatible runner."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation.namespace_sandbox import SandboxLimits, SandboxResult, run_isolated

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "src/evaluation/execeval_task_driver_py311.py"
PY2 = ROOT / "content/tools/python2.7.18/stage/opt/python2"
PYPY2 = ROOT / "content/tools/pypy/stage/pypy2.7-v7.3.9-linux64"
PYPY3 = ROOT / "content/tools/pypy/stage/pypy3.9-v7.3.9-linux64"
PYTHON3 = ROOT / "env"
PYTHON311 = ROOT / "content/tools/python3.11/stage/opt/python3"
JDK = ROOT / "content/tools/temurin/21.0.12.1+1/runtime"
GCC12 = Path("/apps/apps/gcc/gcc-12.3.0/gc")
SECCOMP = ROOT / "content/tools/sandbox/seccomp_exec"
EXECEVAL_ENGINE = ROOT / "content/tools/execeval/official_repo/execution_engine"


def _runtime_mounts(runtime_label: str) -> dict[Path, str]:
    if runtime_label == "Python 2":
        return {PY2: "/opt/python2"}
    if runtime_label == "PyPy 2":
        return {PYPY2: "/opt/pypy2"}
    if runtime_label == "Python 3":
        return {PYTHON311: "/opt/python311"}
    if runtime_label in {"PyPy 3", "PyPy 3-64"}:
        return {PYPY3: "/opt/pypy3"}
    if runtime_label.startswith("Java "):
        return {JDK: "/opt/jdk"}
    if runtime_label.startswith(("GNU C++", "MS C++", "Clang++")):
        return {}
    raise ValueError(f"Unsupported xCodeEval runtime label {runtime_label!r}")


def _exec_eval_sanitize(source: str) -> str:
    if str(EXECEVAL_ENGINE) not in sys.path:
        sys.path.insert(0, str(EXECEVAL_ENGINE))
    from settings import sanitize_c_cpp

    return sanitize_c_cpp(source)


def run_reference(
    source: str,
    language: str,
    runtime_label: str,
    tests: list[dict[str, Any]],
    *,
    repeats: int = 3,
    case_timeout_seconds: float = 6.0,
) -> tuple[dict[str, Any] | None, SandboxResult]:
    """Compile/test a selected canonical program without exposing test files to it."""
    if language not in {"Python", "Java", "C++"}:
        raise ValueError(f"Unsupported language {language!r}")
    if repeats < 1 or not tests:
        raise ValueError("repeats must be positive and tests must not be empty")
    with tempfile.TemporaryDirectory(prefix="sart-xcodeeval-ref-") as temporary:
        assets = Path(temporary)
        if language == "C++":
            source = _exec_eval_sanitize(source)
        (assets / "candidate.source").write_text(source, encoding="utf-8")
        (assets / "execeval_task_driver.py").write_bytes(DRIVER.read_bytes())
        request = {
            "language": language,
            "runtime_label": runtime_label,
            "tests": tests,
            "repeats": repeats,
            "case_timeout_seconds": case_timeout_seconds,
            "case_cpu_seconds": 2,
        }
        payload = json.dumps(request, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        task_wall = min(24 * 3600, 180 + len(tests) * repeats * case_timeout_seconds)
        limits = SandboxLimits(
            wall_seconds=task_wall,
            cpu_seconds=max(120, int(task_wall) + 1),
            address_space_bytes=4 * 1024**3,
            file_size_bytes=64 * 1024**2,
            work_tmpfs_bytes=128 * 1024**2,
            output_bytes=4 * 1024**2,
            open_files=128,
            extra_process_slots=64,
        )
        environment = {
            "PATH": ":".join(
                (
                    str(GCC12 / "bin"),
                    "/opt/jdk/bin",
                    "/opt/python311/bin",
                    "/opt/python2/bin",
                    "/opt/pypy2/bin",
                    "/opt/pypy3/bin",
                    "/opt/venv/bin",
                    "/usr/bin",
                    "/bin",
                )
            ),
            "LD_LIBRARY_PATH": f"{GCC12 / 'lib64'}:{GCC12 / 'lib'}",
            "PYTHONPYCACHEPREFIX": "/work/pycache",
        }
        result = run_isolated(
            ["/opt/venv/bin/python", "/assets/execeval_task_driver.py"],
            assets=assets,
            limits=limits,
            environment=environment,
            runtime_mounts=_runtime_mounts(runtime_label)
            | {
                PYTHON3: "/opt/venv",
                PYTHON311: "/opt/python311",
                SECCOMP: "/opt/sart/seccomp_exec",
            },
            stdin=payload,
            seccomp=False,
        )
    if result.returncode != 0 or result.timed_out or result.output_limited:
        return None, result
    try:
        response = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, result
    return response, result
