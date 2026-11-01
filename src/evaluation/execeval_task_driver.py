"""Compile one canonical xCodeEval reference and run hidden tests in the sandbox."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import resource
import selectors
import signal
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

WORK = Path("/work")
ASSETS = Path("/assets")
SECCOMP = "/opt/sart/seccomp_exec"
OUTPUT_LIMIT = 1024 * 1024


def _match_output(actual: str, expected_options: list[str]) -> bool:
    """Mirror ExecEval's stripped-line, then numeric-token output comparison."""

    def match_pair(expected: str, observed: str) -> bool:
        left = expected.strip().split("\n")
        right = observed.strip().split("\n")
        if len(left) == len(right) and all(
            a.strip() == b.strip() for a, b in zip(left, right, strict=True)
        ):
            return True
        tokens_left, tokens_right = expected.strip().split(), observed.strip().split()
        if len(tokens_left) != len(tokens_right):
            return False
        with localcontext() as context:
            context.prec = 50
            for a, b in zip(tokens_left, tokens_right, strict=True):
                try:
                    if abs(Decimal(a) - Decimal(b)) > Decimal("1e-12"):
                        return False
                except (InvalidOperation, ValueError):
                    yes_no = {"yes", "no", "true", "false"}
                    if a.lower() in yes_no:
                        a = a.lower()
                    if b.lower() in yes_no:
                        b = b.lower()
                    if a != b:
                        return False
        return True

    return any(match_pair(expected, actual) for expected in expected_options)


def _match_avatar_output(actual: str, expected_options: list[str]) -> bool:
    """Apply AVATAR's documented case/whitespace tolerance, retaining punctuation."""
    def normalize(value: str) -> str:
        return "".join(value.casefold().split())

    return any(normalize(actual) == normalize(expected) for expected in expected_options) or (
        _match_output(actual, expected_options)
    )


def _java_public_class_name(source: str) -> str | None:
    """Return the first Java class declaration for its required source filename."""
    match = re.search(
        r"\b(?:(?:public|protected|private|final|abstract|static)\s+)*class\s+(\w+)\b",
        source,
    )
    return match.group(1) if match else None


def _limits(cpu_seconds: int, address_space: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
    resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_bounded(
    command: list[str], stdin: bytes, *, timeout: float, cpu_seconds: int, address_space: int
) -> dict[str, Any]:
    """Run a child with bounded output, wall time and inherited resource limits."""
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=lambda: _limits(cpu_seconds, address_space),
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    streams = (proc.stdin, proc.stdout, proc.stderr)
    for stream in streams:
        os.set_blocking(stream.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    offset = 0
    buffers = {proc.stdout: bytearray(), proc.stderr: bytearray()}
    deadline = time.monotonic() + timeout
    timed_out = False
    output_limited = False
    while selector.get_map() or proc.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        for key, _ in selector.select(min(remaining, 0.1)):
            stream = key.fileobj
            if key.data == "stdin":
                try:
                    offset += os.write(stream.fileno(), stdin[offset : offset + 65536])
                except BrokenPipeError:
                    offset = len(stdin)
                if offset >= len(stdin):
                    selector.unregister(stream)
                    stream.close()
                continue
            chunk = os.read(stream.fileno(), 65536)
            if not chunk:
                selector.unregister(stream)
                continue
            current = sum(len(value) for value in buffers.values())
            room = max(0, OUTPUT_LIMIT - current)
            buffers[stream].extend(chunk[:room])
            if len(chunk) > room:
                output_limited = True
                break
        if output_limited:
            break
    if timed_out or output_limited:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        returncode = proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()
    selector.close()
    return {
        "returncode": returncode,
        "stdout": bytes(buffers[proc.stdout]).decode("utf-8", errors="replace"),
        "stderr": bytes(buffers[proc.stderr]).decode("utf-8", errors="replace"),
        "timed_out": timed_out,
        "output_limited": output_limited,
    }


def _compiler(runtime_label: str, source: str) -> tuple[list[str], list[str] | str, int]:
    if runtime_label.startswith(("GNU C++", "MS C++", "Clang++")):
        command = [
            "/apps/apps/gcc/gcc-12.3.0/gc/bin/g++",
            "-s",
            "-x",
            "c++",
            "-O2",
            "-w",
            "-DONLINE_JUDGE",
            "-include",
            "math.h",
            "-include",
            "limits.h",
            "-static-libstdc++",
            "-static-libgcc",
            "-lm",
        ]
        standards = {
            "GNU C++0x": "c++0x",
            "GNU C++11": "c++11",
            "GNU C++14": "c++14",
            "GNU C++17": "c++17",
            "GNU C++17 (64)": "c++17",
            "GNU C++17 Diagnostics": "c++17",
            "GNU C++20 (64)": "c++20",
            "GNU C++20": "c++20",
            "Clang++17 Diagnostics": "c++17",
        }
        standard = standards.get(runtime_label)
        if standard:
            command.insert(1, f"-std={standard}")
        command.extend(("/work/candidate.cpp", "-o", "/work/program"))
        return command, "/work/program", 4
    if runtime_label.startswith("Java "):
        java_class = _java_public_class_name(source)
        if not java_class:
            raise ValueError("Java public class name not found by pinned ExecEval convention")
        source_path = WORK / f"{java_class}.java"
        auxiliary_java_files = [
            path.relative_to(WORK).as_posix()
            for path in sorted(WORK.rglob("*.java"))
            if path != source_path
        ]
        return (
            [
                "/opt/jdk/bin/javac",
                "-J-Xmx768m",
                "-J-XX:ReservedCodeCacheSize=64m",
                "-J-XX:ActiveProcessorCount=2",
                "-J-XX:+UseSerialGC",
                "--release",
                {"Java 8": "8", "Java 14": "14", "Java 17": "17"}.get(runtime_label, "8"),
                "-nowarn",
                "-cp",
                ".:*",
                "-d",
                "/work",
                source_path.name,
                *auxiliary_java_files,
            ],
            [
                "/opt/jdk/bin/java",
                "-DONLINE_JUDGE=true",
                "-Xlog:disable",
                "-Xmx768m",
                "-XX:ReservedCodeCacheSize=64m",
                "-Xss256k",
                "-XX:ActiveProcessorCount=2",
                "-XX:+UseSerialGC",
                "-cp",
                "/work",
                java_class,
            ],
            4,
        )
    if runtime_label in {"Python 2", "PyPy 2"}:
        runtime = (
            "/opt/python2/bin/python2.7" if runtime_label == "Python 2" else "/opt/pypy2/bin/pypy"
        )
    elif runtime_label == "Python 3":
        runtime = "/opt/venv/bin/python"
    elif runtime_label == "Python 3.11":
        runtime = "/opt/python311/bin/python3.11"
    elif runtime_label in {"PyPy 3", "PyPy 3-64"}:
        runtime = "/opt/pypy3/bin/pypy3"
    else:
        raise ValueError(f"Unsupported xCodeEval runtime label: {runtime_label}")
    compile_command = [runtime, "-m", "py_compile", "/work/candidate.py"]
    return compile_command, [runtime, "-W", "ignore", "-OO", "-s", "-S", "/work/candidate.py"], 2


def main() -> int:
    # Candidate children must not inspect the trusted harness memory or file descriptors.
    if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_DUMPABLE failed")
    request = json.load(sys.stdin)
    sys.stdin.close()
    language = request["language"]
    runtime_label = request["runtime_label"]
    source = (ASSETS / "candidate.source").read_text(encoding="utf-8")
    auxiliary_sources = request.get("auxiliary_sources", {})
    for relative_path, auxiliary_source in auxiliary_sources.items():
        auxiliary_path = Path(relative_path)
        if auxiliary_path.is_absolute() or ".." in auxiliary_path.parts:
            raise ValueError("Unsafe auxiliary source path")
        destination = WORK / auxiliary_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(auxiliary_source, encoding="utf-8")
    if request.get("mode") == "function_harness":
        test_source = (ASSETS / "test.source").read_text(encoding="utf-8")
        interface_source = (ASSETS / "interface.source").read_text(encoding="utf-8")
        repeats = int(request.get("repeats", 3))
        entry_point = request.get("entry_point")
        if language == "Python":
            _, execute, _ = _compiler(runtime_label, source)
            try:
                compile(source, "candidate", "exec")
                compile(test_source, "benchmark_test", "exec")
            except SyntaxError as exc:
                print(json.dumps({"status": "compile_failure", "diagnostic": str(exc)}))
                return 0
            if not isinstance(entry_point, str) or not entry_point.isidentifier():
                print(json.dumps({"status": "setup_failure", "reason": "entry_point"}))
                return 0
            harness = (
                "candidate_namespace = {'__name__': 'candidate'}\n"
                "exec(compile(open('/assets/candidate.source', encoding='utf-8').read(), "
                "'candidate', 'exec'), candidate_namespace)\n"
                f"test_namespace = {{{entry_point!r}: candidate_namespace[{entry_point!r}]}}\n"
                "exec(compile(open('/assets/test.source', encoding='utf-8').read(), "
                "'benchmark_test', 'exec'), test_namespace)\n"
            )
            execute_command = [
                SECCOMP,
                execute[0],
                "-W",
                "ignore",
                "-s",
                "-S",
                "-c",
                harness,
            ]
            compile_result = {"returncode": 0, "timed_out": False, "stderr": ""}
        elif language == "Java":
            (WORK / "Solution.java").write_text(source, encoding="utf-8")
            test_preamble = request.get("test_preamble", "")
            (WORK / "Main.java").write_text(test_preamble + test_source, encoding="utf-8")
            compile_command = [
                "/opt/jdk/bin/javac",
                "-J-Xmx768m",
                "-J-XX:ReservedCodeCacheSize=64m",
                "-J-XX:ActiveProcessorCount=2",
                "-J-XX:+UseSerialGC",
                "--release",
                "17",
                "-nowarn",
                "-d",
                "/work",
                "Solution.java",
                "Main.java",
                *[
                    path.relative_to(WORK).as_posix()
                    for path in sorted(WORK.rglob("*.java"))
                    if path.name not in {"Solution.java", "Main.java"}
                ],
            ]
            execute_command = [
                SECCOMP,
                "/opt/jdk/bin/java",
                "-DONLINE_JUDGE=true",
                "-Xlog:disable",
                "-Xmx768m",
                "-XX:ReservedCodeCacheSize=64m",
                "-Xss256k",
                "-XX:ActiveProcessorCount=2",
                "-XX:+UseSerialGC",
                "-cp",
                "/work",
                "Main",
            ]
            compile_result = _run_bounded(
                compile_command, b"", timeout=120, cpu_seconds=60, address_space=4 * 1024**3
            )
        else:
            (WORK / "candidate.cpp").write_text(source, encoding="utf-8")
            (WORK / "test.cpp").write_text(
                '#include "/work/interface.h"\n' + test_source, encoding="utf-8"
            )
            (WORK / "interface.h").write_text(interface_source, encoding="utf-8")
            compile_command = [
                "/apps/apps/gcc/gcc-12.3.0/gc/bin/g++",
                "-std=c++11",
                "-s",
                "-x",
                "c++",
                "-O2",
                "-w",
                "-DONLINE_JUDGE",
                "-include",
                "math.h",
                "-include",
                "limits.h",
                "-static-libstdc++",
                "-static-libgcc",
                "-lm",
                "/work/candidate.cpp",
                "/work/test.cpp",
                "-o",
                "/work/program",
            ]
            compile_result = _run_bounded(
                compile_command, b"", timeout=120, cpu_seconds=60, address_space=4 * 1024**3
            )
            execute_command = [SECCOMP, "/work/program"]
        if compile_result["returncode"] != 0 or compile_result["timed_out"]:
            print(
                json.dumps(
                    {
                        "status": "compile_failure",
                        "returncode": compile_result["returncode"],
                        "timed_out": compile_result["timed_out"],
                        "diagnostic": compile_result["stderr"][:1000],
                    }
                )
            )
            return 0
        outcomes = []
        for _ in range(repeats):
            result = _run_bounded(
                execute_command,
                b"",
                timeout=float(request.get("program_timeout_seconds", 30)),
                cpu_seconds=int(request.get("program_cpu_seconds", 30)),
                address_space=4 * 1024**3,
            )
            outcomes.append(
                {
                    "returncode": result["returncode"],
                    "timed_out": result["timed_out"],
                    "output_limited": result["output_limited"],
                    "stdout_sha256": hashlib.sha256(result["stdout"].encode("utf-8")).hexdigest(),
                    "stderr_sample": result["stderr"][:500],
                }
            )
        stable = len({item["stdout_sha256"] for item in outcomes}) == 1
        passed = stable and all(
            item["returncode"] == 0 and not item["timed_out"] and not item["output_limited"]
            for item in outcomes
        )
        print(
            json.dumps(
                {
                    "status": "passed" if passed else "failed",
                    "compile_status": "passed",
                    "repeats": repeats,
                    "all_passed": passed,
                    "flaky": not stable,
                    "program_outcomes": outcomes,
                },
                separators=(",", ":"),
            )
        )
        return 0
    if language == "Java":
        java_class = _java_public_class_name(source)
        if not java_class:
            print(json.dumps({"status": "reference_setup_error", "reason": "java_class_name"}))
            return 0
        (WORK / f"{java_class}.java").write_text(source, encoding="utf-8")
    elif language == "C++":
        (WORK / "candidate.cpp").write_text(source, encoding="utf-8")
    else:
        (WORK / "candidate.py").write_text(source, encoding="utf-8")

    try:
        compile_command, execute, address_gib = _compiler(runtime_label, source)
    except ValueError as exc:
        print(json.dumps({"status": "reference_setup_error", "reason": str(exc)}))
        return 0
    compile_result = _run_bounded(
        compile_command,
        b"",
        timeout=120,
        cpu_seconds=60,
        address_space=4 * 1024**3,
    )
    if compile_result["returncode"] != 0 or compile_result["timed_out"]:
        print(
            json.dumps(
                {
                    "status": "compile_failure",
                    "returncode": compile_result["returncode"],
                    "timed_out": compile_result["timed_out"],
                    "stdout": compile_result["stdout"][:500],
                    "diagnostic": compile_result["stderr"][:500],
                }
            )
        )
        return 0

    execute_command = [SECCOMP, "/work/program"] if execute == "/work/program" else execute
    assert isinstance(execute_command, list)
    if request.get("mode") == "program_once":
        if language == "Python":
            execute_command = [
                execute[0],
                "-W",
                "ignore",
                "-s",
                "-S",
                "/work/candidate.py",
            ]
        repeats = int(request.get("repeats", 3))
        outcomes: list[dict[str, Any]] = []
        for _ in range(repeats):
            result = _run_bounded(
                execute_command,
                b"",
                timeout=float(request.get("program_timeout_seconds", 20)),
                cpu_seconds=int(request.get("program_cpu_seconds", 10)),
                address_space=address_gib * 1024**3,
            )
            outcomes.append(
                {
                    "returncode": result["returncode"],
                    "timed_out": result["timed_out"],
                    "output_limited": result["output_limited"],
                    "stdout_sha256": hashlib.sha256(result["stdout"].encode()).hexdigest(),
                    "stdout_sample": result["stdout"][:1000],
                    "stderr_sample": result["stderr"][:500],
                }
            )
        stable = len({item["stdout_sha256"] for item in outcomes}) == 1
        all_passed = stable and all(
            item["returncode"] == 0 and not item["timed_out"] and not item["output_limited"]
            for item in outcomes
        )
        print(
            json.dumps(
                {
                    "status": "passed" if all_passed else "failed",
                    "compile_status": "passed",
                    "repeats": repeats,
                    "all_passed": all_passed,
                    "flaky": not stable,
                    "program_outcomes": outcomes,
                },
                separators=(",", ":"),
            )
        )
        return 0
    tests = request["tests"]
    repeats = int(request.get("repeats", 3))
    case_results: list[list[tuple[str, str]]] = []
    error_counts: dict[str, int] = {}
    mismatch_samples: list[dict[str, Any]] = []
    flaky = False
    for test_index, case in enumerate(tests):
        repeated_results: list[tuple[str, str]] = []
        for _ in range(repeats):
            try:
                stdin = case["input"].encode("ascii")
            except UnicodeEncodeError:
                repeated_results.append(("fixture_encoding_error", ""))
                error_counts["fixture_encoding_error"] = (
                    error_counts.get("fixture_encoding_error", 0) + 1
                )
                continue
            result = _run_bounded(
                execute_command,
                stdin,
                timeout=float(request.get("case_timeout_seconds", 6)),
                cpu_seconds=int(request.get("case_cpu_seconds", 2)),
                address_space=address_gib * 1024**3,
            )
            if result["timed_out"]:
                outcome = "timeout"
            elif result["output_limited"]:
                outcome = "output_limit"
            elif result["returncode"] != 0:
                outcome = "runtime_error"
            elif (
                _match_avatar_output(result["stdout"], case["output"])
                if request.get("comparison_policy") == "avatar_casefold_whitespace"
                else _match_output(result["stdout"], case["output"])
            ):
                outcome = "passed"
            else:
                outcome = "wrong_answer"
            error_counts[outcome] = error_counts.get(outcome, 0) + 1
            if outcome == "wrong_answer" and len(mismatch_samples) < 5:
                mismatch_samples.append(
                    {
                        "test_index": test_index,
                        "expected_options": case["output"],
                        "actual": result["stdout"][:200],
                    }
                )
            output_hash = hashlib.sha256(result["stdout"].encode("utf-8")).hexdigest()
            repeated_results.append((outcome, output_hash))
        case_results.append(repeated_results)
        if (
            len({output_hash for _, output_hash in repeated_results}) > 1
            or len({status for status, _ in repeated_results}) > 1
        ):
            flaky = True
        if any(status != "passed" for status, _ in repeated_results):
            break
        if flaky:
            break
    all_passed = len(case_results) == len(tests) and all(
        len(run) == repeats and all(status == "passed" for status, _ in run) for run in case_results
    )
    print(
        json.dumps(
            {
                "status": "passed" if all_passed else "failed",
                "compile_status": "passed",
                "tests_per_repeat": len(tests),
                "repeats": repeats,
                "all_passed": all_passed,
                "flaky": flaky,
                "outcomes": error_counts,
                "mismatch_samples": mismatch_samples,
                "tests_checked": len(case_results),
                "case_outcomes": [[status for status, _ in run] for run in case_results],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
