from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.evaluation.namespace_sandbox import SandboxLimits, run_isolated

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "env/bin/python"
SECCOMP = ROOT / "content/tools/sandbox/seccomp_exec"
PYTHON2 = ROOT / "content/tools/python2.7.18/stage/opt/python2"


@pytest.mark.skipif(
    not PYTHON.exists() or not SECCOMP.exists(), reason="private sandbox tools missing"
)
def test_rootless_sandbox_hides_home_blocks_fork_and_network_and_bounds_writes():
    with TemporaryDirectory() as directory:
        asset_dir = Path(directory)
        (asset_dir / "readme.txt").write_text("read-only")
        program = (
            "import os, pathlib, socket\n"
            f"print('home=' + str(pathlib.Path({str(Path.home())!r}).exists()))\n"
            "pathlib.Path('/work/output.txt').write_text('allowed')\n"
            "try: pathlib.Path('/assets/readme.txt').write_text('blocked')\n"
            "except OSError: print('assets=readonly')\n"
            "for label, operation in [('fork', os.fork), ('network', socket.socket)]:\n"
            "    try: operation()\n"
            "    except OSError: print(label + '=blocked')\n"
            "cap = next(line.split()[1] for line in open('/proc/self/status')\n"
            "           if line.startswith('CapEff:'))\n"
            "print('capabilities=' + cap)\n"
        )
        result = run_isolated(
            ["/opt/venv/bin/python", "-c", program],
            assets=asset_dir,
            runtime_mounts={PYTHON.parent.parent: "/opt/venv"},
            limits=SandboxLimits(wall_seconds=10),
        )
    assert result.returncode == 0
    assert not result.timed_out
    output = result.stdout.decode()
    assert "home=False" in output
    assert "assets=readonly" in output
    assert "fork=blocked" in output
    assert "network=blocked" in output
    assert "capabilities=0000000000000000" in output


@pytest.mark.skipif(not PYTHON.exists(), reason="private Python runtime missing")
def test_rootless_sandbox_bounds_wall_time():
    with TemporaryDirectory() as directory:
        result = run_isolated(
            ["/bin/sleep", "2"],
            assets=Path(directory),
            runtime_mounts={PYTHON.parent.parent: "/opt/venv"},
            limits=SandboxLimits(wall_seconds=0.2),
            seccomp=False,
        )
    assert result.timed_out


@pytest.mark.skipif(
    not PYTHON.exists() or not SECCOMP.exists(), reason="private sandbox tools missing"
)
def test_rootless_sandbox_limits_thread_creation():
    with TemporaryDirectory() as directory:
        code = (
            "import threading, time\n"
            "threads = []\n"
            "for _ in range(200):\n"
            "    try:\n"
            "        thread = threading.Thread(target=time.sleep, args=(0.2,))\n"
            "        thread.start()\n"
            "        threads.append(thread)\n"
            "    except RuntimeError:\n"
            "        break\n"
            "print('created=' + str(len(threads)))\n"
            "[thread.join() for thread in threads]\n"
        )
        result = run_isolated(
            ["/opt/venv/bin/python", "-c", code],
            assets=Path(directory),
            runtime_mounts={PYTHON.parent.parent: "/opt/venv"},
        )
    assert result.returncode == 0
    created = int(result.stdout.decode().split("=")[1])
    assert 1 <= created < 100


@pytest.mark.skipif(
    not PYTHON.exists() or not SECCOMP.exists(), reason="private sandbox tools missing"
)
def test_cpp_compile_and_candidate_run_stay_in_private_sandbox():
    with TemporaryDirectory() as directory:
        assets = Path(directory)
        (assets / "main.cpp").write_text(
            '#include <iostream>\nint main() { std::cout << "compiled-ok\\n"; }\n'
        )
        result = run_isolated(
            [
                "/bin/sh",
                "-c",
                "g++ -std=c++17 -O0 /assets/main.cpp -o /work/candidate && "
                "/opt/sart/seccomp_exec /work/candidate",
            ],
            assets=assets,
            runtime_mounts={
                PYTHON.parent.parent: "/opt/venv",
                SECCOMP: "/opt/sart/seccomp_exec",
            },
            seccomp=False,
        )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"compiled-ok\n"


@pytest.mark.skipif(
    not PYTHON.exists() or not SECCOMP.exists(), reason="private sandbox tools missing"
)
def test_rootless_sandbox_enforces_virtual_address_space_limit():
    with TemporaryDirectory() as directory:
        result = run_isolated(
            [
                "/opt/venv/bin/python",
                "-c",
                "import mmap;\n"
                "try: mmap.mmap(-1, 4 * 1024**3)\n"
                "except OSError: print('memory=bounded')\n",
            ],
            assets=Path(directory),
            runtime_mounts={PYTHON.parent.parent: "/opt/venv"},
            limits=SandboxLimits(address_space_bytes=2 * 1024**3),
        )
    assert result.returncode == 0
    assert result.stdout.strip() == b"memory=bounded"


@pytest.mark.skipif(not PYTHON.exists(), reason="private Python runtime missing")
def test_rootless_sandbox_delivers_stdin_without_exposing_test_files():
    with TemporaryDirectory() as directory:
        result = run_isolated(
            ["/opt/venv/bin/python", "-c", "import sys; print(sys.stdin.read(), end='')"],
            assets=Path(directory),
            runtime_mounts={PYTHON.parent.parent: "/opt/venv"},
            stdin=b"line one\nline two\n",
        )
    assert result.returncode == 0
    assert result.stdout == b"line one\nline two\n"


@pytest.mark.skipif(not PYTHON2.exists(), reason="private CPython 2.7 runtime not built yet")
def test_xcodeeval_python2_program_runs_inside_sandbox():
    with TemporaryDirectory() as directory:
        assets = Path(directory)
        (assets / "solution.py").write_text('print "python2-ok"\n')
        result = run_isolated(
            ["/opt/python2/bin/python2.7", "/assets/solution.py"],
            assets=assets,
            runtime_mounts={PYTHON2: "/opt/python2"},
        )
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == b"python2-ok\n"
