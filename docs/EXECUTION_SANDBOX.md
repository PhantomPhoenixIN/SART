# External program execution sandbox

## Environment evidence

PBS CPU probe jobs `25576.hn01` and `25577.hn01` ran on `cn01`. No Docker, Podman,
Apptainer, bubblewrap, or nsjail executable is installed. PBS cgroup v1 limits for the
account's `pbs.service` are unlimited (`memory.limit_in_bytes` is effectively unlimited,
`pids.max=max`, and CPU quota `-1`). Rootless user/mount/PID/network/IPC/UTS namespaces
work. `setpriv` can drop all capabilities inside the new user namespace.

## Implemented controls

`src/evaluation/namespace_sandbox.py` provides the private fallback. It creates a per-task
rootless namespace, mounts system/runtime directories read-only, hides the host home,
provides a private writable tmpfs work area and `/tmp`, mounts task assets read-only,
mounts a PID-namespace `/proc`, and starts the command after dropping capabilities and
setting `no_new_privs`. It creates a network namespace and the seccomp launcher blocks
socket operations, fork/vfork, non-thread clone, namespace/mount changes, ptrace, BPF,
io_uring, and other dangerous syscalls. Runtime thread/process growth is additionally
bounded with a dynamic `RLIMIT_NPROC` ceiling based on visible account tasks plus 64 slots.

Per-task defaults are 10 seconds wall time, 8 seconds CPU, 2 GiB address space per
process, 64 MiB maximum file size, 128 MiB writable work tmpfs, 64 open files, and 1 MiB
combined captured output. A supervisor kills the isolated process group on timeout or
output overflow. Benchmark-specific limits may be lower; any higher limit must be set
before the reference manifest is frozen and used uniformly for reference and candidate
programs.

Build the small seccomp launcher with:

```bash
bash scripts/reproduce/build_sandbox_launcher.sh
```

The executable is a generated local artifact under `content/tools/sandbox/`; source is
`src/evaluation/seccomp_exec.c`.

## Validation and limitations

Seven controlled tests pass: home is hidden; mounted assets cannot be written; work files
can be written; capabilities are zero; network socket creation and fork are denied;
thread creation and virtual address space are bounded; wall timeout kills a sleeping
process; and a benign C++ program compiles and runs inside the namespace. Tests are in
`tests/test_namespace_sandbox.py`.

This is a local compatibility sandbox, not the official Docker-based ExecEval service.
PBS does not enforce per-job cgroups on this cluster. `RLIMIT_AS` limits each process,
not the sum of a compiler's parent/child processes; C++ compilation therefore needs
strict wall and address-space limits and low evaluator concurrency. The trusted in-jail
driver launches each program under seccomp and feeds one test input through stdin; test
fixtures are never mounted as files.

### Final canonical-reference preflight status (2026-10-02)

Initial preflight failures were investigated and retained as historical failed attempts;
they are not part of final eligibility. The xCodeEval CPython 3.10 results were superseded
by a complete private CPython 3.11 run. A read-only audit reloaded every pinned candidate
record and reproduced the frozen selection: 810 passing Python, 409 Java, 927 C++
implementations and 332 Common-Core problem IDs. HumanEval-X's first cleaned-declaration/
missing-import attempts and partial C++ parser attempt were invalidated; the harness-v3
recomputation reproduces 157 Common-Core problem IDs and 956 Full-Directional tasks.
TransCoder-Test and AVATAR-TC frozen eligibility were also independently recomputed from
the pinned reference records and match their manifests (321 Common-Core / 1,936
Full-Directional TransCoder tasks; 1,649 AVATAR problems / 3,298 tasks). These checks do
not execute model predictions. Final counts and hashes are in
`reports/data/functional_test_coverage.md` and `data/manifests/`.

Compatibility details remain: xCodeEval's earlier sweep had a CPython 3.10 mismatch;
Java/C++ runtimes use the versions recorded in the manifests, including JDK 21 with
`--release 8` for legacy Java labels and GNU C++11. HumanEval-X uses a separate Java 17
harness because official tests call `List.of`; tests compile separately with required
imports carried over. C++ HumanEval-X tests use the benchmark prompt and canonical
completion, exposing prompt function prototypes to a separate test translation unit.
HumanEval-X's canonical Python assertion harness uses private CPython 3.11 without
optimization so assertions remain active. Keep both the compatibility deviations and
the fact that this is not official ExecEval in the final threats-to-validity section.

The canonical test driver also sets itself nondumpable before retaining the test payload;
seccomp denies `process_vm_readv/writev`, `kcmp`, and `pidfd_getfd`, preventing a candidate
from inspecting the in-memory harness through process-memory or descriptor APIs. A
sandboxed probe attempting to open `/proc/1/mem` was denied.
