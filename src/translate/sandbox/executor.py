"""Sandboxed Python executor for the LLM's code, built on ``unshare`` (no bubblewrap needed).

Isolation, per run (fresh process each time):

* user + mount + pid + net namespaces (``unshare -r -m -p -n -f --kill-child``); the network
  namespace has no usable interface, so every connection fails;
* a new tmpfs root that contains only ``/usr`` (ro), the interpreter prefix at ``/opt/python``
  (ro), numpy at ``/opt/site`` (ro), ``/work`` (rw, persistent between runs), the bundle weights at
  ``/work/weights`` (ro), a size-capped ``/tmp`` and four ``/dev`` nodes; the host root is
  detached with ``pivot_root`` + lazy unmount, so ``$HOME`` and the repository do not exist;
  ``/proc`` is mounted only during setup (host mount paths stay hidden) and the root is read-only;
* the code then runs in a nested, unmapped user namespace, which owns no mounts: remounting the
  read-only weights or the root fails with EPERM;
* ``env -i`` (the API key never reaches the sandbox), ``python -I``, RLIMIT_AS / RLIMIT_FSIZE,
  a wall-clock timeout that kills the whole process group, a stdout cap and a stderr tail.

There is no unsandboxed fallback: if any of this is unavailable the run fails.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from translate.policy.bundle import INDEX_FILE

UNSHARE = "/usr/bin/unshare"
SANDBOX_WORK = "/work"
SANDBOX_WEIGHTS = "/work/weights"
SANDBOX_SITE = "/opt/site"
READ_CHUNK = 65536
BYTES_PER_CHAR_BOUND = 4  # UTF-8 worst case, so the byte cap never cuts below the char cap
DISCOVERY_TIMEOUT_S = 60
EXTRA_TASKS = 16  # processes/threads allowed beyond the BLAS threads (RLIMIT_NPROC inside the sandbox)
KILL_GRACE_S = 5
SANDBOX_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

# Runs as "root" of the new user namespace; every value arrives as a positional argument.
SETUP_SCRIPT = r"""
set -eu
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
R="$1"; PYBASE="$2"; PYREL="$3"; NUMPY="$4"; NUMPY_LIBS="$5"; WORK="$6"; WEIGHTS="$7"; TMP_MB="$8"; THREADS="$9"
ro_bind() { mkdir -p "$2"; mount --bind "$1" "$2"; mount -o remount,ro,bind "$2"; }
mount -t tmpfs -o size=16m,mode=0755 sandbox-root "$R"
for d in bin lib lib32 lib64 libx32 sbin; do
  if [ -L "/$d" ]; then ln -s "$(readlink "/$d")" "$R/$d"; fi
done
mkdir -p "$R/usr" "$R/work" "$R/tmp" "$R/proc" "$R/dev" "$R/.oldroot"
mount --bind /usr "$R/usr"; mount -o remount,ro,bind "$R/usr"  # non-recursive: no writable submounts
ro_bind "$PYBASE" "$R/opt/python"
ro_bind "$NUMPY" "$R/opt/site/numpy"
if [ -n "$NUMPY_LIBS" ]; then ro_bind "$NUMPY_LIBS" "$R/opt/site/numpy.libs"; fi
mount --bind "$WORK" "$R/work"
mount --bind "$WEIGHTS" "$R/work/weights"; mount -o remount,ro,bind "$R/work/weights"
mount -t tmpfs -o "size=${TMP_MB}m" sandbox-tmp "$R/tmp"
for n in null zero random urandom; do touch "$R/dev/$n"; mount --bind "/dev/$n" "$R/dev/$n"; done
mount -t proc proc "$R/proc"
cd "$R"
pivot_root . .oldroot
umount -l /.oldroot
rmdir /.oldroot
mount -o remount,ro,bind /
umount -l /proc
cd /work
exec /usr/bin/unshare --user -- /usr/bin/env -i HOME=/tmp PATH=/usr/bin:/bin LANG=C.UTF-8 \
  OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
  "/opt/python/$PYREL" -I -B -
"""

DISCOVERY_CODE = """\
import json, os, sys
import numpy
exe = os.path.realpath(sys.executable)
base = os.path.realpath(sys.base_prefix)
pkg = os.path.dirname(os.path.realpath(numpy.__file__))
libs = pkg + ".libs"
print(json.dumps({"base_prefix": base, "executable": os.path.relpath(exe, base), "numpy_dir": pkg,
                  "numpy_libs": libs if os.path.isdir(libs) else "", "numpy_version": numpy.__version__,
                  "python_version": sys.version.split()[0]}))
"""


class SandboxError(RuntimeError):
    """The sandbox cannot be built here. Runs are refused; there is no unsandboxed fallback."""


@dataclass(frozen=True)
class Interpreter:
    base_prefix: str
    executable: str  # relative to base_prefix
    numpy_dir: str
    numpy_libs: str
    numpy_version: str
    python_version: str


@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: int
    memory_bytes: int
    file_size_bytes: int
    stdout_cap_chars: int
    stderr_tail_chars: int
    tmp_size_mb: int
    threads: int
    work_max_bytes: int


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stdout_truncated: bool
    stderr_tail: str
    exit_code: int | None
    timed_out: bool
    duration_s: float


def discover_interpreter(python: Path) -> Interpreter:
    """Ask the (trusted, local) interpreter where its prefix and numpy live."""
    if not python.exists():
        raise SandboxError(f"sandbox interpreter not found: {python}")
    completed = subprocess.run(
        [str(python), "-I", "-c", DISCOVERY_CODE],
        capture_output=True,
        text=True,
        timeout=DISCOVERY_TIMEOUT_S,
        env=SANDBOX_ENV,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise SandboxError(f"sandbox interpreter cannot import numpy: {completed.stderr.strip()[-500:]}")
    info = json.loads(completed.stdout)
    if info["executable"].startswith(".."):
        raise SandboxError("interpreter executable lies outside its prefix; cannot mount it")
    return Interpreter(**info)


def build_argv(interp: Interpreter, staging: Path, work: Path, weights: Path, limits: SandboxLimits) -> list[str]:
    """argv for one sandboxed ``python -I -`` that reads its script from stdin."""
    return [
        UNSHARE,
        "--user",
        "--map-root-user",
        "--mount",
        "--pid",
        "--net",
        "--fork",
        "--kill-child",
        "--propagation",
        "private",
        "--",
        "/bin/sh",
        "-c",
        SETUP_SCRIPT,
        "weightreading-sandbox",
        str(staging),
        interp.base_prefix,
        interp.executable,
        interp.numpy_dir,
        interp.numpy_libs,
        str(work),
        str(weights),
        str(limits.tmp_size_mb),
        str(limits.threads),
    ]


def _preload_lines(preload: tuple[tuple[str, str], ...]) -> str:
    """``NAME = {key: array}`` from ``/work/weights/<subdir>/index.json`` for every (NAME, subdir)."""
    lines = []
    for name, subdir in preload:
        root = f"{SANDBOX_WEIGHTS}/{subdir}"
        lines.append(f"with open({root + '/' + INDEX_FILE!r}) as _index:\n    _keys = json.load(_index)\n")
        lines.append(f"{name} = {{_k: np.load({root!r} + '/' + _k + '.npy', mmap_mode='r') for _k in _keys}}\n")
    return "".join(lines)


def wrap_code(code: str, run_label: str, limits: SandboxLimits, preload: tuple[tuple[str, str], ...] = ()) -> str:
    """Prelude that caps processes/CPU, loads ``W`` from the read-only bundle, then runs the model's code.

    ``preload`` adds read-only array dicts from subdirectories of the bundle (e.g. ``("A", "activations")``).
    RLIMIT_NPROC is counted per (user namespace, uid) on this kernel, so it bounds only sandbox
    tasks; the code has no capability in the initial namespace and cannot raise hard limits.
    """
    tasks = limits.threads + EXTRA_TASKS
    names = "".join(f", {name!r}: {name}" for name, _ in preload)
    return (
        "import os, resource, sys\n"
        f"resource.setrlimit(resource.RLIMIT_NPROC, ({tasks}, {tasks}))\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({limits.timeout_s}, {limits.timeout_s}))\n"
        f"sys.path.insert(0, {SANDBOX_SITE!r})\n"
        f"os.chdir({SANDBOX_WORK!r})\n"
        "import json\n"
        "import numpy as np\n"
        f"with open({SANDBOX_WEIGHTS + '/' + INDEX_FILE!r}) as _index:\n"
        "    _keys = json.load(_index)\n"
        f"W = {{_k: np.load({SANDBOX_WEIGHTS!r} + '/' + _k + '.npy') for _k in _keys}}\n"
        f"{_preload_lines(preload)}"
        f"_source = {code!r}\n"
        f"exec(compile(_source, {run_label!r}, 'exec'), {{'__name__': '__main__', 'np': np, 'json': json, 'W': W{names}}})\n"
    )


def _limit_resources(limits: SandboxLimits) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_bytes, limits.file_size_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _read_head(stream: object, cap_bytes: int, sink: list[bytes], totals: list[int]) -> None:
    kept = 0
    for chunk in iter(lambda: stream.read(READ_CHUNK), b""):
        if kept < cap_bytes:
            sink.append(chunk[: cap_bytes - kept])
        kept += len(chunk)
    totals.append(kept)


def _read_tail(stream: object, cap_bytes: int, sink: list[bytes]) -> None:
    tail = b""
    for chunk in iter(lambda: stream.read(READ_CHUNK), b""):
        tail = (tail + chunk)[-cap_bytes:]
    sink.append(tail)


def _feed(stream: object, data: bytes) -> None:
    try:
        stream.write(data)
        stream.close()
    except BrokenPipeError:
        return


def _namespace_init_pids(process: subprocess.Popen) -> list[int]:
    """Host PIDs of ``unshare``'s children: the PID-namespace init. Killing it kills the whole namespace."""
    try:
        text = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text()
    except OSError:
        return []
    return [int(pid) for pid in text.split()]


def _kill_sandbox(process: subprocess.Popen) -> None:
    """SIGKILL the namespace init (code can setsid() or clear its parent-death signal), then the group."""
    for pid in _namespace_init_pids(process):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_sandboxed(script: str, interp: Interpreter, work: Path, weights: Path, limits: SandboxLimits) -> ExecResult:
    """Run ``script`` once in a fresh sandbox. ``work/weights`` must be a real directory (mountpoint)."""
    mountpoint = work / "weights"
    if mountpoint.is_symlink() or not mountpoint.is_dir():
        raise SandboxError("work/weights must be a real directory")
    staging = Path(tempfile.mkdtemp(prefix="wr-sandbox-root-"))
    argv = build_argv(interp, staging, work, weights, limits)
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=SANDBOX_ENV,
            start_new_session=True,
            preexec_fn=_limit_resources(limits),
        )
        stdout_parts: list[bytes] = []
        stdout_totals: list[int] = []
        stderr_parts: list[bytes] = []
        threads = [
            threading.Thread(target=_feed, args=(process.stdin, script.encode("utf-8"))),
            threading.Thread(
                target=_read_head, args=(process.stdout, _stdout_cap_bytes(limits), stdout_parts, stdout_totals)
            ),
            threading.Thread(
                target=_read_tail, args=(process.stderr, limits.stderr_tail_chars * BYTES_PER_CHAR_BOUND, stderr_parts)
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            process.wait(timeout=limits.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_sandbox(process)
            process.wait(timeout=KILL_GRACE_S)
        for thread in threads:
            thread.join(timeout=KILL_GRACE_S)
    finally:
        _remove_staging(staging)
    output = _Output(b"".join(stdout_parts), stdout_totals, b"".join(stderr_parts))
    return _result(output, process.returncode, timed_out, time.monotonic() - started, limits)


def _remove_staging(staging: Path) -> None:
    try:
        staging.rmdir()
    except OSError:
        # Not empty would mean a mount leaked to the host namespace; leave it for inspection.
        return


def _stdout_cap_bytes(limits: SandboxLimits) -> int:
    return limits.stdout_cap_chars * BYTES_PER_CHAR_BOUND


@dataclass(frozen=True)
class _Output:
    stdout: bytes
    stdout_totals: list[int]  # one entry once the reader saw EOF
    stderr: bytes


def _result(
    output: _Output, returncode: int | None, timed_out: bool, duration_s: float, limits: SandboxLimits
) -> ExecResult:
    text = output.stdout.decode("utf-8", errors="replace")
    reader_finished = bool(output.stdout_totals)
    over_cap = not reader_finished or output.stdout_totals[0] > _stdout_cap_bytes(limits)
    stderr = output.stderr.decode("utf-8", errors="replace")[-limits.stderr_tail_chars :]
    return ExecResult(
        stdout=text[: limits.stdout_cap_chars],
        stdout_truncated=over_cap or len(text) > limits.stdout_cap_chars,
        stderr_tail=stderr,
        exit_code=None if timed_out else returncode,
        timed_out=timed_out,
        duration_s=round(duration_s, 3),
    )


class SandboxExecutor:
    """Executor used by the session loop: one persistent ``/work`` dir, fresh process per run."""

    def __init__(
        self,
        interp: Interpreter,
        work: Path,
        weights: Path,
        limits: SandboxLimits,
        preload: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not Path(UNSHARE).exists():
            raise SandboxError(f"{UNSHARE} not found; the sandbox cannot be built")
        (work / "weights").mkdir(parents=True, exist_ok=True)
        self.interp = interp
        self.work = work
        self.weights = weights
        self.limits = limits
        self.preload = preload

    def with_stdout_cap(self, stdout_cap_chars: int) -> SandboxExecutor:
        """Same sandbox, different stdout cap (used by the preflight report)."""
        limits = dataclasses.replace(self.limits, stdout_cap_chars=stdout_cap_chars)
        return SandboxExecutor(self.interp, self.work, self.weights, limits, self.preload)

    def run(self, code: str, run_index: int) -> ExecResult:
        """One sandboxed run. Fails closed if ``/work`` has grown past ``work_max_bytes``."""
        script = wrap_code(code, f"<run {run_index}>", self.limits, self.preload)
        result = run_sandboxed(script, self.interp, self.work, self.weights, self.limits)
        used = directory_bytes(self.work)
        if used > self.limits.work_max_bytes:
            raise SandboxError(f"/work grew to {used} bytes (cap {self.limits.work_max_bytes}); session stopped")
        return result


def directory_bytes(root: Path) -> int:
    """Apparent size of regular files under ``root`` without following symlinks."""
    total = 0
    for directory, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                continue
    return total
