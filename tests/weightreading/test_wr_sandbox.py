"""Sandbox: argv/prelude units everywhere; real isolation checks where unshare works."""

from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from translate import paths
from weightreading.bundle import verify_bundle, write_bundle
from weightreading.preflight import EXPECTED_CHECKS, run_preflight
from weightreading.sandbox import (
    SETUP_SCRIPT,
    UNSHARE,
    Interpreter,
    SandboxError,
    SandboxExecutor,
    SandboxLimits,
    build_argv,
    discover_interpreter,
    run_sandboxed,
    wrap_code,
)
from weightreading.tensors import Tensor

from .wr_helpers import small_arrays

VENV_PYTHON = paths.ROOT / ".venv" / "bin" / "python"
LIMITS = SandboxLimits(
    timeout_s=20,
    memory_bytes=4 << 30,
    file_size_bytes=1 << 20,
    stdout_cap_chars=300,
    stderr_tail_chars=400,
    tmp_size_mb=16,
    threads=1,
    work_max_bytes=1 << 20,
)
INTERP = Interpreter("/py", "bin/python3.12", "/site/numpy", "", "2.0", "3.12.0")


def _userns_available() -> bool:
    if not Path(UNSHARE).exists() or not VENV_PYTHON.exists():
        return False
    probe = subprocess.run([UNSHARE, "-rmn", "true"], capture_output=True, timeout=10)
    return probe.returncode == 0


needs_sandbox = pytest.mark.skipif(not _userns_available(), reason="unprivileged user namespaces unavailable")


def test_build_argv_unshares_all_namespaces_and_passes_paths_as_arguments(tmp_path: Path) -> None:
    argv = build_argv(INTERP, tmp_path / "root", tmp_path / "work", tmp_path / "weights", LIMITS)

    for flag in ["--user", "--map-root-user", "--mount", "--pid", "--net", "--fork", "--kill-child"]:
        assert flag in argv
    assert argv[argv.index("-c") + 1] == SETUP_SCRIPT
    assert str(tmp_path / "work") in argv and str(tmp_path / "weights") in argv
    assert not any(str(tmp_path) in part for part in argv[: argv.index("-c") + 2])


def test_setup_script_drops_privileges_and_hides_host_root() -> None:
    assert "pivot_root . .oldroot" in SETUP_SCRIPT
    assert "--rbind" not in SETUP_SCRIPT
    assert "umount -l /.oldroot" in SETUP_SCRIPT
    assert "umount -l /proc" in SETUP_SCRIPT
    assert 'mount -o remount,ro,bind "$R/work/weights"' in SETUP_SCRIPT
    assert "exec /usr/bin/unshare --user -- /usr/bin/env -i" in SETUP_SCRIPT


def test_wrap_code_quotes_code_and_keeps_run_label() -> None:
    script = wrap_code('print("""hi""")\n', "<run 4>", LIMITS)

    assert "'<run 4>'" in script
    assert "RLIMIT_NPROC, (17, 17)" in script
    assert "compile(_source" in script
    assert 'print("""hi""")' not in script.split("_source = ")[0]


def test_executor_refuses_symlinked_weights_mountpoint(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "weights").symlink_to(tmp_path)

    with pytest.raises(SandboxError, match="real directory"):
        run_sandboxed("print(1)", INTERP, tmp_path / "work", tmp_path, LIMITS)


def test_discover_interpreter_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="not found"):
        discover_interpreter(tmp_path / "nope")


@pytest.fixture
def executor(tmp_path: Path) -> SandboxExecutor:
    tensors = tuple(Tensor(k, "float32", v.shape, v.astype("<f4").tobytes()) for k, v in small_arrays().items())
    bundle = write_bundle(tensors, "net-0123abcd", "abc", tmp_path / "bundles")
    return SandboxExecutor(discover_interpreter(VENV_PYTHON), tmp_path / "work", bundle.weights, LIMITS)


@needs_sandbox
def test_preflight_on_this_host_passes_every_check(executor: SandboxExecutor, tmp_path: Path) -> None:
    result = run_preflight(executor, [paths.ROOT, Path.home(), tmp_path])

    assert result.ok, result.to_dict()
    assert set(result.checks) == EXPECTED_CHECKS
    verify_bundle(executor.weights.parent)


@needs_sandbox
def test_sandbox_run_sees_weights_and_persistent_work_dir(executor: SandboxExecutor) -> None:
    first = executor.run("print(sorted(W)[0], W['rnn.gru.weight_hh_l0'].shape)\nopen('/work/a.txt','w').write('x')", 1)
    second = executor.run("print(open('/work/a.txt').read())", 2)

    assert first.exit_code == 0, first.stderr_tail
    assert "act.action_outs.0.logits_net.bias (12, 4)" in first.stdout
    assert second.stdout.strip() == "x"


@needs_sandbox
def test_sandbox_run_caps_stdout_and_reports_errors(executor: SandboxExecutor) -> None:
    result = executor.run("print('y' * 5000)\nraise ValueError('boom')", 1)

    assert result.stdout_truncated and len(result.stdout) == LIMITS.stdout_cap_chars
    assert result.exit_code == 1
    assert "ValueError: boom" in result.stderr_tail and '"<run 1>"' in result.stderr_tail


@needs_sandbox
def test_sandbox_run_timeout_kills_process(executor: SandboxExecutor) -> None:
    short = SandboxExecutor(executor.interp, executor.work, executor.weights, dataclasses.replace(LIMITS, timeout_s=1))

    result = short.run("import time\ntime.sleep(30)", 1)

    assert result.timed_out and result.exit_code is None
    assert result.duration_s < 10


@needs_sandbox
def test_sandbox_blocks_network_home_and_weight_writes(executor: SandboxExecutor) -> None:
    code = (
        "import os, socket\n"
        f"print(os.path.exists({str(Path.home())!r}), os.path.exists({str(paths.ROOT)!r}))\n"
        "try:\n    socket.create_connection(('1.1.1.1', 443), timeout=2); print('net open')\n"
        "except OSError: print('net blocked')\n"
        "try:\n    open('/work/weights/index.json', 'a'); print('weights writable')\n"
        "except OSError: print('weights read-only')\n"
        "print(sorted(k for k in os.environ if 'KEY' in k))\n"
    )

    result = executor.run(code, 1)

    assert result.stdout.split("\n")[:4] == ["False False", "net blocked", "weights read-only", "[]"]


@needs_sandbox
def test_sandbox_run_removes_its_staging_root(executor: SandboxExecutor, tmp_path: Path, monkeypatch) -> None:
    staging_parent = tmp_path / "staging"
    staging_parent.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(staging_parent))

    executor.run("print(1)", 1)

    assert list(staging_parent.iterdir()) == []


ESCAPE_CODE = """\
import ctypes, os, time
libc = ctypes.CDLL(None)
libc.prctl(1, 0, 0, 0, 0)  # PR_SET_PDEATHSIG = 0
os.setsid()
if os.fork() == 0:
    os.setsid()
    while True:
        with open('/work/alive.txt', 'w') as handle:
            handle.write(str(time.time()))
        time.sleep(0.2)
time.sleep(60)
"""


@needs_sandbox
def test_sandbox_timeout_kills_code_that_detaches_from_its_session(executor: SandboxExecutor) -> None:
    short = SandboxExecutor(executor.interp, executor.work, executor.weights, dataclasses.replace(LIMITS, timeout_s=2))

    result = short.run(ESCAPE_CODE, 1)
    marker = executor.work / "alive.txt"
    before = marker.read_text()
    time.sleep(1.5)

    assert result.timed_out
    assert marker.read_text() == before


@needs_sandbox
def test_sandbox_limits_number_of_processes(executor: SandboxExecutor) -> None:
    code = (
        "import os\npids = []\ntry:\n    for _ in range(64):\n        pid = os.fork()\n"
        "        if pid == 0:\n            import time; time.sleep(5); os._exit(0)\n        pids.append(pid)\n"
        "except OSError as exc:\n    print('blocked after', len(pids), exc.errno)\n"
        "for pid in pids:\n    os.kill(pid, 9)\n"
    )

    result = executor.run(code, 1)

    assert result.stdout.startswith("blocked after"), result
    assert int(result.stdout.split()[2]) < 20


@needs_sandbox
def test_sandbox_work_dir_over_cap_stops_session(executor: SandboxExecutor) -> None:
    with pytest.raises(SandboxError, match="/work grew"):
        executor.run("open('/work/big.bin', 'wb').write(b'0' * (900 * 1024))\n"
                     "open('/work/big2.bin', 'wb').write(b'0' * (900 * 1024))", 1)  # fmt: skip
