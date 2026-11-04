"""Fail-closed sandbox self-test, run before any paid request (and by ``sandbox-selftest``).

Every check must pass: numpy importable, network blocked, repository / ``$HOME`` / bundle host
path invisible, only the expected top-level entries, weights readable but not writable, ``/work``
writable, environment clean (no API key), remount denied, ``/proc`` absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from weightreading.sandbox import SandboxExecutor

PREFLIGHT_STDOUT_CHARS = 4000  # the check report is ~400 chars; independent of the user's cap
RESULT_MARKER = "WEIGHTREADING_PREFLIGHT "
ALLOWED_ROOT_ENTRIES = ["bin", "dev", "lib", "lib32", "lib64", "libx32", "opt", "proc", "sbin", "tmp", "usr", "work"]
EXPECTED_CHECKS = {
    "numpy_import",
    "network_blocked",
    "host_paths_invisible",
    "root_entries_expected",
    "weights_readable",
    "weights_read_only",
    "work_writable",
    "env_clean",
    "remount_denied",
    "proc_absent",
}
ALLOWED_ENV = ["HOME", "PATH", "LANG", "LC_CTYPE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]

# Runs inside the sandbox through the normal prelude, so W is already loaded from /work/weights.
CHECK_CODE = """\
import ctypes, os, socket
hidden = json.loads(HIDDEN_JSON)
checks = {}

def fails(action):
    try:
        action()
    except OSError:
        return True
    return False

checks["numpy_import"] = np.__version__
checks["network_blocked"] = (
    fails(lambda: socket.create_connection(("1.1.1.1", 53), timeout=2))
    and fails(lambda: socket.create_connection(("127.0.0.1", 22), timeout=2))
    and fails(lambda: socket.getaddrinfo("example.com", 443))
)
checks["host_paths_invisible"] = not any(os.path.exists(path) for path in hidden)
checks["root_entries_expected"] = set(os.listdir("/")) <= set(ALLOWED_ROOT)
first = sorted(W)[0]
checks["weights_readable"] = len(W) > 0 and W[first].size > 0
checks["weights_read_only"] = fails(lambda: open("/work/weights/" + first + ".npy", "r+b")) and fails(
    lambda: open("/work/weights/.probe", "wb")
)
def write_probe():
    with open("/work/.preflight_probe", "w") as handle:
        handle.write("ok")
    os.remove("/work/.preflight_probe")
checks["work_writable"] = not fails(write_probe)
checks["env_clean"] = set(os.environ) <= set(ALLOWED_ENV)
libc = ctypes.CDLL(None, use_errno=True)
MS_REMOUNT, MS_BIND = 32, 4096
checks["remount_denied"] = (
    libc.mount(b"none", b"/work/weights", None, MS_REMOUNT | MS_BIND, None) != 0
    and libc.mount(b"none", b"/", None, MS_REMOUNT | MS_BIND, None) != 0
)
checks["proc_absent"] = not os.path.exists("/proc/self")
print(MARKER + json.dumps(checks))
"""


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    checks: dict
    numpy_version: str | None
    exit_code: int | None
    stderr_tail: str

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "numpy_version": self.numpy_version,
            "exit_code": self.exit_code,
            "stderr_tail": self.stderr_tail,
        }


def preflight_code(hidden_paths: list[str]) -> str:
    constants = (
        f"HIDDEN_JSON = {json.dumps(hidden_paths)!r}\n"
        f"ALLOWED_ROOT = {ALLOWED_ROOT_ENTRIES!r}\n"
        f"ALLOWED_ENV = {ALLOWED_ENV!r}\n"
        f"MARKER = {RESULT_MARKER!r}\n"
    )
    return constants + CHECK_CODE


def _parse(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                parsed = json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def run_preflight(executor: SandboxExecutor, hidden_paths: list[Path]) -> PreflightResult:
    """Run the self-test through ``executor``. ``ok`` is True only if every check passed."""
    result = executor.with_stdout_cap(PREFLIGHT_STDOUT_CHARS).run(
        preflight_code([str(path) for path in hidden_paths]), run_index=0
    )
    checks = _parse(result.stdout) or {}
    numpy_version = checks.get("numpy_import")
    booleans = {key: value for key, value in checks.items() if key != "numpy_import"}
    ok = (
        set(checks) == EXPECTED_CHECKS
        and isinstance(numpy_version, str)
        and all(value is True for value in booleans.values())
    )
    return PreflightResult(
        ok=ok and not result.timed_out and result.exit_code == 0,
        checks=checks,
        numpy_version=numpy_version if isinstance(numpy_version, str) else None,
        exit_code=result.exit_code,
        stderr_tail=result.stderr_tail,
    )
