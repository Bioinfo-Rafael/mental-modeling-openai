"""Subcommand bodies. ``cli.py`` owns argument parsing and the paid-call gate."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from mtd_rl_xavier import paths
from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading.bundle import Bundle, verify_bundle, write_bundle
from weightreading.config import RunConfig
from weightreading.preflight import PreflightResult, run_preflight
from weightreading.prompt import ExecutorLimits
from weightreading.sandbox import SandboxExecutor, SandboxLimits, discover_interpreter
from weightreading.tensors import Tensor

DEFAULT_OUTPUT_ROOT = paths.OUTPUTS / "weightreading"
SELFTEST_ALIAS = "net-00000000"
SELFTEST_SHAPE = (2, 3)
FLOAT32_BYTES = 4


def write_json(path: Path, payload: object) -> None:
    derived_path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def executor_limits(config: RunConfig) -> ExecutorLimits:
    sandbox = config.sandbox
    return ExecutorLimits(
        config.limits.max_runs, sandbox.timeout_s, sandbox.stdout_cap_chars, sandbox.stderr_tail_chars
    )


def sandbox_limits(config: RunConfig) -> SandboxLimits:
    s = config.sandbox
    return SandboxLimits(
        s.timeout_s,
        s.memory_bytes,
        s.file_size_bytes,
        s.stdout_cap_chars,
        s.stderr_tail_chars,
        s.tmp_size_mb,
        s.threads,
        s.work_max_bytes,
    )


def make_executor(config: RunConfig, work: Path, weights: Path) -> SandboxExecutor:
    interp = discover_interpreter(config.sandbox.python)
    return SandboxExecutor(interp, work, weights, sandbox_limits(config))


def hidden_paths(*extra: Path) -> list[Path]:
    return [paths.ROOT, Path.home(), *extra]


def _selftest_bundle(directory: Path) -> Bundle:
    numel = SELFTEST_SHAPE[0] * SELFTEST_SHAPE[1]
    tensor = Tensor("selftest.weight", "float32", SELFTEST_SHAPE, bytes(numel * FLOAT32_BYTES))
    return write_bundle((tensor,), SELFTEST_ALIAS, "selftest", directory)


def sandbox_selftest(config: RunConfig, bundle_dir: Path | None) -> PreflightResult:
    """Run the preflight against ``bundle_dir`` (or a tiny throwaway bundle) in a throwaway work dir."""
    scratch = Path(tempfile.mkdtemp(prefix="wr-selftest-"))
    try:
        bundle = verify_bundle(bundle_dir) if bundle_dir else _selftest_bundle(scratch / "bundles")
        executor = make_executor(config, scratch / "work", bundle.weights)
        return run_preflight(executor, hidden_paths(bundle.path))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
