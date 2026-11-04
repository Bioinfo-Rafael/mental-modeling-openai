"""Subcommand bodies. ``cli.py`` owns argument parsing and the paid-call gate."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mtd_rl_xavier import paths
from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading import provenance
from weightreading.backends import OpenAIResponsesBackend
from weightreading.bundle import Bundle, checkpoint_prefix, verify_bundle, write_bundle
from weightreading.config import RunConfig
from weightreading.estimate import preview_estimate
from weightreading.preflight import PreflightResult, run_preflight
from weightreading.prompt import ExecutorLimits, first_message
from weightreading.sandbox import UNSHARE, SandboxExecutor, SandboxLimits, discover_interpreter
from weightreading.session import RunLog, SessionLimits, run_session
from weightreading.spec import Spec, check_against_tensors, load_spec
from weightreading.tensors import Tensor, total_params
from weightreading.torch_reader import Checkpoint, read_checkpoint

DEFAULT_OUTPUT_ROOT = paths.OUTPUTS / "weightreading"
SELFTEST_ALIAS = "net-00000000"
SELFTEST_SHAPE = (2, 3)
FLOAT32_BYTES = 4
VERSION_TIMEOUT_S = 10


@dataclass(frozen=True)
class Prepared:
    checkpoint: Checkpoint
    spec: Spec
    spec_path: Path
    bundle: Bundle


def write_json(path: Path, payload: object) -> None:
    derived_path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def inspect_payload(checkpoint: Checkpoint, tensor_hashes: bool) -> dict:
    if tensor_hashes:
        return {tensor.name: tensor.sha256 for tensor in checkpoint.tensors}
    return {
        "checkpoint_sha256": checkpoint.sha256,
        "tensors": len(checkpoint.tensors),
        "parameters": total_params(checkpoint.tensors),
        "entries": [{"key": t.name, "dtype": t.dtype, "shape": list(t.shape)} for t in checkpoint.tensors],
    }


def load_checked(checkpoint_path: Path, spec_path: Path) -> tuple[Checkpoint, Spec]:
    checkpoint = read_checkpoint(checkpoint_path)
    spec = load_spec(spec_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    return checkpoint, spec


def prepare(checkpoint_path: Path, spec_path: Path, output_root: Path) -> Prepared:
    checkpoint, spec = load_checked(checkpoint_path, spec_path)
    bundle = write_bundle(checkpoint.tensors, spec.alias, checkpoint.sha256, output_root / "bundles")
    return Prepared(checkpoint, spec, spec_path, bundle)


def output_name(prepared: Prepared, config: RunConfig) -> str:
    """Run/preview directory name: ``<UTC>_<alias>-<checkpoint sha12>_<task_mode>``."""
    prefix = checkpoint_prefix(prepared.checkpoint.sha256)
    return f"{utc_stamp()}_{prepared.spec.alias}-{prefix}_{config.task_mode}"


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


def write_preview(prepared: Prepared, config: RunConfig, directory: Path) -> dict:
    """Write ``message.txt`` and ``preview.json`` into ``directory`` (created exclusively)."""
    derived_path(directory).mkdir(parents=True, exist_ok=False)
    message = first_message(prepared.spec, prepared.checkpoint.tensors, config.task_mode, executor_limits(config))
    (directory / "message.txt").write_text(message, encoding="utf-8")
    payload = {
        "network_alias": prepared.spec.alias,
        "task_mode": config.task_mode,
        "bundle": str(prepared.bundle.path),
        "message_sha256": provenance.sha256_text(message),
        "inputs": provenance.input_record(
            prepared.checkpoint.path, prepared.checkpoint.sha256, prepared.spec_path, prepared.bundle.manifest_sha256
        ),
        "estimate": preview_estimate(len(message), config),
        "api_requests_made": 0,
    }
    write_json(directory / "preview.json", payload)
    return payload


def unshare_version() -> str | None:
    try:
        completed = subprocess.run([UNSHARE, "--version"], capture_output=True, text=True, timeout=VERSION_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


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


def run_record(prepared: Prepared, config: RunConfig, message: str) -> dict:
    return {
        "network_alias": prepared.spec.alias,
        "task_mode": config.task_mode,
        "inputs": provenance.input_record(
            prepared.checkpoint.path, prepared.checkpoint.sha256, prepared.spec_path, prepared.bundle.manifest_sha256
        ),
        "message_sha256": provenance.sha256_text(message),
        "code_sha256": provenance.code_hashes(),
        "git": provenance.git_state(paths.ROOT),
        "environment": {**provenance.environment(), "unshare": unshare_version()},
        "model": config.model,
        "backend": config.backend,
        "reasoning_effort": config.reasoning_effort,
        "limits": asdict(config.limits),
        "sandbox": {**asdict(sandbox_limits(config)), "python": str(config.sandbox.python)},
        "status": "started",
    }


def execute_run(
    prepared: Prepared, config: RunConfig, run_dir: Path, client_factory: Callable[[], Any]
) -> tuple[int, dict]:
    """Paid session. Order: run dir -> preflight (fail closed) -> client (reads the key) -> loop."""
    derived_path(run_dir).mkdir(parents=True, exist_ok=False)
    message = first_message(prepared.spec, prepared.checkpoint.tensors, config.task_mode, executor_limits(config))
    (run_dir / "message.txt").write_text(message, encoding="utf-8")
    record = run_record(prepared, config, message)
    executor = make_executor(config, run_dir / "work", prepared.bundle.weights)
    preflight = run_preflight(executor, hidden_paths(prepared.bundle.path, prepared.checkpoint.path.parent))
    record = {
        **record,
        "sandbox_preflight": preflight.to_dict(),
        "interpreter": asdict(executor.interp),
    }
    if not preflight.ok:
        write_json(run_dir / "run.json", {**record, "status": "refused: sandbox preflight failed"})
        return 1, record
    write_json(run_dir / "run.json", record)
    backend = OpenAIResponsesBackend(
        client_factory(), config.model, config.reasoning_effort, config.limits.max_output_tokens, config.retry
    )
    limits = SessionLimits(config.limits.max_runs, config.limits.max_total_tokens, config.limits.max_protocol_errors)
    try:
        state = run_session(backend, executor, message, limits, RunLog(run_dir))
    except Exception as exc:
        write_json(run_dir / "run.json", {**record, "status": f"error: {type(exc).__name__}"})
        raise
    verify_bundle(prepared.bundle.path, prepared.bundle.manifest_sha256)
    final = {**record, "status": "complete", "session": {**asdict(state), "total_tokens": state.total_tokens}}
    write_json(run_dir / "run.json", final)
    return 0, final


def with_task_mode(config: RunConfig, task_mode: str | None) -> RunConfig:
    return config if task_mode is None else replace(config, task_mode=task_mode)
