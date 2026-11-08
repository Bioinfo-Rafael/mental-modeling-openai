"""The one-GPU SLURM template ``slurm/translator_vllm.sbatch``: static checks and one offline run.

Nothing is submitted. The offline run executes the script with ``bash`` (``#SBATCH`` lines are
comments) from the repository root, with the project interpreter and a fake ``vllm`` that records
its argv and environment and exits, so the real launcher refuses to continue. No SLURM, GPU,
network or vLLM is involved.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from translate import paths
from translate.llm import vllm_server

SBATCH = paths.ROOT / "slurm" / "translator_vllm.sbatch"
DEFAULT_CONFIG = paths.ROOT / "configs" / "translator" / "run.yaml"
DEFAULT_UDS_LINE = "uds_path: /tmp/{user}-vllm/clef-flash.sock"
REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"
FAKE_VLLM_EXIT = 3
JOB_TIMEOUT_S = 120


def _sbatch() -> str:
    return SBATCH.read_text(encoding="utf-8")


def _code_lines() -> str:
    return "\n".join(line for line in _sbatch().splitlines() if not line.lstrip().startswith("#"))


# --- static ------------------------------------------------------------------------------------------


def test_sbatch_passes_bash_syntax_check() -> None:
    result = subprocess.run(["bash", "-n", str(SBATCH)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_sbatch_requests_one_gpu_on_the_partition_for_a_short_time() -> None:
    directives = re.findall(r"^#SBATCH (.+)$", _sbatch(), flags=re.MULTILINE)

    assert "--partition=151-partition" in directives and "--gres=gpu:1" in directives
    (time_limit,) = [d.split("=", 1)[1] for d in directives if d.startswith("--time=")]
    hours, minutes, _ = (int(part) for part in time_limit.split(":"))
    assert hours * 60 + minutes <= 60


def test_sbatch_generates_a_per_job_key_and_never_prints_or_passes_it_on_argv() -> None:
    code = _code_lines()

    assert "secrets.token_urlsafe" in code and "export VLLM_API_KEY" in code
    assert not re.search(r"set -[a-z]*x|set -o xtrace", code)
    assert not re.search(r"(echo|printf)[^\n]*\$\{?VLLM_API_KEY", code)
    for forbidden in ("--api-key ", "0.0.0.0", "--enable-log-requests", "--trust-remote-code", "printf -v"):
        assert forbidden not in code


def test_sbatch_has_no_personal_paths_and_requires_placeholders() -> None:
    text = _sbatch()

    assert not re.search(r"/(home|lustre)/user\d+", text) and getpass.getuser() not in text
    for name in ("VLLM_ENV", "HF_CACHE_DIR", "SCENARIO"):
        assert f"${{{name}:?" in text


def test_sbatch_creates_the_socket_directory_0700_and_paid_runs_need_explicit_opt_in() -> None:
    text = _sbatch()

    assert 'mkdir -m 700 "$SOCKET_DIR"' in text
    assert 'MODE="${1:-probe}"' in text and '[ "$MODE" = paid ]' in text
    assert "CONFIRM_PAID" not in text
    assert text.count("--confirm-paid-api") == 1


def test_sbatch_execs_the_launcher_so_signals_reach_it_directly() -> None:
    last = [line for line in _code_lines().splitlines() if line.strip()][-1]

    assert last.startswith('exec "$PYTHON" -m translate.llm.vllm_server')
    assert "uv run" not in _code_lines() and 'PYTHON="$PY"' in _code_lines()  # .venv/bin/python, no uv


def test_sbatch_launcher_flags_are_all_known_to_the_launcher() -> None:
    block = re.search(r"^LAUNCHER=\((.*?)^\)", _sbatch(), flags=re.MULTILINE | re.DOTALL)
    assert block is not None
    used = set(re.findall(r"^\s*(--[a-z-]+)", block.group(1), flags=re.MULTILINE))
    actions = vllm_server.build_parser()._actions
    known = {option for action in actions for option in action.option_strings}
    required = {option for action in actions if action.required for option in action.option_strings}

    assert used and used <= known and required <= used


def test_sbatch_config_reader_runs_offline_against_the_default_config() -> None:
    snippet = re.search(r"<<'PY'\n(.*?)\nPY\n", _sbatch(), flags=re.DOTALL)
    assert snippet is not None

    result = subprocess.run(
        [sys.executable, "-", str(DEFAULT_CONFIG)],
        input=snippet.group(1),
        capture_output=True,
        text=True,
        check=False,
        cwd=paths.ROOT,
    )

    assert result.returncode == 0, result.stderr
    model, revision, served, uds_path, budget = result.stdout.rstrip("\n").split("\t")
    assert (model, revision, served, budget) == ("Cloudflare/clef-flash", REVISION, "clef-flash", "16384")
    assert uds_path == f"/tmp/{getpass.getuser()}-vllm/clef-flash.sock"


def test_sbatch_probe_client_compiles() -> None:
    probe = re.search(r"PROBE='(.*?)'\n", _sbatch(), flags=re.DOTALL)
    assert probe is not None

    compile(probe.group(1), "probe", "exec")


# --- one offline run with a fake vllm ------------------------------------------------------------------


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_sbatch_runs_offline_with_a_fake_vllm_and_never_exposes_the_key(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    socket_root = tmp_path_factory.mktemp("j")  # short: AF_UNIX paths are limited to 107 bytes
    uds_path = socket_root / "s" / "clef-flash.sock"
    dump = tmp_path / "dump"
    dump.mkdir()
    vllm_env = tmp_path / "vllm-env"
    _write_executable(
        vllm_env / "bin" / "vllm",
        f'#!/bin/sh\nprintf "%s\\n" "$@" > "{dump}/argv"\nenv > "{dump}/env"\nexit {FAKE_VLLM_EXIT}\n',
    )
    (tmp_path / "hf").mkdir()
    config_text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert config_text.count(DEFAULT_UDS_LINE) == 1
    config = tmp_path / "run.yaml"
    config.write_text(config_text.replace(DEFAULT_UDS_LINE, f"uds_path: {uds_path}"), encoding="utf-8")
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(tmp_path)),
        "SLURM_SUBMIT_DIR": str(paths.ROOT),
        "VLLM_ENV": str(vllm_env),
        "HF_CACHE_DIR": str(tmp_path / "hf"),
        "SCENARIO": str(tmp_path / "never-read.yaml"),
        "CONFIG": str(config),
        "STARTUP_TIMEOUT_S": "30",
        "SHUTDOWN_TIMEOUT_S": "2",
    }

    job = subprocess.run(
        ["bash", str(SBATCH)], env=env, capture_output=True, text=True, check=False, timeout=JOB_TIMEOUT_S
    )

    assert job.returncode == 1, job.stderr
    assert f"vllm serve exited with code {FAKE_VLLM_EXIT} before it was ready" in job.stderr
    report = json.loads(job.stdout[job.stdout.index("{") :])
    argv = (dump / "argv").read_text(encoding="utf-8").splitlines()
    assert report["argv"] == [str(vllm_env / "bin" / "vllm"), *argv]
    assert argv == [
        "serve",
        "Cloudflare/clef-flash",
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--served-model-name",
        "clef-flash",
        "--max-model-len",
        "16384",
        "--dtype",
        "bfloat16",
        "--uds",
        str(uds_path),
    ]
    child_env = dict(
        line.split("=", 1) for line in (dump / "env").read_text(encoding="utf-8").splitlines() if "=" in line
    )
    key = child_env["VLLM_API_KEY"]
    assert len(key) >= 40 and key not in job.stdout and key not in job.stderr and not any(key in a for a in argv)
    assert child_env["HF_HUB_OFFLINE"] == "1" and child_env["HF_HOME"] == str(tmp_path / "hf")
    assert child_env["PATH"].startswith(f"{vllm_env / 'bin'}{os.pathsep}")
    assert child_env["VIRTUAL_ENV"] == str(vllm_env)
    assert not any(name.startswith("OPENAI_") for name in child_env)
    assert stat.S_IMODE(os.lstat(uds_path.parent).st_mode) == 0o700 and not uds_path.exists()
