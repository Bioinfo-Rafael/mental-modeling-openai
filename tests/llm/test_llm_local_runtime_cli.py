"""vLLM launcher CLI: flags, dry-run preflight, client process, exit codes and signal handling."""

from __future__ import annotations

import json
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from translate.llm import vllm_server
from translate.llm.vllm_server import (
    ServerEndpoint,
    VLLMServerConfig,
    VLLMServerStartupError,
    exit_status,
    main,
    run_client_command,
    split_client_command,
)

from .local_runtime_fakes import FAKE_PID, FakeClock, FakeProcess, GroupSignals, RecordingPopen, make_runtime_dir
from .test_llm_local_runtime import KEY_ENV, KEY_VALUE, REVISION

CLIENT = ["python", "-m", "translate.cli", "--execute"]


@pytest.fixture(autouse=True)
def server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is set, so the no-echo assertions below are meaningful."""
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)


@pytest.fixture
def runtime_dir(tmp_path: Path) -> Path:
    return make_runtime_dir(tmp_path)


def launcher_flags(base: Path, **overrides: str) -> list[str]:
    flags = {
        "--vllm-executable": str(base / "vllm-env" / "bin" / "vllm"),
        "--model": "Cloudflare/clef-flash",
        "--revision": REVISION,
        "--served-model-name": "clef-flash",
        "--max-model-len": "16384",
        "--dtype": "bfloat16",
        "--uds-path": str(base / "sock" / "clef-flash.sock"),
        "--hf-home": str(base / "hf"),
        "--api-key-env": KEY_ENV,
        "--client-input-token-budget": "16384",
        "--startup-timeout-s": "900",
        "--shutdown-timeout-s": "20",
        **overrides,
    }
    return [part for flag, value in flags.items() for part in (flag, value)]


class RecordingServer:
    """``server_factory`` stand-in: records the config and the enter/exit order."""

    def __init__(self, error: Exception | None = None) -> None:
        self.configs: list[VLLMServerConfig] = []
        self.events: list[str] = []
        self._error = error

    @contextmanager
    def __call__(self, config: VLLMServerConfig) -> Iterator[ServerEndpoint]:
        self.configs.append(config)
        if self._error is not None:
            raise self._error
        self.events.append("ready")
        try:
            yield ServerEndpoint(uds_path=config.uds_path, base_url=None, served_model_name="clef-flash", pid=FAKE_PID)
        finally:
            self.events.append("stopped")


# --- run --------------------------------------------------------------------------------------------------


def test_cli_runs_the_client_while_the_server_is_up_and_returns_its_code(tmp_path: Path) -> None:
    server = RecordingServer()
    commands: list[list[str]] = []

    def run_client(command: list[str]) -> int:
        server.events.append("client")
        commands.append(command)
        return 3

    code = main([*launcher_flags(tmp_path), "--", *CLIENT], server_factory=server, run_client=run_client)

    assert code == 3 and commands == [CLIENT]
    assert server.events == ["ready", "client", "stopped"]
    assert server.configs[0].uds_path == tmp_path / "sock" / "clef-flash.sock"


def test_cli_reports_a_client_killed_by_a_signal_as_128_plus_n(tmp_path: Path) -> None:
    code = main([*launcher_flags(tmp_path), "--", *CLIENT], server_factory=RecordingServer(), run_client=lambda _: -9)

    assert code == 137


@pytest.mark.parametrize(("code", "status"), [(0, 0), (3, 3), (-signal.SIGKILL, 137), (-signal.SIGTERM, 143)])
def test_exit_status_maps_signal_deaths(code: int, status: int) -> None:
    assert exit_status(code) == status


def test_cli_ctrl_c_during_the_client_stops_the_server_and_exits_130(tmp_path: Path) -> None:
    server = RecordingServer()

    def interrupted(command: list[str]) -> int:
        raise KeyboardInterrupt

    code = main([*launcher_flags(tmp_path), "--", *CLIENT], server_factory=server, run_client=interrupted)

    assert code == 128 + signal.SIGINT and server.events == ["ready", "stopped"]


def test_cli_startup_failure_exits_1_without_running_the_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ran: list[list[str]] = []
    server = RecordingServer(error=VLLMServerStartupError("vllm serve exited with code 1 before it was ready"))

    def run_client(command: list[str]) -> int:
        ran.append(command)
        return 0

    code = main([*launcher_flags(tmp_path), "--", *CLIENT], server_factory=server, run_client=run_client)

    assert code == 1 and ran == [] and "exited with code 1" in capsys.readouterr().err


# --- dry run ----------------------------------------------------------------------------------------------


def test_cli_dry_run_runs_the_preflight_and_prints_the_argv_without_the_key(
    runtime_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = RecordingServer()

    code = main([*launcher_flags(runtime_dir), "--dry-run"], server_factory=server)

    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == 0 and server.configs == [] and report["preflight"] == "ok"
    assert report["argv"][1:3] == ["serve", "Cloudflare/clef-flash"] and report["serving_budget"] == 16384
    assert KEY_VALUE not in out and "VLLM_API_KEY" in report["child_env_set"]
    assert KEY_ENV in report["child_env_stripped_names"]
    assert not (runtime_dir / "sock").exists()  # the preflight creates nothing


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("executable", "executable"),
        ("hf_home", "hf_home"),
        ("key", "api_key_env"),
        ("socket_dir", "0700"),
    ],
)
def test_cli_dry_run_fails_on_what_a_real_start_would_refuse(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], change: str, match: str
) -> None:
    if change == "executable":
        (runtime_dir / "vllm-env" / "bin" / "vllm").unlink()
    elif change == "hf_home":
        (runtime_dir / "hf").rmdir()
    elif change == "key":
        monkeypatch.delenv(KEY_ENV)
    else:
        (runtime_dir / "sock").mkdir(mode=0o755)
        (runtime_dir / "sock").chmod(0o755)

    code = main([*launcher_flags(runtime_dir), "--dry-run"], server_factory=RecordingServer())

    assert code == 1 and match in capsys.readouterr().err


# --- flags --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        ["--enable-log-requests"],
        ["--api-key", KEY_VALUE],
        ["--trust-remote-code"],
        ["--gpu-memory-utilization", "0.5"],
        ["--served", "clef-flash"],  # abbreviations are disabled
    ],
)
def test_cli_accepts_no_passthrough_or_abbreviated_flags_and_echoes_no_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    server = RecordingServer()

    with pytest.raises(SystemExit) as excinfo:
        main([*launcher_flags(tmp_path), *extra, "--", *CLIENT], server_factory=server)

    err = capsys.readouterr().err
    assert excinfo.value.code == 2 and server.configs == []
    assert "no passthrough" in err and KEY_VALUE not in err and "0.5" not in err


def test_cli_invalid_value_names_the_flag_but_not_the_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flags = launcher_flags(tmp_path, **{"--max-model-len": "SECRET-not-an-int"})

    with pytest.raises(SystemExit):
        main([*flags, "--", *CLIENT], server_factory=RecordingServer())

    err = capsys.readouterr().err
    assert "argument --max-model-len: invalid value (not shown)" in err and "SECRET" not in err


def test_cli_missing_required_flags_are_named(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--model", "Cloudflare/clef-flash", "--", *CLIENT], server_factory=RecordingServer())

    assert "the following arguments are required: --vllm-executable" in capsys.readouterr().err


def test_cli_requires_a_client_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(launcher_flags(tmp_path), server_factory=RecordingServer())

    assert excinfo.value.code == 2 and "client command is required" in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides",
    [{"--client-input-token-budget": "4096"}, {"--api-key-env": "per-job-SECRET"}, {"--revision": "main"}],
)
def test_cli_config_errors_exit_1_before_any_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], overrides: dict[str, str]
) -> None:
    server = RecordingServer()

    code = main([*launcher_flags(tmp_path, **overrides), "--", *CLIENT], server_factory=server)

    err = capsys.readouterr().err
    assert code == 1 and server.configs == [] and "VLLMServerConfigError" in err and "SECRET" not in err


def test_cli_tcp_all_interfaces_is_rejected(tmp_path: Path) -> None:
    flags = [*launcher_flags(tmp_path), "--host", "0.0.0.0", "--port", "8000"]
    flags = [part for part in flags if part not in ("--uds-path", str(tmp_path / "sock" / "clef-flash.sock"))]

    assert main([*flags, "--", *CLIENT], server_factory=RecordingServer()) == 1


def test_split_client_command_uses_the_first_separator_only() -> None:
    assert split_client_command(["--a", "1", "--", "cmd", "--", "x"]) == (["--a", "1"], ["cmd", "--", "x"])
    assert split_client_command(["--a", "1"]) == (["--a", "1"], [])


# --- client process --------------------------------------------------------------------------------------


def test_client_runs_in_its_own_process_group_and_returns_its_exit_code() -> None:
    process = FakeProcess(exit_after_polls=0, exit_code=5)
    popen = RecordingPopen(process)

    assert run_client_command(["true"], popen_factory=popen, killpg=GroupSignals(process)) == 5
    assert popen.calls == [(["true"], {"stdin": subprocess.DEVNULL, "start_new_session": True})]


def interrupt_client(process: FakeProcess) -> tuple[GroupSignals, FakeClock]:
    killpg, clock = GroupSignals(process), FakeClock()
    with pytest.raises((KeyboardInterrupt, SystemExit)):
        run_client_command(
            ["client"], popen_factory=RecordingPopen(process), killpg=killpg, clock=clock, sleep=clock.sleep
        )
    return killpg, clock


def test_interrupted_launcher_sends_sigint_to_the_client_group_then_reraises() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGINT}), wait_raises=KeyboardInterrupt())

    killpg, clock = interrupt_client(process)

    assert killpg.calls == [(FAKE_PID, signal.SIGINT)] and clock.now == 0.0


def test_client_group_ignoring_sigint_is_killed_after_the_grace_period() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGKILL}), wait_raises=SystemExit(143))

    killpg, clock = interrupt_client(process)

    assert killpg.calls == [(FAKE_PID, signal.SIGINT), (FAKE_PID, signal.SIGKILL)]
    assert clock.now == pytest.approx(vllm_server.CLIENT_INTERRUPT_WAIT_S)


def test_client_helpers_outliving_the_client_are_killed_with_its_group() -> None:
    process = FakeProcess(
        exits_on=frozenset({signal.SIGINT}),
        workers_exit_on=frozenset({signal.SIGKILL}),
        wait_raises=KeyboardInterrupt(),
    )

    killpg, _ = interrupt_client(process)

    assert killpg.calls == [(FAKE_PID, signal.SIGINT), (FAKE_PID, signal.SIGKILL)] and not process.workers_alive


# --- signal handling -------------------------------------------------------------------------------------


@pytest.mark.parametrize(("signum", "raised"), [(signal.SIGTERM, SystemExit), (signal.SIGINT, KeyboardInterrupt)])
def test_first_signal_unwinds_and_later_ones_are_ignored(
    monkeypatch: pytest.MonkeyPatch, signum: int, raised: type[BaseException]
) -> None:
    installed: list[tuple[int, Any]] = []
    monkeypatch.setattr(vllm_server.signal, "signal", lambda sig, handler: installed.append((sig, handler)))

    with pytest.raises(raised) as excinfo:
        vllm_server._exit_on_signal(signum, None)

    assert installed == [(signal.SIGTERM, signal.SIG_IGN), (signal.SIGINT, signal.SIG_IGN)]
    if raised is SystemExit:
        assert excinfo.value.code == 128 + signal.SIGTERM  # type: ignore[union-attr]


def test_install_signal_handlers_covers_sigterm_and_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: dict[int, Any] = {}
    monkeypatch.setattr(vllm_server.signal, "signal", lambda sig, handler: installed.update({sig: handler}))

    vllm_server.install_signal_handlers()

    assert installed == {signal.SIGTERM: vllm_server._exit_on_signal, signal.SIGINT: vllm_server._exit_on_signal}


def test_signals_deferred_blocks_sigterm_and_sigint_and_restores_the_mask() -> None:
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    with vllm_server.signals_deferred():
        inside = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    assert {signal.SIGTERM, signal.SIGINT} <= inside
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
