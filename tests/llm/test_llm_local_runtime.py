"""vLLM launcher: allowlisted argv, child environment, budget check, socket directory, lifecycle.

Everything is offline: a fake process, clock and ``killpg``, and ``httpx.MockTransport`` for the
health probe. No vLLM, GPU, network or SLURM is used.
"""

from __future__ import annotations

import os
import signal
import socket
import stat
from contextlib import AbstractContextManager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from translate.llm import vllm_server
from translate.llm.vllm_server import (
    ServerEndpoint,
    VLLMServerConfig,
    VLLMServerConfigError,
    VLLMServerError,
    VLLMServerStartupError,
    build_child_env,
    build_vllm_argv,
    running_vllm_server,
    stop_server,
)

from .llm_fakes import SystemOneServer, json_response, models_body, raise_transport
from .local_runtime_fakes import FAKE_PID, FakeClock, FakeProcess, GroupSignals, RecordingPopen, make_runtime_dir

KEY_ENV = "TEST_VLLM_SERVER_KEY"
KEY_VALUE = "per-job-SECRET-key-0123456789"
REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"
SERVED = "clef-flash"
FORBIDDEN_FLAGS = ("--api-key", "--enable-log-requests", "--trust-remote-code", "0.0.0.0")


@pytest.fixture(autouse=True)
def server_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    return KEY_ENV


@pytest.fixture
def runtime_dir(tmp_path: Path) -> Path:
    """A fake separate vLLM env (executable ``bin/vllm``) and an HF cache directory."""
    return make_runtime_dir(tmp_path)


def make_config(base: Path, **overrides: Any) -> VLLMServerConfig:
    values: dict[str, Any] = {
        "executable": base / "vllm-env" / "bin" / "vllm",
        "model": "Cloudflare/clef-flash",
        "revision": REVISION,
        "served_model_name": SERVED,
        "max_model_len": 16384,
        "dtype": "bfloat16",
        "uds_path": base / "sock" / "clef-flash.sock",
        "host": None,
        "port": None,
        "hf_home": base / "hf",
        "api_key_env": KEY_ENV,
        "client_input_token_budget": 16384,
        "serving_cap_tokens": 16384,
        "startup_timeout_s": 60.0,
        "shutdown_timeout_s": 20.0,
    }
    return VLLMServerConfig(**{**values, **overrides})


def enter_and_exit(manager: AbstractContextManager[Any]) -> None:
    """Enter and leave ``manager``: the start-up checks under test run on entry."""
    with manager:
        return


def tcp(**overrides: Any) -> dict[str, Any]:
    return {"uds_path": None, "host": "127.0.0.1", "port": 8000, **overrides}


def ready() -> httpx.Response:
    return json_response(200, models_body(SERVED))


def start(
    config: VLLMServerConfig,
    health: list[Any],
    process: FakeProcess | None = None,
    *,
    missing_group: bool = False,
) -> tuple[RecordingPopen, GroupSignals, SystemOneServer, FakeClock, Any]:
    process = process or FakeProcess(exits_on=frozenset({signal.SIGTERM}))
    popen, killpg, server, clock = (
        RecordingPopen(process),
        GroupSignals(process, missing_group=missing_group),
        SystemOneServer(health),
        FakeClock(),
    )
    manager = running_vllm_server(
        config,
        popen_factory=popen,
        clock=clock,
        sleep=clock.sleep,
        killpg=killpg,
        transport=server.transport,
    )
    return popen, killpg, server, clock, manager


# --- argv ---------------------------------------------------------------------------------------------


def test_argv_pins_revision_name_length_dtype_and_unix_socket(runtime_dir: Path) -> None:
    config = make_config(runtime_dir)

    argv = build_vllm_argv(config)

    assert argv == (
        str(runtime_dir / "vllm-env" / "bin" / "vllm"),
        "serve",
        "Cloudflare/clef-flash",
        "--revision",
        REVISION,
        "--tokenizer-revision",
        REVISION,
        "--served-model-name",
        SERVED,
        "--max-model-len",
        "16384",
        "--dtype",
        "bfloat16",
        "--uds",
        str(runtime_dir / "sock" / "clef-flash.sock"),
    )


def test_argv_for_tcp_binds_literal_loopback_only(runtime_dir: Path) -> None:
    argv = build_vllm_argv(make_config(runtime_dir, **tcp()))

    assert argv[-4:] == ("--host", "127.0.0.1", "--port", "8000")
    assert "--uds" not in argv


@pytest.mark.parametrize("endpoint", [{}, tcp()], ids=["uds", "tcp"])
def test_argv_never_carries_the_key_or_a_forbidden_flag(runtime_dir: Path, endpoint: dict[str, Any]) -> None:
    argv = build_vllm_argv(make_config(runtime_dir, **endpoint))

    assert not any(KEY_VALUE in part for part in argv)
    assert not any(flag in argv for flag in FORBIDDEN_FLAGS)


def test_all_interfaces_host_is_rejected(runtime_dir: Path) -> None:
    with pytest.raises(VLLMServerConfigError, match="0.0.0.0"):
        make_config(runtime_dir, **tcp(host="0.0.0.0"))


@pytest.mark.parametrize("host", ["localhost", "::1", "10.0.0.5", "127.0.0.2", "", "example.com", "--host"])
def test_non_loopback_host_is_rejected(runtime_dir: Path, host: str) -> None:
    with pytest.raises(VLLMServerConfigError, match="127.0.0.1"):
        make_config(runtime_dir, **tcp(host=host))


@pytest.mark.parametrize(
    "endpoint",
    [
        {"host": "127.0.0.1", "port": 8000},  # UDS and TCP together
        {"uds_path": None},  # neither
        {"uds_path": None, "host": "127.0.0.1"},  # host without port
        {"uds_path": None, "port": 8000},  # port without host
    ],
)
def test_endpoint_must_be_exactly_one_of_uds_or_host_and_port(runtime_dir: Path, endpoint: dict[str, Any]) -> None:
    with pytest.raises(VLLMServerConfigError):
        make_config(runtime_dir, **endpoint)


@pytest.mark.parametrize("port", [0, 80, 1023, 65536, True, "8000"])
def test_privileged_or_invalid_port_is_rejected(runtime_dir: Path, port: object) -> None:
    with pytest.raises(VLLMServerConfigError, match="port"):
        make_config(runtime_dir, **tcp(port=port))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision", "main"),
        ("revision", "fb4417d"),
        ("revision", REVISION.upper()),
        ("revision", "--enable-log-requests"),
        ("model", "--trust-remote-code"),
        ("model", "clef-flash"),
        ("model", "Cloudflare/clef flash"),
        ("served_model_name", "--enable-log-requests"),
        ("served_model_name", "-x"),
        ("served_model_name", ""),
        ("dtype", "float16"),
        ("dtype", "auto"),
        ("dtype", "--api-key"),
        ("max_model_len", 0),
        ("max_model_len", True),
        ("max_model_len", 262145),
        ("api_key_env", "per-job-SECRET-key"),
        ("startup_timeout_s", 0.0),
        ("shutdown_timeout_s", float("inf")),
        ("uds_path", Path("relative/clef.sock")),
        ("hf_home", Path("relative/hf")),
    ],
)
def test_values_outside_the_allowlist_are_rejected(runtime_dir: Path, field: str, value: object) -> None:
    with pytest.raises(VLLMServerConfigError):
        make_config(runtime_dir, **{field: value})


@pytest.mark.parametrize("executable", [Path("bin/vllm"), Path("/bin/sh"), Path("/opt/env/bin/python")])
def test_executable_must_be_an_absolute_vllm(runtime_dir: Path, executable: Path) -> None:
    with pytest.raises(VLLMServerConfigError, match="executable"):
        make_config(runtime_dir, executable=executable)


# --- token budget -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_model_len", "cap", "budget"),
    [(16384, 16384, 16384), (8192, 16384, 8192), (32768, 16384, 16384), (16384, 4096, 4096)],
)
def test_budget_equal_to_min_of_cap_and_max_model_len_is_accepted(
    runtime_dir: Path, max_model_len: int, cap: int, budget: int
) -> None:
    config = make_config(
        runtime_dir, max_model_len=max_model_len, serving_cap_tokens=cap, client_input_token_budget=budget
    )

    assert config.serving_budget == budget


@pytest.mark.parametrize(
    ("max_model_len", "cap", "budget"),
    [(8192, 16384, 16384), (16384, 16384, 4096), (32768, 16384, 32768), (16384, 16384, 16383)],
)
def test_budget_different_from_the_serving_cap_is_refused(
    runtime_dir: Path, max_model_len: int, cap: int, budget: int
) -> None:
    with pytest.raises(VLLMServerConfigError, match="silent truncation"):
        make_config(runtime_dir, max_model_len=max_model_len, serving_cap_tokens=cap, client_input_token_budget=budget)


# --- child environment ---------------------------------------------------------------------------------


def test_child_env_pins_key_offline_cache_and_strips_inherited_switches(runtime_dir: Path) -> None:
    inherited = {
        "PATH": "/usr/bin",
        "HOME": "/home/someone",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_ID": "123",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        KEY_ENV: KEY_VALUE,  # the source variable: the child gets the key only as VLLM_API_KEY
        "OPENAI_API_KEY": "sk-paid-SECRET",
        "OPENAI_LOG": "debug",
        "VLLM_API_KEY": "stale-key",
        "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "HF_TOKEN": "hf_SECRET",
        "HF_HUB_OFFLINE": "0",
        "HF_HUB_CACHE": "/elsewhere",
        "HUGGING_FACE_HUB_TOKEN": "hf_SECRET2",
        "TRANSFORMERS_CACHE": "/elsewhere",
        "PYTHONPATH": "/project/src",
        "PYTHONHOME": "/project/python",
        "VIRTUAL_ENV": "/project/.venv",
        "AWS_SECRET_ACCESS_KEY": "aws-SECRET",
        "GITHUB_TOKEN": "gh-SECRET",
        "WANDB_API_KEY": "wandb-SECRET",
        "DB_PASSWORD": "db-SECRET",
        "my_service_credentials": "cred-SECRET",
    }

    env = build_child_env(make_config(runtime_dir), environ=inherited, api_key=KEY_VALUE)

    vllm_env = runtime_dir / "vllm-env"
    assert env == {
        "PATH": f"{vllm_env / 'bin'}{os.pathsep}/usr/bin",
        "VIRTUAL_ENV": str(vllm_env),
        "HOME": "/home/someone",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_ID": "123",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        "HF_HOME": str(runtime_dir / "hf"),
        "HF_HUB_OFFLINE": "1",
        "VLLM_API_KEY": KEY_VALUE,
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
    }
    assert not any("SECRET" in value for name, value in env.items() if name != "VLLM_API_KEY")


def test_child_env_drops_the_key_source_variable_whatever_its_name(runtime_dir: Path) -> None:
    config = make_config(runtime_dir, api_key_env="SERVER_AUTH")

    env = build_child_env(config, environ={"SERVER_AUTH": KEY_VALUE}, api_key=KEY_VALUE)

    assert "SERVER_AUTH" not in env and list(env.values()).count(KEY_VALUE) == 1


def test_child_env_without_an_inherited_path_still_activates_the_vllm_env(runtime_dir: Path) -> None:
    env = build_child_env(make_config(runtime_dir), environ={}, api_key=KEY_VALUE)

    assert env["PATH"] == str(runtime_dir / "vllm-env" / "bin")


# --- lifecycle -----------------------------------------------------------------------------------------


def test_server_starts_waits_for_authenticated_readiness_and_stops(runtime_dir: Path) -> None:
    config = make_config(runtime_dir)
    health = [raise_transport(httpx.ConnectError), json_response(503, {}), ready()]
    popen, killpg, server, clock, manager = start(config, health)

    with manager as endpoint:
        assert endpoint == ServerEndpoint(
            uds_path=config.uds_path, base_url=None, served_model_name=SERVED, pid=FAKE_PID
        )
        assert killpg.calls == []  # still running inside the block

    argv, kwargs = popen.calls[0]
    assert argv == list(build_vllm_argv(config))
    (lock_fd,) = kwargs["pass_fds"]  # the server inherits the socket lock
    assert isinstance(lock_fd, int) and lock_fd >= 0
    assert kwargs["env"]["VLLM_API_KEY"] == KEY_VALUE and kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["start_new_session"] is True and kwargs["stdin"] is vllm_server.subprocess.DEVNULL
    assert [(r.method, r.path) for r in server.requests] == [("GET", "/v1/models")] * 3
    assert all(r.headers["authorization"] == f"Bearer {KEY_VALUE}" for r in server.requests)
    assert killpg.calls == [(FAKE_PID, signal.SIGTERM)]
    assert len(clock.sleeps) == 2


def test_socket_directory_is_created_with_mode_0700(runtime_dir: Path) -> None:
    config = make_config(runtime_dir)
    old_umask = os.umask(0o000)  # a permissive umask must not widen the directory
    try:
        _, _, _, _, manager = start(config, [ready()])
        with manager:
            mode = stat.S_IMODE(os.lstat(config.uds_path.parent).st_mode)
    finally:
        os.umask(old_umask)

    assert mode == 0o700


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o711, 0o770])
def test_existing_insecure_socket_directory_is_refused_and_left_unchanged(runtime_dir: Path, mode: int) -> None:
    config = make_config(runtime_dir)
    config.uds_path.parent.mkdir()
    os.chmod(config.uds_path.parent, mode)
    popen, _, _, _, manager = start(config, [ready()])

    with pytest.raises(VLLMServerConfigError, match="0700"):
        enter_and_exit(manager)

    assert popen.calls == []
    assert stat.S_IMODE(os.lstat(config.uds_path.parent).st_mode) == mode


def test_socket_directory_owned_by_another_user_is_refused(runtime_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(runtime_dir)
    config.uds_path.parent.mkdir(mode=0o700)
    other_uid = os.getuid() + 1
    monkeypatch.setattr(vllm_server.os, "getuid", lambda: other_uid)
    popen, _, _, _, manager = start(config, [ready()])

    with pytest.raises(VLLMServerConfigError, match="owned"):
        enter_and_exit(manager)

    assert popen.calls == []


def test_stale_own_socket_is_removed_before_start_and_after_stop(
    runtime_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    short = tmp_path_factory.mktemp("u")  # AF_UNIX paths are limited to ~107 bytes
    config = make_config(runtime_dir, uds_path=short / "s" / "c.sock")
    config.uds_path.parent.mkdir(mode=0o700)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(config.uds_path))
    stale.close()
    _, _, _, _, manager = start(config, [ready()])

    with manager:
        assert not config.uds_path.exists()  # removed before vllm serve could fail to bind
        stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stale.bind(str(config.uds_path))  # the "server" socket
        stale.close()

    assert not config.uds_path.exists()


def test_second_launcher_on_the_same_socket_path_is_refused(
    runtime_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    short = tmp_path_factory.mktemp("l")
    config = make_config(runtime_dir, uds_path=short / "s" / "c.sock")
    first_popen, _, _, _, first = start(config, [ready()])
    second_popen, _, _, _, second = start(config, [ready()])

    with first:
        with pytest.raises(VLLMServerConfigError, match="another launcher"):
            enter_and_exit(second)
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.bind(str(config.uds_path))  # the first server's socket is untouched by the refusal
        live.close()

    assert len(first_popen.calls) == 1 and second_popen.calls == []


def test_socket_that_still_accepts_connections_is_refused_and_left_alone(
    runtime_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    short = tmp_path_factory.mktemp("v")
    config = make_config(runtime_dir, uds_path=short / "s" / "c.sock")
    config.uds_path.parent.mkdir(mode=0o700)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(config.uds_path))
    listener.listen(1)
    popen, _, _, _, manager = start(config, [ready()])
    try:
        with pytest.raises(VLLMServerConfigError, match="already listening"):
            enter_and_exit(manager)
        assert config.uds_path.exists() and popen.calls == []
    finally:
        listener.close()


def test_socket_path_beyond_the_af_unix_limit_is_refused_before_start(runtime_dir: Path) -> None:
    config = make_config(runtime_dir, uds_path=runtime_dir / ("d" * 110) / "c.sock")
    popen, _, _, _, manager = start(config, [ready()])

    with pytest.raises(VLLMServerConfigError, match="AF_UNIX"):
        enter_and_exit(manager)

    assert popen.calls == []


def test_non_socket_file_at_the_socket_path_is_refused(runtime_dir: Path) -> None:
    config = make_config(runtime_dir)
    config.uds_path.parent.mkdir(mode=0o700)
    config.uds_path.write_text("not a socket", encoding="utf-8")
    popen, _, _, _, manager = start(config, [ready()])

    with pytest.raises(VLLMServerConfigError, match="not a socket"):
        enter_and_exit(manager)

    assert popen.calls == [] and config.uds_path.read_text(encoding="utf-8") == "not a socket"


def test_tcp_server_is_probed_on_literal_loopback(runtime_dir: Path) -> None:
    config = make_config(runtime_dir, **tcp(port=8123))
    popen, _, server, _, manager = start(config, [ready()])

    with manager as endpoint:
        assert endpoint.base_url == "http://127.0.0.1:8123" and endpoint.uds_path is None

    assert server.requests[0].headers["host"] == "127.0.0.1:8123"
    assert popen.calls[0][1]["pass_fds"] == ()  # no socket, no lock


def test_teardown_runs_with_sigterm_and_sigint_held_back(runtime_dir: Path) -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGKILL}))  # ignores SIGTERM: teardown takes a while
    masks: list[set[int]] = []
    group = GroupSignals(process)

    def killpg(pgid: int, sig: int) -> None:
        masks.append(set(signal.pthread_sigmask(signal.SIG_BLOCK, [])))
        group(pgid, sig)

    clock = FakeClock()
    server = SystemOneServer([ready()])
    manager = running_vllm_server(
        make_config(runtime_dir),
        popen_factory=RecordingPopen(process),
        clock=clock,
        sleep=clock.sleep,
        killpg=killpg,
        transport=server.transport,
    )
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])

    enter_and_exit(manager)

    assert masks and all({signal.SIGTERM, signal.SIGINT} <= mask for mask in masks)
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


def test_child_exiting_before_ready_is_a_startup_error_without_signals(runtime_dir: Path) -> None:
    process = FakeProcess(exit_after_polls=1, exit_code=3)
    popen, killpg, _, _, manager = start(make_config(runtime_dir), [raise_transport(httpx.ConnectError)] * 5, process)

    with pytest.raises(VLLMServerStartupError, match="exited with code 3"):
        enter_and_exit(manager)

    assert len(popen.calls) == 1 and killpg.calls == []


def test_server_not_ready_within_the_timeout_is_stopped(runtime_dir: Path) -> None:
    config = make_config(runtime_dir, startup_timeout_s=9.0)
    health = [raise_transport(httpx.ConnectError)] * 20
    _, killpg, server, clock, manager = start(config, health)

    with pytest.raises(VLLMServerStartupError, match="not ready within 9.0 s"):
        enter_and_exit(manager)

    assert clock.now == pytest.approx(9.0) and sum(clock.sleeps) == pytest.approx(9.0)
    assert len(server.requests) == 5  # probes at t = 0, 2, 4, 6, 8
    assert killpg.calls == [(FAKE_PID, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (json_response(401, {"error": {"message": f"bad key {KEY_VALUE}"}}), "401"),
        (json_response(404, {}), "HTTP 404"),
        (json_response(200, models_body("another-model")), "served_model_name"),
        (json_response(200, {"object": "list"}), "data"),
        (httpx.Response(200, content=b"<html>", headers={"Content-Type": "text/html"}), "non-JSON"),
    ],
)
def test_wrong_server_at_the_endpoint_is_a_startup_error(
    runtime_dir: Path, response: httpx.Response, match: str
) -> None:
    _, killpg, _, _, manager = start(make_config(runtime_dir), [response])

    with pytest.raises(VLLMServerStartupError, match=match) as excinfo:
        enter_and_exit(manager)

    assert KEY_VALUE not in str(excinfo.value)
    assert killpg.calls == [(FAKE_PID, signal.SIGTERM)]


def test_failure_inside_the_block_still_stops_the_server(runtime_dir: Path) -> None:
    _, killpg, _, _, manager = start(make_config(runtime_dir), [ready()])

    with pytest.raises(RuntimeError, match="client failed"), manager:
        raise RuntimeError("client failed")

    assert killpg.calls == [(FAKE_PID, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"executable": Path("/nonexistent-env/bin/vllm")}, "executable"),
        ({"hf_home": Path("/nonexistent-hf-cache")}, "hf_home"),
    ],
)
def test_missing_env_or_cache_is_refused_before_start(runtime_dir: Path, overrides: dict[str, Any], match: str) -> None:
    popen, _, _, _, manager = start(make_config(runtime_dir, **overrides), [ready()])

    with pytest.raises(VLLMServerConfigError, match=match):
        enter_and_exit(manager)

    assert popen.calls == []


@pytest.mark.parametrize("value", [None, "", "has space", "line\nbreak"])
def test_missing_or_unsafe_key_is_refused_before_start_without_echo(
    runtime_dir: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(KEY_ENV)
    else:
        monkeypatch.setenv(KEY_ENV, value)
    popen, _, _, _, manager = start(make_config(runtime_dir), [ready()])

    with pytest.raises(VLLMServerConfigError) as excinfo:
        enter_and_exit(manager)

    assert popen.calls == [] and KEY_ENV not in str(excinfo.value)


# --- teardown -------------------------------------------------------------------------------------------


def stop(
    process: FakeProcess, *, timeout_s: float = 20.0, missing_group: bool = False
) -> tuple[int, GroupSignals, FakeClock]:
    killpg, clock = GroupSignals(process, missing_group=missing_group), FakeClock()
    code = stop_server(process, timeout_s=timeout_s, killpg=killpg, clock=clock, sleep=clock.sleep)
    return code, killpg, clock


def test_teardown_is_terminate_wait_then_kill_wait() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGKILL}))  # ignores SIGTERM

    code, killpg, clock = stop(process)

    assert code == -signal.SIGKILL
    assert killpg.calls == [(FAKE_PID, signal.SIGTERM), (FAKE_PID, signal.SIGKILL)]
    assert clock.now == pytest.approx(20.0)  # the full SIGTERM grace, then an immediate exit


def test_teardown_of_a_graceful_server_never_kills() -> None:
    code, killpg, clock = stop(FakeProcess(exits_on=frozenset({signal.SIGTERM})))

    assert code == -signal.SIGTERM and killpg.calls == [(FAKE_PID, signal.SIGTERM)] and clock.now == 0.0


def test_workers_surviving_the_leader_are_killed_with_the_group() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGTERM}), workers_exit_on=frozenset({signal.SIGKILL}))

    code, killpg, clock = stop(process)

    assert code == -signal.SIGTERM and not process.workers_alive
    assert killpg.calls == [(FAKE_PID, signal.SIGTERM), (FAKE_PID, signal.SIGKILL)]
    assert clock.now == pytest.approx(20.0)


def test_leader_that_already_exited_still_gets_its_group_swept() -> None:
    process = FakeProcess(exit_after_polls=0, exit_code=1, workers_exit_on=frozenset({signal.SIGTERM}))

    code, killpg, _ = stop(process)

    assert code == 1 and not process.workers_alive and killpg.calls == [(FAKE_PID, signal.SIGTERM)]


def test_teardown_reports_a_group_that_survives_sigkill() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGTERM}), workers_exit_on=frozenset())

    with pytest.raises(VLLMServerError, match="survived SIGKILL"):
        stop(process, timeout_s=1.0)


def test_teardown_falls_back_to_the_process_when_it_has_no_own_group() -> None:
    process = FakeProcess(exits_on=frozenset({signal.SIGTERM}))

    code, _, _ = stop(process, missing_group=True)

    assert code == -signal.SIGTERM and process.signals == [signal.SIGTERM]


def test_teardown_of_an_exited_server_with_an_empty_group_sends_nothing() -> None:
    code, killpg, _ = stop(FakeProcess(exit_after_polls=0, exit_code=0))

    assert code == 0 and killpg.calls == [] and killpg.probes == 1


def test_config_is_frozen_and_replace_revalidates(runtime_dir: Path) -> None:
    config = make_config(runtime_dir)

    with pytest.raises(FrozenInstanceError):
        config.revision = "main"  # type: ignore[misc]
    with pytest.raises(VLLMServerConfigError):
        replace(config, revision="main")
