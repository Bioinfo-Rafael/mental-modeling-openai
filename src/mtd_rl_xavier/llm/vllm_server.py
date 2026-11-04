"""Local vLLM launcher: build the ``vllm serve`` argv, start it, wait until ready, tear it down.

- Allowlist only. ``VLLMServerConfig`` has one typed field per flag and ``build_vllm_argv`` emits
  exactly these: the positional model, ``--revision`` (a full commit sha), ``--tokenizer-revision``
  (the same sha; UNVERIFIED flag), ``--served-model-name``, ``--max-model-len``, ``--dtype``, and
  either ``--uds <path>`` or ``--host 127.0.0.1 --port N``. There is no passthrough: no
  ``--api-key``, no ``--enable-log-requests`` (request logging stays off), no
  ``--trust-remote-code``, never ``0.0.0.0``. Values are pattern-checked, so none can start with
  ``-`` and smuggle in another flag. CLI errors never echo argument values.
- Secrets: the Bearer key is read from the variable named by ``api_key_env`` and handed to the
  child ONLY as ``VLLM_API_KEY`` in its environment (never argv, which ``ps`` shows to every user).
- Child environment: the inherited one minus every ``VLLM_*``, ``OPENAI_*``, ``HF_*``,
  ``HUGGING_FACE_*``, ``HUGGINGFACE_*`` and ``TRANSFORMERS_*`` variable, minus ``api_key_env``
  itself and every name containing KEY, TOKEN, SECRET, PASSWORD, PASSWD or CREDENTIAL (no paid key,
  no duplicate of the server key, no hidden vLLM debug or logging switch), and minus
  ``PYTHONPATH`` / ``PYTHONHOME``; plus ``HF_HOME`` from the config, ``HF_HUB_OFFLINE=1``,
  ``VLLM_API_KEY`` and the usage-statistics opt-outs. The separate vLLM env is activated for the
  server only: its ``bin`` first on ``PATH`` and ``VIRTUAL_ENV`` set to it.
- Socket: the parent is created with mode 0700 if missing; an existing insecure one is refused.
  An exclusive lock next to the socket is held for the server's lifetime, so a second launcher on
  the same path fails instead of unlinking a live socket (``mtd_rl_xavier.local_endpoint``).
- Budget: the client's ``input_token_budget`` must equal ``min(serving_cap_tokens, max_model_len)``
  (clef-flash: 16384), otherwise the client's post-hoc truncation check could not see truncation.
- Lifecycle: start (own session) -> authenticated ``GET /v1/models`` until the served name is
  listed, the child exits, or the startup timeout passes -> SIGTERM to the process group -> wait
  until the child AND every group member are gone -> SIGKILL the group -> wait. The group is swept
  even when the child already exited, so leftover workers cannot keep the GPU. SIGTERM and SIGINT
  are held back while teardown runs. The server inherits the socket lock, so the lock lives as long
  as the server does. ``/health`` is not used: vLLM leaves it unauthenticated.
- Tests inject the subprocess factory, clock, sleep, signal function and HTTP transport, so no
  vLLM, GPU or network is needed.

CLI: ``python -m mtd_rl_xavier.llm.vllm_server <flags> -- <client command ...>`` starts the server,
runs the client command (no shell, same environment, own process group) once it is ready, stops the
server and exits with the client's exit code. ``--dry-run`` runs the read-only preflight and prints
the argv (never the key). The first SIGTERM or SIGINT unwinds through teardown; later ones are
ignored so they cannot cut it short.

This module imports no sibling module of ``mtd_rl_xavier.llm`` (the LLM parts stay independent).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, NoReturn, Protocol, TypeVar

import httpx

from mtd_rl_xavier import local_endpoint
from mtd_rl_xavier.local_endpoint import LocalEndpointError

VLLM_EXECUTABLE_NAME = "vllm"
VLLM_SUBCOMMAND = "serve"
LOOPBACK_HOST = "127.0.0.1"
ALLOWED_DTYPES = ("bfloat16",)  # the only dtype the research doc verified for clef-flash
CLEF_FLASH_MAX_LENGTH = 16384  # ClefFlashStrategy.MAX_LENGTH: the System One serving cap
MAX_MODEL_LEN_LIMIT = 262144  # clef-flash max_position_embeddings
MIN_PORT = 1024
MAX_PORT = 65535
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")  # a full commit sha; used with fullmatch only
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
SERVED_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
CHILD_API_KEY_ENV = "VLLM_API_KEY"
CHILD_HF_HOME_ENV = "HF_HOME"
CHILD_OFFLINE_ENV = "HF_HUB_OFFLINE"
# Usage-statistics opt-outs (UNVERIFIED names for vLLM 0.33.0; harmless if unknown).
CHILD_PRIVACY_ENV = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
STRIPPED_ENV_PREFIXES = ("VLLM_", "OPENAI_", "HF_", "HUGGING_FACE_", "HUGGINGFACE_", "TRANSFORMERS_")
# The project's Python must not leak into the separate env; VIRTUAL_ENV is set to that env instead.
STRIPPED_ENV_NAMES = frozenset({"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"})
CREDENTIAL_NAME_PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL")  # used with search
MODELS_PATH = "/v1/models"
HEALTH_POLL_INTERVAL_S = 2.0
HEALTH_PROBE_TIMEOUT_S = 5.0
STOP_POLL_INTERVAL_S = 0.5
KILL_WAIT_S = 30.0
CLIENT_INTERRUPT_WAIT_S = 10.0
PROBE_SIGNAL = 0  # kill(…, 0): existence check, delivers nothing
DEFERRED_SIGNALS = frozenset({signal.SIGTERM, signal.SIGINT})  # held back while teardown runs
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
CLIENT_ERROR_RANGE = range(400, 500)
TRANSPORT_RETRIES = 0
ARGV_SEPARATOR = "--"
EXIT_FAILED = 1
EXIT_USAGE = 2
SIGNAL_EXIT_BASE = 128
REQUIRED_ARGUMENTS_PREFIX = "the following arguments are required"  # argparse text: flag names only
ARGUMENT_FLAG_PATTERN = re.compile(r"argument (--[a-z-]+)")  # used with match only
T = TypeVar("T")


class VLLMServerError(Exception):
    """Launcher failure. Messages carry fixed text, codes and paths; never the key or a response body."""


class VLLMServerConfigError(VLLMServerError):
    """Invalid launcher configuration or failed preflight; raised before any process starts."""


class VLLMServerStartupError(VLLMServerError):
    """The server exited, refused the key, served another model, or was not ready in time."""


class ServerProcess(Protocol):
    """The part of ``subprocess.Popen`` the launcher uses for the server and the client (tests pass fakes)."""

    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def send_signal(self, sig: int) -> None: ...


PopenFactory = Callable[..., ServerProcess]
KillPg = Callable[[int, int], None]


@dataclass(frozen=True)
class VLLMServerConfig:
    executable: Path  # absolute path to `vllm` in the separate vLLM env (never the project .venv)
    model: str  # Hugging Face repo id, e.g. Cloudflare/clef-flash
    revision: str  # full 40-hex commit sha
    served_model_name: str  # the "model" the clients send
    max_model_len: int
    dtype: str
    uds_path: Path | None  # preferred; exactly one of uds_path or host + port
    host: str | None
    port: int | None
    hf_home: Path  # Hugging Face cache (HF_HOME) holding the pinned snapshot; served offline
    api_key_env: str  # NAME of the variable holding the key; the child gets it as VLLM_API_KEY
    client_input_token_budget: int  # the client's input_token_budget; must equal the serving budget
    serving_cap_tokens: int  # the model's own prompt cap (clef-flash: 16384)
    startup_timeout_s: float
    shutdown_timeout_s: float

    def __post_init__(self) -> None:
        _check_path(self.executable, "executable")
        if self.executable.name != VLLM_EXECUTABLE_NAME:
            raise VLLMServerConfigError("executable must be the `vllm` program of the separate vLLM env")
        _check_pattern(self.model, MODEL_ID_PATTERN, "model must be a Hugging Face repo id (owner/name)")
        _check_pattern(self.revision, REVISION_PATTERN, "revision must be a full 40-hex commit sha (pinned)")
        _check_pattern(self.served_model_name, SERVED_NAME_PATTERN, "served_model_name must match [A-Za-z0-9._-]")
        _check_int_range(self.max_model_len, 1, MAX_MODEL_LEN_LIMIT, "max_model_len")
        if not isinstance(self.dtype, str) or self.dtype not in ALLOWED_DTYPES:
            raise VLLMServerConfigError(f"dtype must be one of {list(ALLOWED_DTYPES)}")
        _check_endpoint_fields(self.uds_path, self.host, self.port)
        _check_path(self.hf_home, "hf_home")
        _as_config_error(lambda: local_endpoint.checked_env_name(self.api_key_env))
        _check_int_range(self.serving_cap_tokens, 1, MAX_MODEL_LEN_LIMIT, "serving_cap_tokens")
        _check_int_range(self.client_input_token_budget, 1, MAX_MODEL_LEN_LIMIT, "client_input_token_budget")
        if self.client_input_token_budget != self.serving_budget:
            raise VLLMServerConfigError(
                f"client input_token_budget {self.client_input_token_budget} != serving budget "
                f"min(serving_cap_tokens {self.serving_cap_tokens}, max_model_len {self.max_model_len}) = "
                f"{self.serving_budget}; the client could not detect silent truncation"
            )
        for value, name in (
            (self.startup_timeout_s, "startup_timeout_s"),
            (self.shutdown_timeout_s, "shutdown_timeout_s"),
        ):
            if not _is_positive_finite(value):
                raise VLLMServerConfigError(f"{name} must be a positive finite number")

    @property
    def serving_budget(self) -> int:
        """The server's real prompt cap: ``min(serving_cap_tokens, max_model_len)``."""
        return min(self.serving_cap_tokens, self.max_model_len)


@dataclass(frozen=True)
class ServerEndpoint:
    """Where the ready server listens. Exactly one of ``uds_path`` / ``base_url`` is set."""

    uds_path: Path | None
    base_url: str | None  # loopback TCP root, e.g. http://127.0.0.1:8000
    served_model_name: str
    pid: int


def build_vllm_argv(config: VLLMServerConfig) -> tuple[str, ...]:
    """The complete ``vllm serve`` argv. Nothing else can be added (no passthrough)."""
    if config.uds_path is not None:
        endpoint: tuple[str, ...] = ("--uds", str(config.uds_path))
    else:
        endpoint = ("--host", LOOPBACK_HOST, "--port", str(config.port))
    return (
        str(config.executable),
        VLLM_SUBCOMMAND,
        config.model,
        "--revision",
        config.revision,
        "--tokenizer-revision",  # UNVERIFIED for vLLM 0.33.0: not in the research doc
        config.revision,
        "--served-model-name",
        config.served_model_name,
        "--max-model-len",
        str(config.max_model_len),
        "--dtype",
        config.dtype,
        *endpoint,
    )


def build_child_env(config: VLLMServerConfig, *, environ: Mapping[str, str], api_key: str) -> dict[str, str]:
    """The server's environment (see the module docstring): stripped, pinned, separate env activated."""
    kept = {name: value for name, value in environ.items() if not _is_stripped(name, config.api_key_env)}
    env_bin = config.executable.parent
    inherited_path = kept.get("PATH")
    return {
        **kept,
        **CHILD_PRIVACY_ENV,
        "PATH": str(env_bin) if not inherited_path else f"{env_bin}{os.pathsep}{inherited_path}",
        "VIRTUAL_ENV": str(env_bin.parent),
        CHILD_HF_HOME_ENV: str(config.hf_home),
        CHILD_OFFLINE_ENV: "1",  # UNVERIFIED: that this alone serves the pinned sha from HF_HOME
        CHILD_API_KEY_ENV: api_key,
    }


def _is_stripped(name: str, api_key_env: str) -> bool:
    return (
        name == api_key_env
        or name.startswith(STRIPPED_ENV_PREFIXES)
        or name in STRIPPED_ENV_NAMES
        or CREDENTIAL_NAME_PATTERN.search(name.upper()) is not None
    )


def preflight(config: VLLMServerConfig) -> str:
    """Read-only checks a start needs (no directory is created, no lock taken); returns the key.

    The key variable must be set and header-safe, the executable and the HF cache must exist, and an
    existing socket directory must already be private. ``--dry-run`` runs exactly this.
    """
    api_key = _as_config_error(lambda: local_endpoint.read_api_key(config.api_key_env, required=True))
    if api_key is None:  # read_api_key only returns None when not required
        raise VLLMServerConfigError("the environment variable named by api_key_env is unset or empty")
    if not config.executable.is_file() or not os.access(config.executable, os.X_OK):
        raise VLLMServerConfigError("executable is not an executable file (create the separate vLLM env first)")
    if not config.hf_home.is_dir():
        raise VLLMServerConfigError("hf_home is not a directory (download the pinned snapshot first)")
    uds_path = config.uds_path
    if uds_path is not None:
        _as_config_error(lambda: local_endpoint.check_socket_path_length(uds_path))
        if uds_path.parent.exists():
            _as_config_error(lambda: local_endpoint.check_uds_parent(uds_path))
    return api_key


@contextmanager
def running_vllm_server(
    config: VLLMServerConfig,
    *,
    popen_factory: PopenFactory = subprocess.Popen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    killpg: KillPg = os.killpg,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[ServerEndpoint]:
    """Start ``vllm serve``, yield once it is ready, and always stop it and its process group."""
    api_key = preflight(config)
    lock_fd = _acquire_socket(config.uds_path)
    process: ServerProcess | None = None
    try:
        process = popen_factory(
            list(build_vllm_argv(config)),
            env=build_child_env(config, environ=os.environ, api_key=api_key),
            stdin=subprocess.DEVNULL,
            # Its own process group, so teardown reaches vLLM's worker processes (UNVERIFIED that every
            # vLLM 0.33.0 worker stays in this group; SLURM ends any stray process with the job).
            start_new_session=True,
            close_fds=True,
            # The server inherits the lock, so it stays held for as long as the server lives, even if
            # this launcher is killed (UNVERIFIED that vLLM 0.33.0 keeps inherited descriptors open;
            # if not, the liveness check in acquire_socket_path is the remaining guard).
            pass_fds=() if lock_fd is None else (lock_fd,),
        )
        with _health_client(config, api_key, transport) as http:
            _wait_until_ready(process, http, config, clock=clock, sleep=sleep)
        yield ServerEndpoint(
            uds_path=config.uds_path,
            base_url=None if config.uds_path is not None else f"http://{LOOPBACK_HOST}:{config.port}",
            served_model_name=config.served_model_name,
            pid=process.pid,
        )
    finally:
        with signals_deferred():  # a SIGTERM now must not cut the teardown short; it is handled after
            try:
                if process is not None:
                    stop_server(process, timeout_s=config.shutdown_timeout_s, killpg=killpg, clock=clock, sleep=sleep)
            finally:
                if config.uds_path is not None and lock_fd is not None:
                    local_endpoint.release_socket_path(config.uds_path, lock_fd)


def stop_server(
    process: ServerProcess,
    *,
    timeout_s: float,
    killpg: KillPg = os.killpg,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Stop the child and every process in its group; returns the child's exit code.

    SIGTERM to the group, wait up to ``timeout_s`` until the child has exited AND the group is empty,
    then SIGKILL the group and wait up to ``KILL_WAIT_S``. A child that already exited still gets its
    group swept. The group id stays reserved while any member lives, so the sweep cannot hit an
    unrelated group while workers remain.
    """
    steps = ((signal.SIGTERM, timeout_s), (signal.SIGKILL, KILL_WAIT_S))
    code = _stop_group(process, steps, killpg=killpg, clock=clock, sleep=sleep)
    if code is not None:
        return code
    raise VLLMServerError(f"vllm serve (pid {process.pid}) or a process in its group survived SIGKILL")


@contextmanager
def signals_deferred() -> Iterator[None]:
    """Block SIGTERM and SIGINT for the block; a pending one is handled as soon as it ends.

    Used only around teardown. No process is started inside, so no child inherits the mask.
    """
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, DEFERRED_SIGNALS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


# --- start-up helpers -----------------------------------------------------------------------------


def _acquire_socket(uds_path: Path | None) -> int | None:
    """Create the private socket directory and take the per-path lock (UDS only)."""
    if uds_path is None:
        return None
    _as_config_error(lambda: local_endpoint.ensure_private_socket_dir(uds_path))
    # UNVERIFIED: that `vllm serve --uds` cannot bind over an existing socket file (why a stale one is
    # removed here) was not checked against vLLM 0.33.0.
    return _as_config_error(lambda: local_endpoint.acquire_socket_path(uds_path))


def _health_client(config: VLLMServerConfig, api_key: str, transport: httpx.BaseTransport | None) -> httpx.Client:
    uds = str(config.uds_path) if config.uds_path is not None else None
    base_url = local_endpoint.UDS_BASE_URL if uds else f"http://{LOOPBACK_HOST}:{config.port}"
    return httpx.Client(
        base_url=base_url,
        transport=transport or httpx.HTTPTransport(uds=uds, retries=TRANSPORT_RETRIES, trust_env=False),
        headers={"Accept": "application/json", "Authorization": f"Bearer {api_key}"},
        trust_env=False,  # no proxy or .netrc for a local server
        follow_redirects=False,
    )


def _wait_until_ready(
    process: ServerProcess,
    http: httpx.Client,
    config: VLLMServerConfig,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    deadline = clock() + config.startup_timeout_s
    while True:
        code = process.poll()
        if code is not None:
            raise VLLMServerStartupError(f"vllm serve exited with code {code} before it was ready")
        remaining = deadline - clock()
        if remaining <= 0:
            raise VLLMServerStartupError(f"vllm serve was not ready within {config.startup_timeout_s} s")
        if _probe_ready(http, config.served_model_name, timeout_s=min(HEALTH_PROBE_TIMEOUT_S, remaining)):
            return
        sleep(max(0.0, min(HEALTH_POLL_INTERVAL_S, deadline - clock())))


def _probe_ready(http: httpx.Client, served_model_name: str, *, timeout_s: float) -> bool:
    """True once ``/v1/models`` lists the served name; False while the server is still starting.

    UNVERIFIED: that vLLM 0.33.0 refuses connections (or answers 5xx) until the model is loaded.
    """
    try:
        response = http.get(MODELS_PATH, timeout=timeout_s)
    except httpx.TransportError:  # not listening yet: socket missing, connection refused, timeout
        return False
    status = response.status_code
    if status == HTTP_UNAUTHORIZED:
        raise VLLMServerStartupError(
            "/v1/models rejected the Bearer key (HTTP 401); another server may own the endpoint"
        )
    if status in CLIENT_ERROR_RANGE:
        raise VLLMServerStartupError(f"/v1/models returned HTTP {status}")
    if status != HTTP_OK:
        return False  # 5xx while the engine starts
    if served_model_name not in _served_ids(response):
        raise VLLMServerStartupError("/v1/models does not list the configured served_model_name")
    return True


def _served_ids(response: httpx.Response) -> set[str]:
    try:
        raw = response.json()
    except ValueError:
        raw = None
    if raw is None:
        raise VLLMServerStartupError("/v1/models returned a non-JSON body")
    data = raw.get("data") if isinstance(raw, dict) else None
    if not isinstance(data, list):
        raise VLLMServerStartupError("/v1/models response has no 'data' list")
    return {item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)}


# --- process groups -------------------------------------------------------------------------------


def _signal_group(process: ServerProcess, sig: int, killpg: KillPg) -> None:
    """Signal the child's process group; fall back to the child itself if it has no own group."""
    try:
        killpg(process.pid, sig)
    except ProcessLookupError:
        if process.poll() is None:
            process.send_signal(sig)


def _group_alive(pgid: int, killpg: KillPg) -> bool:
    try:
        killpg(pgid, PROBE_SIGNAL)
    except ProcessLookupError:
        return False
    return True


def _gone_code(process: ServerProcess, killpg: KillPg) -> int | None:
    """The child's exit code once it has exited and no process is left in its group, else None."""
    code = process.poll()
    if code is None or _group_alive(process.pid, killpg):
        return None
    return code


def _wait_until_gone(
    process: ServerProcess,
    killpg: KillPg,
    wait_s: float,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> int | None:
    deadline = clock() + wait_s
    while True:
        code = _gone_code(process, killpg)
        if code is not None:
            return code
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        sleep(min(STOP_POLL_INTERVAL_S, remaining))


def _stop_group(
    process: ServerProcess,
    steps: tuple[tuple[int, float], ...],
    *,
    killpg: KillPg,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> int | None:
    """For each ``(signal, wait_s)``: signal the group and wait until the child has exited and the
    group is empty. Returns the child's exit code, or None if the group outlived every step."""
    code = _gone_code(process, killpg)
    for sig, wait_s in steps:
        if code is not None:
            return code
        _signal_group(process, sig, killpg)
        code = _wait_until_gone(process, killpg, wait_s, clock=clock, sleep=sleep)
    return code


# --- validation -----------------------------------------------------------------------------------


def _as_config_error(check: Callable[[], T]) -> T:
    """Run a shared endpoint check; its fixed message is re-raised as ``VLLMServerConfigError`` outside
    the ``except`` clause, so no exception is chained."""
    try:
        return check()
    except LocalEndpointError as exc:
        message = str(exc)
    raise VLLMServerConfigError(message)


def _check_path(value: object, name: str) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise VLLMServerConfigError(f"{name} must be an absolute Path")


def _check_pattern(value: object, pattern: re.Pattern[str], message: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise VLLMServerConfigError(message)


def _check_int_range(value: object, low: int, high: int, name: str) -> None:
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if not is_int or not low <= value <= high:
        raise VLLMServerConfigError(f"{name} must be an integer in {low}..{high}")


def _is_positive_finite(value: object) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value > 0


def _check_endpoint_fields(uds_path: object, host: object, port: object) -> None:
    tcp_given = host is not None or port is not None
    if (uds_path is None) == (not tcp_given):
        raise VLLMServerConfigError("set exactly one of uds_path or host + port")
    if uds_path is not None:
        _check_path(uds_path, "uds_path")
        return
    if host == "0.0.0.0":
        raise VLLMServerConfigError("host must not be 0.0.0.0 (all interfaces); use 127.0.0.1")
    if host != LOOPBACK_HOST:
        raise VLLMServerConfigError("host must be literally 127.0.0.1; other hosts are rejected")
    _check_int_range(port, MIN_PORT, MAX_PORT, "port")


# --- CLI ------------------------------------------------------------------------------------------


ServerFactory = Callable[[VLLMServerConfig], AbstractContextManager[ServerEndpoint]]


class QuietArgumentParser(argparse.ArgumentParser):
    """argparse whose errors never echo argument values (a mistaken ``--api-key <key>`` must not
    reach the job log). Flag names are kept; values are replaced by fixed text."""

    def error(self, message: str) -> NoReturn:
        if message.startswith(REQUIRED_ARGUMENTS_PREFIX):
            shown = message  # flag names only
        elif (flag := ARGUMENT_FLAG_PATTERN.match(message)) is not None:
            shown = f"argument {flag.group(1)}: invalid value (not shown)"
        else:
            shown = "unknown or malformed arguments (not shown); the launcher accepts no passthrough flags"
        self.usage_error(shown)

    def usage_error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> QuietArgumentParser:
    parser = QuietArgumentParser(
        prog="python -m mtd_rl_xavier.llm.vllm_server",
        description="Start vllm serve, run one client command once it is ready, then stop the server.",
        usage="%(prog)s [flags] -- CLIENT_COMMAND [ARG ...]",
        allow_abbrev=False,  # no prefix matching: every flag is spelt out
    )
    parser.add_argument("--vllm-executable", type=Path, required=True, help="the separate env's bin/vllm")
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--revision", required=True, help="full 40-hex commit sha")
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--dtype", required=True, choices=ALLOWED_DTYPES)
    parser.add_argument("--uds-path", type=Path, help="Unix socket (preferred); parent created 0700")
    parser.add_argument("--host", help="loopback TCP only: 127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--hf-home", type=Path, required=True, help="HF cache holding the pinned snapshot")
    parser.add_argument("--api-key-env", required=True, help="NAME of the variable holding the Bearer key")
    parser.add_argument("--client-input-token-budget", type=int, required=True)
    parser.add_argument("--serving-cap-tokens", type=int, default=CLEF_FLASH_MAX_LENGTH)
    parser.add_argument("--startup-timeout-s", type=float, required=True)
    parser.add_argument("--shutdown-timeout-s", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true", help="run the read-only preflight, print the argv")
    return parser


def config_from_args(args: argparse.Namespace) -> VLLMServerConfig:
    return VLLMServerConfig(
        executable=args.vllm_executable,
        model=args.model,
        revision=args.revision,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        uds_path=args.uds_path,
        host=args.host,
        port=args.port,
        hf_home=args.hf_home,
        api_key_env=args.api_key_env,
        client_input_token_budget=args.client_input_token_budget,
        serving_cap_tokens=args.serving_cap_tokens,
        startup_timeout_s=args.startup_timeout_s,
        shutdown_timeout_s=args.shutdown_timeout_s,
    )


def split_client_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Launcher flags before the first ``--``; the client command after it (empty if no ``--``)."""
    if ARGV_SEPARATOR not in argv:
        return list(argv), []
    index = argv.index(ARGV_SEPARATOR)
    return list(argv[:index]), list(argv[index + 1 :])


def run_client_command(
    command: list[str],
    *,
    popen_factory: PopenFactory = subprocess.Popen,
    killpg: KillPg = os.killpg,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run the client without a shell, in the launcher's environment, in its own process group.

    If the launcher is interrupted (Ctrl-C, or SIGTERM from ``scancel`` / the time limit), the client's
    group first gets SIGINT, so a Python CLI can record its own interrupted status, then SIGKILL if
    the client or any process in its group is still there after ``CLIENT_INTERRUPT_WAIT_S``. The
    interruption is then re-raised and the server is stopped.
    """
    process = popen_factory(command, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        with signals_deferred():
            steps = ((signal.SIGINT, CLIENT_INTERRUPT_WAIT_S), (signal.SIGKILL, KILL_WAIT_S))
            _stop_group(process, steps, killpg=killpg, clock=clock, sleep=sleep)
        raise


def exit_status(code: int) -> int:
    """A process exit status for ``sys.exit``: a child killed by signal N (code -N) becomes 128 + N."""
    return SIGNAL_EXIT_BASE - code if code < 0 else code


def dry_run_report(config: VLLMServerConfig) -> dict[str, Any]:
    """What a real start would execute. Never includes the key value."""
    pinned = ["PATH", "VIRTUAL_ENV", CHILD_HF_HOME_ENV, CHILD_OFFLINE_ENV, CHILD_API_KEY_ENV, *CHILD_PRIVACY_ENV]
    return {
        "argv": list(build_vllm_argv(config)),
        "child_env_set": pinned,
        "child_env_stripped_prefixes": list(STRIPPED_ENV_PREFIXES),
        "child_env_stripped_names": sorted({*STRIPPED_ENV_NAMES, config.api_key_env}),
        "child_env_stripped_pattern": CREDENTIAL_NAME_PATTERN.pattern,
        "serving_budget": config.serving_budget,
        "preflight": "ok",
    }


def main(
    argv: list[str] | None = None,
    *,
    server_factory: ServerFactory = running_vllm_server,
    run_client: Callable[[list[str]], int] = run_client_command,
) -> int:
    parser = build_parser()
    launcher_argv, client_command = split_client_command(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(launcher_argv)  # unknown flags exit 2: nothing is passed through
    if not args.dry_run and not client_command:
        parser.usage_error(f"a client command is required after '{ARGV_SEPARATOR}'")
    try:
        config = config_from_args(args)
        if args.dry_run:
            preflight(config)
            print(json.dumps(dry_run_report(config), indent=2))
            return 0
        with server_factory(config) as endpoint:
            where = endpoint.uds_path if endpoint.uds_path is not None else endpoint.base_url
            print(f"vllm_server: ready at {where} (pid {endpoint.pid}); running the client", file=sys.stderr)
            code = exit_status(run_client(client_command))
        print(f"vllm_server: stopped; client exit code {code}", file=sys.stderr)
        return code
    except VLLMServerError as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED
    except KeyboardInterrupt:  # Ctrl-C: the server was already stopped on the way out of the block
        print("vllm_server: interrupted; server stopped", file=sys.stderr)
        return SIGNAL_EXIT_BASE + signal.SIGINT


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """The first SIGTERM (scancel, time limit) or SIGINT unwinds through ``finally``, so the server is
    stopped. Both signals are ignored afterwards, so a second one cannot cut the teardown short."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(SIGNAL_EXIT_BASE + signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGINT, _exit_on_signal)


if __name__ == "__main__":
    install_signal_handlers()
    sys.exit(main())
