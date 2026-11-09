"""Shared local-endpoint rules (``translate.local_endpoint``) and parity of the clients using them."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

import pytest

from translate import local_endpoint
from translate.llm import system_one, vllm_server
from translate.local_endpoint import LocalEndpointError

KEY_ENV = "TEST_LOCAL_ENDPOINT_KEY"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# --- socket directory ---------------------------------------------------------------------------------


def test_missing_socket_directory_is_created_0700_whatever_the_umask(tmp_path: Path) -> None:
    socket_path = tmp_path / "vllm" / "clef.sock"
    old_umask = os.umask(0o000)
    try:
        local_endpoint.ensure_private_socket_dir(socket_path)
    finally:
        os.umask(old_umask)

    assert _mode(socket_path.parent) == 0o700


def test_existing_private_directory_is_accepted_unchanged(secure_uds_path: Path) -> None:
    before = os.lstat(secure_uds_path.parent)

    local_endpoint.ensure_private_socket_dir(secure_uds_path)

    assert os.lstat(secure_uds_path.parent).st_ino == before.st_ino and _mode(secure_uds_path.parent) == 0o700


@pytest.mark.parametrize("mode", [0o755, 0o711, 0o770, 0o500])
def test_existing_insecure_directory_is_refused_not_repaired(secure_uds_path: Path, mode: int) -> None:
    os.chmod(secure_uds_path.parent, mode)

    with pytest.raises(LocalEndpointError, match="0700"):
        local_endpoint.ensure_private_socket_dir(secure_uds_path)

    assert _mode(secure_uds_path.parent) == mode


def test_directory_created_by_another_user_is_refused(secure_uds_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other_uid = os.getuid() + 1
    monkeypatch.setattr(local_endpoint.os, "getuid", lambda: other_uid)

    with pytest.raises(LocalEndpointError, match="owned by the current user"):
        local_endpoint.ensure_private_socket_dir(secure_uds_path)


def test_symlinked_socket_directory_is_refused(secure_uds_path: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(secure_uds_path.parent, target_is_directory=True)

    with pytest.raises(LocalEndpointError, match="canonical"):
        local_endpoint.ensure_private_socket_dir(link / "clef.sock")


def test_directory_under_a_missing_parent_is_not_created(tmp_path: Path) -> None:
    with pytest.raises(LocalEndpointError, match="could not be created"):
        local_endpoint.ensure_private_socket_dir(tmp_path / "missing" / "vllm" / "clef.sock")

    assert not (tmp_path / "missing").exists()


def test_relative_socket_path_is_refused_before_anything_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(LocalEndpointError, match="absolute"):
        local_endpoint.ensure_private_socket_dir(Path("vllm/clef.sock"))

    assert not (tmp_path / "vllm").exists()


# --- endpoint and key ------------------------------------------------------------------------------------


def test_checked_endpoint_returns_the_socket_placeholder_or_the_tcp_root(secure_uds_path: Path) -> None:
    assert local_endpoint.checked_endpoint(None, secure_uds_path) == local_endpoint.UDS_BASE_URL
    assert local_endpoint.checked_endpoint("http://127.0.0.1:8000/", None) == "http://127.0.0.1:8000"


def test_checked_endpoint_needs_exactly_one(secure_uds_path: Path) -> None:
    for base_url, uds_path in (("http://127.0.0.1:8000", secure_uds_path), (None, None)):
        with pytest.raises(LocalEndpointError, match="exactly one"):
            local_endpoint.checked_endpoint(base_url, uds_path)


def test_read_api_key_required_and_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "value-0123")

    assert local_endpoint.read_api_key(KEY_ENV, required=True) == "value-0123"
    assert local_endpoint.read_api_key(None, required=False) is None
    with pytest.raises(LocalEndpointError, match="api_key_env"):
        local_endpoint.read_api_key(None, required=True)


# --- parity: every client re-raises the shared messages as its own config error -------------------------


@pytest.mark.parametrize(
    "base_url",
    ["http://0.0.0.0:8000", "http://user:pw@127.0.0.1:8000", "http://10.0.0.5:8000", "http://127.0.0.1:8000/v1", ""],
)
def test_system_one_tcp_check_keeps_the_shared_message_and_type(base_url: str) -> None:
    with pytest.raises(LocalEndpointError) as shared:
        local_endpoint.checked_tcp_url(base_url)
    with pytest.raises(system_one.SystemOneConfigError) as wrapped:
        system_one.checked_tcp_url(base_url)

    assert str(wrapped.value) == str(shared.value)
    assert wrapped.value.__cause__ is None and wrapped.value.__context__ is None


@pytest.mark.parametrize(
    "base_url",
    ["http://user" + chr(0xFF1A) + "SECRETPW@127.0.0.1:8000", "http://user:SECRETPW@[::1:8000"],
)
def test_unparsable_url_with_a_password_is_not_reachable_through_any_chain(base_url: str) -> None:
    with pytest.raises(LocalEndpointError) as shared:
        local_endpoint.checked_tcp_url(base_url)
    with pytest.raises(system_one.SystemOneConfigError) as wrapped:
        system_one.checked_tcp_url(base_url)

    for error in (shared.value, wrapped.value):
        assert error.__cause__ is None and error.__context__ is None and "SECRETPW" not in str(error)


@pytest.mark.parametrize("name", ["sk-not-a-name", "lower_case", "", None])
def test_system_one_env_name_check_keeps_the_shared_message_and_type(name: object) -> None:
    with pytest.raises(LocalEndpointError) as shared:
        local_endpoint.checked_env_name(name)
    with pytest.raises(system_one.SystemOneConfigError) as wrapped:
        system_one.checked_env_name(name)

    assert str(wrapped.value) == str(shared.value)


def test_system_one_still_accepts_valid_values() -> None:
    assert system_one.checked_tcp_url("https://[::1]:8443") == "https://[::1]:8443"
    assert system_one.checked_env_name("VLLM_API_KEY") == "VLLM_API_KEY"


@pytest.mark.parametrize(
    "config_error",
    [system_one.SystemOneConfigError, vllm_server.VLLMServerConfigError],
)
def test_no_client_exposes_the_shared_error_type(config_error: type[Exception]) -> None:
    assert not issubclass(config_error, LocalEndpointError)


# --- serving on a socket: lock, stale socket, release ----------------------------------------------------


@pytest.fixture
def short_socket(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A socket path in a 0700 directory, short enough for AF_UNIX."""
    directory = tmp_path_factory.mktemp("e") / "s"
    directory.mkdir(mode=0o700)
    return directory / "c.sock"


def _bind(path: Path, *, listen: bool) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    if listen:
        sock.listen(1)
    return sock


def test_lock_is_exclusive_until_released_and_release_removes_our_socket(short_socket: Path) -> None:
    lock_fd = local_endpoint.acquire_socket_path(short_socket)
    _bind(short_socket, listen=False).close()  # what the server leaves behind

    with pytest.raises(LocalEndpointError, match="another launcher"):
        local_endpoint.acquire_socket_path(short_socket)
    local_endpoint.release_socket_path(short_socket, lock_fd)

    assert not short_socket.exists()
    local_endpoint.release_socket_path(short_socket, local_endpoint.acquire_socket_path(short_socket))


def test_stale_socket_is_removed_but_a_listening_one_is_refused(short_socket: Path) -> None:
    _bind(short_socket, listen=False).close()
    local_endpoint.release_socket_path(short_socket, local_endpoint.acquire_socket_path(short_socket))
    listener = _bind(short_socket, listen=True)
    try:
        with pytest.raises(LocalEndpointError, match="already listening"):
            local_endpoint.acquire_socket_path(short_socket)
        assert short_socket.exists()
    finally:
        listener.close()


def test_a_refused_acquire_leaves_the_lock_free(short_socket: Path) -> None:
    short_socket.write_text("not a socket", encoding="utf-8")

    with pytest.raises(LocalEndpointError, match="not a socket"):
        local_endpoint.acquire_socket_path(short_socket)
    short_socket.unlink()

    local_endpoint.release_socket_path(short_socket, local_endpoint.acquire_socket_path(short_socket))


def test_socket_path_length_is_checked(tmp_path: Path) -> None:
    with pytest.raises(LocalEndpointError, match="AF_UNIX"):
        local_endpoint.check_socket_path_length(tmp_path / ("x" * 120))
