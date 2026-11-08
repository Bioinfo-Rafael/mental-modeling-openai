"""System One part: endpoint security, enforced before any transport exists."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from translate.llm import system_one
from translate.llm.system_one import SystemOneConfigError

from .conftest import SYSTEMONE_KEY_ENV
from .llm_fakes import SystemOneServer
from .test_llm_system_one import make_client

pytestmark = pytest.mark.usefixtures("systemone_key")


def _never_called() -> SystemOneServer:
    return SystemOneServer([])


# --- endpoint security ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:pw@127.0.0.1:8000",
        "http://user@127.0.0.1:8000",
        "http://127.0.0.1:8000@evil.example.com",
        "http://0.0.0.0:8000",
        "http://10.0.0.5:8000",
        "http://example.com:8000",
        "http://localhost:8000",
        "http://127.0.0.2:8000",
        "http://[::]:8000",
        "ftp://127.0.0.1:8000",
        "127.0.0.1:8000",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:8000/?x=1",
        "http://127.0.0.1:8000/#frag",
        "http://127.0.0.1:notaport",
        "http://127.0.0.1:8000:9000",
        "http://127.0.0.1:0",
        "http://127.0.0.1.:8000",
        "http://[::1%25eth0]:8000",
        "http://[0:0:0:0:0:0:0:1]:8000",
        "http://127.0.0.1%3A8000",
        " http://127.0.0.1:8000",
        "http://127.0.0.1:8000\n",
        "http://127.0.0.1:8000\t",
        "http://user:hunter2@exam\uff03ple.com",
        "http://user:hunter2@[::1:8000",
        "",
    ],
)
def test_unsafe_or_malformed_base_url_is_rejected_before_any_request(base_url: str) -> None:
    server = _never_called()

    with pytest.raises(SystemOneConfigError):
        make_client(server, base_url=base_url)

    assert server.requests == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:hunter2@127.0.0.1:8000",
        "http://user:hunter2@exam\uff03ple.com",  # urlsplit's own ValueError would echo the netloc
        "http://user:hunter2@[::1:8000",
    ],
)
def test_rejection_of_userinfo_never_echoes_the_password(base_url: str) -> None:
    with pytest.raises(SystemOneConfigError) as excinfo:
        make_client(_never_called(), base_url=base_url)

    assert "hunter2" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize("api_key_env", [None, ""])
def test_tcp_without_api_key_env_is_rejected(api_key_env: str | None) -> None:
    with pytest.raises(SystemOneConfigError, match="api_key_env"):
        make_client(_never_called(), api_key_env=api_key_env)


def test_api_key_env_naming_an_unset_variable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UNSET_SYSTEMONE_KEY", raising=False)

    with pytest.raises(SystemOneConfigError):
        make_client(_never_called(), api_key_env="UNSET_SYSTEMONE_KEY")


@pytest.mark.parametrize("name", ["sk-not-a-name", "has space", "1STARTS_WITH_DIGIT"])
def test_api_key_env_that_is_not_a_variable_name_is_rejected_without_echo(name: str) -> None:
    with pytest.raises(SystemOneConfigError) as excinfo:
        make_client(_never_called(), api_key_env=name)

    assert name not in str(excinfo.value)


@pytest.mark.parametrize("bad_key", ["line\nbreak", "secret\n", "has space", "tab\tkey", "nön-ascii"])
def test_key_value_unsafe_for_a_header_is_rejected_without_echo(monkeypatch: pytest.MonkeyPatch, bad_key: str) -> None:
    monkeypatch.setenv(SYSTEMONE_KEY_ENV, bad_key)

    with pytest.raises(SystemOneConfigError) as excinfo:
        make_client(_never_called())

    assert bad_key not in str(excinfo.value)


def test_tcp_and_uds_together_are_rejected(secure_uds_path: Path) -> None:
    with pytest.raises(SystemOneConfigError, match="exactly one"):
        make_client(_never_called(), uds_path=secure_uds_path)


def test_neither_tcp_nor_uds_is_rejected() -> None:
    with pytest.raises(SystemOneConfigError, match="exactly one"):
        make_client(_never_called(), base_url=None)


@pytest.mark.parametrize("mode", [0o755, 0o750, 0o711, 0o701, 0o770, 0o500])
def test_uds_parent_not_mode_0700_is_rejected(secure_uds_path: Path, mode: int) -> None:
    os.chmod(secure_uds_path.parent, mode)

    with pytest.raises(SystemOneConfigError, match="0700"):
        make_client(_never_called(), base_url=None, uds_path=secure_uds_path)


def test_uds_parent_owned_by_another_user_is_rejected(secure_uds_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    other_uid = os.getuid() + 1
    monkeypatch.setattr(system_one.os, "getuid", lambda: other_uid)

    with pytest.raises(SystemOneConfigError, match="owned"):
        make_client(_never_called(), base_url=None, uds_path=secure_uds_path)


def test_uds_parent_symlink_is_rejected(secure_uds_path: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(secure_uds_path.parent, target_is_directory=True)

    with pytest.raises(SystemOneConfigError, match="symlink"):
        make_client(_never_called(), base_url=None, uds_path=link / "clef.sock")


def test_uds_missing_parent_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemOneConfigError):
        make_client(_never_called(), base_url=None, uds_path=tmp_path / "missing" / "clef.sock")


def test_uds_relative_path_is_rejected() -> None:
    with pytest.raises(SystemOneConfigError, match="absolute"):
        make_client(_never_called(), base_url=None, uds_path=Path("relative/clef.sock"))


@pytest.mark.parametrize("overrides", [{"model": ""}, {"timeout_s": 0.0}, {"timeout_s": float("nan")}])
def test_invalid_model_or_timeout_is_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(SystemOneConfigError):
        make_client(_never_called(), **overrides)


def test_uds_path_through_an_intermediate_symlink_is_rejected(secure_uds_path: Path, tmp_path: Path) -> None:
    link = tmp_path / "via-link"
    link.symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(SystemOneConfigError, match="canonical"):
        make_client(_never_called(), base_url=None, uds_path=link / "vllm" / "clef.sock")


def test_uds_path_with_dotdot_component_is_rejected(secure_uds_path: Path) -> None:
    dotted = secure_uds_path.parent / ".." / "vllm" / "clef.sock"

    with pytest.raises(SystemOneConfigError, match="canonical"):
        make_client(_never_called(), base_url=None, uds_path=dotted)


@pytest.mark.parametrize("mode", [0o777, 0o775, 0o757])
def test_uds_ancestor_writable_by_others_without_sticky_bit_is_rejected(secure_uds_path: Path, mode: int) -> None:
    ancestor = secure_uds_path.parent.parent
    os.chmod(ancestor, mode)

    with pytest.raises(SystemOneConfigError, match="sticky"):
        make_client(_never_called(), base_url=None, uds_path=secure_uds_path)


def test_uds_ancestor_writable_by_others_with_sticky_bit_is_accepted(secure_uds_path: Path) -> None:
    os.chmod(secure_uds_path.parent.parent, 0o1777)

    client = make_client(SystemOneServer([]), base_url=None, uds_path=secure_uds_path)

    assert client.model == "clef-flash"


def test_uds_ancestor_owned_by_another_user_is_rejected(secure_uds_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ancestor = secure_uds_path.parent.parent
    real_lstat = os.lstat
    other_uid = os.getuid() + 1

    def fake_lstat(path: Any) -> os.stat_result:
        info = real_lstat(path)
        if Path(path) != ancestor:
            return info
        fields = list(info)
        fields[4] = other_uid  # st_uid
        return os.stat_result(fields)

    monkeypatch.setattr(system_one.os, "lstat", fake_lstat)

    with pytest.raises(SystemOneConfigError, match="owned by root or the current user"):
        make_client(_never_called(), base_url=None, uds_path=secure_uds_path)


@pytest.mark.parametrize("use_uds", [True, False])
def test_client_builds_its_own_transport_without_connecting(secure_uds_path: Path, use_uds: bool) -> None:
    endpoint: dict[str, Any] = {"base_url": None, "uds_path": secure_uds_path} if use_uds else {}

    with make_client(SystemOneServer([]), transport=None, **endpoint) as client:
        assert client.input_token_budget == 16384


# --- public validators shared with the Translator config (D38) ---------------------------------------


@pytest.mark.parametrize(
    "name",
    ["vllm_api_key", "deadbeefcafe0123", "0123456789ABCDEF", "SK-PROJ-KEY", "A" * 65, "", None, 7],
)
def test_checked_env_name_rejects_key_like_values_without_echo(name: object) -> None:
    with pytest.raises(SystemOneConfigError, match="NAME") as excinfo:
        system_one.checked_env_name(name)

    assert not (isinstance(name, str) and name and name in str(excinfo.value))


@pytest.mark.parametrize("name", ["VLLM_API_KEY", "_K", "K2", "A" * 64])
def test_checked_env_name_accepts_posix_names(name: str) -> None:
    assert system_one.checked_env_name(name) == name


@pytest.mark.parametrize(
    ("base_url", "root"),
    [("http://127.0.0.1:8000/", "http://127.0.0.1:8000"), ("https://[::1]:8443", "https://[::1]:8443")],
)
def test_checked_tcp_url_returns_the_server_root(base_url: str, root: str) -> None:
    assert system_one.checked_tcp_url(base_url) == root


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:SECRETPW@127.0.0.1:8000",
        "http://user" + chr(0xFF1A) + "SECRETPW@127.0.0.1:8000",  # fullwidth colon (NFKC ":")
        "http://SECRETPW" + chr(0xFF20) + "127.0.0.1:8000",  # fullwidth at sign (NFKC "@")
    ],
)
def test_checked_tcp_url_never_echoes_credentials(base_url: str) -> None:
    with pytest.raises(SystemOneConfigError) as excinfo:
        system_one.checked_tcp_url(base_url)

    assert "SECRETPW" not in str(excinfo.value) and excinfo.value.__cause__ is None
