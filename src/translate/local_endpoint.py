"""Endpoint rules for local model servers on a shared node (shared by the local LLM parts).

The LLM parts in ``translate.llm`` must not import each other (D2), so the endpoint checks
they all need live here, outside that package:

- TCP: ``checked_tcp_url`` accepts only a server root whose host is literally ``127.0.0.1`` or
  ``::1``, with no userinfo, path, query, fragment or whitespace.
- Unix socket: ``check_uds_parent`` requires an absolute, canonical path whose parent is a real
  directory owned by the user with mode 0700, and whose ancestors are owned by root or the user and
  are not writable by others unless sticky. ``ensure_private_socket_dir`` creates that parent
  (0700) when it is missing and then applies the same check, so an existing insecure one is refused.
- Keys: ``checked_env_name`` accepts only a POSIX variable NAME, and ``read_api_key`` reads the
  value from the environment and requires it to be safe for an HTTP header.
- Serving on a socket: ``acquire_socket_path`` takes an exclusive ``flock`` next to the socket for
  the server's lifetime (one launcher per path), refuses a socket that still accepts connections,
  and removes a stale one; ``release_socket_path`` removes our socket and drops the lock.

Every message is fixed text: it never echoes a URL, a variable name, a key or a path, and no
error is chained to an exception that could (``__cause__`` and ``__context__`` stay empty). All
errors are ``LocalEndpointError``; each client re-raises the message as its own configuration
error type, outside any ``except`` clause, so nothing is chained there either.
"""

from __future__ import annotations

import fcntl
import os
import re
import socket
import stat
from contextlib import suppress
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
ALLOWED_SCHEMES = frozenset({"http", "https"})
UDS_BASE_URL = "http://localhost"  # httpx needs a URL; the Unix socket transport ignores the host
SECURE_DIR_MODE = 0o700
ROOT_UID = 0
# POSIX portable environment variable name (uppercase letters, digits, underscore). Typical keys
# (hex, base64, "sk-...") contain lowercase letters or punctuation, so a pasted value is rejected.
ENV_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")  # used with fullmatch only
HEADER_SAFE_KEY_PATTERN = re.compile(r"[\x21-\x7e]+")  # used with fullmatch only
MAX_UDS_PATH_BYTES = 107  # sockaddr_un.sun_path is 108 bytes on Linux, including the final NUL
LOCK_SUFFIX = ".lock"
LOCK_FILE_MODE = 0o600
SOCKET_PROBE_TIMEOUT_S = 1.0


class LocalEndpointError(ValueError):
    """An endpoint, socket directory or key variable is unsafe or malformed (fixed message text)."""


def checked_endpoint(base_url: str | None, uds_path: Path | None) -> str:
    """Validate exactly one of a loopback TCP root or a Unix socket; return the httpx base URL."""
    if (base_url is None) == (uds_path is None):
        raise LocalEndpointError("set exactly one of base_url (loopback TCP) or uds_path")
    if uds_path is not None:
        check_uds_parent(Path(uds_path))
        return UDS_BASE_URL
    return checked_tcp_url(base_url)


def checked_tcp_url(base_url: object) -> str:
    """The loopback server root ``scheme://host:port``. Pure; messages are fixed and never echo the URL."""
    if not isinstance(base_url, str) or not base_url:
        raise LocalEndpointError("base_url must be a non-empty string")
    if any(char.isspace() or not char.isprintable() for char in base_url):
        raise LocalEndpointError("base_url must not contain whitespace or control characters")
    split = _split_url(base_url)
    if split is None:  # raised outside the except clause: urlsplit's error can echo a password
        raise LocalEndpointError("base_url is not a valid URL")
    parts, hostname, port = split
    if parts.scheme not in ALLOWED_SCHEMES:
        raise LocalEndpointError("base_url must start with http:// or https://")
    if "@" in parts.netloc:
        raise LocalEndpointError("base_url must not contain credentials (userinfo)")
    if hostname == "0.0.0.0":
        raise LocalEndpointError("base_url must not use 0.0.0.0; use 127.0.0.1 or ::1")
    if hostname not in LOOPBACK_HOSTS:
        raise LocalEndpointError("base_url host must be literally 127.0.0.1 or ::1; remote hosts are rejected")
    if port == 0:
        raise LocalEndpointError("base_url has an invalid port")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise LocalEndpointError("base_url must be the server root (no path, query or fragment)")
    return f"{parts.scheme}://{parts.netloc}"


def _split_url(base_url: str) -> tuple[SplitResult, str | None, int | None] | None:
    """``urlsplit`` plus host and port, or None. The ``ValueError`` is dropped here, never chained,
    because its text can contain the netloc (and so a password)."""
    try:
        parts = urlsplit(base_url)
        return parts, parts.hostname, parts.port
    except ValueError:
        return None


def check_uds_parent(uds_path: Path) -> None:
    """Parent: real directory, owned by the user, mode 0700. Ancestors: no symlink or ``..``, owned
    by root or the user, and not writable by others unless sticky, so no other user can swap a
    component between this check and the connection."""
    if not uds_path.is_absolute():
        raise LocalEndpointError("uds_path must be absolute")
    parent = uds_path.parent
    info = _lstat_or_none(parent)
    if info is None:
        raise LocalEndpointError("uds_path parent directory does not exist")
    if os.path.realpath(parent) != str(parent):
        raise LocalEndpointError("uds_path must be canonical: no symlink or '..' components")
    if not stat.S_ISDIR(info.st_mode):
        raise LocalEndpointError("uds_path parent must be a real directory, not a symlink or file")
    if info.st_uid != os.getuid():
        raise LocalEndpointError("uds_path parent directory must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != SECURE_DIR_MODE:
        raise LocalEndpointError("uds_path parent directory must have mode 0700")
    for ancestor in parent.parents:
        _check_uds_ancestor(ancestor)


def ensure_private_socket_dir(uds_path: Path) -> None:
    """Create the socket's parent with mode 0700 if it is missing, then apply ``check_uds_parent``.

    An existing parent is never modified: if another user created it first, or its mode is not
    0700, the check refuses it. Only the last component is created; its own parent must exist.
    """
    if not uds_path.is_absolute():
        raise LocalEndpointError("uds_path must be absolute")
    _make_private_dir(uds_path.parent)
    check_uds_parent(uds_path)


def _make_private_dir(directory: Path) -> bool:
    """``mkdir`` with mode 0700 (exact, whatever the umask); False when the path already exists."""
    try:
        os.mkdir(directory, SECURE_DIR_MODE)
    except FileExistsError:
        return False  # an existing directory is checked by the caller, never changed
    except OSError:
        raise LocalEndpointError("uds_path parent directory could not be created") from None
    os.chmod(directory, SECURE_DIR_MODE)
    return True


def check_socket_path_length(uds_path: Path) -> None:
    """AF_UNIX addresses are short: a longer path could neither be bound nor probed."""
    if len(os.fsencode(uds_path)) > MAX_UDS_PATH_BYTES:
        raise LocalEndpointError(f"uds_path is longer than {MAX_UDS_PATH_BYTES} bytes (the AF_UNIX limit)")


def acquire_socket_path(uds_path: Path) -> int:
    """Lock ``uds_path`` for one server's lifetime and clear a stale socket; returns the lock fd.

    Call after ``ensure_private_socket_dir``. The lock (``<socket>.lock``, mode 0600, never deleted)
    is an exclusive non-blocking ``flock``: a second launcher on the same path fails at once instead
    of unlinking a live socket. An existing socket that still accepts connections (a server that no
    launcher holds, e.g. one left by a killed launcher) is refused; one that refuses them is stale
    and removed. Anything at the path that is not a socket owned by the user is refused.
    """
    check_socket_path_length(uds_path)
    lock_fd = _locked_fd(uds_path.with_name(uds_path.name + LOCK_SUFFIX))
    try:
        _clear_stale_socket(uds_path)
    except LocalEndpointError:
        os.close(lock_fd)
        raise
    return lock_fd


def release_socket_path(uds_path: Path, lock_fd: int) -> None:
    """After the server has stopped: remove our socket (if any) and release the lock."""
    try:
        if _is_own_socket(uds_path):
            with suppress(FileNotFoundError):
                os.unlink(uds_path)
    finally:
        os.close(lock_fd)


def _locked_fd(lock_path: Path) -> int:
    try:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, LOCK_FILE_MODE)
    except OSError:
        lock_fd = None
    if lock_fd is None:
        raise LocalEndpointError("the uds_path lock file could not be opened")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        lock_fd = None
    if lock_fd is None:
        raise LocalEndpointError("another launcher is serving on uds_path (its lock is held)")
    return lock_fd


def _clear_stale_socket(uds_path: Path) -> None:
    if _lstat_or_none(uds_path) is None:
        return
    if not _is_own_socket(uds_path):
        raise LocalEndpointError("uds_path exists and is not a socket owned by the current user")
    if _socket_accepts_connections(uds_path):
        raise LocalEndpointError("a server is already listening on uds_path")
    with suppress(FileNotFoundError):
        os.unlink(uds_path)


def _is_own_socket(uds_path: Path) -> bool:
    info = _lstat_or_none(uds_path)
    return info is not None and stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid()


def _socket_accepts_connections(uds_path: Path) -> bool:
    """False only when the connection is refused or the socket is gone; anything else counts as live."""
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(SOCKET_PROBE_TIMEOUT_S)
    try:
        probe.connect(str(uds_path))
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError:
        return True  # e.g. a timeout: cannot tell, so never remove it
    finally:
        probe.close()
    return True


def _check_uds_ancestor(directory: Path) -> None:
    info = _lstat_or_none(directory)
    if info is None or info.st_uid not in (ROOT_UID, os.getuid()):
        raise LocalEndpointError("every directory above uds_path must be owned by root or the current user")
    writable_by_others = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not info.st_mode & stat.S_ISVTX:
        raise LocalEndpointError("a directory above uds_path is writable by other users without the sticky bit")


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError:
        return None


def checked_env_name(api_key_env: object) -> str:
    """A POSIX environment variable NAME (``[A-Z_][A-Z0-9_]{0,63}``). Pure; the message never echoes it."""
    if not isinstance(api_key_env, str) or not ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise LocalEndpointError(
            "api_key_env must be an environment variable NAME (A-Z, 0-9, _; at most 64), not a key value"
        )
    return api_key_env


def read_api_key(api_key_env: str | None, *, required: bool) -> str | None:
    """Read the Bearer key from the named variable. Messages never echo the name or the value."""
    if not api_key_env:
        if required:
            raise LocalEndpointError("TCP endpoints require api_key_env (the env variable holding the Bearer key)")
        return None
    value = os.environ.get(checked_env_name(api_key_env))
    if not value:
        raise LocalEndpointError("the environment variable named by api_key_env is unset or empty")
    if not HEADER_SAFE_KEY_PATTERN.fullmatch(value):
        raise LocalEndpointError("the Bearer key must be printable ASCII without whitespace")
    return value
