"""clef-flash, the change source of online runs: one System One choice question per tick (D9, D26, D27).

The state text is the config's ``change.state`` filled with the detector's own inputs (the same
INPUTS functions, on ``View(t)``). The question asks ``none`` or ``change``; a tick adds a change
reason when the choice is ``change`` and ``p(change) >= threshold``. vLLM truncates an over-long
state silently, so every state is bounded before it is sent (UTF-8 bytes plus fixed reserves) and
the server's reported input tokens are checked after. The server is local: a Unix socket in a 0700
directory owned by the user (default, D39) or a loopback TCP root with a Bearer key from an
environment variable NAME. Messages never echo a URL, a key or a body; nothing is retried.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from translate.episode import Target, View
from translate.inputs import INPUTS
from translate.prompts import fill

QUESTION_ID = "point_kind"
LABELS = ("none", "change")
FRAMING_RESERVE_TOKENS = 1024 + 64  # the server's fixed framing plus one question's labels
MAX_BYTES_PER_CHAR = 4  # worst case for a previous intent in the budget preflight
LOOPBACK = ("127.0.0.1", "::1")


class DetectorError(Exception):
    """The detector could not be asked, or answered badly. ``uncertain``: it may have run (a timeout)."""

    def __init__(self, message: str, *, uncertain: bool = False, raw: Any = None) -> None:
        super().__init__(message)
        self.uncertain, self.raw = uncertain, raw


def request_body(change: Any, state: str) -> dict[str, Any]:
    question = {"type": "choice", "instructions": change.question.instructions}
    criteria = {"none": change.question.none, "change": change.question.change}
    return {"model": change.model, "state": state, "questions": {QUESTION_ID: {**question, "criteria": criteria}}}


def token_bound(body: dict[str, Any]) -> int:
    """Upper bound on prompt tokens: byte-level BPE never yields more tokens than UTF-8 bytes."""
    questions = json.dumps(body["questions"], ensure_ascii=False, separators=(",", ":"))
    return len(body["state"].encode("utf-8")) + len(questions.encode("utf-8")) + FRAMING_RESERVE_TOKENS


def state_text(change: Any, options: dict[str, Any], view: View, target: Target, base: dict[str, Any]) -> str:
    """``change.state`` filled with the base values and the detector's own inputs at t."""
    values = {name: INPUTS[name].render(view, target, opts) for name, opts in options.items()}
    return fill(change.state, {**base, **values})


def check_budget(change: Any, body: dict[str, Any], intent_reserve_chars: int = 0) -> int:
    """The bound of a state (plus a worst-case intent for the preflight); over the budget is an error."""
    bound = token_bound(body) + MAX_BYTES_PER_CHAR * intent_reserve_chars
    if bound >= change.input_token_budget:
        raise DetectorError(f"detector state bound {bound} >= input_token_budget {change.input_token_budget}")
    return bound


def parse_decision(raw: Any, change: Any) -> tuple[str, dict[str, float], dict[str, int]]:
    """(label, probabilities, usage) of a ``/v1/systemone`` answer; a malformed one is an error."""
    try:
        answer = raw["answers"][QUESTION_ID]
        probabilities = {label: float(answer["probabilities"][label]) for label in LABELS}
        usage = {key: int(raw["usage"][key]) for key in ("input_tokens", "output_tokens")}
        ok = answer["type"] == "choice" and answer["choice"] in LABELS and set(answer["probabilities"]) == set(LABELS)
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok or not all(0 <= p <= 1 for p in probabilities.values()):
        raise DetectorError("the detector answer does not match the question", raw=raw)
    if usage["input_tokens"] >= change.input_token_budget:
        raise DetectorError(
            "the server reports input_tokens >= input_token_budget: the state may be truncated", raw=raw
        )
    return answer["choice"], probabilities, usage


# --- the local endpoint ---------------------------------------------------------------------------


def check_socket(path: Path) -> None:
    """The parent is a real directory owned by the user with mode 0700, and no ancestor can be swapped."""
    parent = path.parent
    info = os.lstat(parent) if parent.exists() else None
    if not path.is_absolute() or info is None or os.path.realpath(parent) != str(parent):
        raise DetectorError("the socket directory must exist as an absolute, canonical path")
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise DetectorError("the socket directory must be a real directory owned by you with mode 0700")
    for ancestor in parent.parents:
        mode = os.lstat(ancestor)
        if mode.st_uid not in (0, os.getuid()) or (
            mode.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not mode.st_mode & stat.S_ISVTX
        ):
            raise DetectorError("a directory above the socket is writable by other users")


def check_tcp(base_url: str) -> str:
    """A loopback server root: host literally 127.0.0.1 or ::1, no credentials, path, query or fragment."""
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
    except ValueError:
        parts, host = urlsplit(""), None  # the parser's message could echo the URL
    if host not in LOOPBACK or parts.scheme not in ("http", "https") or "@" in parts.netloc:
        raise DetectorError("base_url must be http(s)://127.0.0.1:<port> or [::1] without credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise DetectorError("base_url must be the server root")
    return f"{parts.scheme}://{parts.netloc}"


def read_key(name: str | None) -> str | None:
    value = os.environ.get(name) if name else None
    if name and (not value or not value.isascii() or not value.isprintable() or " " in value):
        raise DetectorError("the variable named by api_key_env must hold a printable key without spaces")
    return value


class SystemOneClient:
    """``POST /v1/systemone`` over the configured local endpoint; the checks run before any transport."""

    def __init__(self, change: Any, transport: httpx.BaseTransport | None = None) -> None:
        socket = change.socket_path
        if socket is not None:
            check_socket(socket)
        base_url = "http://localhost" if socket is not None else check_tcp(change.base_url)
        key = read_key(change.api_key_env)
        headers = {"Accept": "application/json"} | ({"Authorization": f"Bearer {key}"} if key else {})
        transport = transport or httpx.HTTPTransport(uds=str(socket) if socket else None, retries=0)
        self.change = change
        self.http = httpx.Client(
            base_url=base_url,
            transport=transport,
            timeout=change.timeout_s,
            headers=headers,
            trust_env=False,
            follow_redirects=False,
        )

    def _send(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        content = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            response = self.http.request(method, path, content=content, headers={"Content-Type": "application/json"})
        except httpx.TimeoutException:
            raise DetectorError(f"System One {path} timed out; not retried", uncertain=True) from None
        except httpx.HTTPError as exc:
            raise DetectorError(f"System One {path} failed ({type(exc).__name__})") from None
        try:
            raw = response.json()
        except ValueError:
            raw = None
        if response.status_code != httpx.codes.OK or not isinstance(raw, dict):
            raise DetectorError(f"System One {path} returned HTTP {response.status_code}", raw=raw)
        return raw

    def health(self) -> None:
        """``GET /v1/models`` must list the configured model."""
        served = {item.get("id") for item in self._send("GET", "/v1/models").get("data", []) if isinstance(item, dict)}
        if self.change.model not in served:
            raise DetectorError("the configured detector model is not served by this endpoint")

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        check_budget(self.change, body)
        return self._send("POST", "/v1/systemone", body)
