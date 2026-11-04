"""System One part: typed decision questions against a local ``POST /v1/systemone`` server.

- Server-agnostic: vLLM >= 0.33 (e.g. ``clef-flash``) and OpenJev ``wev serve``. Only the fields
  both return are read: ``model``, ``answers``, ``usage``; per answer ``type`` plus ``noul`` (noul),
  ``choice`` + ``probabilities`` (choice), ``score`` + ``probabilities`` (score). ``confidence``,
  ``legend``, ``diagnostics``, ``joint``, ``id`` and latency fields are ignored.
- The question set is fixed for the client's lifetime (clef-flash scores questions jointly);
  ``question_set_sha256`` hashes the exact questions JSON that every request carries.
- Endpoint security is enforced here, before any transport exists: exactly one of a loopback TCP
  URL (host literally ``127.0.0.1`` or ``::1``, no userinfo, Bearer key required) or a Unix socket
  whose parent directory is a real directory owned by the user with mode 0700. The key comes from
  an environment variable whose NAME is configured; it never appears in argv, config or errors.
- Token budget: no tokenizer is installed, so ``conservative_input_token_bound`` counts UTF-8
  bytes (byte-level BPE never yields more tokens than bytes) plus fixed framing reserves. A bound
  ``>= input_token_budget`` is rejected before sending; ``usage.input_tokens >= budget`` after the
  response is an error because vLLM silently truncates the END of an over-long state.
- No retries. Error messages never contain the request body, the response body or the key; the
  parsed error body is kept on ``SystemOneError.raw`` for the caller's private records.

This module imports no sibling module of ``mtd_rl_xavier.llm`` (the LLM parts stay independent).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 26
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
CHOICE_OPTION_FIELDS = 2  # (name, description)
# Not configurable: system prompt, chat template, section headers and suffix of the server prompt.
FIXED_FRAMING_RESERVE_TOKENS = 1024
# Not configurable: per-question labels the server adds (FIELD n / ID / TYPE / OPTIONS ...).
PER_QUESTION_FRAMING_RESERVE_TOKENS = 64
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
ALLOWED_SCHEMES = frozenset({"http", "https"})
UDS_BASE_URL = "http://localhost"
SECURE_DIR_MODE = 0o700
ROOT_UID = 0
# POSIX portable environment variable name (uppercase letters, digits, underscore). Typical keys
# (hex, base64, "sk-...") contain lowercase letters or punctuation, so a pasted value is rejected.
ENV_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")  # used with fullmatch only
HEADER_SAFE_KEY_PATTERN = re.compile(r"[\x21-\x7e]+")  # used with fullmatch only
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_NOT_IMPLEMENTED = 501
CLIENT_ERROR_RANGE = range(400, 500)
SERVER_ERROR_RANGE = range(500, 600)


class SystemOneError(Exception):
    """Base error. ``raw`` is the parsed JSON body of a received response, if any (never in the message)."""

    def __init__(self, message: str, *, status_code: int | None = None, raw: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw


class SystemOneConfigError(SystemOneError):
    """Invalid questions, endpoint, key variable, budget or timeout; raised before any transport."""


class SystemOneBudgetError(SystemOneError):
    """Conservative bound ``>= budget`` before sending, or ``usage.input_tokens >= budget`` after."""


class SystemOneRequestError(SystemOneError):
    """Invalid local input, or HTTP 400 / 422 / any other 4xx not mapped below."""


class SystemOneAuthError(SystemOneError):
    """HTTP 401: missing or wrong Bearer key."""


class SystemOneModelError(SystemOneError):
    """HTTP 404 (model name mismatch), or the model is not listed by ``/v1/models``."""


class SystemOneServerError(SystemOneError):
    """HTTP 500 and every other 5xx except 501."""


class SystemOneUnsupportedError(SystemOneError):
    """HTTP 501: the loaded model has no System One strategy."""


class SystemOneTransportError(SystemOneError):
    """Connection failure or timeout; ``timed_out`` tells them apart. Never retried."""

    def __init__(self, message: str, *, timed_out: bool) -> None:
        super().__init__(message)
        self.timed_out = timed_out


class SystemOneProtocolError(SystemOneError):
    """Unexpected status (1xx/2xx/3xx other than 200) or a 200 body missing common fields."""


@dataclass(frozen=True)
class NoulQuestion:
    question_id: str
    instructions: str
    true_criterion: str
    false_criterion: str

    def __post_init__(self) -> None:
        for field in ("question_id", "instructions", "true_criterion", "false_criterion"):
            _require_text(getattr(self, field), field)


@dataclass(frozen=True)
class ChoiceQuestion:
    question_id: str
    instructions: str
    criteria: tuple[tuple[str, str | None], ...]  # (option name, description or None), request order

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_text(self.instructions, "instructions")
        if not isinstance(self.criteria, tuple) or not 1 <= len(self.criteria) <= MAX_CHOICE_OPTIONS:
            raise SystemOneConfigError(f"choice criteria must be a tuple of 1..{MAX_CHOICE_OPTIONS} options")
        for option in self.criteria:
            if not isinstance(option, tuple) or len(option) != CHOICE_OPTION_FIELDS:
                raise SystemOneConfigError("each choice option must be a (name, description) tuple")
            _require_text(option[0], "choice option name")
            if option[1] is not None and not _is_utf8_text(option[1]):
                raise SystemOneConfigError("choice option description must be UTF-8 text or None")
        _require_unique([name for name, _ in self.criteria], "choice option names")


@dataclass(frozen=True)
class ScoreQuestion:
    question_id: str
    instructions: str
    levels: tuple[str, ...]  # level names, lowest first; answers use their 0-based index

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_text(self.instructions, "instructions")
        if not isinstance(self.levels, tuple) or not MIN_SCORE_LEVELS <= len(self.levels) <= MAX_SCORE_LEVELS:
            raise SystemOneConfigError(f"score levels must be a tuple of {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} names")
        for level in self.levels:
            _require_text(level, "score level")
        _require_unique(list(self.levels), "score levels")


SystemOneQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion


@dataclass(frozen=True)
class NoulAnswer:
    question_id: str
    p_yes: float


@dataclass(frozen=True)
class ChoiceAnswer:
    question_id: str
    choice: str
    probabilities: tuple[tuple[str, float], ...]  # in the question's option order


@dataclass(frozen=True)
class ScoreAnswer:
    question_id: str
    score: float  # expected 0-based level index
    probabilities: tuple[tuple[str, float], ...]  # ("0", p0), ("1", p1), ...


SystemOneAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer


@dataclass(frozen=True)
class SystemOneUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class SystemOneResult:
    model: str
    answers: tuple[SystemOneAnswer, ...]  # in question-set order
    usage: SystemOneUsage
    raw: dict[str, Any]

    def answer(self, question_id: str) -> SystemOneAnswer:
        for candidate in self.answers:
            if candidate.question_id == question_id:
                return candidate
        raise KeyError(question_id)


def question_payload(question: SystemOneQuestion) -> dict[str, Any]:
    """The wire form of one question (``type``, ``instructions``, ``criteria``; nothing else)."""
    if isinstance(question, NoulQuestion):
        criteria: Any = {"true": question.true_criterion, "false": question.false_criterion}
        return {"type": "noul", "instructions": question.instructions, "criteria": criteria}
    if isinstance(question, ChoiceQuestion):
        return {"type": "choice", "instructions": question.instructions, "criteria": dict(question.criteria)}
    if isinstance(question, ScoreQuestion):
        return {"type": "score", "instructions": question.instructions, "criteria": list(question.levels)}
    raise SystemOneConfigError("questions must be NoulQuestion, ChoiceQuestion or ScoreQuestion")


def questions_payload(questions: tuple[SystemOneQuestion, ...]) -> dict[str, dict[str, Any]]:
    """The ``questions`` object of a request, keyed by question id in tuple order."""
    if not isinstance(questions, tuple) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise SystemOneConfigError(f"questions must be a tuple of 1..{MAX_QUESTIONS} questions")
    payload = {getattr(question, "question_id", None): question_payload(question) for question in questions}
    if len(payload) != len(questions):
        raise SystemOneConfigError("question ids must be unique")
    return payload


def question_set_sha256(questions: tuple[SystemOneQuestion, ...]) -> str:
    """SHA-256 of the exact questions JSON sent with every request (content and order)."""
    return hashlib.sha256(_canonical_json(questions_payload(questions)).encode("utf-8")).hexdigest()


def conservative_input_token_bound(*, state: str, questions: tuple[SystemOneQuestion, ...]) -> int:
    """Upper bound on prompt tokens: UTF-8 bytes of state + questions JSON + fixed reserves."""
    question_bytes = len(_canonical_json(questions_payload(questions)).encode("utf-8"))
    reserve = FIXED_FRAMING_RESERVE_TOKENS + PER_QUESTION_FRAMING_RESERVE_TOKENS * len(questions)
    return len(state.encode("utf-8")) + question_bytes + reserve


class SystemOneClient:
    """``/v1/systemone`` client for one fixed question set. Validates the endpoint before any transport."""

    def __init__(
        self,
        *,
        base_url: str | None,
        uds_path: Path | None,
        model: str,
        questions: tuple[SystemOneQuestion, ...],
        input_token_budget: int,
        timeout_s: float,
        api_key_env: str | None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        _require_text(model, "model")
        self._questions_payload = questions_payload(questions)
        self._questions = questions
        self._model = model
        self._budget = _checked_budget(input_token_budget, questions)
        self._question_set_sha256 = question_set_sha256(questions)
        if not _is_positive_finite(timeout_s):
            raise SystemOneConfigError("timeout_s must be a positive finite number")
        http_base_url = _checked_endpoint(base_url, uds_path)
        api_key = _read_api_key(api_key_env, required=uds_path is None)
        headers = {"Accept": "application/json"}
        if api_key is not None:
            headers = {**headers, "Authorization": f"Bearer {api_key}"}
        self._http = httpx.Client(
            base_url=http_base_url,
            transport=transport or httpx.HTTPTransport(uds=str(uds_path) if uds_path else None, retries=0),
            timeout=timeout_s,
            headers=headers,
            trust_env=False,  # no proxy or .netrc from the environment for a local server
            follow_redirects=False,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def questions(self) -> tuple[SystemOneQuestion, ...]:
        return self._questions

    @property
    def question_set_sha256(self) -> str:
        return self._question_set_sha256

    @property
    def input_token_budget(self) -> int:
        return self._budget

    def health(self) -> None:
        """Authenticated ``GET /v1/models``; the configured model must be listed."""
        raw = _json_object(self._send("GET", MODELS_PATH, content=None))
        data = raw.get("data")
        if not isinstance(data, list):
            raise SystemOneProtocolError("/v1/models response has no 'data' list", status_code=HTTP_OK, raw=raw)
        served = {item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)}
        if self._model not in served:
            raise SystemOneModelError(
                "the configured model is not served by this endpoint", status_code=HTTP_OK, raw=raw
            )

    def decide(self, state: str) -> SystemOneResult:
        """One ``POST /v1/systemone`` with the fixed question set; budget-checked before and after."""
        self._check_state_fits(state)
        body = {"model": self._model, "state": state, "questions": self._questions_payload}
        raw = _json_object(self._send("POST", SYSTEMONE_PATH, content=_canonical_json(body).encode("utf-8")))
        result = _parse_result(raw, self._questions)
        if result.usage.input_tokens >= self._budget:
            raise SystemOneBudgetError(
                f"server reported input_tokens {result.usage.input_tokens} >= input_token_budget {self._budget}; "
                "the state may have been truncated silently",
                status_code=HTTP_OK,
                raw=raw,
            )
        return result

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SystemOneClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _check_state_fits(self, state: object) -> None:
        if not isinstance(state, str) or not state or not _is_utf8_text(state):
            raise SystemOneRequestError("state must be a non-empty UTF-8 string")
        bound = conservative_input_token_bound(state=state, questions=self._questions)
        if bound >= self._budget:
            raise SystemOneBudgetError(
                f"conservative input bound {bound} >= input_token_budget {self._budget}; request not sent"
            )

    def _send(self, method: str, path: str, *, content: bytes | None) -> httpx.Response:
        headers = {"Content-Type": "application/json"} if content is not None else None
        try:
            response = self._http.request(method, path, content=content, headers=headers)
        except httpx.TimeoutException as exc:
            message = f"System One {path} timed out ({type(exc).__name__}); not retried"
            raise SystemOneTransportError(message, timed_out=True) from exc
        except httpx.TransportError as exc:
            message = f"System One {path} transport failure ({type(exc).__name__})"
            raise SystemOneTransportError(message, timed_out=False) from exc
        except httpx.RequestError as exc:  # e.g. an undecodable response body
            raise SystemOneProtocolError(f"System One {path} response unreadable ({type(exc).__name__})") from exc
        _raise_for_status(response, path)
        return response


# --- validation helpers ---------------------------------------------------------------------------


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or not _is_utf8_text(value):
        raise SystemOneConfigError(f"{field} must be a non-empty UTF-8 string")


def _is_utf8_text(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _require_unique(names: list[str], field: str) -> None:
    if len(set(names)) != len(names):
        raise SystemOneConfigError(f"{field} must be unique")


def _is_positive_finite(value: object) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value > 0


def _checked_budget(budget: object, questions: tuple[SystemOneQuestion, ...]) -> int:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise SystemOneConfigError("input_token_budget must be a positive integer")
    if conservative_input_token_bound(state="", questions=questions) >= budget:
        raise SystemOneConfigError("input_token_budget leaves no room for a state after the framing reserve")
    return budget


def _checked_endpoint(base_url: str | None, uds_path: Path | None) -> str:
    """Validate the endpoint and return the httpx base URL (a fixed placeholder for UDS)."""
    if (base_url is None) == (uds_path is None):
        raise SystemOneConfigError("set exactly one of base_url (loopback TCP) or uds_path")
    if uds_path is not None:
        _check_uds_parent(Path(uds_path))
        return UDS_BASE_URL
    return checked_tcp_url(base_url)


def checked_tcp_url(base_url: object) -> str:
    """The loopback server root ``scheme://host:port``. Pure; messages are fixed and never echo the URL."""
    if not isinstance(base_url, str) or not base_url:
        raise SystemOneConfigError("base_url must be a non-empty string")
    if any(char.isspace() or not char.isprintable() for char in base_url):
        raise SystemOneConfigError("base_url must not contain whitespace or control characters")
    try:
        parts = urlsplit(base_url)
        hostname, port = parts.hostname, parts.port
    except ValueError:
        raise SystemOneConfigError("base_url is not a valid URL") from None
    if parts.scheme not in ALLOWED_SCHEMES:
        raise SystemOneConfigError("base_url must start with http:// or https://")
    if "@" in parts.netloc:
        raise SystemOneConfigError("base_url must not contain credentials (userinfo)")
    if hostname == "0.0.0.0":
        raise SystemOneConfigError("base_url must not use 0.0.0.0; use 127.0.0.1 or ::1")
    if hostname not in LOOPBACK_HOSTS:
        raise SystemOneConfigError("base_url host must be literally 127.0.0.1 or ::1; remote hosts are rejected")
    if port == 0:
        raise SystemOneConfigError("base_url has an invalid port")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise SystemOneConfigError("base_url must be the server root (no path, query or fragment)")
    return f"{parts.scheme}://{parts.netloc}"


def _check_uds_parent(uds_path: Path) -> None:
    """Parent: real directory, owned by the user, mode 0700. Ancestors: no symlink or ``..``, owned
    by root or the user, and not writable by others unless sticky, so no other user can swap a
    component between this check and the connection."""
    if not uds_path.is_absolute():
        raise SystemOneConfigError("uds_path must be absolute")
    parent = uds_path.parent
    info = _lstat_or_none(parent)
    if info is None:
        raise SystemOneConfigError("uds_path parent directory does not exist")
    if os.path.realpath(parent) != str(parent):
        raise SystemOneConfigError("uds_path must be canonical: no symlink or '..' components")
    if not stat.S_ISDIR(info.st_mode):
        raise SystemOneConfigError("uds_path parent must be a real directory, not a symlink or file")
    if info.st_uid != os.getuid():
        raise SystemOneConfigError("uds_path parent directory must be owned by the current user")
    if stat.S_IMODE(info.st_mode) != SECURE_DIR_MODE:
        raise SystemOneConfigError("uds_path parent directory must have mode 0700")
    for ancestor in parent.parents:
        _check_uds_ancestor(ancestor)


def _check_uds_ancestor(directory: Path) -> None:
    info = _lstat_or_none(directory)
    if info is None or info.st_uid not in (ROOT_UID, os.getuid()):
        raise SystemOneConfigError("every directory above uds_path must be owned by root or the current user")
    writable_by_others = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable_by_others and not info.st_mode & stat.S_ISVTX:
        raise SystemOneConfigError("a directory above uds_path is writable by other users without the sticky bit")


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError:
        return None


def checked_env_name(api_key_env: object) -> str:
    """A POSIX environment variable NAME (``[A-Z_][A-Z0-9_]{0,63}``). Pure; the message never echoes it."""
    if not isinstance(api_key_env, str) or not ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise SystemOneConfigError(
            "api_key_env must be an environment variable NAME (A-Z, 0-9, _; at most 64), not a key value"
        )
    return api_key_env


def _read_api_key(api_key_env: str | None, *, required: bool) -> str | None:
    """Read the Bearer key from the named variable. Messages never echo the name or the value."""
    if not api_key_env:
        if required:
            raise SystemOneConfigError("TCP endpoints require api_key_env (the env variable holding the Bearer key)")
        return None
    value = os.environ.get(checked_env_name(api_key_env))
    if not value:
        raise SystemOneConfigError("the environment variable named by api_key_env is unset or empty")
    if not HEADER_SAFE_KEY_PATTERN.fullmatch(value):
        raise SystemOneConfigError("the Bearer key must be printable ASCII without whitespace")
    return value


# --- response handling ----------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _raise_for_status(response: httpx.Response, path: str) -> None:
    status = response.status_code
    if status == HTTP_OK:
        return
    raw = _error_body(response)  # kept on .raw only; response text never reaches the message
    raise _status_error_type(status)(f"System One {path} returned HTTP {status}", status_code=status, raw=raw)


def _status_error_type(status: int) -> type[SystemOneError]:
    if status == HTTP_UNAUTHORIZED:
        return SystemOneAuthError
    if status == HTTP_NOT_FOUND:
        return SystemOneModelError
    if status == HTTP_NOT_IMPLEMENTED:
        return SystemOneUnsupportedError
    if status in SERVER_ERROR_RANGE:
        return SystemOneServerError
    if status in CLIENT_ERROR_RANGE:
        return SystemOneRequestError
    return SystemOneProtocolError


def _error_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        raw = response.json()
    except ValueError:
        raise SystemOneProtocolError("System One returned a non-JSON body", status_code=HTTP_OK) from None
    if not isinstance(raw, dict):
        raise SystemOneProtocolError("System One returned a non-object JSON body", status_code=HTTP_OK, raw=raw)
    return raw


def _protocol(message: str, raw: dict[str, Any]) -> SystemOneProtocolError:
    return SystemOneProtocolError(f"System One response {message}", status_code=HTTP_OK, raw=raw)


def _parse_result(raw: dict[str, Any], questions: tuple[SystemOneQuestion, ...]) -> SystemOneResult:
    model = raw.get("model")
    answers = raw.get("answers")
    if not isinstance(model, str) or not model:
        raise _protocol("has no model name", raw)
    if not isinstance(answers, dict) or set(answers) != {question.question_id for question in questions}:
        raise _protocol("answers do not match the question set", raw)
    parsed = []
    for index, question in enumerate(questions, start=1):
        answer = _parse_answer(question, answers[question.question_id])
        if answer is None:
            raise _protocol(f"answer for question {index} of {len(questions)} is malformed", raw)
        parsed.append(answer)
    usage = raw.get("usage")
    counts = [usage.get(key) if isinstance(usage, dict) else None for key in ("input_tokens", "output_tokens")]
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts):
        raise _protocol("usage must carry non-negative integer input_tokens and output_tokens", raw)
    return SystemOneResult(model=model, answers=tuple(parsed), usage=SystemOneUsage(*counts), raw=raw)


def _parse_answer(question: SystemOneQuestion, answer: object) -> SystemOneAnswer | None:
    """The typed answer, or ``None`` when the answer does not match the question."""
    if not isinstance(answer, dict) or answer.get("type") != question_payload(question)["type"]:
        return None
    if isinstance(question, NoulQuestion):
        p_yes = _probability(answer.get("noul"))
        return None if p_yes is None else NoulAnswer(question.question_id, p_yes)
    if isinstance(question, ChoiceQuestion):
        names = tuple(name for name, _ in question.criteria)
        probabilities = _probabilities(answer.get("probabilities"), names)
        if probabilities is None or answer.get("choice") not in names:
            return None
        return ChoiceAnswer(question.question_id, answer["choice"], probabilities)
    keys = tuple(str(index) for index in range(len(question.levels)))
    probabilities = _probabilities(answer.get("probabilities"), keys)
    score = answer.get("score")
    if probabilities is None or not _is_number(score) or not 0 <= score <= len(question.levels) - 1:
        return None
    return ScoreAnswer(question.question_id, float(score), probabilities)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(value: object) -> float | None:
    return float(value) if _is_number(value) and 0 <= value <= 1 else None


def _probabilities(value: object, keys: tuple[str, ...]) -> tuple[tuple[str, float], ...] | None:
    if not isinstance(value, dict) or set(value) != set(keys):
        return None
    pairs = tuple((key, _probability(value[key])) for key in keys)
    if any(p is None for _, p in pairs):
        return None
    return tuple((key, float(p)) for key, p in pairs if p is not None)
