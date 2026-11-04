"""Local chat part: OpenAI-compatible ``POST /v1/chat/completions`` against a local vLLM server.

- For a GENERATIVE model served by ``vllm serve``. The default detector, clef-flash, is a pooling
  model that answers ``/v1/systemone`` only (``system_one.py``); it does not serve chat.
- Endpoint: exactly one of a Unix socket (preferred) or a literal loopback TCP root, validated by
  ``mtd_rl_xavier.local_endpoint`` before any client exists (the same rules as ``system_one``).
  The Bearer key is required on both, read from the environment variable whose NAME is
  configured; it never appears in argv, config or errors.
- ``make_vllm_chat_client`` imports the ``openai`` SDK lazily and passes an explicit base URL, the
  key and ``max_retries=0``, plus its own httpx client: ``trust_env=False`` (no proxy or ``.netrc``
  for a local server), no redirects, no transport retries, ``uds=`` for a socket. It refuses to
  start while ``OPENAI_CUSTOM_HEADERS`` or ``OPENAI_LOG`` is set, and it omits the
  ``OpenAI-Organization`` / ``OpenAI-Project`` headers the SDK would take from the environment.
- Nothing touches the network at import or construction; ``health()`` runs only when called.
- Structured output is optional: ``schema_name`` + ``schema`` send ``response_format`` =
  ``json_schema`` (strict). UNVERIFIED: the research doc does not cover vLLM 0.33.0's chat
  request fields (``max_tokens``, ``temperature``, ``seed``, ``response_format``).
- Errors mirror ``openai_responses``: ``VLLMChatError.uncertain`` is True when the server may have
  run the request without us knowing the outcome (connection failure, timeout, HTTP 5xx, an
  unreadable response). Nothing is retried. Messages carry only the error class, HTTP status and
  fixed text, never request or response content, and never the key. Unlike ``openai_responses``,
  SDK exceptions are NOT chained (``__cause__``/``__context__`` stay empty): their message can quote
  the response body and their ``request`` carries the Bearer key.
- It refuses to build a client, and to send each request, while the SDK logger is enabled for DEBUG
  (it would log the messages).

This module imports no sibling module of ``mtd_rl_xavier.llm`` (the LLM parts stay independent).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx

from mtd_rl_xavier import local_endpoint
from mtd_rl_xavier.local_endpoint import LocalEndpointError

API_PATH_PREFIX = "/v1"
SDK_MAX_RETRIES = 0
TRANSPORT_RETRIES = 0
SDK_PACKAGE = "openai"
# Environment variables the SDK would apply to every request; refused rather than silently honoured.
REFUSED_SDK_ENV = ("OPENAI_CUSTOM_HEADERS", "OPENAI_LOG")
# Headers the SDK fills from OPENAI_ORG_ID / OPENAI_PROJECT_ID; a local server needs neither.
OMITTED_SDK_HEADERS = ("OpenAI-Organization", "OpenAI-Project")
# SDK loggers: at DEBUG the SDK logs request options, i.e. the chat messages.
SDK_LOGGER_NAMES = ("openai", "openai._base_client")
CHAT_ROLES = frozenset({"system", "user", "assistant"})
RESPONSE_ROLE = "assistant"
RESPONSE_OBJECT = "chat.completion"
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
# finish_reason values that may appear in a message; any other server value is shown as "other".
DISPLAYED_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call", "abort"})
T = TypeVar("T")
SCHEMA_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")  # used with fullmatch only
SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # used with fullmatch only
SERVER_ERROR_STATUS_MIN = 500
UNREADABLE_MESSAGE = "chat completion returned an unreadable response; outcome uncertain, not retried"


class VLLMChatError(Exception):
    """A chat request was invalid or the call failed.

    ``uncertain`` is True when the server may have run the request without us knowing the
    outcome. ``error_code`` (a sanitised SDK code), ``result`` and ``raw`` are attributes only;
    messages never carry response content.
    """

    def __init__(
        self,
        message: str,
        *,
        uncertain: bool = False,
        status_code: int | None = None,
        error_code: str | None = None,
        result: VLLMChatResult | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.uncertain = uncertain
        self.status_code = status_code
        self.error_code = error_code
        self.result = result
        self.raw = raw


class VLLMChatConfigError(VLLMChatError):
    """Invalid endpoint, key variable, timeout or request; raised before anything is sent."""


class VLLMChatRefusalError(VLLMChatError):
    """The model refused; ``result`` keeps the refusal text and the raw response."""


class VLLMChatIncompleteError(VLLMChatError):
    """``finish_reason == "length"``: ``max_tokens`` ran out; ``result`` keeps the partial text."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in CHAT_ROLES:
            raise VLLMChatConfigError(f"message role must be one of {sorted(CHAT_ROLES)}")
        _require_text(self.content, "message content")


@dataclass(frozen=True)
class VLLMChatRequest:
    model: str  # the served model name (vllm serve --served-model-name)
    messages: tuple[ChatMessage, ...]
    max_tokens: int
    temperature: float | None = None  # None: not sent (server default)
    seed: int | None = None  # None: not sent
    schema_name: str | None = None  # set together with schema for json_schema output (unverified)
    schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.model, "model")
        if not isinstance(self.messages, tuple) or not self.messages:
            raise VLLMChatConfigError("messages must be a non-empty tuple of ChatMessage")
        if not all(isinstance(message, ChatMessage) for message in self.messages):
            raise VLLMChatConfigError("messages must be a non-empty tuple of ChatMessage")
        if not _is_positive_int(self.max_tokens):
            raise VLLMChatConfigError("max_tokens must be a positive integer")
        if self.temperature is not None and not _is_non_negative_finite(self.temperature):
            raise VLLMChatConfigError("temperature must be a non-negative finite number or None")
        if self.seed is not None and not _is_int(self.seed):
            raise VLLMChatConfigError("seed must be an integer or None")
        _validate_schema(self.schema_name, self.schema)


@dataclass(frozen=True)
class VLLMChatUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class VLLMChatResult:
    response_id: str | None
    model: str
    finish_reason: str | None
    content: str
    refusal: str | None
    usage: VLLMChatUsage
    raw: dict[str, Any]


def build_chat_body(request: VLLMChatRequest) -> dict[str, Any]:
    """The exact keyword arguments sent to ``chat.completions.create`` (a deep copy; safe to persist).

    UNVERIFIED for vLLM 0.33.0 (not in the research doc): ``max_tokens``, ``temperature``, ``seed``
    and ``response_format`` = ``json_schema`` with ``strict``.
    """
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [{"role": message.role, "content": message.content} for message in request.messages],
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        body = {**body, "temperature": request.temperature}
    if request.seed is not None:
        body = {**body, "seed": request.seed}
    if request.schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": request.schema_name, "schema": copy.deepcopy(request.schema), "strict": True},
        }
        body = {**body, "response_format": response_format}
    return body


class VLLMChatClient:
    """Chat caller. ``client`` is ``make_vllm_chat_client(...)`` or any fake with
    ``chat.completions.create`` and ``models.list``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, request: VLLMChatRequest) -> VLLMChatResult:
        body = build_chat_body(request)
        _refuse_sdk_debug_logging()  # checked per call: logging may be reconfigured after construction
        response = _call_sdk(lambda: self._client.chat.completions.create(**body))
        result = _parse_result(_response_dict(response))
        _raise_for_outcome(result)
        return result

    def health(self, model: str) -> None:
        """Authenticated ``GET /v1/models`` on demand; ``model`` (the served name) must be listed."""
        _require_text(model, "model")
        _refuse_sdk_debug_logging()
        raw = _response_dict(_call_sdk(self._client.models.list))
        data = raw.get("data")
        if not isinstance(data, list) or not all(_has_string_id(item) for item in data):
            raise _unreadable(raw)
        if model not in {item["id"] for item in data}:
            raise VLLMChatError("the configured model is not served by this endpoint", raw=raw)


def make_vllm_chat_client(
    *,
    base_url: str | None,
    uds_path: Path | None,
    api_key_env: str,
    timeout_s: float,
    transport: httpx.BaseTransport | None = None,
) -> Any:
    """Build the SDK client for a local server. Validates everything before the SDK is imported."""
    if not _is_positive_finite(timeout_s):
        raise VLLMChatConfigError("timeout_s must be a positive finite number")
    if not api_key_env:
        raise VLLMChatConfigError("api_key_env is required (the env variable holding the Bearer key)")
    root = _as_config_error(lambda: local_endpoint.checked_endpoint(base_url, uds_path))
    api_key = _as_config_error(lambda: local_endpoint.read_api_key(api_key_env, required=True))
    refused = [name for name in REFUSED_SDK_ENV if name in os.environ]
    if refused:
        raise VLLMChatConfigError(f"unset {', '.join(refused)}: the local client takes no headers from the environment")
    import openai  # lazy: previews and tests that use fakes never import the SDK

    _refuse_sdk_debug_logging()
    omitted: dict[str, Any] = {name: openai.Omit() for name in OMITTED_SDK_HEADERS}
    http_client = build_local_http_client(uds_path=uds_path, timeout_s=timeout_s, transport=transport)
    try:
        return openai.OpenAI(
            api_key=api_key,
            base_url=f"{root}{API_PATH_PREFIX}",
            max_retries=SDK_MAX_RETRIES,
            timeout=timeout_s,
            http_client=http_client,
            default_headers=omitted,
        )
    except BaseException:
        http_client.close()  # nothing owns it yet; the error itself propagates unchanged
        raise


def _refuse_sdk_debug_logging() -> None:
    """At DEBUG the SDK logs each request's options, including the chat messages."""
    if any(logging.getLogger(name).isEnabledFor(logging.DEBUG) for name in SDK_LOGGER_NAMES):
        raise VLLMChatConfigError("the openai SDK logger is enabled for DEBUG, which logs request bodies")


def build_local_http_client(
    *, uds_path: Path | None, timeout_s: float, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """httpx client for a local server: no environment proxies or ``.netrc``, no redirects, no retries."""
    if transport is None:
        transport = httpx.HTTPTransport(
            uds=str(uds_path) if uds_path else None, retries=TRANSPORT_RETRIES, trust_env=False
        )
    return httpx.Client(transport=transport, timeout=timeout_s, trust_env=False, follow_redirects=False)


# --- validation -----------------------------------------------------------------------------------


def _as_config_error(check: Callable[[], T]) -> T:
    """Run a shared endpoint check; its fixed message is re-raised as ``VLLMChatConfigError`` outside
    the ``except`` clause, so no exception is chained."""
    try:
        return check()
    except LocalEndpointError as exc:
        message = str(exc)
    raise VLLMChatConfigError(message)


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or not _is_utf8(value):
        raise VLLMChatConfigError(f"{field} must be a non-empty UTF-8 string")


def _is_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_finite(value: object) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value > 0


def _is_non_negative_finite(value: object) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value >= 0


def _validate_schema(schema_name: object, schema: object) -> None:
    if schema_name is None and schema is None:
        return
    if not isinstance(schema_name, str) or not SCHEMA_NAME_PATTERN.fullmatch(schema_name):
        raise VLLMChatConfigError("schema_name must match [A-Za-z0-9_-]{1,64} and be set together with schema")
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise VLLMChatConfigError("schema must be a JSON schema whose root type is 'object'")


def _safe_identifier(value: object) -> str | None:
    if isinstance(value, str) and SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        return value
    return None


# --- response parsing -----------------------------------------------------------------------------


def _unreadable(raw: dict[str, Any] | None) -> VLLMChatError:
    """The call ran but its response cannot be read: the outcome is unknown."""
    return VLLMChatError(UNREADABLE_MESSAGE, uncertain=True, raw=raw)


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return copy.deepcopy(response)
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        raise _unreadable(None)
    try:
        raw = dump(mode="json")
    except Exception:  # any serialisation failure of a received response; never chained (it may quote it)
        raw = None
    if not isinstance(raw, dict):
        raise _unreadable(None)
    return raw


def _parse_result(raw: dict[str, Any]) -> VLLMChatResult:
    """UNVERIFIED for vLLM 0.33.0: ``choices[0].message.{content,refusal}``, ``finish_reason``, usage."""
    model, choices = raw.get("model"), raw.get("choices")
    if raw.get("object") != RESPONSE_OBJECT or not isinstance(model, str) or not model:
        raise _unreadable(raw)
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise _unreadable(raw)
    choice = choices[0]
    message, finish_reason = choice.get("message"), choice.get("finish_reason")
    if choice.get("index") != 0 or not isinstance(message, dict) or not _is_optional_str(finish_reason):
        raise _unreadable(raw)
    if message.get("role") != RESPONSE_ROLE:
        raise _unreadable(raw)
    content, refusal = message.get("content"), message.get("refusal")
    if not _is_optional_str(content) or not _is_optional_str(refusal):
        raise _unreadable(raw)
    response_id = raw.get("id")
    return VLLMChatResult(
        response_id=response_id if isinstance(response_id, str) else None,
        model=model,
        finish_reason=finish_reason,
        content=content if content is not None else "",
        refusal=refusal or None,
        usage=_usage(raw.get("usage"), raw),
        raw=raw,
    )


def _has_string_id(item: object) -> bool:
    return isinstance(item, dict) and isinstance(item.get("id"), str)


def _is_optional_str(value: object) -> bool:
    return value is None or isinstance(value, str)


def _usage(raw_usage: object, raw: dict[str, Any]) -> VLLMChatUsage:
    """Token counts; missing counts are 0, malformed ones make the response unreadable."""
    if raw_usage is None:
        return VLLMChatUsage(0, 0)
    if not isinstance(raw_usage, dict):
        raise _unreadable(raw)
    counts = (raw_usage.get("prompt_tokens"), raw_usage.get("completion_tokens"))
    if not all(value is None or _is_count(value) for value in counts):
        raise _unreadable(raw)
    return VLLMChatUsage(*(value or 0 for value in counts))


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _raise_for_outcome(result: VLLMChatResult) -> None:
    if result.refusal is not None:
        raise VLLMChatRefusalError("model refused the request (see result.refusal)", result=result)
    if result.finish_reason == FINISH_LENGTH:
        raise VLLMChatIncompleteError("response incomplete (finish_reason=length)", result=result)
    if result.finish_reason != FINISH_STOP:
        reason = result.finish_reason if result.finish_reason in DISPLAYED_FINISH_REASONS else "other"
        raise VLLMChatError(f"unexpected finish_reason ({reason})", result=result)
    if not result.content:
        raise VLLMChatError("chat completion has no message content", result=result)


# --- SDK errors -------------------------------------------------------------------------------------


def _call_sdk(call: Callable[[], T]) -> T:
    """Run one SDK call. An SDK error becomes ``VLLMChatError``, raised outside the ``except`` clause:
    the SDK exception is not chained, because its message can quote the response body and its
    ``request`` carries the Bearer key. Anything else propagates unchanged."""
    try:
        return call()
    except Exception as exc:
        wrapped = _wrap_sdk_error(exc)
        if wrapped is None:
            raise
    raise wrapped


def _sdk_class_names(exc: BaseException) -> frozenset[str]:
    """Names of the exception's classes defined in the ``openai`` package (no SDK import needed)."""
    return frozenset(cls.__name__ for cls in type(exc).__mro__ if cls.__module__.split(".")[0] == SDK_PACKAGE)


def _wrap_sdk_error(exc: Exception) -> VLLMChatError | None:
    names = _sdk_class_names(exc)
    if "APIError" not in names:
        return None
    raw_status = getattr(exc, "status_code", None) if "APIStatusError" in names else None
    status = raw_status if _is_positive_int(raw_status) else None
    uncertain = status is None or status >= SERVER_ERROR_STATUS_MIN
    code = _safe_identifier(getattr(exc, "code", None))  # attribute only, never in the message
    parts = [f"local chat call failed: {type(exc).__name__}"]
    if status is not None:
        parts.append(f"HTTP {status}")
    if uncertain:
        parts.append("outcome uncertain, not retried")
    return VLLMChatError(", ".join(parts), uncertain=uncertain, status_code=status, error_code=code)
