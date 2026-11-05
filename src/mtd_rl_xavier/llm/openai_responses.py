"""OpenAI Responses API part: one strict-JSON-schema call per request, text and image input.

- The SDK is imported lazily, and ``make_openai_client`` is the one OpenAI Responses client factory
  of the library: the Translator, the judge, weightreading and both methods of the intent
  comparison build their SDK client here, and it is their only reader of ``OPENAI_API_KEY``
  (legacy chat-completions tools in ``experiment/``, ``tools/`` and ``lag/`` still read the key
  themselves). Composition roots call it after the paid-call gate; previews and tests never
  import the SDK.
- ``OpenAIResponsesClient`` accepts any object with ``responses.create`` (a scripted fake in tests).
- Every request sends ``store=False``, ``reasoning.effort`` and a strict ``json_schema`` text format.
  ``temperature``, ``top_p`` and ``top_logprobs`` are never sent (reasoning models reject or ignore
  them). ``max_output_tokens`` covers reasoning plus visible output.
- No retries here (SDK ``max_retries=0``): a retried paid call may be billed twice. The intent
  comparison adds the narrow, logged retry of ``mtd_rl_xavier.llm.retry`` on top of this client.
- The endpoint is pinned: ``make_openai_client`` passes ``base_url=API_BASE_URL`` explicitly, so
  ``OPENAI_BASE_URL`` in the environment cannot send the key or the prompts elsewhere, and it
  refuses to start while ``OPENAI_CUSTOM_HEADERS`` (the SDK would inject those headers) or
  ``OPENAI_LOG`` (the SDK would log request bodies) is set. Proxy, CA-bundle and organisation or
  project variables stay honoured; ``set_provenance_env_names`` reports which of them are set so
  run records can name them (names only, never values: proxy URLs can carry credentials).
- Refusals and incomplete responses raise typed errors that keep the parsed result and raw response.
- Error messages carry only the SDK error class, HTTP status and a safe error code; never the SDK
  message (it can echo request text or a masked key) and never the key.

This module imports no sibling module of ``mtd_rl_xavier.llm`` (the LLM parts stay independent).
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections.abc import Container
from dataclasses import dataclass
from typing import Any

API_KEY_ENV = "OPENAI_API_KEY"
API_BASE_URL = "https://api.openai.com/v1"  # explicit: never taken from OPENAI_BASE_URL
# Environment variables the SDK would apply to every request; refused rather than silently honoured.
REFUSED_SDK_ENV = ("OPENAI_CUSTOM_HEADERS", "OPENAI_LOG")
# Environment variables that stay honoured but change where or how a request travels; run records
# list the ones that are set (names only).
PROVENANCE_ENV = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
)
SDK_MAX_RETRIES = 0
DEFAULT_TIMEOUT_S = 1800.0
# The one reasoning-effort allowlist of the repo (Translator, judge, weightreading, both methods of
# the intent comparison): the gpt-6.1-sol subset of the openai SDK ``ReasoningEffort`` literal
# (which also has "none" and "minimal"; not all reasoning models support every value).
REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
IMAGE_DETAILS = frozenset({"low", "high", "original", "auto"})
IMAGE_URL_PREFIXES = ("data:image/", "https://")
INPUT_ITEM_KEYS = {
    "input_text": frozenset({"type", "text"}),
    "input_image": frozenset({"type", "image_url", "detail"}),
}
SCHEMA_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")  # used with fullmatch only
SAFE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")  # used with fullmatch only
STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
FAILED_STATUSES = frozenset({"failed", "cancelled"})  # terminal statuses whose outcome is known
INCOMPLETE_REASONS = frozenset({"max_output_tokens", "content_filter"})  # allowlist for messages
UNREADABLE_MESSAGE = "Responses call returned an unreadable response; outcome uncertain, not retried"
SERVER_ERROR_STATUS_MIN = 500
SDK_PACKAGE = "openai"


class OpenAIClientError(Exception):
    """A Responses request was invalid or the call failed.

    ``uncertain`` is True when the server may have run (and billed) the request without us
    knowing the outcome: connection failures, timeouts, HTTP 5xx, unknown statuses and
    unreadable responses. ``error_code`` (a sanitised SDK error code) and ``raw`` (a response
    that could not be parsed) are attributes only; messages never carry response content.
    """

    def __init__(
        self,
        message: str,
        *,
        uncertain: bool = False,
        status_code: int | None = None,
        error_code: str | None = None,
        result: OpenAIResult | None = None,
        raw: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.uncertain = uncertain
        self.status_code = status_code
        self.error_code = error_code
        self.result = result
        self.raw = raw


class OpenAIRefusalError(OpenAIClientError):
    """The model refused; ``result`` keeps the refusal text and the raw response."""


class OpenAIIncompleteError(OpenAIClientError):
    """``status == "incomplete"``; ``result.incomplete_reason`` says why (e.g. ``max_output_tokens``)."""


@dataclass(frozen=True)
class OpenAIRequest:
    model: str
    instructions: str
    input_items: tuple[dict[str, Any], ...]  # input_text / input_image content parts, in order
    schema_name: str
    schema: dict[str, Any]
    reasoning_effort: str
    max_output_tokens: int

    def __post_init__(self) -> None:
        _require_text(self.model, "model")
        _require_text(self.instructions, "instructions")
        if not isinstance(self.schema_name, str) or not SCHEMA_NAME_PATTERN.fullmatch(self.schema_name):
            raise OpenAIClientError("schema_name must match [A-Za-z0-9_-]{1,64}")
        if not isinstance(self.schema, dict) or self.schema.get("type") != "object":
            raise OpenAIClientError("schema must be a JSON schema whose root type is 'object'")
        if not _is_one_of(self.reasoning_effort, REASONING_EFFORTS):
            raise OpenAIClientError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
        if not _is_positive_int(self.max_output_tokens):
            raise OpenAIClientError("max_output_tokens must be a positive integer")
        _validate_input_items(self.input_items)


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class OpenAIResult:
    response_id: str | None
    model: str
    status: str
    output_text: str
    output_items: tuple[dict[str, Any], ...]
    refusal: str | None
    incomplete_reason: str | None
    usage: LLMUsage
    raw: dict[str, Any]


def input_text(text: str) -> dict[str, Any]:
    """A text content part for ``OpenAIRequest.input_items``."""
    return {"type": "input_text", "text": text}


def input_image(image_url: str, *, detail: str) -> dict[str, Any]:
    """An image content part; ``image_url`` is a ``data:image/...`` URL or an https URL."""
    return {"type": "input_image", "image_url": image_url, "detail": detail}


def build_responses_body(request: OpenAIRequest) -> dict[str, Any]:
    """The exact keyword arguments sent to ``responses.create`` (a deep copy; safe to persist)."""
    return {
        "model": request.model,
        "instructions": request.instructions,
        "input": [{"role": "user", "content": copy.deepcopy(list(request.input_items))}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": request.schema_name,
                "schema": copy.deepcopy(request.schema),
                "strict": True,
            }
        },
        "reasoning": {"effort": request.reasoning_effort},
        "max_output_tokens": request.max_output_tokens,
        "store": False,
    }


class OpenAIResponsesClient:
    """Responses API caller. ``client`` is ``make_openai_client()`` or any fake with ``responses.create``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, request: OpenAIRequest) -> OpenAIResult:
        body = build_responses_body(request)
        try:
            response = self._client.responses.create(**body)
        except Exception as exc:
            wrapped = _wrap_sdk_error(exc)
            if wrapped is None:
                raise
            raise wrapped from exc
        result = _parse_result(_response_dict(response))
        _raise_for_outcome(result)
        return result


def set_provenance_env_names() -> list[str]:
    """Names (never values) of the ``PROVENANCE_ENV`` variables set in the environment, in list order."""
    return [name for name in PROVENANCE_ENV if name in os.environ]


def make_openai_client(*, timeout_s: float = DEFAULT_TIMEOUT_S) -> Any:
    """Build the SDK client for ``API_BASE_URL`` (the library's only factory); call it after the paid gate."""
    if not _is_positive_finite(timeout_s):
        raise OpenAIClientError("timeout_s must be a positive finite number")
    refused = [name for name in REFUSED_SDK_ENV if name in os.environ]
    if refused:
        raise OpenAIClientError(f"unset {', '.join(refused)}: the paid client takes no headers from the environment")
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise OpenAIClientError(f"{API_KEY_ENV} must be set in the environment")
    import openai  # lazy: previews and tests never import the SDK

    return openai.OpenAI(api_key=api_key, base_url=API_BASE_URL, max_retries=SDK_MAX_RETRIES, timeout=timeout_s)


# --- validation -----------------------------------------------------------------------------------


def _require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or not _is_utf8(value):
        raise OpenAIClientError(f"{field} must be a non-empty UTF-8 string")


def _is_utf8(value: str) -> bool:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _is_one_of(value: object, choices: Container[str]) -> bool:
    """Membership for untrusted values: an unhashable value is simply not a member."""
    return isinstance(value, str) and value in choices


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_finite(value: object) -> bool:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return is_number and math.isfinite(value) and value > 0


def _validate_input_items(items: object) -> None:
    if not isinstance(items, tuple) or not items:
        raise OpenAIClientError("input_items must be a non-empty tuple of content parts")
    for item in items:
        _validate_input_item(item)


def _validate_input_item(item: object) -> None:
    if not isinstance(item, dict) or not _is_one_of(item.get("type"), INPUT_ITEM_KEYS):
        raise OpenAIClientError(f"each input item must be one of {sorted(INPUT_ITEM_KEYS)}")
    if set(item) != INPUT_ITEM_KEYS[item["type"]]:
        raise OpenAIClientError(f"{item['type']} items take exactly the keys {sorted(INPUT_ITEM_KEYS[item['type']])}")
    if item["type"] == "input_text":
        _require_text(item["text"], "input_text.text")
        return
    image_url = item["image_url"]
    if not isinstance(image_url, str) or not image_url.startswith(IMAGE_URL_PREFIXES):
        raise OpenAIClientError("input_image.image_url must be a data:image/ URL or an https URL")
    if not _is_one_of(item["detail"], IMAGE_DETAILS):
        raise OpenAIClientError(f"input_image.detail must be one of {sorted(IMAGE_DETAILS)}")


def _safe_identifier(value: object) -> str | None:
    if isinstance(value, str) and SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        return value
    return None


# --- response parsing -----------------------------------------------------------------------------


def _unreadable(raw: dict[str, Any] | None) -> OpenAIClientError:
    """The call ran but its response cannot be read: the outcome (and billing) is unknown."""
    return OpenAIClientError(UNREADABLE_MESSAGE, uncertain=True, raw=raw)


def _response_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return copy.deepcopy(response)
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        raise _unreadable(None)
    try:
        raw = dump(mode="json")
    except Exception as exc:  # any serialisation failure of a response we were billed for
        raise _unreadable(None) from exc
    if not isinstance(raw, dict):
        raise _unreadable(None)
    return raw


def _parse_result(raw: dict[str, Any]) -> OpenAIResult:
    status, model, output = raw.get("status"), raw.get("model"), raw.get("output")
    if not isinstance(status, str) or not status or not isinstance(model, str):
        raise _unreadable(raw)
    if not isinstance(output, list) or not all(isinstance(item, dict) for item in output):
        raise _unreadable(raw)
    items = tuple(output)
    response_id = raw.get("id")
    return OpenAIResult(
        response_id=response_id if isinstance(response_id, str) else None,
        model=model,
        status=status,
        output_text="".join(_content_parts(items, "output_text", "text", raw)),
        output_items=items,
        refusal="".join(_content_parts(items, "refusal", "refusal", raw)) or None,
        incomplete_reason=_incomplete_reason(raw.get("incomplete_details")),
        usage=_usage(raw.get("usage"), raw),
        raw=raw,
    )


def _content_parts(items: tuple[dict[str, Any], ...], part_type: str, key: str, raw: dict[str, Any]) -> list[str]:
    """Strings of ``part_type`` content parts in assistant message items, in output order."""
    texts: list[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list) or not all(isinstance(part, dict) for part in content):
            raise _unreadable(raw)
        for part in content:
            if part.get("type") != part_type:
                continue
            if not isinstance(part.get(key), str):
                raise _unreadable(raw)
            texts.append(part[key])
    return texts


def _incomplete_reason(details: object) -> str | None:
    if not isinstance(details, dict):
        return None
    reason = details.get("reason")
    return reason if isinstance(reason, str) else None


def _usage(raw_usage: object, raw: dict[str, Any]) -> LLMUsage:
    """Token counts; missing counts are 0, malformed ones make the response unreadable."""
    if raw_usage is None:
        return LLMUsage(0, 0, 0, 0)
    if not isinstance(raw_usage, dict):
        raise _unreadable(raw)
    input_details = raw_usage.get("input_tokens_details") or {}
    output_details = raw_usage.get("output_tokens_details") or {}
    if not isinstance(input_details, dict) or not isinstance(output_details, dict):
        raise _unreadable(raw)
    counts = (
        raw_usage.get("input_tokens"),
        input_details.get("cached_tokens"),
        raw_usage.get("output_tokens"),
        output_details.get("reasoning_tokens"),
    )
    if not all(value is None or _is_count(value) for value in counts):
        raise _unreadable(raw)
    return LLMUsage(*(value or 0 for value in counts))


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _raise_for_outcome(result: OpenAIResult) -> None:
    if result.refusal is not None:
        raise OpenAIRefusalError("model refused the request (see result.refusal)", result=result)
    if result.status == STATUS_INCOMPLETE:
        reason = result.incomplete_reason if result.incomplete_reason in INCOMPLETE_REASONS else "other"
        raise OpenAIIncompleteError(f"response incomplete (reason={reason})", result=result)
    if result.status in FAILED_STATUSES:
        raise OpenAIClientError(f"response {result.status}", result=result)
    if result.status != STATUS_COMPLETED:
        message = "unexpected response status; outcome uncertain, not retried"
        raise OpenAIClientError(message, uncertain=True, result=result)


# --- SDK errors -------------------------------------------------------------------------------------


def _sdk_class_names(exc: BaseException) -> frozenset[str]:
    """Names of the exception's classes defined in the ``openai`` package (no SDK import needed)."""
    return frozenset(cls.__name__ for cls in type(exc).__mro__ if cls.__module__.split(".")[0] == SDK_PACKAGE)


def _wrap_sdk_error(exc: Exception) -> OpenAIClientError | None:
    """Translate an SDK error into ``OpenAIClientError``; ``None`` for anything that is not one."""
    names = _sdk_class_names(exc)
    if "APIError" not in names:
        return None
    raw_status = getattr(exc, "status_code", None) if "APIStatusError" in names else None
    status = raw_status if _is_positive_int(raw_status) else None
    uncertain = status is None or status >= SERVER_ERROR_STATUS_MIN
    code = _safe_identifier(getattr(exc, "code", None))  # attribute only, never in the message
    parts = [f"OpenAI Responses call failed: {type(exc).__name__}"]
    if status is not None:
        parts.append(f"HTTP {status}")
    if uncertain:
        parts.append("outcome uncertain, not retried")
    return OpenAIClientError(", ".join(parts), uncertain=uncertain, status_code=status, error_code=code)
