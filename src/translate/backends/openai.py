"""The OpenAI Responses backend: one strict-JSON-schema request per call, never retried.

The client is pinned to ``https://api.openai.com/v1`` (``OPENAI_BASE_URL`` is ignored), SDK retries
are off, and it refuses to start while ``OPENAI_CUSTOM_HEADERS`` or ``OPENAI_LOG`` (it logs request
bodies) is set. ``OPENAI_API_KEY`` is read only here, after the paid gate. Every request sends
``store: false`` and a strict ``json_schema``; no temperature. Outcomes (no automatic retry, design
arbitration): an HTTP error, a connection error, a refusal or an incomplete response ends the run
``failed``; a timeout, an unknown status, an unreadable response or any other error ends it
``uncertain``. Messages carry the error class and HTTP status only, never a body or the key.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from translate.backends import BackendError, Prompt, Reply

API_BASE_URL = "https://api.openai.com/v1"
API_KEY_ENV = "OPENAI_API_KEY"
REFUSED_ENV = ("OPENAI_CUSTOM_HEADERS", "OPENAI_LOG")
USAGE = (("input_tokens", "input_tokens", None), ("cached_input_tokens", "input_tokens_details", "cached_tokens"))
USAGE += (("output_tokens", "output_tokens", None), ("reasoning_tokens", "output_tokens_details", "reasoning_tokens"))


def make_client(*, timeout_s: float) -> Any:
    """The SDK client (imported lazily: previews and tests never import it)."""
    refused = [name for name in REFUSED_ENV if name in os.environ]
    if refused:
        raise BackendError(
            f"unset {', '.join(refused)}: the paid client takes no headers or logging from the environment",
            uncertain=False,
        )
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise BackendError(f"{API_KEY_ENV} must be set in the environment", uncertain=False)
    import openai

    return openai.OpenAI(api_key=key, base_url=API_BASE_URL, max_retries=0, timeout=timeout_s)


def _sdk_error(exc: Exception) -> BackendError | None:
    names = {cls.__name__ for cls in type(exc).__mro__ if cls.__module__.split(".")[0] == "openai"}
    if "APIError" not in names:
        return None
    status = getattr(exc, "status_code", None) if "APIStatusError" in names else None
    uncertain = "APITimeoutError" in names or not ({"APIStatusError", "APIConnectionError"} & names)
    message = f"OpenAI call failed: {type(exc).__name__}" + (f", HTTP {status}" if isinstance(status, int) else "")
    return BackendError(message + ("; outcome uncertain, not retried" if uncertain else ""), uncertain=uncertain)


def _unreadable(raw: Any = None) -> BackendError:
    return BackendError("the response is unreadable; outcome uncertain, not retried", uncertain=True, raw=raw)


def _texts(output: list[Any], part_type: str, key: str, raw: dict[str, Any]) -> str:
    parts = [
        part
        for item in output
        if isinstance(item, dict) and item.get("type") == "message"
        for part in (item.get("content") if isinstance(item.get("content"), list) else [])
        if isinstance(part, dict) and part.get("type") == part_type
    ]
    if not all(isinstance(part.get(key), str) for part in parts):
        raise _unreadable(raw)
    return "".join(part[key] for part in parts)


def _usage(raw: dict[str, Any]) -> dict[str, int]:
    usage = raw.get("usage") or {}
    counts = {name: (usage.get(key) or {}).get(sub) if sub else usage.get(key) for name, key, sub in USAGE}
    if not all(value is None or (isinstance(value, int) and value >= 0) for value in counts.values()):
        raise _unreadable(raw)
    return {name: value or 0 for name, value in counts.items()}


def parse_reply(response: Any) -> Reply:
    raw = response if isinstance(response, dict) else getattr(response, "model_dump", lambda **_: None)(mode="json")
    if not isinstance(raw, dict) or not isinstance(raw.get("output"), list) or not isinstance(raw.get("model"), str):
        raise _unreadable(raw if isinstance(raw, dict) else None)
    raw = copy.deepcopy(raw)
    status = raw.get("status")
    if _texts(raw["output"], "refusal", "refusal", raw):
        raise BackendError("the model refused the request", uncertain=False, raw=raw)
    if status in ("incomplete", "failed", "cancelled"):
        reason = (raw.get("incomplete_details") or {}).get("reason")
        raise BackendError(
            f"response {status}" + (f" ({reason})" if reason in ("max_output_tokens", "content_filter") else ""),
            uncertain=False,
            raw=raw,
        )
    if status != "completed":
        raise BackendError("unexpected response status; outcome uncertain, not retried", uncertain=True, raw=raw)
    response_id = raw.get("id") if isinstance(raw.get("id"), str) else None
    return Reply(_texts(raw["output"], "output_text", "text", raw), raw["model"], response_id, _usage(raw), raw)


class OpenAIBackend:
    def __init__(self, model: Any, client: Any | None = None) -> None:
        self.model = model
        self.client = client if client is not None else make_client(timeout_s=model.timeout_s)

    @staticmethod
    def request_body(model: Any, prompt: Prompt) -> dict[str, Any]:
        """The exact keyword arguments of ``responses.create`` (a deep copy; safe to journal)."""
        schema = {
            "type": "json_schema",
            "name": prompt.schema_name,
            "schema": copy.deepcopy(prompt.schema),
            "strict": True,
        }
        return {
            "model": model.name,
            "instructions": prompt.system,
            "input": [{"role": "user", "content": copy.deepcopy(list(prompt.items))}],
            "text": {"format": schema},
            "reasoning": {"effort": model.reasoning_effort},
            "max_output_tokens": model.max_output_tokens,
            "store": False,
        }

    def send(self, body: dict[str, Any]) -> Reply:
        try:
            response = self.client.responses.create(**copy.deepcopy(body))
        except Exception as exc:
            wrapped = _sdk_error(exc)
            if wrapped is None:
                raise
            raise wrapped from None
        return parse_reply(response)
