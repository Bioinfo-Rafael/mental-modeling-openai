"""Chat backends. The model side is tool-free by construction: no ``tools`` are ever sent.

``OpenAIResponsesBackend`` replays the whole conversation statelessly (``store=False``) and keeps
reasoning across turns through ``reasoning.encrypted_content`` items, like ``codex exec resume``.
The ``openai`` SDK is imported lazily and the client is built only by ``make_openai_client``,
which the CLI calls after the paid-call gate and the sandbox preflight have passed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

API_KEY_ENV = "OPENAI_API_KEY"
SDK_MAX_RETRIES = 0  # a retried paid request may be billed twice
REQUEST_TIMEOUT_S = 1800.0


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached_input_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class BackendReply:
    text: str
    output_items: tuple[dict, ...]  # appended to the history for the next turn
    usage: Usage
    status: str  # "completed" or "incomplete"
    response_id: str | None
    request_record: dict  # request options + input hash (never credentials)


class ChatBackend(Protocol):
    model: str

    def respond(self, history: tuple[dict, ...]) -> BackendReply: ...


def user_item(text: str) -> dict:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def history_sha256(history: tuple[dict, ...]) -> str:
    return hashlib.sha256(json.dumps(list(history), sort_keys=True).encode("utf-8")).hexdigest()


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage(0, 0, 0, 0)
    output_details = getattr(raw, "output_tokens_details", None)
    input_details = getattr(raw, "input_tokens_details", None)
    return Usage(
        input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
    )


def _dump_item(item: Any) -> dict:
    if isinstance(item, dict):
        return item
    return item.model_dump(mode="json", exclude_none=True)


class OpenAIResponsesBackend:
    """Responses API backend. ``client`` is any object with ``responses.create`` (a fake in tests)."""

    def __init__(self, client: Any, model: str, reasoning_effort: str, max_output_tokens: int) -> None:
        self.client = client
        self.model = model
        self.options = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }

    def respond(self, history: tuple[dict, ...]) -> BackendReply:
        response = self.client.responses.create(input=list(history), **self.options)
        items = tuple(_dump_item(item) for item in (getattr(response, "output", None) or []))
        return BackendReply(
            text=getattr(response, "output_text", "") or "",
            output_items=items,
            usage=_usage(getattr(response, "usage", None)),
            status=str(getattr(response, "status", "completed") or "completed"),
            response_id=getattr(response, "id", None),
            request_record={
                **self.options,
                "input_items": len(history),
                "input_sha256": history_sha256(history),
                "tools": None,
            },
        )


def make_openai_client() -> Any:
    """Build the SDK client. Reads the API key from the environment only; never logs it."""
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} must be set in the environment")
    import openai  # lazy: previews and tests never import the SDK

    return openai.OpenAI(api_key=api_key, max_retries=SDK_MAX_RETRIES, timeout=REQUEST_TIMEOUT_S)
