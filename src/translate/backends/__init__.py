"""LLM backends: ``request_body(model, prompt)`` builds the exact body (journaled before the send, and
shown by a preview without any backend), ``send(body)`` makes exactly one call, never retried.

A new backend is one module with those two methods plus one entry in ``BACKENDS``; a config selects
it with ``model.backend``. Its key is read in its ``__init__``, which only runs after the paid gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Prompt:
    system: str
    items: tuple[dict[str, Any], ...]  # content items of the user message (input_text; later input_image)
    schema_name: str
    schema: dict[str, Any]  # the strict JSON schema of the answer


@dataclass(frozen=True)
class Reply:
    text: str
    model: str
    response_id: str | None
    usage: dict[str, int]
    raw: dict[str, Any]


class BackendError(Exception):
    """A call failed. ``uncertain``: it may have run and been billed; never resend it. ``raw``: a body."""

    def __init__(self, message: str, *, uncertain: bool, raw: Any = None) -> None:
        super().__init__(message)
        self.uncertain, self.raw = uncertain, raw


def input_text(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}


from translate.backends.openai import OpenAIBackend  # noqa: E402  (it uses the types above)

BACKENDS = {"openai": OpenAIBackend}
