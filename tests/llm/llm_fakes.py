"""Offline fakes for the LLM parts: a scripted Responses client and a scripted System One server.

Nothing here touches the network, a GPU or a real key. Later phases (translator, judge E2E tests)
may import these helpers as ``tests.llm.llm_fakes``.
"""

from __future__ import annotations

import copy
import json
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

FAKE_MODEL = "gpt-6.1-test"
FAKE_RESPONSE_ID = "resp_fake_1"


class ScriptedResponsesClient:
    """Stands in for ``openai.OpenAI``: ``responses.create`` returns or raises scripted outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self.responses = self
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.requests.append(copy.deepcopy(kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class DumpableResponse:
    """Mimics an SDK pydantic response: only ``model_dump(mode="json")`` is used by the client."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = copy.deepcopy(raw)

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        return copy.deepcopy(self._raw)


class RecordingOpenAI:
    """Replaces ``openai.OpenAI`` in a fake ``openai`` module; records constructor arguments."""

    instances: list[RecordingOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        RecordingOpenAI.instances.append(self)


def fake_openai_module() -> types.ModuleType:
    """A stand-in ``openai`` module (importing the real SDK costs seconds)."""
    RecordingOpenAI.instances = []
    module = types.ModuleType("openai")
    module.OpenAI = RecordingOpenAI  # type: ignore[attr-defined]
    return module


# SDK-shaped exception doubles. The client classifies SDK errors by the class names in the MRO of
# classes defined in the ``openai`` package, so these mirror openai 2.54.0's hierarchy:
# OpenAIError > APIError > {APIStatusError, APIConnectionError > APITimeoutError}.
class OpenAIError(Exception):
    __module__ = "openai._exceptions"


class APIError(OpenAIError):
    __module__ = "openai._exceptions"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class APIStatusError(APIError):
    __module__ = "openai._exceptions"

    def __init__(self, message: str, *, status_code: int, code: str | None = None) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code


class APIConnectionError(APIError):
    __module__ = "openai._exceptions"


class APITimeoutError(APIConnectionError):
    __module__ = "openai._exceptions"


class APIResponseValidationError(APIError):
    __module__ = "openai._exceptions"


def message_item(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def refusal_item(refusal: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": refusal}],
    }


def usage_body(
    *, input_tokens: int = 120, cached: int = 20, output_tokens: int = 40, reasoning: int = 30
) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def responses_body(
    *,
    output: list[dict[str, Any]] | None = None,
    status: str = "completed",
    incomplete_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A Responses API response as returned by ``model_dump(mode="json")``."""
    reasoning_item = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    return {
        "id": FAKE_RESPONSE_ID,
        "object": "response",
        "model": FAKE_MODEL,
        "status": status,
        "output": [reasoning_item, *(output if output is not None else [message_item('{"ok": true}')])],
        "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
        "usage": usage if usage is not None else usage_body(),
        "error": None,
    }


# ---------------------------------------------------------------------------------------------
# System One (httpx.MockTransport)


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None
    raw_body: bytes


Outcome = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class SystemOneServer:
    """Scripted local server for ``httpx.MockTransport``. Records every request it receives."""

    def __init__(self, outcomes: list[Outcome]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[RecordedRequest] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        raw_body = request.read()
        body = json.loads(raw_body) if raw_body else None
        self.requests.append(RecordedRequest(request.method, request.url.path, dict(request.headers), body, raw_body))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, httpx.Response):
            return outcome
        return outcome(request)


def raise_transport(error_type: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    """An outcome that raises an httpx transport error bound to the request (as httpcore does)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error_type("simulated transport failure", request=request)

    return handler


def noul_answer(p_yes: float, *, vllm_extras: bool = False) -> dict[str, Any]:
    answer: dict[str, Any] = {"type": "noul", "noul": p_yes}
    if vllm_extras:
        answer = {**answer, "probabilities": {"yes": p_yes, "no": 1 - p_yes}, "confidence": 0.9}
    return answer


def choice_answer(choice: str, probabilities: dict[str, float]) -> dict[str, Any]:
    return {"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": 0.5}


def score_answer(score: float, probabilities: dict[str, float]) -> dict[str, Any]:
    legend = {key: f"level-{key}" for key in probabilities}
    return {"type": "score", "score": score, "probabilities": probabilities, "legend": legend, "confidence": 0.4}


def systemone_body(
    answers: dict[str, Any], *, input_tokens: int = 500, output_tokens: int = 0, vllm_extras: bool = False
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "clef-flash",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }
    if vllm_extras:
        extras = {"id": "sd-1", "object": "structured_decision", "created": 1, "diagnostics": {}, "joint": None}
        body = {**body, **extras}
    return body


def json_response(status_code: int, body: Any) -> httpx.Response:
    """Serialised like a lax server: NaN/Infinity literals pass through (httpx's ``json=`` refuses them)."""
    content = json.dumps(body).encode("utf-8")
    return httpx.Response(status_code, content=content, headers={"Content-Type": "application/json"})


def models_body(*model_ids: str) -> dict[str, Any]:
    return {"object": "list", "data": [{"id": model_id, "object": "model"} for model_id in model_ids]}
