"""The shared retry policy (``translate.llm.retry``) as a one-shot translation call uses it: ``RetryingClient`` below
``OpenAIResponsesClient``. Offline, real ``openai`` exception instances, no sleeping."""

from __future__ import annotations

import httpx
import openai
import pytest

from translate.llm.openai_responses import (
    OpenAIClientError,
    OpenAIRequest,
    OpenAIResponsesClient,
    build_responses_body,
    input_text,
)
from translate.llm.retry import RetryEvent, RetryingClient, RetryPolicy, is_transient

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")
SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
COMPLETED = {
    "id": "resp_1",
    "model": "m",
    "status": "completed",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
}


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=REQUEST, headers={"x-request-id": "req_1"})
    return cls("boom", response=response, body=None)


class _Responses:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.bodies: list[dict] = []

    def create(self, **kwargs: object) -> dict:
        self.bodies.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return COMPLETED


class _SDK:
    def __init__(self, failures: list[BaseException]) -> None:
        self.responses = _Responses(failures)


def _request() -> OpenAIRequest:
    return OpenAIRequest("m", "instructions", (input_text("hi"),), "s", SCHEMA, "high", 10)


def _client(failures: list[BaseException]) -> tuple[OpenAIResponsesClient, _SDK, list[float], list[RetryEvent]]:
    sdk, sleeps, events = _SDK(failures), [], []
    wrapped = RetryingClient(sdk, RetryPolicy(2, 10.0, 3.0), sleep=sleeps.append, log=events.append)
    return OpenAIResponsesClient(wrapped), sdk, sleeps, events


@pytest.mark.parametrize(
    ("error", "transient"),
    [
        (_status_error(openai.InternalServerError, 500), True),
        (_status_error(openai.InternalServerError, 503), True),
        (openai.APIConnectionError(request=REQUEST), True),
        (openai.APITimeoutError(request=REQUEST), False),
        (_status_error(openai.RateLimitError, 429), False),
        (_status_error(openai.BadRequestError, 400), False),
        (_status_error(openai.AuthenticationError, 401), False),
        (ValueError("not an SDK error"), False),
    ],
    ids=["500", "503", "connection", "timeout", "429", "400", "401", "non-sdk"],
)
def test_is_transient_classifies_only_5xx_and_non_timeout_connection_errors(error: BaseException, transient: bool) -> None:  # fmt: skip
    assert is_transient(error) is transient


def test_retrying_client_after_one_5xx_succeeds_with_the_identical_body() -> None:
    client, sdk, sleeps, events = _client([_status_error(openai.InternalServerError, 502)])

    result = client.create(_request())

    assert result.status == "completed"
    assert sdk.responses.bodies == [build_responses_body(_request())] * 2
    assert sleeps == [10.0]
    assert [(e.attempt, e.status_code, e.delay_s) for e in events] == [(1, 502, 10.0)]


def test_retrying_client_exhausted_5xx_surfaces_as_uncertain_client_error() -> None:
    errors = [_status_error(openai.InternalServerError, 500) for _ in range(3)]
    client, sdk, sleeps, events = _client(errors)

    with pytest.raises(OpenAIClientError) as caught:
        client.create(_request())

    assert caught.value.uncertain is True
    assert caught.value.status_code == 500
    assert len(sdk.responses.bodies) == 3
    assert sleeps == [10.0, 30.0]
    assert events[-1].delay_s is None


@pytest.mark.parametrize(
    "error",
    [openai.APITimeoutError(request=REQUEST), _status_error(openai.RateLimitError, 429)],
    ids=["timeout", "429"],
)
def test_retrying_client_never_resends_timeouts_or_4xx(error: BaseException) -> None:
    client, sdk, sleeps, events = _client([error])

    with pytest.raises(OpenAIClientError):
        client.create(_request())

    assert len(sdk.responses.bodies) == 1
    assert sleeps == []
    assert events == []
