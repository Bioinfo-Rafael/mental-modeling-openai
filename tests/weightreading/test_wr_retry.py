"""Transient-error retry of the OpenAI backend: offline, real ``openai`` exception instances."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr

import httpx
import openai
import pytest

from weightreading.backends import OpenAIResponsesBackend, make_openai_client, user_item
from weightreading.config import ConfigError, parse_config
from weightreading.retry import RetryEvent, RetryPolicy, call_with_retry, log_to_stderr
from weightreading.session import RunLog, SessionState

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")
HISTORY = (user_item("hi"),)
REQUEST_ID = "req_abc123"


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=REQUEST, headers={"x-request-id": REQUEST_ID})
    return cls("boom", response=response, body={"error": {"type": "server_error"}})


class _Response:
    id = "resp_1"
    status = "completed"
    output_text = "ok"
    output: list = []
    usage = None


class _Responses:
    """Raises the queued exceptions one per call, then returns a response."""

    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.calls = 0

    def create(self, **kwargs: object) -> _Response:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Response()


def _backend(failures: list[BaseException], sleeps: list[float], log: list[RetryEvent] | None = None):
    client = type("Client", (), {})()
    client.responses = _Responses(failures)
    sink = [] if log is None else log
    backend = OpenAIResponsesBackend(client, "m", "high", 10, sleep=sleeps.append, log=sink.append)
    return backend, client.responses, sink


def test_respond_after_one_server_error_returns_success_and_records_retry() -> None:
    sleeps: list[float] = []
    backend, responses, log = _backend([_status_error(openai.InternalServerError, 500)], sleeps)

    reply = backend.respond(HISTORY)

    assert reply.text == "ok" and responses.calls == 2
    assert sleeps == [10.0]
    assert reply.retries == (
        {
            "attempt": 1,
            "max_attempts": 3,
            "error_type": "InternalServerError",
            "status_code": 500,
            "request_id": REQUEST_ID,
            "delay_s": 10.0,
        },
    )
    assert len(log) == 1


def test_respond_after_two_server_errors_uses_backoff_schedule() -> None:
    sleeps: list[float] = []
    failures = [_status_error(openai.InternalServerError, 500), _status_error(openai.InternalServerError, 503)]
    backend, responses, _ = _backend(failures, sleeps)

    reply = backend.respond(HISTORY)

    assert responses.calls == 3 and sleeps == [10.0, 30.0]
    assert [r["status_code"] for r in reply.retries] == [500, 503]


def test_respond_with_three_server_errors_raises_the_last_one() -> None:
    sleeps: list[float] = []
    errors = [_status_error(openai.InternalServerError, 500) for _ in range(3)]
    backend, responses, log = _backend(list(errors), sleeps)

    with pytest.raises(openai.InternalServerError) as caught:
        backend.respond(HISTORY)

    assert caught.value is errors[-1]
    assert responses.calls == 3 and sleeps == [10.0, 30.0]
    assert [event.delay_s for event in log] == [10.0, 30.0, None]


def test_respond_after_connection_error_retries() -> None:
    sleeps: list[float] = []
    backend, responses, _ = _backend([openai.APIConnectionError(request=REQUEST)], sleeps)

    reply = backend.respond(HISTORY)

    assert reply.text == "ok" and responses.calls == 2 and sleeps == [10.0]
    assert reply.retries[0]["error_type"] == "APIConnectionError"
    assert reply.retries[0]["status_code"] is None


@pytest.mark.parametrize(
    "error",
    [
        openai.APITimeoutError(request=REQUEST),
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.AuthenticationError, 401),
        ValueError("not an openai error"),
    ],
    ids=lambda error: type(error).__name__,
)
def test_respond_with_non_transient_error_raises_after_single_attempt(error: BaseException) -> None:
    sleeps: list[float] = []
    backend, responses, log = _backend([error], sleeps)

    with pytest.raises(type(error)) as caught:
        backend.respond(HISTORY)

    assert caught.value is error
    assert responses.calls == 1 and sleeps == [] and log == []


def test_call_with_retry_with_zero_retries_raises_after_one_attempt() -> None:
    calls: list[int] = []

    def failing() -> None:
        calls.append(1)
        raise _status_error(openai.InternalServerError, 500)

    with pytest.raises(openai.InternalServerError):
        call_with_retry(failing, RetryPolicy(max_retries=0), sleep=lambda _: None, log=lambda _: None)

    assert calls == [1]


def test_retry_policy_defaults_give_ten_then_thirty_seconds() -> None:
    policy = RetryPolicy()

    assert policy.max_retries == 2
    assert [policy.delay_s(1), policy.delay_s(2)] == [10.0, 30.0]


def test_log_to_stderr_names_attempt_error_and_request_id_only() -> None:
    event = RetryEvent(1, 3, "InternalServerError", 500, REQUEST_ID, 10.0)
    buffer = io.StringIO()

    with redirect_stderr(buffer):
        log_to_stderr(event)

    line = buffer.getvalue()
    assert "attempt 1/3" in line and "InternalServerError" in line and "status=500" in line
    assert REQUEST_ID in line and "retrying in 10 s" in line


def test_run_log_response_persists_retry_count(tmp_path) -> None:
    backend, _, _ = _backend([_status_error(openai.InternalServerError, 500)], [])
    reply = backend.respond(HISTORY)

    RunLog(tmp_path).response(SessionState(turn=1), reply)

    entry = json.loads((tmp_path / "transcript.jsonl").read_text(encoding="utf-8"))
    assert entry["retry_count"] == 1
    assert entry["retries"][0]["request_id"] == REQUEST_ID


def _raw_config(retry: object | None) -> dict:
    raw = {
        "backend": "openai-responses",
        "model": "m",
        "reasoning_effort": "high",
        "task_mode": "shown",
        "limits": {"max_runs": 1, "max_output_tokens": 1, "max_total_tokens": 1, "max_protocol_errors": 1},
        "sandbox": {
            "python": "py",
            "timeout_s": 1,
            "memory_bytes": 1,
            "file_size_bytes": 1,
            "stdout_cap_chars": 1,
            "stderr_tail_chars": 1,
            "tmp_size_mb": 1,
            "threads": 1,
            "work_max_bytes": 1,
        },
    }
    return raw if retry is None else {**raw, "retry": retry}


def test_parse_config_without_retry_block_uses_default_policy(tmp_path) -> None:
    assert parse_config(_raw_config(None), tmp_path).retry == RetryPolicy()


def test_parse_config_with_retry_block_overrides_policy(tmp_path) -> None:
    block = {"max_retries": 1, "initial_backoff_s": 1, "backoff_factor": 2}

    assert parse_config(_raw_config(block), tmp_path).retry == RetryPolicy(1, 1.0, 2.0)


@pytest.mark.parametrize(
    "block",
    [
        {"max_retries": 2},
        {"max_retries": -1, "initial_backoff_s": 1, "backoff_factor": 2},
        {"max_retries": 3, "initial_backoff_s": 1, "backoff_factor": 2},
        {"max_retries": True, "initial_backoff_s": 1, "backoff_factor": 2},
        {"max_retries": 2, "initial_backoff_s": -1, "backoff_factor": 2},
        {"max_retries": 2, "initial_backoff_s": 1, "backoff_factor": 0.5},
    ],
)
def test_parse_config_with_invalid_retry_block_raises(tmp_path, block: dict) -> None:
    with pytest.raises(ConfigError):
        parse_config(_raw_config(block), tmp_path)


@pytest.mark.parametrize(("status", "retried"), [(499, False), (500, True), (599, True), (600, False)])
def test_respond_retries_only_inside_the_5xx_range(status: int, retried: bool) -> None:
    sleeps: list[float] = []
    backend, responses, _ = _backend([_status_error(openai.APIStatusError, status)], sleeps)

    if retried:
        backend.respond(HISTORY)
        assert responses.calls == 2 and sleeps == [10.0]
        return
    with pytest.raises(openai.APIStatusError):
        backend.respond(HISTORY)
    assert responses.calls == 1 and sleeps == []


def test_retry_log_and_record_omit_the_error_message() -> None:
    secret = "sk-test-secret-body"
    response = httpx.Response(500, request=REQUEST, headers={"x-request-id": REQUEST_ID})
    error = openai.InternalServerError(secret, response=response, body={"echo": secret})
    backend, _, _ = _backend([error], [])
    buffer = io.StringIO()
    backend.log = log_to_stderr

    with redirect_stderr(buffer):
        reply = backend.respond(HISTORY)

    assert secret not in buffer.getvalue()
    assert secret not in json.dumps(reply.retries)


def test_make_openai_client_disables_sdk_retries(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    assert make_openai_client().max_retries == 0
