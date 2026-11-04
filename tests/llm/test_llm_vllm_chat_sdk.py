"""Local chat part against the REAL ``openai`` SDK over ``httpx.MockTransport`` (no network).

These prove what the fakes cannot: the SDK accepts the local httpx client, sends the Bearer key
and no organisation/project header, makes exactly one attempt (``max_retries=0``), and raises the
error classes the client maps. Importing the SDK costs a few seconds, so these live apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mtd_rl_xavier.llm.vllm_chat import (
    ChatMessage,
    VLLMChatClient,
    VLLMChatError,
    VLLMChatRequest,
    make_vllm_chat_client,
)

from .llm_fakes import SystemOneServer, json_response, models_body, raise_transport
from .local_runtime_fakes import CHAT_MODEL, chat_body

KEY_ENV = "TEST_VLLM_CHAT_SDK_KEY"
KEY_VALUE = "sdk-SECRET-key-0123456789"
SCHEMA = {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}


@pytest.fixture(autouse=True)
def sdk_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    monkeypatch.setenv("OPENAI_ORG_ID", "org-canary")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-canary")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-paid-canary")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://elsewhere.example.com/v1")
    for name in ("OPENAI_LOG", "OPENAI_CUSTOM_HEADERS"):
        monkeypatch.delenv(name, raising=False)


def client_over(server: SystemOneServer, uds_path: Path) -> VLLMChatClient:
    sdk = make_vllm_chat_client(
        base_url=None, uds_path=uds_path, api_key_env=KEY_ENV, timeout_s=5.0, transport=server.transport
    )
    return VLLMChatClient(sdk)


def request() -> VLLMChatRequest:
    return VLLMChatRequest(
        model=CHAT_MODEL,
        messages=(ChatMessage("user", "Describe the motion."),),
        max_tokens=64,
        temperature=0.0,
        schema_name="intent_v1",
        schema=SCHEMA,
    )


def test_real_sdk_client_makes_no_request_at_construction(secure_uds_path: Path) -> None:
    server = SystemOneServer([])

    client_over(server, secure_uds_path)

    assert server.requests == []


def test_real_sdk_sends_one_keyed_request_to_the_local_server(secure_uds_path: Path) -> None:
    server = SystemOneServer([json_response(200, chat_body('{"intent": "climb"}'))])

    result = client_over(server, secure_uds_path).complete(request())

    assert result.content == '{"intent": "climb"}'
    (sent,) = server.requests
    assert (sent.method, sent.path) == ("POST", "/v1/chat/completions")
    assert sent.headers["host"] == "localhost"
    assert sent.headers["authorization"] == f"Bearer {KEY_VALUE}"
    assert "openai-organization" not in sent.headers and "openai-project" not in sent.headers
    assert "canary" not in json.dumps(sent.headers)
    body: dict[str, Any] = sent.body or {}
    assert body["response_format"]["json_schema"]["strict"] is True and body["max_tokens"] == 64


@pytest.mark.parametrize(
    ("outcome", "status"),
    [(json_response(500, {"error": {"message": "engine died"}}), 500), (raise_transport(httpx.ReadTimeout), None)],
    ids=["http-500", "timeout"],
)
def test_real_sdk_makes_exactly_one_attempt_and_reports_uncertain(
    secure_uds_path: Path, outcome: Any, status: int | None
) -> None:
    server = SystemOneServer([outcome, json_response(200, chat_body())])  # a retry would succeed

    with pytest.raises(VLLMChatError) as excinfo:
        client_over(server, secure_uds_path).complete(request())

    assert len(server.requests) == 1
    assert excinfo.value.uncertain is True and excinfo.value.status_code == status
    assert "engine died" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None  # no key via the chain


def test_real_sdk_client_error_is_certain(secure_uds_path: Path) -> None:
    server = SystemOneServer([json_response(400, {"error": {"message": "bad schema", "code": "invalid"}})])

    with pytest.raises(VLLMChatError) as excinfo:
        client_over(server, secure_uds_path).complete(request())

    assert excinfo.value.uncertain is False and excinfo.value.status_code == 400


def test_real_sdk_health_lists_models_with_the_key(secure_uds_path: Path) -> None:
    server = SystemOneServer([json_response(200, models_body(CHAT_MODEL))])

    client_over(server, secure_uds_path).health(CHAT_MODEL)

    (sent,) = server.requests
    assert (sent.method, sent.path) == ("GET", "/v1/models")
    assert sent.headers["authorization"] == f"Bearer {KEY_VALUE}"
