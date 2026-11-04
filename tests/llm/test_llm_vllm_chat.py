"""Local chat part: request body, response parsing, typed errors, endpoint checks, lazy SDK.

Uses a scripted SDK-shaped client and a fake ``openai`` module only; the real SDK contract is in
``test_llm_vllm_chat_sdk.py``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from mtd_rl_xavier.llm import vllm_chat
from mtd_rl_xavier.llm.vllm_chat import (
    ChatMessage,
    VLLMChatClient,
    VLLMChatConfigError,
    VLLMChatError,
    VLLMChatIncompleteError,
    VLLMChatRefusalError,
    VLLMChatRequest,
    build_chat_body,
    make_vllm_chat_client,
)

from .llm_fakes import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    DumpableResponse,
    SystemOneServer,
    models_body,
)
from .local_runtime_fakes import (
    CHAT_MODEL,
    FakeOmit,
    RecordingChatOpenAI,
    ScriptedChatClient,
    chat_body,
    fake_chat_openai_module,
)

KEY_ENV = "TEST_VLLM_CHAT_KEY"
KEY_VALUE = "chat-SECRET-key-0123456789"
SCHEMA = {"type": "object", "properties": {"intent": {"type": "string"}}, "required": ["intent"]}
MESSAGES = (ChatMessage("system", "Answer in JSON."), ChatMessage("user", "Describe the motion."))


@pytest.fixture
def chat_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(KEY_ENV, KEY_VALUE)
    return KEY_ENV


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[RecordingChatOpenAI]:
    monkeypatch.setitem(sys.modules, "openai", fake_chat_openai_module())
    for name in vllm_chat.REFUSED_SDK_ENV:
        monkeypatch.delenv(name, raising=False)
    return RecordingChatOpenAI


def request(**overrides: Any) -> VLLMChatRequest:
    values: dict[str, Any] = {"model": CHAT_MODEL, "messages": MESSAGES, "max_tokens": 256}
    return VLLMChatRequest(**{**values, **overrides})


def complete(outcome: object, **overrides: Any) -> tuple[Any, ScriptedChatClient]:
    scripted = ScriptedChatClient([outcome])
    return VLLMChatClient(scripted).complete(request(**overrides)), scripted


# --- request ------------------------------------------------------------------------------------------


def test_body_sends_only_model_messages_and_max_tokens_by_default() -> None:
    assert build_chat_body(request()) == {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "Answer in JSON."},
            {"role": "user", "content": "Describe the motion."},
        ],
        "max_tokens": 256,
    }


def test_body_adds_temperature_seed_and_strict_json_schema_when_set() -> None:
    schema = {"type": "object", "properties": {}}
    body = build_chat_body(request(temperature=0.0, seed=7, schema_name="intent_v1", schema=schema))
    schema["properties"]["late"] = {"type": "string"}  # the body is a deep copy

    assert body["temperature"] == 0.0 and body["seed"] == 7
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "intent_v1", "schema": {"type": "object", "properties": {}}, "strict": True},
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"messages": ()},
        {"messages": [ChatMessage("user", "x")]},
        {"messages": ({"role": "user", "content": "x"},)},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": True},
        {"seed": True},
        {"seed": 1.5},
        {"schema": SCHEMA},
        {"schema_name": "intent_v1"},
        {"schema_name": "bad name", "schema": SCHEMA},
        {"schema_name": "intent_v1", "schema": {"type": "array"}},
    ],
)
def test_invalid_request_is_rejected_before_sending(overrides: dict[str, Any]) -> None:
    with pytest.raises(VLLMChatConfigError):
        request(**overrides)


@pytest.mark.parametrize(("role", "content"), [("tool", "x"), ("developer", "x"), ("user", ""), ("user", "  ")])
def test_invalid_message_is_rejected(role: str, content: str) -> None:
    with pytest.raises(VLLMChatConfigError):
        ChatMessage(role, content)


# --- response ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("wrap", [dict, DumpableResponse], ids=["dict", "sdk-object"])
def test_completed_response_is_parsed_and_raw_kept(wrap: Any) -> None:
    result, scripted = complete(wrap(chat_body('{"intent": "climb"}')))

    assert result.content == '{"intent": "climb"}' and result.finish_reason == "stop"
    assert result.model == CHAT_MODEL and result.response_id == "chatcmpl-1"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (50, 7)
    assert result.raw == chat_body('{"intent": "climb"}')
    assert scripted.requests == [build_chat_body(request())]


def test_missing_usage_counts_as_zero() -> None:
    body = {**chat_body(), "usage": None}

    result, _ = complete(body)

    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (0, 0)


def test_length_finish_is_incomplete_and_keeps_the_partial_text() -> None:
    with pytest.raises(VLLMChatIncompleteError, match="length") as excinfo:
        complete(chat_body('{"intent": "cl', finish_reason="length"))

    assert excinfo.value.result is not None and excinfo.value.result.content == '{"intent": "cl'
    assert excinfo.value.uncertain is False


def test_refusal_raises_and_keeps_the_text_out_of_the_message() -> None:
    with pytest.raises(VLLMChatRefusalError) as excinfo:
        complete(chat_body(None, refusal="I-REFUSE-SECRET"))

    assert "I-REFUSE-SECRET" not in str(excinfo.value)
    assert excinfo.value.result is not None and excinfo.value.result.refusal == "I-REFUSE-SECRET"


@pytest.mark.parametrize(
    ("finish", "shown"),
    [("tool_calls", "tool_calls"), (None, "other"), ("bad reason!", "other"), ("customer-secret-123", "other")],
)
def test_unexpected_finish_reason_is_a_known_failure(finish: str | None, shown: str) -> None:
    with pytest.raises(VLLMChatError, match=f"unexpected finish_reason \\({shown}\\)") as excinfo:
        complete(chat_body(finish_reason=finish))

    assert excinfo.value.uncertain is False


@pytest.mark.parametrize("content", [None, ""])
def test_stop_without_content_is_an_error(content: str | None) -> None:
    with pytest.raises(VLLMChatError, match="no message content"):
        complete(chat_body(content))


@pytest.mark.parametrize(
    "body",
    [
        {**chat_body(), "model": None},
        {**chat_body(), "object": "text_completion"},
        chat_body(choices=[{"index": 1, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]),
        chat_body(choices=[{"index": 0, "message": {"role": "user", "content": "x"}, "finish_reason": "stop"}]),
        {**chat_body(), "choices": []},
        {**chat_body(), "choices": [chat_body()["choices"][0]] * 2},
        chat_body(choices=[{"index": 0, "message": "text", "finish_reason": "stop"}]),
        chat_body(choices=[{"index": 0, "message": {"content": 5}, "finish_reason": "stop"}]),
        chat_body(choices=[{"index": 0, "message": {"content": "x"}, "finish_reason": 1}]),
        {**chat_body(), "usage": {"prompt_tokens": -1, "completion_tokens": 0}},
        {**chat_body(), "usage": {"prompt_tokens": True, "completion_tokens": 0}},
        {**chat_body(), "usage": "lots"},
    ],
)
def test_unreadable_response_is_uncertain_and_keeps_raw(body: dict[str, Any]) -> None:
    with pytest.raises(VLLMChatError, match="unreadable") as excinfo:
        complete(body)

    assert excinfo.value.uncertain is True and excinfo.value.raw == body


def test_response_whose_dump_fails_is_uncertain() -> None:
    class Broken:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            raise ValueError("cannot serialise")

    with pytest.raises(VLLMChatError, match="unreadable") as excinfo:
        complete(Broken())

    assert excinfo.value.uncertain is True


# --- SDK errors ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "uncertain"),
    [
        (APIStatusError("bad SECRET-PROMPT", status_code=400, code="invalid_request"), 400, False),
        (APIStatusError("auth SECRET-PROMPT", status_code=401), 401, False),
        (APIStatusError("boom SECRET-PROMPT", status_code=500), 500, True),
        (APIConnectionError("refused SECRET-PROMPT"), None, True),
        (APITimeoutError("timeout SECRET-PROMPT"), None, True),
    ],
)
def test_sdk_errors_become_typed_errors_without_sdk_text(error: Exception, status: int | None, uncertain: bool) -> None:
    with pytest.raises(VLLMChatError) as excinfo:
        complete(error)

    assert excinfo.value.status_code == status and excinfo.value.uncertain is uncertain
    assert "SECRET-PROMPT" not in str(excinfo.value) and type(error).__name__ in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__context__ is None  # SDK error not chained


def test_unsafe_sdk_error_code_is_dropped() -> None:
    with pytest.raises(VLLMChatError) as excinfo:
        complete(APIStatusError("x", status_code=400, code="code with spaces <script>"))

    assert excinfo.value.error_code is None


def test_non_sdk_exception_propagates_unchanged() -> None:
    with pytest.raises(KeyError, match="not-sdk"):
        complete(KeyError("not-sdk"))


# --- health ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("wrap", [dict, DumpableResponse], ids=["dict", "sdk-page"])
def test_health_passes_when_the_model_is_listed(wrap: Any) -> None:
    scripted = ScriptedChatClient([wrap(models_body("other", CHAT_MODEL))])

    VLLMChatClient(scripted).health(CHAT_MODEL)

    assert scripted.model_lists == 1


def test_health_fails_when_the_model_is_not_listed() -> None:
    with pytest.raises(VLLMChatError, match="not served") as excinfo:
        VLLMChatClient(ScriptedChatClient([models_body("other")])).health(CHAT_MODEL)

    assert excinfo.value.uncertain is False


def test_health_with_malformed_list_or_sdk_error_raises_typed_errors() -> None:
    malformed = [{"object": "list"}, {"data": [{"id": CHAT_MODEL}, {"id": 7}]}, {"data": [CHAT_MODEL]}]
    client = VLLMChatClient(ScriptedChatClient([*malformed, APIConnectionError("down")]))

    for _ in malformed:
        with pytest.raises(VLLMChatError, match="unreadable") as excinfo:
            client.health(CHAT_MODEL)
        assert excinfo.value.uncertain is True
    with pytest.raises(VLLMChatError, match="APIConnectionError"):
        client.health(CHAT_MODEL)


def test_constructing_the_wrapper_calls_nothing() -> None:
    scripted = ScriptedChatClient([])

    VLLMChatClient(scripted)

    assert scripted.requests == [] and scripted.model_lists == 0


# --- client factory ---------------------------------------------------------------------------------------


@pytest.mark.usefixtures("chat_key")
def test_factory_over_uds_pins_base_url_key_no_retries_and_a_local_http_client(
    fake_openai: type[RecordingChatOpenAI], secure_uds_path: Path
) -> None:
    make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=30.0)

    kwargs = fake_openai.instances[0].kwargs
    assert kwargs["api_key"] == KEY_VALUE and kwargs["base_url"] == "http://localhost/v1"
    assert kwargs["max_retries"] == 0 and kwargs["timeout"] == 30.0
    http_client = kwargs["http_client"]
    assert isinstance(http_client, httpx.Client)
    assert http_client.trust_env is False and http_client.follow_redirects is False
    assert set(kwargs["default_headers"]) == {"OpenAI-Organization", "OpenAI-Project"}
    assert all(isinstance(value, FakeOmit) for value in kwargs["default_headers"].values())


@pytest.mark.usefixtures("chat_key")
def test_factory_over_tcp_uses_the_validated_loopback_root(fake_openai: type[RecordingChatOpenAI]) -> None:
    make_vllm_chat_client(base_url="http://127.0.0.1:8001/", uds_path=None, api_key_env=KEY_ENV, timeout_s=5)

    assert fake_openai.instances[0].kwargs["base_url"] == "http://127.0.0.1:8001/v1"


@pytest.mark.parametrize("uds", [True, False])
def test_local_http_client_uses_the_socket_and_no_transport_retries(
    monkeypatch: pytest.MonkeyPatch, secure_uds_path: Path, uds: bool
) -> None:
    calls: list[dict[str, Any]] = []

    def recording_transport(**kwargs: Any) -> httpx.MockTransport:
        calls.append(kwargs)
        return httpx.MockTransport(lambda request: httpx.Response(500))

    monkeypatch.setattr(vllm_chat.httpx, "HTTPTransport", recording_transport)

    vllm_chat.build_local_http_client(uds_path=secure_uds_path if uds else None, timeout_s=1.0)

    assert calls == [{"uds": str(secure_uds_path) if uds else None, "retries": 0, "trust_env": False}]


@pytest.mark.usefixtures("chat_key")
@pytest.mark.parametrize(
    "endpoint",
    [
        {"base_url": "http://0.0.0.0:8000", "uds_path": None},
        {"base_url": "http://10.0.0.5:8000", "uds_path": None},
        {"base_url": "http://user:pw@127.0.0.1:8000", "uds_path": None},
        {"base_url": "http://localhost:8000", "uds_path": None},
        {"base_url": None, "uds_path": None},
        {"base_url": None, "uds_path": Path("relative.sock")},
    ],
)
def test_factory_rejects_unsafe_endpoints_before_building_a_client(
    fake_openai: type[RecordingChatOpenAI], endpoint: dict[str, Any]
) -> None:
    with pytest.raises(VLLMChatConfigError) as excinfo:
        make_vllm_chat_client(**endpoint, api_key_env=KEY_ENV, timeout_s=5.0)

    assert fake_openai.instances == [] and "pw" not in str(excinfo.value)


@pytest.mark.usefixtures("chat_key")
def test_factory_rejects_an_insecure_socket_directory(
    fake_openai: type[RecordingChatOpenAI], secure_uds_path: Path
) -> None:
    os.chmod(secure_uds_path.parent, 0o755)

    with pytest.raises(VLLMChatConfigError, match="0700"):
        make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=5.0)

    assert fake_openai.instances == []


@pytest.mark.parametrize("api_key_env", [None, "", "chat-SECRET-key", "UNSET_VLLM_CHAT_KEY"])
def test_factory_requires_a_set_key_variable_even_over_uds(
    fake_openai: type[RecordingChatOpenAI], secure_uds_path: Path, monkeypatch: pytest.MonkeyPatch, api_key_env: Any
) -> None:
    monkeypatch.delenv("UNSET_VLLM_CHAT_KEY", raising=False)

    with pytest.raises(VLLMChatConfigError) as excinfo:
        make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=api_key_env, timeout_s=5.0)

    assert fake_openai.instances == [] and "SECRET" not in str(excinfo.value)


@pytest.mark.usefixtures("chat_key")
@pytest.mark.parametrize("name", ["OPENAI_LOG", "OPENAI_CUSTOM_HEADERS"])
def test_factory_refuses_sdk_variables_that_log_or_inject_headers(
    fake_openai: type[RecordingChatOpenAI], secure_uds_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "debug-VALUE-canary")

    with pytest.raises(VLLMChatConfigError, match=name) as excinfo:
        make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=5.0)

    assert fake_openai.instances == [] and "VALUE-canary" not in str(excinfo.value)


@pytest.mark.usefixtures("chat_key")
@pytest.mark.parametrize("logger_name", vllm_chat.SDK_LOGGER_NAMES)
def test_factory_refuses_an_sdk_logger_enabled_for_debug(
    fake_openai: type[RecordingChatOpenAI], secure_uds_path: Path, logger_name: str
) -> None:
    logger = logging.getLogger(logger_name)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(VLLMChatConfigError, match="DEBUG"):
            make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=5.0)
    finally:
        logger.setLevel(previous)

    assert fake_openai.instances == []


@pytest.mark.parametrize("call", ["complete", "health"])
def test_sdk_debug_logging_enabled_after_construction_blocks_each_call(call: str) -> None:
    scripted = ScriptedChatClient([chat_body(), models_body(CHAT_MODEL)])
    client = VLLMChatClient(scripted)
    logger = logging.getLogger(vllm_chat.SDK_LOGGER_NAMES[0])
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        with pytest.raises(VLLMChatConfigError, match="DEBUG"):
            client.complete(request()) if call == "complete" else client.health(CHAT_MODEL)
    finally:
        logger.setLevel(previous)

    assert scripted.requests == [] and scripted.model_lists == 0


@pytest.mark.usefixtures("chat_key")
def test_factory_closes_its_http_client_when_the_sdk_constructor_fails(
    monkeypatch: pytest.MonkeyPatch, secure_uds_path: Path
) -> None:
    received: list[httpx.Client] = []

    class FailingOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            received.append(kwargs["http_client"])
            raise ValueError("constructor failed")

    module = fake_chat_openai_module()
    module.OpenAI = FailingOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)

    with pytest.raises(ValueError, match="constructor failed"):
        make_vllm_chat_client(base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=5.0)

    assert len(received) == 1 and received[0].is_closed


@pytest.mark.parametrize("module", ["mtd_rl_xavier.llm.vllm_chat", "mtd_rl_xavier.llm.vllm_server"])
def test_importing_the_local_parts_does_not_import_the_sdk(module: str) -> None:
    code = f"import sys, {module}; sys.exit(1 if 'openai' in sys.modules else 0)"

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


@pytest.mark.usefixtures("chat_key")
@pytest.mark.parametrize("timeout_s", [0, -1.0, float("nan"), float("inf"), True])
def test_factory_rejects_invalid_timeouts(fake_openai: type[RecordingChatOpenAI], timeout_s: Any) -> None:
    with pytest.raises(VLLMChatConfigError, match="timeout_s"):
        make_vllm_chat_client(base_url="http://127.0.0.1:8000", uds_path=None, api_key_env=KEY_ENV, timeout_s=timeout_s)


@pytest.mark.usefixtures("chat_key", "fake_openai")
def test_factory_makes_no_request_at_construction(secure_uds_path: Path) -> None:
    server = SystemOneServer([])

    make_vllm_chat_client(
        base_url=None, uds_path=secure_uds_path, api_key_env=KEY_ENV, timeout_s=5.0, transport=server.transport
    )

    assert server.requests == []
