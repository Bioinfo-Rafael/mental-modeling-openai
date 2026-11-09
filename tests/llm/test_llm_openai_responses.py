"""OpenAI Responses part: exact request body, result parsing, refusal/incomplete, errors, lazy key."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from translate.llm import openai_responses as oai
from translate.llm.openai_responses import (
    LLMUsage,
    OpenAIClientError,
    OpenAIIncompleteError,
    OpenAIRefusalError,
    OpenAIRequest,
    OpenAIResponsesClient,
    build_responses_body,
    input_image,
    input_text,
    make_openai_client,
)

from .conftest import OPENAI_KEY_VALUE
from .llm_fakes import (
    FAKE_MODEL,
    FAKE_RESPONSE_ID,
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    DumpableResponse,
    RecordingOpenAI,
    ScriptedResponsesClient,
    fake_openai_module,
    message_item,
    refusal_item,
    responses_body,
    usage_body,
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
IMAGE_URL = "data:image/jpeg;base64,AAAA"
EXPECTED_BODY_KEYS = {"model", "instructions", "input", "text", "reasoning", "max_output_tokens", "store"}


def _request(**overrides: Any) -> OpenAIRequest:
    fields: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "instructions": "You are a careful judge.",
        "input_items": (input_text("Trajectory text"),),
        "schema_name": "judge_verdict",
        "schema": SCHEMA,
        "reasoning_effort": "medium",
        "max_output_tokens": 25_000,
    }
    return OpenAIRequest(**{**fields, **overrides})


def _create(outcome: object, request: OpenAIRequest | None = None) -> tuple[Any, ScriptedResponsesClient]:
    fake = ScriptedResponsesClient([outcome])
    return OpenAIResponsesClient(fake).create(request or _request()), fake


# --- request body -------------------------------------------------------------------------------


def test_create_sends_exact_strict_body_without_sampling_parameters() -> None:
    _, fake = _create(responses_body())

    sent = fake.requests[0]
    assert set(sent) == EXPECTED_BODY_KEYS
    assert sent["model"] == "gpt-6.1-sol"
    assert sent["instructions"] == "You are a careful judge."
    assert sent["text"] == {
        "format": {"type": "json_schema", "name": "judge_verdict", "schema": SCHEMA, "strict": True}
    }
    assert sent["reasoning"] == {"effort": "medium"}
    assert sent["max_output_tokens"] == 25_000
    assert sent["store"] is False
    assert not {"temperature", "top_p", "top_logprobs"} & set(sent)


def test_create_sends_text_and_image_items_in_order_in_one_user_message() -> None:
    items = (input_text("Frame at t=1.0s"), input_image(IMAGE_URL, detail="high"), input_text("Rubric"))

    _, fake = _create(responses_body(), _request(input_items=items))

    assert fake.requests[0]["input"] == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Frame at t=1.0s"},
                {"type": "input_image", "image_url": IMAGE_URL, "detail": "high"},
                {"type": "input_text", "text": "Rubric"},
            ],
        }
    ]


def test_build_responses_body_returns_a_copy_that_cannot_alter_the_request() -> None:
    request = _request()

    body = build_responses_body(request)
    body["text"]["format"]["schema"]["properties"]["x"] = {"type": "string"}
    body["input"][0]["content"][0]["text"] = "changed"

    assert "x" not in request.schema["properties"]
    assert request.input_items[0]["text"] == "Trajectory text"


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_request_accepts_every_supported_reasoning_effort(effort: str) -> None:
    assert _request(reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasoning_effort": "none"},
        {"reasoning_effort": "minimal"},
        {"reasoning_effort": "turbo"},  # not in the SDK's ReasoningEffort set
        {"reasoning_effort": "High"},  # matching is exact
        {"model": ""},
        {"instructions": "   "},
        {"schema_name": "has space"},
        {"schema_name": "x" * 65},
        {"schema": {"type": "array", "items": {}}},
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"max_output_tokens": 1.5},
        {"input_items": ()},
        {"input_items": [input_text("list, not tuple")]},
        {"input_items": ({"type": "input_file", "file_id": "f"},)},
        {"input_items": ({"type": "input_text", "text": "x", "extra": 1},)},
        {"input_items": (input_text(""),)},
        {"input_items": (input_image("http://example.com/a.jpg", detail="high"),)},
        {"input_items": (input_image(IMAGE_URL, detail="ultra"),)},
        {"input_items": (input_image(IMAGE_URL, detail=["high"]),)},  # type: ignore[arg-type]
        {"input_items": ({"type": ["input_text"], "text": "x"},)},
        {"reasoning_effort": ["low"]},
        {"schema_name": "judge_verdict\n"},
        {"instructions": "lone \ud800 surrogate"},
    ],
)
def test_request_with_invalid_field_raises_client_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(OpenAIClientError):
        _request(**overrides)


def test_request_accepts_https_image_url_and_each_detail() -> None:
    items = tuple(input_image("https://example.com/f.png", detail=d) for d in ("low", "high", "original", "auto"))

    assert len(_request(input_items=items).input_items) == 4


# --- result parsing -----------------------------------------------------------------------------


def test_create_returns_result_with_text_usage_items_and_raw() -> None:
    body = responses_body(output=[message_item('{"ok": '), message_item("true}")])

    result, _ = _create(body)

    assert result.response_id == FAKE_RESPONSE_ID
    assert result.model == FAKE_MODEL
    assert result.status == "completed"
    assert result.output_text == '{"ok": true}'
    assert result.refusal is None
    assert result.incomplete_reason is None
    assert result.usage == LLMUsage(input_tokens=120, cached_input_tokens=20, output_tokens=40, reasoning_tokens=30)
    assert [item["type"] for item in result.output_items] == ["reasoning", "message", "message"]
    assert result.raw == body


def test_create_accepts_sdk_style_object_with_model_dump() -> None:
    body = responses_body()

    result, _ = _create(DumpableResponse(body))

    assert result.raw == body
    assert result.output_text == '{"ok": true}'


def test_create_with_missing_usage_reports_zero_tokens() -> None:
    body = {**responses_body(), "usage": None}

    result, _ = _create(body)

    assert result.usage == LLMUsage(0, 0, 0, 0)


def test_create_with_partial_usage_details_reports_available_counts() -> None:
    body = responses_body(usage={"input_tokens": 7, "output_tokens": 3})

    result, _ = _create(body)

    assert result.usage == LLMUsage(input_tokens=7, cached_input_tokens=0, output_tokens=3, reasoning_tokens=0)


def test_create_with_completed_but_empty_output_returns_empty_text() -> None:
    result, _ = _create(responses_body(output=[]))

    assert result.output_text == ""


def test_create_with_unreadable_response_object_raises_uncertain_error() -> None:
    with pytest.raises(OpenAIClientError) as excinfo:
        _create(object())

    assert excinfo.value.uncertain is True


class _FailingDump:
    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        raise ValueError(f"serialisation failed {OPENAI_KEY_VALUE}")


class _ListDump:
    def model_dump(self, *, mode: str = "python") -> list[str]:
        return ["not", "a", "dict"]


@pytest.mark.parametrize("response", [_FailingDump(), _ListDump()])
def test_create_with_failing_model_dump_raises_uncertain_error(response: object) -> None:
    with pytest.raises(OpenAIClientError) as excinfo:
        _create(response)

    assert excinfo.value.uncertain is True
    assert OPENAI_KEY_VALUE not in str(excinfo.value)


def _message_with(content: object) -> dict[str, Any]:
    return {**message_item("x"), "content": content}


@pytest.mark.parametrize(
    "changes",
    [
        {"status": None},
        {"status": ""},
        {"status": 1},
        {"model": None},
        {"output": None},
        {"output": "text"},
        {"output": ["not a dict"]},
        {"output": [_message_with("not a list")]},
        {"output": [_message_with(["not a dict"])]},
        {"output": [_message_with([{"type": "output_text", "text": 1}])]},
        {"output": [_message_with([{"type": "refusal", "refusal": None}])]},
        {"usage": "many"},
        {"usage": {"input_tokens": "10"}},
        {"usage": {"input_tokens": -1}},
        {"usage": {"input_tokens": True}},
        {"usage": {"input_tokens": 1, "input_tokens_details": ["cached"]}},
        {"usage": {"output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 1.5}}},
    ],
)
def test_create_with_malformed_response_shape_raises_uncertain_error_keeping_raw(changes: dict[str, Any]) -> None:
    body = {**responses_body(), **changes}

    with pytest.raises(OpenAIClientError) as excinfo:
        _create(body)

    assert excinfo.value.uncertain is True
    assert excinfo.value.raw == body
    assert excinfo.value.result is None


# --- refusal / incomplete / failed --------------------------------------------------------------


def test_create_with_refusal_raises_refusal_error_keeping_raw() -> None:
    body = responses_body(output=[refusal_item("I can't help with that.")])

    with pytest.raises(OpenAIRefusalError) as excinfo:
        _create(body)

    assert excinfo.value.result is not None
    assert excinfo.value.result.refusal == "I can't help with that."
    assert excinfo.value.result.raw == body
    assert excinfo.value.uncertain is False
    assert "I can't help" not in str(excinfo.value)


def test_create_with_incomplete_status_raises_incomplete_error_with_reason() -> None:
    body = responses_body(status="incomplete", incomplete_reason="max_output_tokens", output=[])

    with pytest.raises(OpenAIIncompleteError) as excinfo:
        _create(body)

    assert excinfo.value.result is not None
    assert excinfo.value.result.incomplete_reason == "max_output_tokens"
    assert excinfo.value.result.raw == body
    assert "max_output_tokens" in str(excinfo.value)


def test_create_with_refusal_and_incomplete_reports_refusal_first() -> None:
    body = responses_body(status="incomplete", incomplete_reason="content_filter", output=[refusal_item("no")])

    with pytest.raises(OpenAIRefusalError):
        _create(body)


@pytest.mark.parametrize(
    ("status", "uncertain"),
    [("failed", False), ("cancelled", False), ("in_progress", True), ("queued", True), ("<script>", True)],
)
def test_create_with_unexpected_status_raises_client_error_with_result(status: str, uncertain: bool) -> None:
    body = responses_body(status=status)

    with pytest.raises(OpenAIClientError) as excinfo:
        _create(body)

    assert not isinstance(excinfo.value, (OpenAIRefusalError, OpenAIIncompleteError))
    assert excinfo.value.result is not None
    assert excinfo.value.result.status == status
    assert excinfo.value.uncertain is uncertain
    assert "<script>" not in str(excinfo.value)


def test_incomplete_reason_outside_the_allowlist_is_not_echoed() -> None:
    body = responses_body(status="incomplete", incomplete_reason="echo_of_secret_state", output=[])

    with pytest.raises(OpenAIIncompleteError) as excinfo:
        _create(body)

    assert "reason=other" in str(excinfo.value)
    assert "echo_of_secret_state" not in str(excinfo.value)
    assert excinfo.value.result is not None
    assert excinfo.value.result.incomplete_reason == "echo_of_secret_state"


# --- SDK errors -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status_code", "uncertain"),
    [
        (APIStatusError("bad", status_code=400, code="invalid_json_schema"), 400, False),
        (APIStatusError("auth", status_code=401, code="invalid_api_key"), 401, False),
        (APIStatusError("rate", status_code=429), 429, False),
        (APIStatusError("boom", status_code=500), 500, True),
        (APIStatusError("gateway", status_code=504), 504, True),
        (APIConnectionError("Connection error."), None, True),
        (APITimeoutError("Request timed out."), None, True),
        (APIError("other sdk failure"), None, True),
        (APIResponseValidationError("bad 200 body"), None, True),
    ],
)
def test_create_wraps_sdk_errors_with_status_and_uncertainty(
    error: Exception, status_code: int | None, uncertain: bool
) -> None:
    with pytest.raises(OpenAIClientError) as excinfo:
        _create(error)

    assert excinfo.value.status_code == status_code
    assert excinfo.value.uncertain is uncertain
    assert excinfo.value.__cause__ is error


def test_sdk_error_message_never_contains_sdk_message_or_key() -> None:
    leaky = APIStatusError(f"Incorrect API key provided: {OPENAI_KEY_VALUE}; body echo", status_code=401)

    with pytest.raises(OpenAIClientError) as excinfo:
        _create(leaky)

    message = str(excinfo.value)
    assert OPENAI_KEY_VALUE not in message
    assert "body echo" not in message
    assert "401" in message


def test_sdk_error_code_is_reported_only_when_it_is_a_safe_identifier() -> None:
    with pytest.raises(OpenAIClientError) as safe:
        _create(APIStatusError("x", status_code=400, code="context_length_exceeded"))
    with pytest.raises(OpenAIClientError) as unsafe:
        _create(APIStatusError("x", status_code=400, code="secret text with spaces"))

    assert safe.value.error_code == "context_length_exceeded"
    assert unsafe.value.error_code is None
    assert "context_length_exceeded" not in str(safe.value)  # attribute only, never in the message
    assert "secret text" not in str(unsafe.value)


def test_non_sdk_exception_propagates_unchanged() -> None:
    with pytest.raises(TypeError, match="programming bug"):
        _create(TypeError("programming bug"))


# --- lazy key and client factory ----------------------------------------------------------------


def test_client_wrapper_reads_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result, _ = _create(responses_body())

    assert result.status == "completed"


def test_make_openai_client_without_key_raises_naming_only_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())

    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        make_openai_client()

    assert RecordingOpenAI.instances == []


def test_make_openai_client_with_empty_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())

    with pytest.raises(OpenAIClientError):
        make_openai_client()


def test_make_openai_client_passes_env_key_and_disables_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_VALUE)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())

    client = make_openai_client(timeout_s=60.0)

    assert isinstance(client, RecordingOpenAI)
    assert client.kwargs == {
        "api_key": OPENAI_KEY_VALUE,
        "base_url": "https://api.openai.com/v1",
        "max_retries": 0,
        "timeout": 60.0,
    }


def test_make_openai_client_ignores_openai_base_url_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real SDK (no fake module): OPENAI_BASE_URL cannot redirect the key or prompts. No request is sent."""
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_VALUE)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)

    client = make_openai_client(timeout_s=60.0)

    try:
        assert str(client.base_url) == "https://api.openai.com/v1/"
        assert (client.base_url.scheme, client.base_url.host) == ("https", "api.openai.com")
        assert oai.API_BASE_URL == "https://api.openai.com/v1" and client.max_retries == 0
    finally:
        client.close()


def test_make_openai_client_refuses_custom_headers_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = "X-Injected: SECRET-header-value"
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_VALUE)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", injected)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())

    with pytest.raises(OpenAIClientError, match="OPENAI_CUSTOM_HEADERS") as excinfo:
        make_openai_client()

    assert RecordingOpenAI.instances == [] and "SECRET" not in str(excinfo.value)


def test_make_openai_client_refuses_openai_log_without_echoing_its_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY_VALUE)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)
    monkeypatch.setenv("OPENAI_LOG", "debug-SECRET-level")
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module())

    with pytest.raises(OpenAIClientError, match="OPENAI_LOG") as excinfo:
        make_openai_client()

    message = str(excinfo.value)
    assert RecordingOpenAI.instances == [] and "debug" not in message and "SECRET" not in message
    assert OPENAI_KEY_VALUE not in message


def test_refused_sdk_env_lists_both_variables() -> None:
    assert set(oai.REFUSED_SDK_ENV) == {"OPENAI_CUSTOM_HEADERS", "OPENAI_LOG"}


def test_set_provenance_env_names_lists_only_set_names_never_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in oai.PROVENANCE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert oai.set_provenance_env_names() == []
    monkeypatch.setenv("https_proxy", "http://user:PW-canary@proxy.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/PW-canary.pem")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-PW-canary")

    names = oai.set_provenance_env_names()

    assert names == ["https_proxy", "SSL_CERT_FILE", "OPENAI_ORG_ID"]
    assert "PW-canary" not in " ".join(names)


def test_provenance_env_covers_the_decided_variables() -> None:
    expected = {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    expected |= {"OPENAI_ORG_ID", "OPENAI_PROJECT_ID"}
    expected |= {name.lower() for name in expected if name.endswith("_PROXY")}

    assert set(oai.PROVENANCE_ENV) == expected and len(oai.PROVENANCE_ENV) == len(expected)
    assert not set(oai.PROVENANCE_ENV) & set(oai.REFUSED_SDK_ENV)


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("inf"), float("nan")])
def test_make_openai_client_with_invalid_timeout_raises_before_reading_key(
    monkeypatch: pytest.MonkeyPatch, timeout_s: float
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIClientError, match="timeout_s"):
        make_openai_client(timeout_s=timeout_s)


def test_module_imports_the_sdk_only_inside_the_factory() -> None:
    tree = ast.parse(Path(oai.__file__).read_text(encoding="utf-8"))
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_openai_client")
    factory_imports = {alias.name for n in ast.walk(factory) if isinstance(n, ast.Import) for alias in n.names}

    assert "openai" not in top_level
    assert factory_imports == {"openai"}


def test_usage_keys_match_plan_interface() -> None:
    assert list(LLMUsage.__dataclass_fields__) == [
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
    ]


def test_usage_body_helper_round_trips() -> None:
    result, _ = _create(responses_body(usage=usage_body(input_tokens=1, cached=0, output_tokens=2, reasoning=1)))

    assert result.usage == LLMUsage(1, 0, 2, 1)
