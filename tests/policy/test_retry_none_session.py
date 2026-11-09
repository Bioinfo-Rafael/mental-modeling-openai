"""A code session under ``translation.retry: none`` sends each request once (no crash, no retry)."""

from __future__ import annotations

from pathlib import Path

import httpx
import openai
import pytest

from tests.llm.llm_fakes import ScriptedResponsesClient
from translate.config import parse_config
from translate.session_backend import OpenAIResponsesBackend, user_item

from .policy_fakes import policy_raw
from .test_wr_cli_gate import ScriptedClient

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _retry_none_backend(tmp_path: Path, client: object) -> OpenAIResponsesBackend:
    raw = policy_raw(tmp_path / "rollout", tmp_path / "spec.json")
    raw = {**raw, "translation": {**raw["translation"], "retry": "none"}}
    config = parse_config(raw, path=tmp_path / "config.yaml", sha256="test")
    translation = config.translation
    return OpenAIResponsesBackend(
        client, translation.model, translation.reasoning_effort, translation.max_output_tokens, translation.retry
    )


def test_session_backend_with_retry_none_sends_and_answers(tmp_path: Path) -> None:
    client = ScriptedClient(["```json\n{}\n```"])

    reply = _retry_none_backend(tmp_path, client).respond((user_item("first"),))

    assert reply.text.startswith("```json") and reply.retries == () and len(client.requests) == 1


def test_session_backend_with_retry_none_does_not_resend_a_5xx(tmp_path: Path) -> None:
    error = openai.InternalServerError("boom", response=httpx.Response(503, request=REQUEST), body=None)
    client = ScriptedResponsesClient([error])

    with pytest.raises(openai.InternalServerError):
        _retry_none_backend(tmp_path, client).respond((user_item("first"),))

    assert len(client.requests) == 1
