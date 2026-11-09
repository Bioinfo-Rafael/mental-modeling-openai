"""Shared OpenAI client safety for the unified translate CLI."""

from __future__ import annotations

import pytest

from translate import cli
from translate.llm.openai_responses import API_BASE_URL, OpenAIClientError, make_openai_client


class ScriptedClient:
    """Stands in for ``openai.OpenAI``: ``responses.create`` returns scripted replies, no network."""

    def __init__(self, texts: list[str]) -> None:
        self.responses = self
        self.texts = texts
        self.requests: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        text = self.texts.pop(0)
        usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
        return type(
            "Response", (), {"id": "r", "status": "completed", "output_text": text, "output": [], "usage": usage}
        )()


def test_make_openai_client_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        make_openai_client()


def test_cli_default_client_factory_is_the_shared_pinned_factory() -> None:
    assert cli.Factories().openai_client is make_openai_client


def test_cli_default_client_ignores_openai_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
    monkeypatch.delenv("OPENAI_LOG", raising=False)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)

    client = cli.Factories().openai_client()

    assert str(client.base_url).rstrip("/") == API_BASE_URL
    assert client.max_retries == 0


def test_cli_default_client_refuses_openai_log(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_LOG", "debug")

    with pytest.raises(OpenAIClientError, match="OPENAI_LOG"):
        cli.Factories().openai_client()
