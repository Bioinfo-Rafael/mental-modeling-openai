"""CLI: only method B's commands remain, and the default client is the shared, pinned one.

The paid-call gate of ``intent-run`` and its preview path are covered in ``test_wr_intent_run.py``.
"""

from __future__ import annotations

import pytest

from mtd_rl_xavier.llm.openai_responses import API_BASE_URL, OpenAIClientError, make_openai_client
from weightreading import cli, intent_run
from weightreading.sandbox import Interpreter

from .test_wr_session import FakeExecutor

METHOD_B_COMMANDS = {"sandbox-selftest", "intent-preview", "intent-run"}
# Shared with the intent-run tests (test_wr_intent_run.py, test_wr_intent_limits.py).
DUMMY_INTERPRETER = Interpreter("/opt/python", "bin/python3", "/opt/site/numpy", "", "2.0", "3.12.0")


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


def _fake_executor() -> FakeExecutor:
    executor = FakeExecutor()
    executor.interp = DUMMY_INTERPRETER
    return executor


def test_cli_offers_only_method_b_commands() -> None:
    subparsers = next(action for action in cli.build_parser()._actions if action.dest == "command")

    assert set(subparsers.choices) == METHOD_B_COMMANDS


@pytest.mark.parametrize("command", ["run", "evaluate", "inspect", "bundle", "preview", "audit", "validate-spec"])
def test_removed_weights_only_commands_are_usage_errors(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([command])

    assert raised.value.code == 2


def test_sandbox_selftest_defaults_to_the_method_b_config() -> None:
    args = cli.build_parser().parse_args(["sandbox-selftest"])

    assert args.config == intent_run.DEFAULT_CONFIG


def test_make_openai_client_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        make_openai_client()


def test_cli_default_client_factory_is_the_shared_pinned_factory() -> None:
    assert cli.main.__defaults__[-1] is make_openai_client


def test_cli_default_client_ignores_openai_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://attacker.invalid/v1")
    monkeypatch.delenv("OPENAI_LOG", raising=False)
    monkeypatch.delenv("OPENAI_CUSTOM_HEADERS", raising=False)

    client = cli.main.__defaults__[-1]()

    assert str(client.base_url).rstrip("/") == API_BASE_URL
    assert client.max_retries == 0


def test_cli_default_client_refuses_openai_log(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_LOG", "debug")

    with pytest.raises(OpenAIClientError, match="OPENAI_LOG"):
        cli.main.__defaults__[-1]()
