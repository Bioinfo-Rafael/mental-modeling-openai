"""One-shot calls on the shared retry policy: 5xx resent with the same body and journaled, timeouts never resent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import openai
import pandas as pd
import pytest

from translate import cli, paths
from translate.call_one_shot import (
    OUTPUT_SCHEMA_NAME,
    IntentModelOutput,
    OneShotCall,
    RetryJournal,
    TranslationPrompt,
    output_schema,
)
from translate.config import RunConfig, load_config
from translate.events import ModelCallError
from translate.llm.retry import RetryEvent
from translate.pipeline import Translator

from .tr_fakes import FakeIntentModel, MemorySink, fake_translation_renderer, instruction, make_config, make_scenario

REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")
TRAJECTORY_CONFIG = paths.CONFIGS / "translate" / "seed100601-trajectory.yaml"
ONLINE_CONFIG = paths.CONFIGS / "translate" / "online.example.yaml"
ANSWER = '{"intent": "i", "evidence": ["e"]}'


class _Responses:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.bodies: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> dict[str, Any]:
        self.bodies.append(dict(kwargs))
        if self.failures:
            raise self.failures.pop(0)
        return {"id": "r", "model": "m", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": ANSWER}]}]}  # fmt: skip


class _SDK:
    def __init__(self, failures: list[BaseException]) -> None:
        self.responses = _Responses(failures)


def _server_error() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=httpx.Response(503, request=REQUEST), body=None)


def _prompt() -> TranslationPrompt:
    return TranslationPrompt("instructions", "user text", OUTPUT_SCHEMA_NAME, output_schema())


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("translate.llm.retry.time.sleep", lambda _: None)


def _call(config: RunConfig, failures: list[BaseException]) -> tuple[OneShotCall, _SDK, RetryJournal]:
    sdk, journal = _SDK(failures), RetryJournal()
    return cli._one_shot_call(config, sdk, journal), sdk, journal


def test_one_shot_call_with_default_retry_resends_a_5xx_with_the_same_body_and_journals_it() -> None:
    call, sdk, journal = _call(load_config(TRAJECTORY_CONFIG), [_server_error()])

    result = call.run(_prompt())

    assert result.response_id == "r"
    assert len(sdk.responses.bodies) == 2 and sdk.responses.bodies[0] == sdk.responses.bodies[1]
    assert [(event["attempt"], event["status_code"]) for event in journal.drain()] == [(1, 503)]
    assert journal.drain() == ()


def test_one_shot_call_with_default_retry_never_resends_a_timeout() -> None:
    call, sdk, journal = _call(load_config(TRAJECTORY_CONFIG), [openai.APITimeoutError(request=REQUEST)])

    with pytest.raises(ModelCallError) as caught:
        call.run(_prompt())

    assert caught.value.uncertain is True
    assert len(sdk.responses.bodies) == 1
    assert journal.drain() == ()


def test_one_shot_call_with_retry_none_sends_a_5xx_once() -> None:
    call, sdk, journal = _call(load_config(ONLINE_CONFIG), [_server_error()])

    with pytest.raises(ModelCallError):
        call.run(_prompt())

    assert len(sdk.responses.bodies) == 1
    assert journal.drain() == ()


class _RetriedModel(FakeIntentModel):
    """Records one failed attempt in the journal before answering, as RetryingClient would."""

    def __init__(self, journal: RetryJournal) -> None:
        super().__init__()
        self._journal = journal

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        self._journal.record(RetryEvent(1, 3, "InternalServerError", 503, "req-1", 10.0))
        return super().generate(prompt)


def test_pipeline_writes_the_retried_attempts_into_the_response_line(short_frame: pd.DataFrame) -> None:
    config = make_config(triggers__source="preset", triggers__tick_interval_steps=None,
                         triggers__points=[{"step": 0, "kind": "initial", "targets": None}], detector=None)  # fmt: skip
    journal, sink = RetryJournal(), MemorySink()
    translator = Translator(
        call=OneShotCall(_RetriedModel(journal), config.prompt, config.answer.limits),
        translation_renderer=fake_translation_renderer,
        retries=journal,
    )
    scenario = make_scenario([instruction("i-1", 0, "aircraft", "A0100")])

    summary = translator.run(frame=short_frame, scenario=scenario, config=config, sink=sink)

    assert summary.status == "completed"
    retries = [line["retries"] for line in sink.responses]
    assert retries == [[{"attempt": 1, "max_attempts": 3, "error_type": "InternalServerError", "status_code": 503,
                         "request_id": "req-1", "delay_s": 10.0}]]  # fmt: skip
    assert json.loads(json.dumps(sink.responses))  # standard JSON


def test_one_shot_call_journal_is_empty_without_failures(tmp_path: Path) -> None:
    call, sdk, journal = _call(load_config(TRAJECTORY_CONFIG), [])

    call.run(_prompt())

    assert len(sdk.responses.bodies) == 1 and journal.drain() == ()
