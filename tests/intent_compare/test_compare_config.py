"""The one compare.yaml parser shared by methods A and B, and method A's use of the shared retry policy."""

from __future__ import annotations

import hashlib

import httpx
import openai
import pytest
import yaml

import mtd_rl_xavier.intent_compare.method_a_cli as run_a
from mtd_rl_xavier import intent_compare, paths
from mtd_rl_xavier.intent_compare import (
    DEFAULT_CONFIG,
    REQUIRED_KEYS,
    CompareConfigError,
    check_run_id,
    load_compare,
    parse_compare,
)
from mtd_rl_xavier.intent_compare.method_a_requests import TranslationSettings
from mtd_rl_xavier.llm import openai_responses
from mtd_rl_xavier.llm.openai_responses import REASONING_EFFORTS
from mtd_rl_xavier.translator.translation import OUTPUT_SCHEMA_NAME, TranslationPrompt, output_schema
from weightreading import intent_cli, intent_run

COMMITTED_CONFIGS = sorted((paths.CONFIGS / "intent_compare").glob("compare*.yaml"))  # not pricing.yaml
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _parse(raw: dict) -> intent_compare.CompareConfig:
    return parse_compare(raw, DEFAULT_CONFIG, "sha")


# --- one parser for both methods -----------------------------------------------------------------


def test_methods_a_and_b_use_the_same_parser_and_run_id_rule() -> None:
    assert run_a.load_compare is load_compare
    assert intent_run.load_compare is load_compare
    assert intent_cli.check_run_id is check_run_id


@pytest.mark.parametrize("path", COMMITTED_CONFIGS, ids=lambda path: path.name)
def test_every_committed_config_loads_with_a_shared_effort(path) -> None:
    config = load_compare(path)

    assert config.reasoning_effort in REASONING_EFFORTS
    assert config.steps == tuple(sorted(set(config.steps)))
    assert config.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_configs_are_found() -> None:
    assert {path.name for path in COMMITTED_CONFIGS} >= {"compare.yaml", "compare.ja.yaml", "compare.ja-short.yaml"}


# --- tightened rules -----------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [[145, 45], [45, 245, 145], [45, 45]])
def test_parse_compare_with_unordered_or_repeated_steps_raises(steps: list[int]) -> None:
    with pytest.raises(CompareConfigError, match="steps"):
        _parse({**_raw(), "steps": steps})


@pytest.mark.parametrize("seed", [True, False, "7", 7.0, 7.9, None, -1])
def test_parse_compare_with_non_int_or_negative_seed_raises(seed: object) -> None:
    with pytest.raises(CompareConfigError, match="seed"):
        _parse({**_raw(), "seed": seed})


@pytest.mark.parametrize("effort", ["turbo", "HIGH", "high ", "", None, 1])
def test_parse_compare_with_effort_outside_the_shared_allowlist_raises(effort: object) -> None:
    with pytest.raises(CompareConfigError, match="reasoning_effort"):
        _parse({**_raw(), "reasoning_effort": effort})


@pytest.mark.parametrize("effort", sorted(REASONING_EFFORTS))
def test_parse_compare_accepts_every_shared_effort(effort: str) -> None:
    assert _parse({**_raw(), "reasoning_effort": effort}).reasoning_effort == effort


def test_parse_compare_strips_the_instruction_and_rejects_a_blank_one() -> None:
    assert _parse({**_raw(), "instruction": "  go  \n"}).instruction == "go"
    with pytest.raises(CompareConfigError, match="instruction"):
        _parse({**_raw(), "instruction": " \n "})


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_parse_compare_without_a_required_key_raises(key: str) -> None:
    raw = {name: value for name, value in _raw().items() if name != key}

    with pytest.raises(CompareConfigError, match=key):
        _parse(raw)


@pytest.mark.parametrize("value", ["0.2", True, 0, -0.2])
def test_parse_compare_with_non_positive_or_non_numeric_step_seconds_raises(value: object) -> None:
    with pytest.raises(CompareConfigError, match="step_seconds"):
        _parse({**_raw(), "step_seconds": value})


@pytest.mark.parametrize("run_id", ["run1\n", "run1\r\n", "\nrun1", "run 1", "a..b", "../x", "", None])
def test_check_run_id_with_trailing_newline_or_unsafe_text_raises(run_id: object) -> None:
    with pytest.raises(CompareConfigError):
        check_run_id(run_id)


def test_run_dir_with_trailing_newline_raises_for_method_a_and_b_paths() -> None:
    config = load_compare(DEFAULT_CONFIG)

    with pytest.raises(CompareConfigError):
        config.run_dir("seed100601\n")
    with pytest.raises(CompareConfigError):
        config.cues_path("seed100601\n", "B")


def test_check_run_id_accepts_the_default_and_variant_ids() -> None:
    assert [check_run_id(name) for name in ("seed100601", "seed100601-ja", "seed100601-ja-short")]


def test_load_compare_with_invalid_yaml_raises(tmp_path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("seed: [1\n", encoding="utf-8")

    with pytest.raises(CompareConfigError, match="YAML"):
        load_compare(path)


# --- method A on the shared client factory and retry policy --------------------------------------


class _Responses:
    def __init__(self, failures: list[BaseException]) -> None:
        self.failures = list(failures)
        self.bodies: list[dict] = []

    def create(self, **kwargs: object) -> dict:
        self.bodies.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        text = '{"intent": "i", "evidence": ["e"]}'
        return {"id": "r", "model": "m", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}  # fmt: skip


class _SDK:
    def __init__(self, failures: list[BaseException]) -> None:
        self.responses = _Responses(failures)


def _server_error() -> openai.InternalServerError:
    return openai.InternalServerError("boom", response=httpx.Response(503, request=REQUEST), body=None)


def _model(monkeypatch, failures: list[BaseException]) -> tuple[object, _SDK, run_a.RetryJournal]:
    sdk, journal = _SDK(failures), run_a.RetryJournal()
    monkeypatch.setattr(openai_responses, "make_openai_client", lambda **_: sdk)
    monkeypatch.setattr("mtd_rl_xavier.llm.retry.time.sleep", lambda _: None)
    monkeypatch.setattr(run_a, "log_to_stderr", lambda _: None)
    settings = TranslationSettings("English", 1000, 100, 10.0)
    config = load_compare(DEFAULT_CONFIG)
    return run_a.make_model(config, settings, journal), sdk, journal


def _prompt() -> TranslationPrompt:
    return TranslationPrompt("instructions", "user text", OUTPUT_SCHEMA_NAME, output_schema())


def test_method_a_resends_a_5xx_with_the_same_body_and_journals_it(monkeypatch) -> None:
    model, sdk, journal = _model(monkeypatch, [_server_error()])

    output = model.generate(_prompt())
    retries = journal.drain()

    assert output.response_id == "r"
    assert sdk.responses.bodies == [model.request_payload(_prompt())] * 2
    assert [(event["attempt"], event["status_code"]) for event in retries] == [(1, 503)]
    assert run_a.attempts_of(retries) == 2
    assert journal.drain() == []


def test_method_a_never_resends_a_timeout(monkeypatch) -> None:
    from mtd_rl_xavier.translator.schema import ModelCallError

    model, sdk, journal = _model(monkeypatch, [openai.APITimeoutError(request=REQUEST)])

    with pytest.raises(ModelCallError) as caught:
        model.generate(_prompt())

    assert caught.value.uncertain is True
    assert len(sdk.responses.bodies) == 1
    assert run_a.attempts_of(journal.drain()) == 1


def test_attempts_of_counts_a_give_up_without_an_extra_attempt() -> None:
    gave_up = [{"delay_s": 10.0}, {"delay_s": 30.0}, {"delay_s": None}]

    assert run_a.attempts_of([]) == 1
    assert run_a.attempts_of(gave_up[:1]) == 2
    assert run_a.attempts_of(gave_up) == 3
