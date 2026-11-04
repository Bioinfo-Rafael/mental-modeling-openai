"""Method B answer-length limits: compare-config keys, prompt wording, validator, and the invalid-answer path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from weightreading import cli
from weightreading.config import ConfigError
from weightreading.intent_config import parse_compare
from weightreading.intent_protocol import (
    DEFAULT_LIMITS,
    MAX_INTENT_CHARS,
    AnswerLimits,
    IntentAnswerError,
    parse_answer,
    parse_intent_reply,
)
from weightreading.protocol import KIND_FINAL

from .test_wr_cli_gate import ScriptedClient
from .test_wr_intent_run import PASSED, Factory, _args, _fake_sandbox
from .test_wr_intent_units import _answer_text, _compare_raw, _mappo_message
from .wr_intent_helpers import IntentFiles, write_intent_files

SHORT = AnswerLimits(max_intent_chars=200, max_evidence_items=4, max_evidence_chars=80)
LIMIT_KEYS = ("max_intent_chars", "max_evidence_items", "max_evidence_chars")


def _parse(raw: dict, tmp_path: Path):
    return parse_compare(raw, tmp_path / "c.yaml", "sha", tmp_path)


def _message(limits: AnswerLimits, language: str | None = None) -> str:
    return _mappo_message(None, language=language, limits=limits)


# --- compare config ----------------------------------------------------------------------------


def test_parse_compare_without_limit_keys_keeps_todays_limits(tmp_path: Path) -> None:
    assert (
        _parse(_compare_raw(), tmp_path).answer_limits == DEFAULT_LIMITS == AnswerLimits(MAX_INTENT_CHARS, None, None)
    )


def test_parse_compare_reads_the_three_limit_keys(tmp_path: Path) -> None:
    raw = {**_compare_raw(), "max_intent_chars": 200, "max_evidence_items": 4, "max_evidence_chars": 80}

    assert _parse(raw, tmp_path).answer_limits == SHORT


@pytest.mark.parametrize("key", LIMIT_KEYS)
@pytest.mark.parametrize("value", [0, -1, 1.5, "200", None, True, [200]])
def test_parse_compare_rejects_invalid_limit_values(tmp_path: Path, key: str, value: object) -> None:
    with pytest.raises(ConfigError, match=key):
        _parse({**_compare_raw(), key: value}, tmp_path)


# --- prompt ------------------------------------------------------------------------------------


def test_message_with_default_limits_is_unchanged_and_has_no_limits_sentence() -> None:
    message = _message(DEFAULT_LIMITS)

    assert message == _mappo_message(None)
    assert "Length limits" not in message and f"at most {MAX_INTENT_CHARS} characters>" in message


@pytest.mark.parametrize("language", ["en", "ja"])
def test_message_states_all_limits_when_set(language: str) -> None:
    message = _message(SHORT, language)

    assert "intent at most 200 characters; at most 4 evidence items, each at most 80 characters." in message
    assert "at most 200 characters>" in message and "at most 1000 characters" not in message
    assert message.split("# Final answer")[0] == _mappo_message(None).split("# Final answer")[0]


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (AnswerLimits(max_evidence_items=3), "intent at most 1000 characters; at most 3 evidence items."),
        (
            AnswerLimits(max_evidence_chars=60),
            "intent at most 1000 characters; each evidence item at most 60 characters.",
        ),
        (AnswerLimits(max_intent_chars=300), "intent at most 300 characters."),
    ],
)
def test_message_states_only_the_limits_that_are_set(limits: AnswerLimits, expected: str) -> None:
    assert expected in _message(limits)


# --- validator ---------------------------------------------------------------------------------


def _answer(intent: str = "Pursue.", evidence: list[str] | None = None) -> str:
    return _answer_text({"intent": intent, "evidence": evidence or ["a"]})


def test_parse_answer_with_default_limits_ignores_evidence_size() -> None:
    answer = parse_answer(_answer(evidence=["e" * 5000] * 20))

    assert len(answer.evidence) == 20


def test_parse_answer_accepts_answers_exactly_at_the_limits() -> None:
    text = _answer("a" * 200, ["b" * 80] * 4)

    assert parse_answer(text, SHORT).intent == "a" * 200


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_answer("a" * 201), "'\"intent\" has 201 characters; the limit is 200'"),
        (_answer(evidence=["x"] * 5), "5 items; the limit is 4"),
        (_answer(evidence=["x", "y" * 81]), "81 characters; the limit is 80"),
    ],
)
def test_parse_answer_rejects_answers_over_the_limits(text: str, message: str) -> None:
    assert parse_intent_reply(text).kind == KIND_FINAL  # a json attempt still ends the session
    with pytest.raises(IntentAnswerError, match=message.strip("'")):
        parse_answer(text, SHORT)


def test_parse_answer_counts_characters_not_bytes_for_the_limits() -> None:
    at_limit = _answer("あ" * 200, ["根" * 80] * 4)  # 600 / 240 UTF-8 bytes
    assert parse_answer(at_limit, SHORT).evidence[0] == "根" * 80
    with pytest.raises(IntentAnswerError):
        parse_answer(_answer("あ" * 201, ["根"]), SHORT)
    with pytest.raises(IntentAnswerError):
        parse_answer(_answer("あ", ["根" * 81]), SHORT)


# --- end to end --------------------------------------------------------------------------------


def _short_files(small_files, tmp_path: Path) -> IntentFiles:
    checkpoint, spec = small_files
    files = write_intent_files(tmp_path, checkpoint, spec)
    raw = yaml.safe_load(files.compare.read_text(encoding="utf-8"))
    tight = {"max_intent_chars": 20, "max_evidence_items": 2, "max_evidence_chars": 10}
    files.compare.write_text(yaml.safe_dump({**raw, **tight}), encoding="utf-8")
    return files


def test_over_length_answer_is_recorded_invalid_like_other_schema_violations(
    small_files, tmp_path, monkeypatch
) -> None:
    files = _short_files(small_files, tmp_path)
    _fake_sandbox(monkeypatch, PASSED)
    ok = '```json\n{"intent": "Hold course.", "evidence": ["logit gap"]}\n```'
    too_long = '```json\n{"intent": "' + "x" * 21 + '", "evidence": ["logit gap"]}\n```'
    factory = Factory(ScriptedClient([ok, too_long, ok]))

    assert cli.main(_args(files, "intent-run", "--run-id", "r4", "--execute", "--confirm-paid-api"), factory) == 0

    cues = [json.loads(line) for line in (files.cues_root / "r4" / "cues_B.jsonl").read_text().splitlines()]
    assert cues[0]["intent"] == "Hold course."
    assert "invalid final answer" in cues[1]["skipped"] and "the limit is 20" in cues[1]["skipped"]
    run_dir = files.output_root / "intent" / "runs" / "r4"
    message = (run_dir / "sessions" / "s0003_A0100" / "message.txt").read_text()
    assert "intent at most 20 characters; at most 2 evidence items, each at most 10 characters." in message
    assert json.loads((run_dir / "run.json").read_text())["answer_limits"] == {
        "max_intent_chars": 20, "max_evidence_items": 2, "max_evidence_chars": 10,
    }  # fmt: skip


def test_preview_with_limits_reserves_only_limit_sized_previous_answer(small_files, tmp_path, capsys) -> None:
    files = _short_files(small_files, tmp_path)

    assert cli.main(_args(files, "intent-preview"), client_factory=Factory()) == 0
    preview_dir = Path(json.loads(capsys.readouterr().out)["preview_dir"])

    messages = sorted((preview_dir / "sessions").glob("*/message.txt"))
    chained = [m.read_text() for m in messages if "From step" in m.read_text()]
    assert chained and all("intent at most 20 characters" in text for text in chained)
    assert all(len(text.split("evidence:")[1].split("Keep what")[0].strip().splitlines()) == 2 for text in chained)
