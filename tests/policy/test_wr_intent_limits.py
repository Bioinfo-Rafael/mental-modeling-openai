"""Character-based answer limits shared by one-shot and code-session translations."""

from __future__ import annotations

import json

import pytest

from translate.answers import DEFAULT_LIMITS, MAX_INTENT_CHARS, AnswerLimits
from translate.call_one_shot import CallResult
from translate.call_session import SessionCall
from translate.sandbox.protocol import IntentAnswerError, parse_answer

SHORT = AnswerLimits(max_intent_chars=200, max_evidence_items=4, max_evidence_chars=80)


def _answer(intent: str = "Pursue.", evidence: list[str] | None = None) -> str:
    return "```json\n" + json.dumps({"intent": intent, "evidence": evidence or ["e"]}) + "\n```"


def test_parse_answer_with_default_limits_ignores_evidence_size() -> None:
    answer = parse_answer(_answer(evidence=["e" * 10_000] * 100), DEFAULT_LIMITS)

    assert answer.intent == "Pursue."
    assert len(answer.evidence) == 100 and len(answer.evidence[0]) == 10_000


def test_parse_answer_accepts_answers_exactly_at_the_limits() -> None:
    answer = parse_answer(_answer("x" * 200, ["e" * 80] * 4), SHORT)

    assert len(answer.intent) == 200
    assert len(answer.evidence) == 4
    assert all(len(item) == 80 for item in answer.evidence)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (_answer("x" * 201), '"intent" has 201 characters; the limit is 200'),
        (_answer(evidence=["e"] * 5), '"evidence" has 5 items; the limit is 4'),
        (_answer(evidence=["e" * 81]), 'an "evidence" item has 81 characters; the limit is 80'),
    ],
)
def test_parse_answer_rejects_answers_over_the_limits(text: str, message: str) -> None:
    with pytest.raises(IntentAnswerError, match=message):
        parse_answer(text, SHORT)


def test_parse_answer_counts_characters_not_bytes_for_the_limits() -> None:
    assert parse_answer(_answer("あ" * 200, ["根" * 80] * 4), SHORT).intent == "あ" * 200

    with pytest.raises(IntentAnswerError, match='"intent" has 201 characters'):
        parse_answer(_answer("あ" * 201), SHORT)

    assert MAX_INTENT_CHARS == DEFAULT_LIMITS.max_intent_chars


def test_over_length_answer_is_recorded_invalid_like_other_schema_violations() -> None:
    call = object.__new__(SessionCall)
    call._limits = SHORT
    result = CallResult({}, _answer("x" * 201), None, None, {})

    translation, problem = call.answer(result)

    assert translation is None
    assert problem == 'invalid final answer: "intent" has 201 characters; the limit is 200'
