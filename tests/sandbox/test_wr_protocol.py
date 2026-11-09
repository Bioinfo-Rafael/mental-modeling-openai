"""Reply protocol for policy code sessions: block merging and intent answers."""

from __future__ import annotations

import pytest

from translate.answers import AnswerLimits
from translate.sandbox.protocol import (
    KIND_CODE,
    KIND_ERROR,
    KIND_FINAL,
    IntentAnswer,
    IntentAnswerError,
    answer_problems,
    parse_answer,
    parse_reply,
)

WELL_FORMED = 'analysis\n```json\n{"intent": "Pursue.", "evidence": ["A[step] changed"]}\n```'


def test_parse_reply_two_blocks_merges_in_order_with_separator() -> None:
    parsed = parse_reply("```python\na = 1\n```\n```python\nb = 2\n```\n```python\nc = 3\n```")

    assert parsed.kind == KIND_CODE
    assert parsed.block_count == 3
    assert parsed.code == "a = 1\n\n# --- block 2 ---\nb = 2\n\n# --- block 3 ---\nc = 3\n"


def test_parse_answer_well_formed_returns_intent_answer() -> None:
    assert answer_problems(WELL_FORMED) == ()
    assert parse_answer(WELL_FORMED) == IntentAnswer("Pursue.", ("A[step] changed",))
    assert parse_reply(WELL_FORMED).kind == KIND_FINAL


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "found 0"),
        (
            '```json\n{"intent": "x", "evidence": ["e"]}\n```\n```json\n{"intent": "x", "evidence": ["e"]}\n```',
            "found 2",
        ),
        ("```json\n{not json}\n```", "not valid JSON"),
        ('```json\n{"intent": "x", "evidence": []}\n```', "non-empty list"),
        ('```json\n{"intent": "x", "intent": "y", "evidence": ["e"]}\n```', "repeats a key"),
    ],
)
def test_parse_answer_invalid_payload_reports_exact_problem(text: str, expected: str) -> None:
    problems = answer_problems(text)

    assert len(problems) == 1
    assert expected in problems[0]
    with pytest.raises(IntentAnswerError, match=expected):
        parse_answer(text)


def test_parse_answer_limits_count_characters_and_evidence() -> None:
    limits = AnswerLimits(max_intent_chars=2, max_evidence_items=1, max_evidence_chars=2)
    valid = '```json\n{"intent": "あい", "evidence": ["根拠"]}\n```'

    assert parse_answer(valid, limits).intent == "あい"
    assert "3 characters" in answer_problems(valid.replace("あい", "あいう"), limits)[0]
    assert "2 items" in answer_problems(valid.replace('["根拠"]', '["根拠", "e"]'), limits)[0]
    assert "3 characters" in answer_problems(valid.replace("根拠", "根拠長"), limits)[0]


def test_parse_reply_error_carries_problems() -> None:
    parsed = parse_reply("I'm unable to analyze the network.")

    assert parsed.kind == KIND_ERROR
    assert parsed.problems == ('exactly one ```json block {"intent": ..., "evidence": [...]} (found 0)',)
