"""Answer limits and skip-on-invalid behavior in the unified pipeline."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from translate.answers import AnswerLimits, payload_problem
from translate.pipeline import STATUS_COMPLETED

from .tr_fakes import FakeIntentModel, MemorySink, instruction, make_config, make_scenario, make_translator, request

AIRCRAFT = ("aircraft", "A0100")
LIMITS = AnswerLimits(max_intent_chars=200, max_evidence_items=4, max_evidence_chars=80)


def _answer(intent: str, evidence: list[str] | None = None) -> str:
    return json.dumps({"intent": intent, "evidence": evidence or ["e1"]}, ensure_ascii=False)


def test_payload_problem_at_and_above_character_and_item_limits_returns_exact_problem() -> None:
    assert payload_problem({"intent": "あ" * 200, "evidence": ["根" * 80] * 4}, LIMITS) is None
    assert payload_problem({"intent": "あ" * 201, "evidence": ["e"]}, LIMITS) == (
        '"intent" has 201 characters; the limit is 200'
    )
    assert payload_problem({"intent": "x", "evidence": ["e"] * 5}, LIMITS) == ('"evidence" has 5 items; the limit is 4')
    assert payload_problem({"intent": "x", "evidence": ["根" * 81]}, LIMITS) == (
        'an "evidence" item has 81 characters; the limit is 80'
    )


@pytest.mark.parametrize(
    ("invalid", "reason"),
    [
        (_answer("あ" * 201), "intent has 201 characters; the limit is 200"),
        (_answer("x", ["e"] * 5), '"evidence" has 5 items; the limit is 4'),
        (_answer("x", ["根" * 81]), 'an "evidence" item has 81 characters; the limit is 80'),
    ],
)
def test_pipeline_on_invalid_skip_records_blank_event_and_chains_last_valid_answer(
    short_frame: pd.DataFrame, invalid: str, reason: str
) -> None:
    first = _answer("VALID-INITIAL", ["first evidence"])
    last = _answer("VALID-LAST", ["last evidence"])
    intent = FakeIntentModel([first, invalid, last])
    config = make_config(
        answer__max_intent_chars=200,
        answer__max_evidence_items=4,
        answer__max_evidence_chars=80,
        answer__on_invalid="skip",
    )
    scenario = make_scenario(
        [instruction("a1", 0, *AIRCRAFT)],
        [request("r1", 7, AIRCRAFT), request("r2", 9, AIRCRAFT)],
    )
    sink = MemorySink()

    summary = make_translator(intent=intent, config=config).run(
        frame=short_frame,
        scenario=scenario,
        config=config,
        sink=sink,
    )

    events = sink.of_type("translation_event")
    assert summary.status == STATUS_COMPLETED and [event["step"] for event in events] == [0, 7, 9]
    assert events[1]["output"] == {"intent": "", "evidence": []}
    assert events[1]["answer_problem"].startswith("invalid final answer: ")
    assert reason in events[1]["answer_problem"]
    assert events[2]["input"]["previous_translation"]["intent"] == "VALID-INITIAL"
    assert "VALID-INITIAL" in intent.prompts[2].user_text and invalid not in intent.prompts[2].user_text
    translation_responses = [item for item in sink.responses if item["layer"] == "translation"]
    assert [item["raw"]["text"] for item in translation_responses] == [first, invalid, last]
