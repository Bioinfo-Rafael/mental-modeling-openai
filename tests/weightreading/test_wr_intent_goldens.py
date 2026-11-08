"""Frozen message bytes of the policy-input prompt ``intent-b-v1`` (restructure guard).

Each test rebuilds a message (or a request body) over the committed synthetic network and compares
its SHA-256 with ``tests/fixtures/intent-b-v1-goldens.json``. A mismatch means a prompt or request
byte changed: fix the change, do not update the golden.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from mtd_rl_xavier.intent_compare import AnswerLimits
from translate.events import canonical_json, sha256_text
from weightreading.backends import OpenAIResponsesBackend, user_item
from weightreading.intent_prompt import PreviousTranslation
from weightreading.intent_protocol import IntentAnswer
from weightreading.intent_run import message_for, prepare
from weightreading.prompt import budget_exhausted_message, protocol_error_message, run_result_message

from .wr_intent_helpers import write_intent_files

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures" / "intent-b-v1-goldens.json"
PREVIOUS = PreviousTranslation(3, IntentAnswer("[PLACEHOLDER] climb and turn left", ("e1: step 3", "e2: logits")))
SHORT_LIMITS = AnswerLimits(max_intent_chars=200, max_evidence_items=4, max_evidence_chars=80)


def goldens() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text(encoding="utf-8"))


def messages(tmp_path: Path, small_files: tuple[Path, Path]) -> dict[str, str]:
    """First and chained messages in English, Japanese and Japanese with short answer limits."""
    files = write_intent_files(tmp_path, *small_files)
    prepared = prepare(files.compare, files.config, files.spec)
    first, chained = (session for session in prepared.sessions if session.aircraft == "A0100")
    variants = {
        "en": prepared.compare,
        "ja": replace(prepared.compare, output_language="ja"),
        "ja-short": replace(prepared.compare, output_language="ja", answer_limits=SHORT_LIMITS),
    }
    found = {}
    for name, compare in variants.items():
        variant = replace(prepared, compare=compare)
        found[f"{name}/first"] = message_for(variant, first, None)
        found[f"{name}/chained"] = message_for(variant, chained, PREVIOUS)
    return found


def test_intent_b_v1_messages_are_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    found = {name: sha256_text(text) for name, text in messages(tmp_path, small_files).items()}

    assert found == goldens()["messages_sha256"]


def test_intent_b_v1_first_request_body_is_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    message = messages(tmp_path, small_files)["en/first"]
    backend = OpenAIResponsesBackend(client=None, model="fake-model", reasoning_effort="high", max_output_tokens=25000)
    body = {"input": [user_item(message)], **backend.options}

    assert sha256_text(canonical_json(body)) == goldens()["first_request_body_sha256"]


def test_intent_b_v1_follow_up_turns_are_unchanged() -> None:
    turns = {
        "run_result": run_result_message(2, 12, "out", "", 0, False, False),
        "run_result_all_notes": run_result_message(12, 12, "out", "err tail", None, True, True, 3),
        "protocol_error": protocol_error_message(("missing a block",), 5),
        "protocol_error_no_runs": protocol_error_message(("a", "b"), 0),
        "budget_exhausted": budget_exhausted_message(),
    }

    assert {name: sha256_text(text) for name, text in turns.items()} == goldens()["turns_sha256"]
