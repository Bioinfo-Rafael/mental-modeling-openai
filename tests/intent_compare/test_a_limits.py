"""Method A answer-length limits: the shared prompt sentence and the skip-and-continue handling."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
from intent_compare_a_requests import (
    INSTRUCTIONS_JOINER,
    TranslationSettings,
    answer_problem,
    build_request,
    load_frame,
    load_settings,
    plan_slots,
    prompt_for,
    provenance,
    sha256_text,
)
from intent_compare_run_a import EXIT_OK, RunPaths, execute

from mtd_rl_xavier.intent_compare import DEFAULT_CONFIG, DEFAULT_LIMITS, AnswerLimits, CompareConfig, limits_sentence
from mtd_rl_xavier.intent_compare import load_compare as load_config
from mtd_rl_xavier.translator.translation import (
    IntentModelOutput,
    TranslationPrompt,
    build_prompt,
)
from tests.weightreading.test_wr_intent_units import _mappo_message

MAX_OUTPUT_TOKENS = 25_000
JA_CONFIG = DEFAULT_CONFIG.with_name("compare.ja.yaml")
JA_SHORT_CONFIG = DEFAULT_CONFIG.with_name("compare.ja-short.yaml")
SHORT = AnswerLimits(max_intent_chars=200, max_evidence_items=4, max_evidence_chars=80)
SHORT_SENTENCE = (
    "Length limits (counted in characters, not bytes; a longer answer is rejected): "
    "intent at most 200 characters; at most 4 evidence items, each at most 80 characters."
)
BAD_STEP = 145
TARGET = "A0200"


def _frame_and_slots(config: CompareConfig):
    if not config.episode_csv.exists():
        pytest.skip("episode CSV (generated output) is not available")
    frame, facts = load_frame(config)
    return frame, plan_slots(config, facts)


# --- prompt ------------------------------------------------------------------------------------


def test_limits_sentence_is_empty_for_default_limits_and_matches_the_documented_wording() -> None:
    assert limits_sentence(DEFAULT_LIMITS) == ""
    assert limits_sentence(SHORT) == SHORT_SENTENCE


@pytest.mark.parametrize("config_path", [DEFAULT_CONFIG, JA_CONFIG])
def test_prompt_for_without_limits_is_byte_identical_to_the_translator_prompt(config_path: Path) -> None:
    config = load_config(config_path)
    frame, slots = _frame_and_slots(config)
    request = build_request(config, frame, slots[0], previous=None)
    settings = load_settings(config, MAX_OUTPUT_TOKENS)

    expected = build_prompt(request, output_language=settings.output_language, max_intent_chars=1000)

    assert settings.limits_sentence == ""
    assert prompt_for(request, settings) == expected


def test_prompt_for_with_limits_appends_exactly_the_shared_sentence() -> None:
    config = load_config(JA_SHORT_CONFIG)
    frame, slots = _frame_and_slots(config)
    request = build_request(config, frame, slots[0], previous=None)
    settings = load_settings(config, MAX_OUTPUT_TOKENS)

    prompt = prompt_for(request, settings)
    base = build_prompt(request, output_language="Japanese", max_intent_chars=200)

    assert prompt.instructions == f"{base.instructions}{INSTRUCTIONS_JOINER}{SHORT_SENTENCE}"
    assert prompt.user_text == base.user_text and prompt.schema == base.schema


@pytest.mark.parametrize("language", ["en", "ja"])
def test_a_and_b_state_the_same_length_sentence(language: str) -> None:
    config = dataclasses.replace(load_config(JA_SHORT_CONFIG), output_language=language)
    frame, slots = _frame_and_slots(config)
    request = build_request(config, frame, slots[0], previous=None)
    a_instructions = prompt_for(request, load_settings(config, MAX_OUTPUT_TOKENS)).instructions

    b_message = _mappo_message(None, language=language, limits=SHORT)

    assert SHORT_SENTENCE in a_instructions and SHORT_SENTENCE in b_message
    assert a_instructions.endswith(SHORT_SENTENCE)


def test_provenance_records_the_limits_and_the_sentence() -> None:
    config = load_config(JA_SHORT_CONFIG)
    frame, _ = _frame_and_slots(config)
    settings = load_settings(config, MAX_OUTPUT_TOKENS)

    record = provenance(config, frame, settings)

    assert record["answer_limits"] == dataclasses.asdict(SHORT)
    assert record["limits_sentence"] == SHORT_SENTENCE
    assert record["limits_sentence_sha256"] == sha256_text(SHORT_SENTENCE)


# --- validation (method B's rules) -------------------------------------------------------------


def _settings(limits: AnswerLimits) -> TranslationSettings:
    return TranslationSettings(
        "Japanese",
        limits.max_intent_chars,
        MAX_OUTPUT_TOKENS,
        60.0,
        limits.max_evidence_items,
        limits.max_evidence_chars,
    )


def test_answer_problem_uses_method_b_rules_and_counts_characters() -> None:
    settings = _settings(SHORT)

    assert answer_problem("あ" * 200, ("根" * 80,) * 4, settings) is None
    assert answer_problem("x", ("a",) * 5, settings) == '"evidence" has 5 items; the limit is 4'
    assert answer_problem("x", ("a", "b" * 81), settings) == 'an "evidence" item has 81 characters; the limit is 80'
    assert answer_problem("x" * 201, ("a",), settings) == '"intent" has 201 characters; the limit is 200'


def test_answer_problem_without_limits_ignores_evidence_size() -> None:
    assert answer_problem("x", ("y" * 5000,) * 50, _settings(DEFAULT_LIMITS)) is None


# --- execute: skip and continue ----------------------------------------------------------------


class ScriptedModel:
    """Fake IntentModel: one bad answer for ``(BAD_STEP, TARGET)``, a valid one otherwise; records every prompt."""

    def __init__(self, bad_text: str) -> None:
        self.bad_text = bad_text
        self.prompts: list[tuple[int, str, TranslationPrompt]] = []

    def request_payload(self, prompt: TranslationPrompt) -> dict:
        return {"instructions": prompt.instructions, "input": prompt.user_text}

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        step = int(re.search(r"Current step: (\d+)", prompt.user_text).group(1))
        aircraft = re.search(r"Translation target: aircraft (\w+)", prompt.user_text).group(1)
        self.prompts.append((step, aircraft, prompt))
        text = self.bad_text if (step, aircraft) == (BAD_STEP, TARGET) else _valid(f"VALID-{step}-{aircraft}")
        return IntentModelOutput("fake", f"resp-{len(self.prompts)}", text, {}, {"text": text})


def _valid(intent: str) -> str:
    return json.dumps({"intent": intent, "evidence": ["e1", "e2"]}, ensure_ascii=False)


def _cues(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path: Path, bad_text: str):
    config = load_config(JA_SHORT_CONFIG)
    frame, slots = _frame_and_slots(config)
    model = ScriptedModel(bad_text)
    paths = RunPaths(tmp_path / "run", tmp_path / "run" / "method_a", tmp_path / "run" / "cues_A.jsonl")
    code = execute(config, frame, slots, paths, load_settings(config, MAX_OUTPUT_TOKENS), model)
    return code, paths, slots, model


BAD_ANSWERS = {
    "intent-over-length": (
        _valid("あ" * 201),
        "intent has 201 characters; the limit is 200",
    ),  # Translator parser (TranslationOutputError)
    "too-many-evidence": (json.dumps({"intent": "x", "evidence": ["a"] * 5}), "has 5 items; the limit is 4"),
    "evidence-too-long": (json.dumps({"intent": "x", "evidence": ["b" * 81]}), "has 81 characters; the limit is 80"),
}


@pytest.mark.parametrize("case", sorted(BAD_ANSWERS))
def test_execute_skips_an_invalid_answer_keeps_the_raw_and_continues(tmp_path: Path, case: str) -> None:
    bad_text, reason_part = BAD_ANSWERS[case]

    code, paths, slots, model = _run(tmp_path, bad_text)

    cues = _cues(paths.cues)
    assert code == EXIT_OK and len(cues) == len(slots) and len(model.prompts) == len(slots)
    rejected = [cue for cue in cues if "skipped" in cue]
    assert [(c["step"], c["aircraft"]) for c in rejected] == [(BAD_STEP, TARGET)]
    assert rejected[0]["skipped"].startswith("invalid final answer: ") and reason_part in rejected[0]["skipped"]
    assert "intent" not in rejected[0] and "evidence" not in rejected[0]
    run = json.loads((paths.method_dir / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "completed" and run["rejected_answers"] == [
        {"call_id": f"step{BAD_STEP:03d}_{TARGET}", "reason": rejected[0]["skipped"]}
    ]
    assert run["provenance"]["limits_sentence"] == SHORT_SENTENCE
    journal = [json.loads(line) for line in (paths.method_dir / "responses.jsonl").read_text("utf-8").splitlines()]
    assert len(journal) == len(slots)
    assert [r["raw"]["text"] for r in journal if r["call_id"] == f"step{BAD_STEP:03d}_{TARGET}"] == [bad_text]


def test_execute_chains_the_next_call_from_the_last_valid_answer(tmp_path: Path) -> None:
    code, _, _, model = _run(tmp_path, _valid("BAD-" + "x" * 300))

    next_prompt = next(p for step, aircraft, p in model.prompts if (step, aircraft) == (245, TARGET))
    other_aircraft = next(p for step, aircraft, p in model.prompts if (step, aircraft) == (245, "A0100"))
    assert code == EXIT_OK
    assert "VALID-45-A0200" in next_prompt.user_text and "BAD-" not in next_prompt.user_text
    assert "VALID-145-A0100" in other_aircraft.user_text  # the other aircraft chains its own answers
    assert SHORT_SENTENCE in next_prompt.instructions
