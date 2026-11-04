"""Offline tests of method A slot planning (needs the Translator package)."""

from __future__ import annotations

import dataclasses

import pytest
import yaml
from intent_compare_a_requests import (
    build_request,
    load_frame,
    load_settings,
    plan_slots,
    prompt_for,
)
from intent_compare_config import DEFAULT_CONFIG, CompareConfigError, load_config, parse_config

from mtd_rl_xavier.translator.pipeline import EpisodeFacts
from mtd_rl_xavier.translator.translation import system_prompt

DEFAULT_MAX_OUTPUT_TOKENS = 25_000


def test_plan_slots_skips_target_down_at_step() -> None:
    config = load_config(DEFAULT_CONFIG)
    facts = EpisodeFacts("e", 354, {"A": ("A0100", "A0200")}, {"A0100": 145, "A0200": None})

    slots = plan_slots(config, facts)

    skipped = [(s.step, s.aircraft) for s in slots if s.skipped]
    assert skipped == [(145, "A0100"), (245, "A0100"), (315, "A0100")]
    assert len(slots) == 8


@pytest.mark.parametrize(("language", "name"), [("en", "English"), ("ja", "Japanese")])
def test_load_settings_language_comes_from_compare_config_not_translator_run_yaml(language: str, name: str) -> None:
    config = dataclasses.replace(load_config(DEFAULT_CONFIG), output_language=language)

    settings = load_settings(config, DEFAULT_MAX_OUTPUT_TOKENS)
    prompt = system_prompt(output_language=settings.output_language, max_intent_chars=settings.max_intent_chars)

    assert settings.output_language == name
    assert prompt.endswith(f"Write both in {name}.")


JA_CONFIG = DEFAULT_CONFIG.with_name("compare.ja.yaml")
JA_SHORT_CONFIG = DEFAULT_CONFIG.with_name("compare.ja-short.yaml")


def test_load_settings_without_limit_keys_uses_the_compare_default_cap() -> None:
    # The cap comes from the compare config (shared with method B), not from translator run.yaml.
    settings = load_settings(load_config(DEFAULT_CONFIG), DEFAULT_MAX_OUTPUT_TOKENS)

    assert settings.max_intent_chars == 1000
    assert settings.max_evidence_items is None and settings.max_evidence_chars is None


def test_load_settings_takes_the_limits_from_the_compare_config_not_translator_run_yaml() -> None:
    config = load_config(JA_SHORT_CONFIG)

    settings = load_settings(config, DEFAULT_MAX_OUTPUT_TOKENS)

    assert (settings.max_intent_chars, settings.max_evidence_items, settings.max_evidence_chars) == (200, 4, 80)
    assert "at most 200 characters" in system_prompt(output_language="Japanese", max_intent_chars=200)


def test_ja_short_config_is_the_ja_config_plus_exactly_the_three_limits() -> None:
    japanese = yaml.safe_load(JA_CONFIG.read_text(encoding="utf-8"))
    short = yaml.safe_load(JA_SHORT_CONFIG.read_text(encoding="utf-8"))

    assert short == {**japanese, "max_intent_chars": 200, "max_evidence_items": 4, "max_evidence_chars": 80}


@pytest.mark.parametrize("key", ["max_intent_chars", "max_evidence_items", "max_evidence_chars"])
@pytest.mark.parametrize("value", [0, -3, 1.5, "80", None, True])
def test_parse_config_rejects_invalid_limit_values(key: str, value: object) -> None:
    raw = {**yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")), key: value}

    with pytest.raises(CompareConfigError, match=key):
        parse_config(raw, DEFAULT_CONFIG)


def test_translator_prompt_has_no_evidence_limit_slot() -> None:
    """The Translator prompt itself has no evidence-limit slot; ``prompt_for`` appends B's sentence instead."""
    with_limits = system_prompt(output_language="Japanese", max_intent_chars=200)

    assert "evidence items" not in with_limits


def test_prompt_for_passes_max_intent_chars_from_the_compare_config() -> None:
    short, default = load_config(JA_SHORT_CONFIG), load_config(JA_CONFIG)
    if not short.episode_csv.exists():
        pytest.skip("episode CSV (generated output) is not available")
    frame, facts = load_frame(short)
    slot = plan_slots(short, facts)[0]
    request = build_request(short, frame, slot, previous=None)

    settings = load_settings(short, DEFAULT_MAX_OUTPUT_TOKENS)
    short_prompt = prompt_for(request, settings)
    default_prompt = prompt_for(request, load_settings(default, DEFAULT_MAX_OUTPUT_TOKENS))

    assert "in at most 200 characters" in short_prompt.instructions
    assert "in at most 1000 characters" in default_prompt.instructions
    assert short_prompt.user_text == default_prompt.user_text
    translator_part = short_prompt.instructions.removesuffix(settings.limits_sentence).removesuffix("\n\n")
    assert translator_part.replace("at most 200 characters", "at most 1000 characters") == default_prompt.instructions
