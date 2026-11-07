"""Frozen request bytes of the Translator and method A (restructure guard).

Each test rebuilds one serialized Responses request body from committed fixtures and compares its
SHA-256 with ``tests/fixtures/prompt-request-goldens.json``. A mismatch means a prompt or request
byte changed: fix the change, do not update the golden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mtd_rl_xavier.intent_compare import DEFAULT_CONFIG, load_compare
from mtd_rl_xavier.intent_compare.method_a_requests import Slot, build_request, load_settings, prompt_for, request_body
from mtd_rl_xavier.lag.dataset import load_episode
from mtd_rl_xavier.llm.openai_responses import build_responses_body
from mtd_rl_xavier.translator.cli import _mandatory_requests, load_inputs
from mtd_rl_xavier.translator.config import load_config
from mtd_rl_xavier.translator.schema import canonical_json, sha256_text
from mtd_rl_xavier.translator.translation import build_openai_request, build_prompt
from tests.translator.tr_fakes import EPISODE_CSV, EXAMPLE_SCENARIO, RUN_CONFIG, WIN01

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDENS = FIXTURES / "prompt-request-goldens.json"
CONVERTER = Path(__file__).resolve().parents[2] / "src" / "mtd_rl_xavier" / "intent_compare" / "to_translator_run.py"
METHOD_A_MAX_OUTPUT_TOKENS = 25_000  # the method A CLI default
METHOD_A_SLOT = Slot(step=45, aircraft="A0100", skipped=None)
JA_SHORT_CONFIG = DEFAULT_CONFIG.with_name("compare.ja-short.yaml")


def goldens() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text(encoding="utf-8"))


def body_sha256(body: dict[str, Any]) -> str:
    return sha256_text(canonical_json(body))


def translator_request_sha256() -> str:
    """The example scenario's first mandatory translation, exactly as the CLI preview writes it."""
    config = load_config(RUN_CONFIG)
    inputs = load_inputs(RUN_CONFIG, EXAMPLE_SCENARIO, config)
    translation = config.translation
    prompt = build_prompt(
        _mandatory_requests(inputs)[0],
        output_language=translation.output_language,
        max_intent_chars=translation.max_intent_chars,
    )
    request = build_openai_request(
        prompt,
        model=translation.model,
        reasoning_effort=translation.reasoning_effort,
        max_output_tokens=translation.max_output_tokens,
    )
    return body_sha256(build_responses_body(request))


def method_a_request_sha256(config_path: Path) -> str:
    """One fixed method A slot over the committed MHI episode (no previous translation)."""
    config = load_compare(config_path)
    frame = load_episode(EPISODE_CSV[WIN01])
    settings = load_settings(config, METHOD_A_MAX_OUTPUT_TOKENS)
    request = build_request(config, frame, METHOD_A_SLOT, previous=None)
    return body_sha256(request_body(prompt_for(request, settings), config, settings))


def test_translator_translation_request_bytes_are_unchanged() -> None:
    assert translator_request_sha256() == goldens()["translator_translation_request_sha256"]


def test_method_a_fixed_slot_request_bytes_are_unchanged() -> None:
    expected = goldens()["method_a_fixed_slot_request_sha256"]

    assert method_a_request_sha256(DEFAULT_CONFIG) == expected["compare.yaml"]
    assert method_a_request_sha256(JA_SHORT_CONFIG) == expected["compare.ja-short.yaml"]


def test_rescued_converter_is_byte_identical() -> None:
    assert hashlib.sha256(CONVERTER.read_bytes()).hexdigest() == goldens()["converter_sha256"]
