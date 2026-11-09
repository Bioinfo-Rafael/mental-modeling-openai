"""Frozen request bytes of the shipped prompt configurations (restructure guard).

Each test renders one request through the config-driven pipeline (inputs -> prompt file -> request
body) and compares its SHA-256 with a golden computed by the code before the restructure:
``tests/fixtures/prompt-request-goldens.json`` (the online Translator and the trajectory-input
configurations) and ``tests/fixtures/intent-b-v1-goldens.json`` (the policy-input configuration).
A mismatch means a prompt or request byte changed: fix the change, never update the golden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from translate import paths
from translate.cli import _request_body, load_inputs, mandatory_bodies
from translate.config import load_config
from translate.events import (
    TARGET_AIRCRAFT,
    IntentTranslation,
    PreviousTranslation,
    TargetRef,
    canonical_json,
    sha256_text,
)
from translate.inputs import RunInputs, SlotState, slot_values, trajectory_window
from translate.lag.dataset import load_episode
from translate.lag.rendering import render_lag17_jsonl
from translate.policy.material import load_policy_material
from translate.prompts import ConfiguredTurns
from translate.scenario import episode_facts, instructions_in_effect, load_scenario

from ..policy.policy_fakes import SEED, STEPS, policy_config, policy_inputs, write_rollout
from .tr_fakes import EPISODE_CSV, WIN01

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRANSLATE = paths.CONFIGS / "translate"
CONVERTER = FIXTURES / "frozen" / "to_translator_run_v2.py.txt"
SLOT_STEP, SLOT_AIRCRAFT = 45, "A0100"  # the golden trajectory-input slot (no previous translation)
PREVIOUS = PreviousTranslation(
    "previous", 3, IntentTranslation("[PLACEHOLDER] climb and turn left", ("e1: step 3", "e2: logits"))
)


def goldens(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def body_sha256(body: dict[str, Any]) -> str:
    return sha256_text(canonical_json(body))


def test_online_translator_first_request_is_unchanged() -> None:
    config = load_config(TRANSLATE / "online.example.yaml")

    body = mandatory_bodies(load_inputs(config, None))[0]

    assert body_sha256(body) == goldens("prompt-request-goldens.json")["translator_translation_request_sha256"]


def trajectory_slot_sha256(config_name: str) -> str:
    """The golden slot over the committed MHI episode (as the former fixed-slot golden built it)."""
    config = load_config(TRANSLATE / config_name)
    scenario = load_scenario(config.scenario_path, limits=config.scenario_limits)
    frame = load_episode(EPISODE_CSV[WIN01])
    inputs = RunInputs(frame, scenario, episode_facts(frame), config, render_lag17_jsonl)
    target = TargetRef(TARGET_AIRCRAFT, SLOT_AIRCRAFT)
    instruction = next(i for i in instructions_in_effect(scenario, SLOT_STEP) if i.target == target)
    slot = SlotState(SLOT_STEP, target, instruction, None)
    return body_sha256(_request_body(config, slot_values(inputs, slot, trajectory_window(inputs, SLOT_STEP))))


def test_trajectory_input_requests_are_unchanged() -> None:
    expected = goldens("prompt-request-goldens.json")["method_a_fixed_slot_request_sha256"]

    assert trajectory_slot_sha256("seed100601-trajectory.yaml") == expected["compare.yaml"]
    assert trajectory_slot_sha256("seed100601-trajectory-ja-short.yaml") == expected["compare.ja-short.yaml"]


# --- policy input (code session) ------------------------------------------------------------------


def policy_values(tmp_path: Path, small_files: tuple[Path, Path]) -> dict[str, dict[str, object]]:
    """Prompt values of the first and the chained session in English, Japanese and Japanese-short."""
    checkpoint, spec = small_files
    write_rollout(tmp_path / "rollout", checkpoint, spec)
    found = {}
    for name, config_name in (("en", "seed100601-policy.yaml"), ("ja", "seed100601-policy-ja.yaml"),
                              ("ja-short", "seed100601-policy-ja-short.yaml")):  # fmt: skip
        config = policy_config(tmp_path / "rollout", spec, TRANSLATE / config_name)
        inputs = policy_inputs(config, load_policy_material(config.policy.rollout_dir, SEED, spec, None))
        instruction = inputs.scenario.instructions[0]
        for label, step, previous in (("first", STEPS[0], None), ("chained", STEPS[1], PREVIOUS)):
            slot = SlotState(step, instruction.target, instruction, previous)
            found[f"{name}/{label}"] = (config, slot_values(inputs, slot, None))
    return found


def test_policy_input_messages_are_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    found = {
        name: sha256_text(config.prompt.render(values)[1])
        for name, (config, values) in policy_values(tmp_path, small_files).items()
    }

    assert found == goldens("intent-b-v1-goldens.json")["messages_sha256"]


def test_policy_input_first_request_body_is_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    config, values = policy_values(tmp_path, small_files)["en/first"]

    body = _request_body(config, values)

    assert body_sha256(body) == goldens("intent-b-v1-goldens.json")["first_request_body_sha256"]


def test_policy_input_follow_up_turns_are_unchanged() -> None:
    turns = ConfiguredTurns(load_config(TRANSLATE / "seed100601-policy.yaml").prompt.turns)
    found = {
        "run_result": turns.run_result(2, 12, "out", "", 0, False, False, 1),
        "run_result_all_notes": turns.run_result(12, 12, "out", "err tail", None, True, True, 3),
        "protocol_error": turns.protocol_error(("missing a block",), 5),
        "protocol_error_no_runs": turns.protocol_error(("a", "b"), 0),
        "budget_exhausted": turns.budget_exhausted(),
    }

    assert {name: sha256_text(text) for name, text in found.items()} == goldens("intent-b-v1-goldens.json")[
        "turns_sha256"
    ]


def test_converter_provenance_blob_is_unchanged() -> None:
    digest = hashlib.sha256(CONVERTER.read_bytes()).hexdigest()

    assert digest == goldens("prompt-request-goldens.json")["converter_sha256"]
