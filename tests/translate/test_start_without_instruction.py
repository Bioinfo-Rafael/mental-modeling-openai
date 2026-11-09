"""An episode may start with or without an instruction in effect (2026-11-09): every input, every
prompt and the detector state word both cases."""

from __future__ import annotations

import pandas as pd
import pytest

from translate.cli import _request_body
from translate.config import load_config
from translate.events import TargetRef
from translate.inputs import RunInputs, SlotState, event_input, slot_values, trajectory_window
from translate.lag.dataset import load_episode
from translate.lag.rendering import render_lag17_jsonl
from translate.scenario import episode_facts, instructions_in_effect

from .tr_fakes import EPISODE_CSV, RUN_CONFIG, WIN01, instruction, make_scenario

A0100 = TargetRef("aircraft", "A0100")
LATE = 30  # A0100's first instruction


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return load_episode(EPISODE_CSV[WIN01])


def _trajectory_slot(frame: pd.DataFrame, step: int) -> tuple[str, dict]:
    """(user text of the online example prompt, event input) for A0100 at ``step``."""
    config = load_config(RUN_CONFIG)
    scenario = make_scenario([instruction("a1", LATE, "aircraft", "A0100")])
    inputs = RunInputs(frame, scenario, episode_facts(frame), config, render_lag17_jsonl)
    own = next((item for item in instructions_in_effect(scenario, step) if item.target == A0100), None)
    slot = SlotState(step, A0100, own, None)
    trajectory = trajectory_window(inputs, step)
    body = _request_body(config, slot_values(inputs, slot, trajectory))
    user = body["input"][0]["content"][0]["text"]
    return user, event_input(inputs, slot, trajectory)


def test_the_trajectory_prompt_says_none_when_no_instruction_is_in_effect(frame: pd.DataFrame) -> None:
    user, record = _trajectory_slot(frame, 0)

    assert "Instructions in effect:\n(none)\n" in user
    assert record["instructions_in_effect"] == [] and record["primary_instruction_id"] is None


def test_the_trajectory_prompt_lists_the_instruction_once_it_is_in_effect(frame: pd.DataFrame) -> None:
    user, record = _trajectory_slot(frame, LATE)

    assert f"Instructions in effect:\n- issued at step {LATE} for aircraft A0100 | THIS TARGET'S INSTRUCTION:" in user
    assert record["primary_instruction_id"] == "a1"
