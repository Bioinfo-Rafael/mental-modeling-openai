"""An episode may start with or without an instruction in effect (2026-11-09): every input, every
prompt and the detector state word both cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from translate import paths
from translate.cli import _request_body
from translate.config import ConfigError, load_config
from translate.events import TargetRef
from translate.inputs import RunInputs, SlotState, event_input, slot_values, trajectory_window
from translate.lag.dataset import load_episode
from translate.lag.rendering import render_lag17_jsonl
from translate.policy.material import load_policy_material
from translate.scenario import episode_facts, instructions_in_effect

from ..policy.policy_fakes import POLICY_CONFIG, SEED, STEPS, policy_config, policy_inputs, policy_raw, write_rollout
from .tr_fakes import EPISODE_CSV, RUN_CONFIG, WIN01, instruction, make_scenario, write_yaml

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


def _policy_message(tmp_path: Path, small_files: tuple[Path, Path], with_instruction: bool) -> str:
    checkpoint, spec = small_files
    write_rollout(tmp_path / "rollout", checkpoint, spec)
    config = policy_config(tmp_path / "rollout", spec)
    inputs = policy_inputs(config, load_policy_material(config.policy.rollout_dir, SEED, spec, None))
    own = inputs.scenario.instructions[0]
    slot = SlotState(STEPS[0], own.target, own if with_instruction else None, None)
    return config.prompt.render(slot_values(inputs, slot, None))[1]


@pytest.mark.parametrize("with_instruction", [True, False])
def test_the_policy_prompt_words_the_targets_own_instruction_or_its_absence(
    tmp_path: Path, small_files: tuple[Path, Path], with_instruction: bool
) -> None:
    message = _policy_message(tmp_path, small_files, with_instruction)

    line = next(line for line in message.split("\n") if "Human instruction in effect" in line)
    expected = "No operator instruction was given." if with_instruction else "(none: no instruction is in effect)"
    assert line.endswith(f"): {expected}")


def test_a_prompt_that_uses_primary_instruction_needs_its_block(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    raw = yaml.safe_load(POLICY_CONFIG.read_text(encoding="utf-8"))
    prompt = yaml.safe_load((paths.ROOT / raw["prompt"]["file"]).read_text(encoding="utf-8"))
    without = {**prompt, "blocks": {k: v for k, v in prompt["blocks"].items() if k != "primary_instruction"}}
    prompt_file = write_yaml(tmp_path / "prompt.yaml", without)
    checkpoint, spec = small_files
    write_rollout(tmp_path / "rollout", checkpoint, spec)
    config = policy_raw(tmp_path / "rollout", spec, POLICY_CONFIG)

    with pytest.raises(ConfigError, match="uses primary_instruction but has no primary_instruction block"):
        load_config(
            write_yaml(tmp_path / "config.yaml", {**config, "prompt": {**config["prompt"], "file": str(prompt_file)}})
        )
