"""Input processors: the selected inputs of one slot become the prompt's named values.

Each processor reads only what is visible at the slot's step t (instructions issued at or before
t, the alarms that reach the target by t, the request at t, the target's own latest valid
translation, the trajectory window ending at t, the policy window ending at t) and returns named
values; list and optional inputs are turned into text by the prompt file's ``blocks``. The values
every slot has (``BASE_VALUES`` in ``config.py``) come first. ``event_input`` records what a human
rater sees for the slot (scenario texts and the trajectory window), whatever the translator read.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from translate.config import INPUT_POLICY, INPUT_TRAJECTORY, RunConfig
from translate.events import (
    Instruction,
    PreviousTranslation,
    Scenario,
    TargetRef,
    step_time_s,
    trajectory_json,
)
from translate.lag.rendering import STEP_SECONDS, RenderedTrajectory
from translate.policy.material import PolicyMaterial, policy_values
from translate.prompts import list_block, optional_block
from translate.scenario import EpisodeFacts, instructions_in_effect, target_alarms, target_request

T_S_DECIMALS = 6  # the policy prompt's "time t_s s" (formatted with :g)

Renderer = Callable[..., RenderedTrajectory]


@dataclass(frozen=True)
class RunInputs:
    """Everything the processors read, loaded once per run."""

    frame: pd.DataFrame
    scenario: Scenario
    facts: EpisodeFacts
    config: RunConfig
    renderer: Renderer
    policy: PolicyMaterial | None = None


@dataclass(frozen=True)
class SlotState:
    step: int
    target: TargetRef
    instruction: Instruction | None  # the target's current instruction; None before its first one
    previous: PreviousTranslation | None  # the target's latest valid translation


def trajectory_window(inputs: RunInputs, step: int) -> RenderedTrajectory:
    """The window ``[max(0, t - past + 1), t]``: judge material always, prompt input when selected."""
    past = inputs.config.trajectory.past_steps
    return inputs.renderer(
        inputs.frame,
        start_step=max(0, step - past + 1),
        end_step=step,
        anchor_step=step,
        episode_alias=inputs.scenario.episode.alias,
    )


def _base(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    return {
        "episode_alias": inputs.scenario.episode.alias,
        "step": slot.step,
        "time_s": step_time_s(slot.step),
        "t_s": round(slot.step * STEP_SECONDS, T_S_DECIMALS),
        "step_seconds": STEP_SECONDS,
        "target_label": slot.target.label,
        "target_aircraft": ", ".join(inputs.facts.target_aircraft(slot.target)),
        "max_intent_chars": inputs.config.answer.limits.max_intent_chars,
    }


def _instructions(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    """All instructions in effect as a block and this target's own one through its optional block (each
    when the prompt defines it); either may be empty: no instruction need be in effect."""
    blocks, own = inputs.config.prompt.blocks, slot.instruction
    values: dict[str, object] = {}
    if "primary_instruction" in blocks:
        values = {
            "primary_instruction": optional_block(blocks["primary_instruction"], {"text": own.text} if own else None)
        }
    block = blocks.get("instructions")
    if block is None:
        return values
    own_id = own.instruction_id if own else None
    items = [
        {
            "issue_step": item.issue_step,
            "target": item.target.label,
            "primary": block["primary_marker"] if item.instruction_id == own_id else "",
            "text": item.text,
        }
        for item in instructions_in_effect(inputs.scenario, slot.step)
    ]
    return {**values, "instructions": list_block(block, items)}


def _alarms(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    alarms = target_alarms(inputs.scenario, inputs.facts, slot.target, slot.step)
    items = [{"step": a.step, "targets": ", ".join(t.label for t in a.targets), "text": a.text} for a in alarms]
    return {"alarms": list_block(inputs.config.prompt.blocks["alarms"], items)}


def _question(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    request = target_request(inputs.scenario, slot.target, slot.step)
    item = None if request is None else {"step": request.step, "text": request.question}
    return {"question": optional_block(inputs.config.prompt.blocks["question"], item)}


def _previous(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    block = inputs.config.prompt.blocks["previous_translation"]
    previous = slot.previous
    if previous is None:
        return {"previous_translation": optional_block(block, None)}
    evidence = block["evidence_joiner"].join(
        block["evidence_item"].format(item=item) for item in previous.translation.evidence
    )
    item = {"step": previous.step, "intent": previous.translation.intent, "evidence": evidence}
    return {"previous_translation": optional_block(block, item)}


def _trajectory(trajectory: RenderedTrajectory) -> dict[str, object]:
    return {
        "trajectory": trajectory.text,
        "trajectory_start": trajectory.start_step,
        "trajectory_end": trajectory.end_step,
        "trajectory_points": trajectory.end_step - trajectory.start_step + 1,
        "trajectory_anchor": trajectory.anchor_step,
    }


def _policy(inputs: RunInputs, slot: SlotState) -> dict[str, object]:
    if inputs.policy is None or inputs.config.policy is None:
        raise ValueError("the policy input is selected but no policy material was loaded")
    return policy_values(inputs.policy, slot.step, inputs.config.policy.past_steps)


def _session_values(config: RunConfig) -> dict[str, object]:
    session = config.session
    if session is None:
        return {}
    sandbox = session.sandbox
    return {
        "max_runs": session.max_runs,
        "timeout_s": sandbox.timeout_s,
        "stdout_cap_chars": sandbox.stdout_cap_chars,
        "stderr_tail_chars": sandbox.stderr_tail_chars,
    }


PROCESSORS: dict[str, Callable[[RunInputs, SlotState], dict[str, object]]] = {
    "instructions": _instructions,
    "alarms": _alarms,
    "question": _question,
    "previous_translation": _previous,
    INPUT_POLICY: _policy,
}


def slot_values(inputs: RunInputs, slot: SlotState, trajectory: RenderedTrajectory) -> dict[str, object]:
    """Base values, then each selected input's values, then the session limits (code session)."""
    values = {**_base(inputs, slot), **_session_values(inputs.config)}
    for name in inputs.config.inputs:
        found = _trajectory(trajectory) if name == INPUT_TRAJECTORY else PROCESSORS[name](inputs, slot)
        values = {**values, **found}
    return values


def event_input(inputs: RunInputs, slot: SlotState, trajectory: RenderedTrajectory) -> dict[str, Any]:
    """The event's ``input``: what a human rater sees for this slot, plus what the translator read."""
    request = target_request(inputs.scenario, slot.target, slot.step)
    config = inputs.config
    record = {
        "prompt_version": config.prompt.version,
        "target_aircraft": list(inputs.facts.target_aircraft(slot.target)),
        "instructions_in_effect": [item.to_json() for item in instructions_in_effect(inputs.scenario, slot.step)],
        "primary_instruction_id": slot.instruction.instruction_id if slot.instruction else None,
        "alarms": [item.to_json() for item in target_alarms(inputs.scenario, inputs.facts, slot.target, slot.step)],
        "question": None
        if request is None
        else {"request_id": request.request_id, "step": request.step, "text": request.question},
        "trajectory": trajectory_json(trajectory),
        "previous_translation": slot.previous.to_json() if slot.previous else None,
        "output_language": config.prompt.vars.get("output_language"),
        "translator_inputs": list(config.inputs),
    }
    if inputs.policy is None:
        return record
    # The window and the bundle the sandbox read are added by the call (pipeline: observed_inputs).
    policy = {
        "network_alias": inputs.policy.spec.alias,
        "checkpoint_sha256": inputs.policy.checkpoint.sha256,
        "rollout_sha256": inputs.policy.rollout.sha256,
    }
    return {**record, "policy": policy}
