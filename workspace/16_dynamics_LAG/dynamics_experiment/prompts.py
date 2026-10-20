"""Joint trajectory prompts based on Experiment 14’s environment description.

LLM-Xavier owns the outer system/user wrappers. This module owns the LAG
description, history formatting, and task-specific question/answer contract.
Only the visible query is accepted; ground truth stays in runner/evaluation.
"""
from dataclasses import dataclass
import json
from types import SimpleNamespace

from lag_data.definitions import ACTION_LABELS, DT
from llm_x import prompting as llmx
from .dynamics import prior_text

PROMPT_VERSION = "lag-dynamics-v1"
TRAJECTORY_MARKER = ">>Final trajectory:"

# Reference ranges reviewed against valid aircraft telemetry in the three
# Experiment 13 recordings. These are prompt descriptions, not clipping bounds.
STATE_DESCRIPTIONS = {
    "lon_deg": "longitude in degrees; [119, 121]",
    "lat_deg": "latitude in degrees; [59, 61]",
    "alt_m": "altitude above mean sea level in meters; [3500, 6500]",
    "roll_deg": "roll angle in degrees; [-180, 180]",
    "pitch_deg": "pitch angle in degrees; [-40, 40]",
    "yaw_deg": "heading angle in degrees; [0, 360)",
    "tas_mps": "true airspeed in meters per second; [170, 360]",
    "vn_mps": "northward velocity in meters per second; [-350, 370]",
    "ve_mps": "eastward velocity in meters per second; [-370, 380]",
    "vu_mps": "upward velocity in meters per second; [-150, 170]",
}


def system_prompt(query):
    """Use the LLM-Xavier scene structure, omitting the reward section."""
    fields = ", ".join(query["state_fields"])
    state_definitions = "\n".join(
        f"- {name}: {STATE_DESCRIPTIONS[name]}." for name in query["state_fields"]
    )
    action_definitions = "\n".join(
        f"- {field}: " + "; ".join(f"{i} = {label}" for i, label in enumerate(labels)) + "."
        for field, labels in ACTION_LABELS.items()
    )
    description = SimpleNamespace(data={
        "task_name": "LAG-2v2",
        "task_description": """Light Aircraft Game (LAG) is an aircraft combat simulation. This
experiment uses recorded episodes of a two-versus-two engagement
between teams of F-16 aircraft equipped with missiles. The task
objective is to defeat the opposing aircraft while surviving.

Team A consists of A0100 and A0200, and Team B consists of B0100
and B0200. In the supplied recordings, Team A uses learned policies
and Team B uses rule-based pursuit controllers.

Control is hierarchical: a high-level policy selects altitude,
heading, speed, and firing commands. A fixed low-level policy
translates the movement commands into aircraft control inputs.""",
        "observation_space": f"""The state is a {len(query['state_fields'])}-dimensional telemetry vector in the following
fixed order:

[{fields}]

The fields and their reference ranges are:

{state_definitions}

Heading is measured clockwise from north: 0 degrees is north,
90 degrees is east, 180 degrees is south, and 270 degrees is west.
Account for angular wraparound when interpreting changes in
heading and roll.

The telemetry vector is not the complete observation originally
available to the policy. The upstream policy also uses information
about other aircraft and missile threats.

A null value means missing or unavailable information, not zero.
A state marked invalid must not be interpreted as active flight.""",
        "action_space": f"""Each action contains four discrete components in this order:

[altitude_action, heading_action, speed_action, shoot_action]

Each component takes one of the following values:

{action_definitions}

The movement commands correspond to altitude offsets of
+100 / 0 / -100 meters, heading offsets of
-30 / -15 / 0 / +15 / +30 degrees, and forward-speed offsets
of approximately +17 / 0 / -17 meters per second.

These are inputs to the low-level controller. They do not specify
the actual state change completed within one time step.
The speed command concerns forward speed, whereas tas_mps
records true airspeed.

The firing component specifies a launch request. A request does
not guarantee that a missile is launched.

Historical actions contain all four component IDs, including the
firing command. For action prediction, predict only the altitude,
heading, and speed components, in that order.""",
        "reward_space": "",
        "transition_dynamics": """Aircraft motion is simulated using JSBSim. Movement commands pass
through the low-level controller before affecting the aircraft's
subsequent position, attitude, and velocity.

Recorded steps are separated by 0.2 seconds. At step t, s_t denotes
the state before action a_t is applied. The action acts over the
interval from step t to step t+1, producing s_(t+1).

s_t -- a_t --> s_(t+1)""",
        "init_state": """In the supplied recordings, aircraft start near longitude
120 degrees and latitude 60 degrees, at approximately 6096 meters
altitude and 243.84 meters per second true airspeed.
Initial positions and headings vary between scenarios.""",
        "termination": """The upstream task checks termination for each aircraft upon
destruction or a flight-safety violation, at the time limit,
or when all opponents are destroyed and no incoming missiles
remain.""",
    })
    description.data["transition_dynamics"] += prior_text()
    if query.get("observation_scope", "self") == "all_aircraft_missiles":
        description.data["observation_space"] += """

The reference ranges above describe aircraft telemetry.
Missile states use the same field order and units. Their recorded
positions and attitudes are provided; unrecorded speed and velocity
components are null. Aircraft reference ranges do not bound missile states."""
    # The upstream wrapper requires all seven fields. Remove only its empty
    # reward section locally, preserving the upstream implementation unchanged.
    return llmx.system_prompt(description).replace("\n\nReward space:\n", "", 1)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _render_block(block):
    lines = [f"Step {block['step']} (time_s={block['time_s']}):"]
    for state in block.get("states", []):
        metadata = ""
        if "entity_type" in state:
            metadata = "; " + "; ".join(
                f"{key}={_json(state[key])}"
                for key in ("entity_type", "team", "alive", "launcher_id", "event")
            )
        lines.append(f"  state {state['object_id']}: {_json(state['values'])}; valid={state['valid']}{metadata}")
    for action in block.get("actions", []):
        lines.append(f"  action {action['object_id']}: {_json(action['values'])}; status={action['status']}")
    return "\n".join(lines)


@dataclass(frozen=True)
class HistoryView:
    """Minimal history_text interface for unmodified llmx.user_prompt.

    start/end index the selected blocks, not the original simulation step.
    The original step numbers remain in each block, including sparse histories.
    """
    blocks: tuple
    prefix: str = ""
    suffix: str = ""

    def history_text(self, start, end, *, indexed, drop_last_state_feature):
        if not indexed or drop_last_state_feature or not 0 <= start < end <= len(self.blocks):
            raise ValueError("Unsupported history rendering parameters")
        return self.prefix + "\n\n".join(
            _render_block(block) for block in self.blocks[start:end]
        ) + self.suffix


def user_prompt(query):
    """One joint forecast, with no teacher forcing or future context."""
    target = query["target_id"]
    steps = query["target_steps"]
    scope = query["observation_scope"]
    history_description = (
        "Each history entry contains the target aircraft's state and recorded action.\n"
        if scope == "self" else
        "Each history step contains states of all aircraft and missiles recorded at that step,\n"
        "plus available aircraft actions. Unrecorded or inactive actions are null.\n"
        "Missile control actions are not recorded. Missing values are not filled in.\n"
        "Metadata gives entity type, team, alive flag, launcher, and recorded event.\n"
        "An alive flag of null is unavailable; 0 means inactive. Invalid states are null.\n"
        "A remove event records disappearance and does not by itself establish a missile hit.\n"
    )
    prefix = (
        f"Target aircraft: {target}\n"
        f"Forecast steps (in order): {_json(steps)}\n"
        f"State field order: [{', '.join(query['state_fields'])}]\n"
        f"Historical action ID order: [{', '.join(query['history_action_fields'])}]\n\n"
        f"History: {len(query['history'])} consecutive steps, separated by {DT} seconds.\n"
        + history_description +
        "Each pair contains state s_t before action a_t, which acts from t to t+1.\n\n"
    )
    template = _json([
        {"step": step, "state": [f"<{name}>" for name in query["state_fields"]],
         "action": ["<altitude_id>", "<heading_id>", "<speed_id>"]}
        for step in steps
    ])
    question = (
        f"Predict the complete future trajectory of aircraft {target} for the next "
        f"{query['horizon']} consecutive steps ({steps[0]} through {steps[-1]} inclusive).\n"
        "Predict BOTH the ten-dimensional state s_t and the three movement action IDs a_t "
        "at EVERY requested step in ONE response. State and action in an entry refer to "
        "the SAME step: a_t is selected at s_t and affects s_(t+1).\n"
        "No future state or action is supplied, for the target or other entities. "
        "Continue from the final historical state/action pair and predict the whole sequence "
        "using the supplied history and your own predicted trajectory.\n\n"
        "For each state, return exactly ten finite numerical values in the stated field order.\n"
        "For each action, return exactly three integer IDs in this order: "
        "[altitude_action, heading_action, speed_action]. "
        "Allowed IDs are altitude 0..2, heading 0..4, speed 0..2. "
        "Do not predict a shoot_action ID.\n"
        "Choose one value per component, even if multiple values seem possible.\n\n"
        "Please first provide compact reasoning before your prediction.\n"
        "Use the following template in your answer:\n\n"
        "1. [Reasoning]:\n"
        "2. [Prediction]:\n"
        "3. [Formatting]:\n"
        f"{TRAJECTORY_MARKER} {template}\n\n"
        "Replace every placeholder string with a numerical value. "
        "Use the final-answer marker exactly once. Its final JSON array must occupy "
        "a single line, with exactly ten objects in the requested step order. "
        "Each object must have exactly the keys step, state, action."
    )
    history = HistoryView(tuple(query["history"]), prefix)
    return llmx.user_prompt(
        history, history_start=0, history_end=len(history.blocks),
        indexed_history=True, drop_last_state_feature=False, question=question,
    )


def build_messages(query):
    """The only public message-construction entry point used by make_plan."""
    return [
        {"role": "system", "content": system_prompt(query)},
        {"role": "user", "content": user_prompt(query)},
    ]
