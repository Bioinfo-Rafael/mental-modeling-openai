"""Reviewed English prompts for LAG mental-modeling experiments.

LLM-Xavier owns the outer system/user wrappers. This module owns the LAG
description, history formatting, and task-specific question/answer contract.
Only the visible query is accepted; ground truth stays in runner/evaluation.
"""
from dataclasses import dataclass
import json
from types import SimpleNamespace

from lag_data.definitions import ACTION_LABELS, DT
from llm_x import prompting as llmx

PROMPT_VERSION = "lag-context-v3"
ACTION_MARKER = ">>Final action IDs:"
STATE_MARKER = ">>Final state values:"

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
    """Describe the visible evidence and ask one of the four temporal questions."""
    context = "\n\n".join(_render_block(block) for block in query["query_context"])
    task = query["task"]
    target = query["target_id"]
    step = query["target_step"]
    scope = query.get("observation_scope", "self")
    questions = {
        "next-action": (
            f"At step {step}, the agent is in the state shown above. "
            "Based on your observation and understanding of the agent's "
            f"behavior, predict the movement action that aircraft {target} "
            "will most likely select at this step."
        ),
        "last-action": (
            f"The states of aircraft {target} at steps {step} and {step + 1} "
            "are shown above. Based on your observation and understanding of "
            "the agent's behavior, infer the movement action it most likely "
            f"selected at step {step} to produce this transition."
        ),
        "next-state": (
            f"The state and action of aircraft {target} at step {step - 1} "
            "are shown above. Based on the supplied trajectory and action, "
            f"predict its state at step {step}."
        ),
        "last-state": (
            f"The action of aircraft {target} at step {step} and its state "
            f"at step {step + 1} are shown above. Based on the supplied "
            f"trajectory, infer its state at step {step}, before that action "
            "was applied."
        ),
    }
    if task.endswith("action"):
        allowed = "\n".join(
            f"- {field}: " + ", ".join(f"{i} = {label}" for i, label in enumerate(ACTION_LABELS[field])) + "."
            for field in ("altitude_action", "heading_action", "speed_action")
        )
        instruction = (
            "Predict exactly three integer IDs in this order:\n"
            "[altitude_action, heading_action, speed_action]\n\n"
            f"Allowed IDs:\n{allowed}"
        )
        answer = f"{ACTION_MARKER} [<altitude_id>, <heading_id>, <speed_id>]"
        closing = (
            "Choose exactly one ID for each component, even if multiple actions seem possible. "
            "Do not include a firing-command ID in the final answer."
        )
    else:
        instruction = (
            f"Return exactly {len(query['state_fields'])} finite numerical values in this order: "
            + ", ".join(query["state_fields"]) + "."
        )
        answer = STATE_MARKER + " [" + ", ".join(f"<{name}>" for name in query["state_fields"]) + "]"
        closing = "Choose exactly one numerical value for each state component, even if multiple values seem possible."
    history_description = (
        f"History:\nThe following {len(query['history'])} state-action pairs describe the\n"
        "target aircraft before the prediction step.\n"
    )
    if scope == "all_aircraft_missiles":
        history_description = (
            f"History:\nThe following {len(query['history'])} time steps contain recorded states\n"
            "of all four aircraft and all missiles recorded at those steps,\n"
            "together with available aircraft actions.\n"
            "Aircraft actions marked not_recorded or inactive are null.\n"
            "Missile control actions are not recorded. No missing values are filled in.\n"
            "Metadata identifies the entity type, team, aircraft alive flag,\n"
            "missile launcher, and recorded event (update, remove, or none).\n"
            "An alive flag of null means unavailable; 0 means inactive.\n"
            "Invalid state vectors are null. A remove event records disappearance\n"
            "and does not by itself establish a missile hit.\n"
            "Additional information follows the same object and field conventions;\n"
            "the target quantity requested below is withheld.\n"
        )
    prefix = (
        f"Target aircraft: {target}\n"
        f"Prediction step: {step}\n"
        f"Prediction time: {DT * step} seconds\n\n"
        f"State field order:\n[{', '.join(query['state_fields'])}]\n\n"
        f"Historical action ID order:\n[{', '.join(query['history_action_fields'])}]\n\n"
        + history_description +
        "Consecutive entries are separated by 0.2 seconds.\n"
        "For each aircraft, a recorded pair contains state s_t and action a_t\n"
        "selected at that state and applied over the following time step.\n\n"
    )
    context_heading = "Current state" if task == "next-action" else "Additional information for this question"
    history = HistoryView(tuple(query["history"]), prefix, f"\n\n{context_heading}:\n{context}")
    question = (
        questions[task] + "\n\n" + instruction + "\n\n"
        "Please first provide compact reasoning before your prediction.\n"
        "Use the following template in your answer:\n\n"
        "1. [Reasoning]:\n"
        "2. [Prediction]:\n"
        "3. [Formatting]:\n"
        + answer + "\n\n" + closing + "\n"
        "Use the final-answer marker exactly once and keep the final answer on a single line."
    )
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
