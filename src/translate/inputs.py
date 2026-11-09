"""INPUTS: the named functions ``View(t) x target x options -> text`` that fill prompt placeholders.

A config selects inputs under ``inputs:`` (each with typed options, defaults below); each selected
input fills the placeholder of its name. An input sees only ``View(t)``, so it cannot leak data
issued after t (the poison test runs every input). A new input is one function here plus its key in
a config and its placeholder in the prompt. Lists render in a fixed documented format; free text is
quoted as a JSON string.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from translate.episode import Target, View, render_compact, render_lag17
from translate.journal import quoted, repo_path

NO_INSTRUCTION = "(no instruction in effect)"
NONE = "(none)"


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreviousOptions(Options):
    evidence: bool = True  # false: the intent only (the detector state)


class RecentOptions(Options):
    past_steps: int = Field(25, gt=0)  # alarms and questions issued in [t - past_steps + 1, t]


class TrajectoryOptions(Options):
    past_steps: int = Field(60, gt=0)  # the window [max(0, t - past_steps + 1), t]
    objects: Literal["all", "target_and_opponents"] = "all"
    fields: tuple[str, ...] | None = None  # None: every entity in the 17 format; else these columns
    precision: int = Field(5, ge=0)  # decimals of float fields (fields only)


class PolicyOptions(Options):
    rollout: Path  # manifest.json + episode_seed<seed>.npz + head_weights.npz (slurm/lag/record_rollouts.sbatch)
    seed: int
    arrays: tuple[Literal["obs", "gru_out", "act_in", "logits", "actions"], ...] = (
        "obs",
        "gru_out",
        "act_in",
        "logits",
        "actions",
    )
    head_weights: bool = True  # the 13 x 128 head matrix and its biases: logits = W_head @ act_in + b_head
    past_steps: int = Field(60, gt=0)
    precision: int = Field(4, ge=0)


@dataclass(frozen=True)
class InputSpec:
    render: Callable[[View, Target, Any], str]
    options: type[Options]


INPUTS: dict[str, InputSpec] = {}


def register(name: str, options: type[Options] = Options) -> Callable:
    def add(function: Callable[[View, Target, Any], str]) -> Callable[[View, Target, Any], str]:
        INPUTS[name] = InputSpec(function, options)
        return function

    return add


# --- the messages a target sees at t (shared with the event record) -------------------------------


def instructions_in_effect(view: View) -> list[tuple[str, Any]]:
    """(target id, its latest instruction) for every instructed target, oldest issue first."""
    latest: dict[str, Any] = {}
    for message in sorted((m for m in view.messages if m.kind == "instruction"), key=lambda m: (m.step, m.id)):
        latest = {**latest, **dict.fromkeys(message.to, message)}
    return sorted(latest.items(), key=lambda item: (item[1].step, item[1].id, item[0]))


def addressed_aircraft(to: str, view: View) -> tuple[str, ...]:
    kind, name = to.split(":")
    return (name,) if kind == "aircraft" else tuple(view.teams.get(name, ()))


def alarms_for(view: View, target: Target) -> list[Any]:
    """Every alarm issued by t that reaches the target: it shares an aircraft with an addressee."""
    reach = set(target.aircraft)
    return [
        m
        for m in view.messages
        if m.kind == "alarm" and reach & {name for to in m.to for name in addressed_aircraft(to, view)}
    ]


def question_at(view: View, target: Target) -> Any | None:
    return next((m for m in view.messages if m.kind == "question" and m.step == view.step and target.id in m.to), None)


# --- inputs ---------------------------------------------------------------------------------------


@register("instructions")
def instructions(view: View, target: Target, options: Options) -> str:
    """Every instructed target's current instruction, oldest first; this target's own is marked."""
    lines = [
        f"- {to.replace(':', ' ')}, issued at step {m.step}: {quoted(m.text)}"
        + (" (this target)" if to == target.id else "")
        for to, m in instructions_in_effect(view)
    ]
    return "\n".join(lines) or NO_INSTRUCTION


@register("alarms")
def alarms(view: View, target: Target, options: Options) -> str:
    """Every alarm that reached the target by t, oldest first."""
    found = alarms_for(view, target)
    return "\n".join(f"- step {m.step}: {quoted(m.text)}" for m in found) or NONE


@register("question")
def question(view: View, target: Target, options: Options) -> str:
    """The human question to the target arriving at t."""
    found = question_at(view, target)
    return NONE if found is None else quoted(found.text)


@register("recent_messages", RecentOptions)
def recent_messages(view: View, target: Target, options: RecentOptions) -> str:
    """Alarms that reached the target and questions to it, issued in the last ``past_steps``, oldest first."""
    first = view.step - options.past_steps + 1
    found = [m for m in alarms_for(view, target) if m.step >= first]
    found += [m for m in view.messages if m.kind == "question" and target.id in m.to and m.step >= first]
    ordered = sorted(found, key=lambda m: (m.step, m.kind, m.id))
    return "\n".join(f"- {m.kind} at step {m.step}: {quoted(m.text)}" for m in ordered) or NONE


@register("previous_translation", PreviousOptions)
def previous_translation(view: View, target: Target, options: PreviousOptions) -> str:
    """The target's latest valid translation before t, or that there is none yet."""
    previous = view.translations.get(target.id)
    if previous is None:
        return "(none: this is the first translation for this target)"
    text = f"from step {previous.step}: {quoted(previous.intent)}"
    if not options.evidence:
        return text
    return text + "".join(f"\n  - {quoted(item)}" for item in previous.evidence)


@register("trajectory", TrajectoryOptions)
def trajectory(view: View, target: Target, options: TrajectoryOptions) -> str:
    """The window ending at t: every entity (17 format), or the chosen fields of the chosen objects."""
    start = max(0, view.step - options.past_steps + 1)
    if options.fields is None:
        return render_lag17(view.rows, start, view.step, view.step)
    objects = (*target.aircraft, *target.opponents) if options.objects == "target_and_opponents" else _all(view)
    return render_compact(view.rows, start, view.step, view.step, objects, options.fields, options.precision)


def _all(view: View) -> tuple[str, ...]:
    return tuple(sorted(view.rows.loc[view.rows["entity_type"].eq("aircraft"), "object_id"].unique()))


# --- the policy trace (U-c option b: recorded activations plus the action-head weights) -----------


@dataclass(frozen=True)
class Rollout:
    learners: tuple[str, ...]  # the aircraft the recorded policy flies, in row order
    arrays: dict[str, np.ndarray]  # name -> (decisions, rows, ...)
    weight: np.ndarray | None  # (logits, act_in width)
    bias: np.ndarray | None


@lru_cache(maxsize=4)
def load_rollout(directory: Path, seed: int, names: tuple[str, ...], with_weights: bool) -> Rollout:
    """One recorded episode of the policy, read once (npz only, no pickle)."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    files = {entry["seed"]: entry["file"] for entry in manifest["episodes"]}
    if seed not in files:
        raise ValueError(f"seed {seed} is not in the rollout {directory}")
    with np.load(directory / files[seed], allow_pickle=False) as data:
        missing = sorted(set(names) - set(data.files))
        if missing:
            raise ValueError(f"the rollout lacks {missing}: re-record it (slurm/lag/record_rollouts.sbatch)")
        arrays = {name: np.array(data[name]) for name in names}
    if not with_weights:
        return Rollout(tuple(manifest["learner_uids"]), arrays, None, None)
    with np.load(directory / "head_weights.npz", allow_pickle=False) as weights:
        return Rollout(tuple(manifest["learner_uids"]), arrays, np.array(weights["weight"]), np.array(weights["bias"]))


def _rounded(values: np.ndarray, precision: int) -> list[Any]:
    return np.round(values, precision).tolist() if values.dtype.kind == "f" else values.tolist()


@register("policy_trace", PolicyOptions)
def policy_trace(view: View, target: Target, options: PolicyOptions) -> str:
    """The recorded policy rows of the window ending at t for each aircraft of the target, then W_head and b_head."""
    rollout = load_rollout(repo_path(options.rollout), options.seed, options.arrays, options.head_weights)
    start = max(0, view.step - options.past_steps + 1)
    if view.step >= len(next(iter(rollout.arrays.values()))):
        raise ValueError(f"step {view.step} is after the end of the recorded rollout")
    sections = []
    for aircraft in target.aircraft:
        if aircraft not in rollout.learners:
            sections.append(f"{aircraft}: not flown by the recorded policy")
            continue
        row = rollout.learners.index(aircraft)
        lines = [
            json.dumps(
                {
                    "step": s,
                    "relative_step": s - view.step,
                    **{n: _rounded(rollout.arrays[n][s, row], options.precision) for n in options.arrays},
                },
                separators=(",", ":"),
            )
            for s in range(start, view.step + 1)
        ]
        title = f"Policy trace of {aircraft}, steps {start}-{view.step}, one JSON line per step, oldest first:"
        sections.append("\n".join([title, *lines]))
    if options.head_weights:
        rows = [
            json.dumps(
                {
                    "logit": i,
                    "bias": _rounded(rollout.bias[i], options.precision),
                    "weight": _rounded(rollout.weight[i], options.precision),
                },
                separators=(",", ":"),
            )
            for i in range(len(rollout.bias))
        ]
        sections.append(
            "\n".join(["Action heads, one JSON line per logit (logits = W_head @ act_in + b_head):", *rows])
        )
    return "\n\n".join(sections)
