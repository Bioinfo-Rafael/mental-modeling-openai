"""The scenario: episode, instructions, alarms and requests, and what of it a target sees at step t.

Strict loading (exact key sets) of scenario YAML schema v2, the rules that need no episode
(``check_scenario_rules``), the episode facts (last step, teams, down steps) and the rules that do
(``validate_scenario``), and the per-step views every input uses: the instructions in effect, the
alarms that reach a target, and the request at a step. Paths in a scenario file resolve relative
to that file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from translate.events import (
    TARGET_AIRCRAFT,
    TARGET_KINDS,
    TARGET_TEAM,
    Alarm,
    EpisodeRef,
    Instruction,
    RequestPoint,
    Scenario,
    TargetRef,
)
from translate.lag.dataset import last_complete_step
from translate.strict_yaml import mapping, nonnegative_int, read_yaml, safe_id, text

SCENARIO_SCHEMA_VERSION = 2
# D58 alias outcome check on lowercase alphanumeric tokens: the alias shares no token with
# episode_id (outcome, seed or IC component) and no token is an outcome word.
ALIAS_TOKEN_SEPARATORS = re.compile(r"[^a-z0-9]+")
OUTCOME_WORDS = ("win", "loss", "lose", "won", "lost")
SCENARIO_KEYS = {"schema_version", "scenario_id", "episode", "instructions", "alarms", "requests"}
EPISODE_KEYS = {"episode_id", "alias", "csv_path", "video_path"}
INSTRUCTION_KEYS = {"instruction_id", "issue_step", "target", "text"}
ALARM_KEYS = {"alarm_id", "step", "targets", "text"}
REQUEST_KEYS = {"request_id", "step", "targets", "source", "question"}
TARGET_KEYS = {"kind", "id"}


class ScenarioError(ValueError):
    """The scenario file is malformed or inconsistent."""


class PipelineError(ValueError):
    """The scenario does not fit the episode, or the detector budget cannot hold a scheduled state."""


@dataclass(frozen=True)
class ScenarioLimitsConfig:
    """Character limits of the scenario texts; they bound every text that reaches a model."""

    max_instruction_chars: int
    max_alarm_chars: int
    max_question_chars: int


# --- loading --------------------------------------------------------------------------------------


def _target(raw: object, name: str) -> TargetRef:
    section = mapping(raw, TARGET_KEYS, name, ScenarioError)
    if section["kind"] not in TARGET_KINDS:
        raise ScenarioError(f"{name}.kind must be one of {list(TARGET_KINDS)}")
    return TargetRef(kind=section["kind"], target_id=safe_id(section["id"], f"{name}.id", ScenarioError))


def _list(raw: object, name: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ScenarioError(f"{name} must be a list (use [] for none)")
    return raw


def _alias_tokens(value: str) -> set[str]:
    """Lowercase alphanumeric tokens: ``mhi_v1.3_ic100601_win`` -> {mhi, v1, 3, ic100601, win}."""
    return {token for token in ALIAS_TOKEN_SEPARATORS.split(value.lower()) if token}


def _checked_alias(alias: str, episode_id: str) -> str:
    """D58: the alias shares no token with the episode id, and none of its tokens is an outcome word."""
    tokens = _alias_tokens(alias)
    if tokens & _alias_tokens(episode_id):
        raise ScenarioError("episode.alias must differ from episode_id and share no token with it")
    if tokens.intersection(OUTCOME_WORDS):
        raise ScenarioError(f"episode.alias must not have an outcome word {list(OUTCOME_WORDS)} as a token")
    return alias


def _episode(raw: object, base: Path) -> EpisodeRef:
    section = mapping(raw, EPISODE_KEYS, "episode", ScenarioError)
    episode_id = text(section["episode_id"], "episode.episode_id", ScenarioError)
    alias = _checked_alias(safe_id(section["alias"], "episode.alias", ScenarioError), episode_id)
    video = section["video_path"]
    return EpisodeRef(
        episode_id=episode_id,
        alias=alias,
        csv_path=(base / text(section["csv_path"], "episode.csv_path", ScenarioError)).resolve(),
        video_path=None if video is None else (base / text(video, "episode.video_path", ScenarioError)).resolve(),
    )


def _instruction(raw: object, index: int) -> Instruction:
    name = f"instructions[{index}]"
    section = mapping(raw, INSTRUCTION_KEYS, name, ScenarioError)
    return Instruction(
        instruction_id=safe_id(section["instruction_id"], f"{name}.instruction_id", ScenarioError),
        issue_step=nonnegative_int(section["issue_step"], f"{name}.issue_step", ScenarioError),
        target=_target(section["target"], f"{name}.target"),
        text=text(section["text"], f"{name}.text", ScenarioError),
    )


def _targets(raw: object, name: str) -> tuple[TargetRef, ...]:
    items = _list(raw, f"{name}.targets")
    targets = tuple(_target(item, f"{name}.targets[{i}]") for i, item in enumerate(items))
    if not targets or len(set(targets)) != len(targets):
        raise ScenarioError(f"{name}.targets must be a non-empty list of distinct targets")
    return targets


def _alarm(raw: object, index: int) -> Alarm:
    name = f"alarms[{index}]"
    section = mapping(raw, ALARM_KEYS, name, ScenarioError)
    return Alarm(
        alarm_id=safe_id(section["alarm_id"], f"{name}.alarm_id", ScenarioError),
        step=nonnegative_int(section["step"], f"{name}.step", ScenarioError),
        targets=_targets(section["targets"], name),
        text=text(section["text"], f"{name}.text", ScenarioError),
    )


def _request(raw: object, index: int) -> RequestPoint:
    name = f"requests[{index}]"
    section = mapping(raw, REQUEST_KEYS, name, ScenarioError)
    return RequestPoint(
        request_id=safe_id(section["request_id"], f"{name}.request_id", ScenarioError),
        step=nonnegative_int(section["step"], f"{name}.step", ScenarioError),
        targets=_targets(section["targets"], name),
        source=safe_id(section["source"], f"{name}.source", ScenarioError),
        question=text(section["question"], f"{name}.question", ScenarioError),
    )


def _check_text_limits(scenario: Scenario, limits: ScenarioLimitsConfig) -> None:
    """Every scenario text within its configured length; nothing is ever truncated."""
    texts = [
        *((item.instruction_id, item.text, limits.max_instruction_chars) for item in scenario.instructions),
        *((item.alarm_id, item.text, limits.max_alarm_chars) for item in scenario.alarms),
        *((item.request_id, item.question, limits.max_question_chars) for item in scenario.requests),
    ]
    for item_id, content, limit in texts:
        if len(content) > limit:
            raise ScenarioError(f"{item_id} text has {len(content)} characters; the limit is {limit} (scenario_limits)")


def check_scenario_rules(scenario: Scenario, limits: ScenarioLimitsConfig) -> None:
    """Rules that need no episode: ids, step-0 instruction, per-step uniqueness, references, text limits.

    ``parse_scenario`` and ``pipeline.validate_scenario`` both run it, so a ``Scenario`` built in
    code obeys the same rules as a loaded one (and ``plan_schedule`` matches the run).
    """
    if not scenario.instructions:
        raise ScenarioError("instructions must not be empty")
    ids = [
        *(item.instruction_id for item in scenario.instructions),
        *(item.alarm_id for item in scenario.alarms),
        *(item.request_id for item in scenario.requests),
    ]
    if len(set(ids)) != len(ids):
        raise ScenarioError("instruction, alarm and request ids must be unique across the scenario")
    if not any(item.issue_step == 0 for item in scenario.instructions):
        raise ScenarioError("at least one instruction must be issued at step 0")
    keys = [(item.target, item.issue_step) for item in scenario.instructions]
    if len(set(keys)) != len(keys):
        raise ScenarioError("at most one instruction per exact target per step")
    request_keys = [(target, item.step) for item in scenario.requests for target in item.targets]
    if len(set(request_keys)) != len(request_keys):
        raise ScenarioError("at most one request per exact target per step")
    targets = {item.target for item in scenario.instructions}
    first_step = {t: min(i.issue_step for i in scenario.instructions if i.target == t) for t in targets}
    references = [(item.request_id, t, item.step) for item in scenario.requests for t in item.targets]
    references += [(item.alarm_id, t, item.step) for item in scenario.alarms for t in item.targets]
    for reference_id, target, step in references:
        if target not in first_step or first_step[target] > step:
            raise ScenarioError(
                f"{reference_id} references {target.label}, which has no instruction at or before {step}"
            )
    _check_text_limits(scenario, limits)


def parse_scenario(raw: object, *, base: Path, limits: ScenarioLimitsConfig) -> Scenario:
    section = mapping(raw, SCENARIO_KEYS, "scenario", ScenarioError)
    if section["schema_version"] != SCENARIO_SCHEMA_VERSION or isinstance(section["schema_version"], bool):
        raise ScenarioError(f"scenario schema_version must be {SCENARIO_SCHEMA_VERSION}")
    instructions = _list(section["instructions"], "instructions")
    if not instructions:
        raise ScenarioError("instructions must not be empty")
    scenario = Scenario(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=safe_id(section["scenario_id"], "scenario_id", ScenarioError),
        episode=_episode(section["episode"], base),
        instructions=tuple(_instruction(item, i) for i, item in enumerate(instructions)),
        alarms=tuple(_alarm(item, i) for i, item in enumerate(_list(section["alarms"], "alarms"))),
        requests=tuple(_request(item, i) for i, item in enumerate(_list(section["requests"], "requests"))),
    )
    check_scenario_rules(scenario, limits)
    return scenario


def load_scenario(path: Path, *, limits: ScenarioLimitsConfig) -> Scenario:
    """Load a scenario; ``csv_path`` / ``video_path`` resolve relative to the scenario file."""
    path = Path(path)
    return parse_scenario(read_yaml(path, ScenarioError), base=path.resolve().parent, limits=limits)


# --- the episode and what a target sees at step t -------------------------------------------------


@dataclass(frozen=True)
class EpisodeFacts:
    episode_id: str
    last_step: int
    teams: Mapping[str, tuple[str, ...]]  # team id -> its aircraft ids, sorted
    down_steps: Mapping[str, int | None]  # aircraft id -> first step <= last_step with alive == 0

    def team_of(self, aircraft_id: str) -> str:
        return next(team for team, members in self.teams.items() if aircraft_id in members)

    def target_aircraft(self, target: TargetRef) -> tuple[str, ...]:
        return (target.target_id,) if target.kind == TARGET_AIRCRAFT else self.teams[target.target_id]

    def opponents(self, target: TargetRef) -> tuple[str, ...]:
        own_team = self.team_of(target.target_id) if target.kind == TARGET_AIRCRAFT else target.target_id
        return tuple(sorted(a for team, members in self.teams.items() if team != own_team for a in members))

    def down_step(self, target: TargetRef) -> int | None:
        """Aircraft: its first ``alive == 0`` step. Team: the step its last aircraft is down, if all are."""
        steps = [self.down_steps[aircraft] for aircraft in self.target_aircraft(target)]
        return None if any(step is None for step in steps) else max(s for s in steps if s is not None)


def episode_facts(frame: pd.DataFrame) -> EpisodeFacts:
    last_step = last_complete_step(frame)
    aircraft = frame[frame["entity_type"].eq("aircraft") & frame["step"].le(last_step)]
    teams = {str(team): tuple(sorted(rows["object_id"].unique())) for team, rows in aircraft.groupby("team")}
    down_steps: dict[str, int | None] = {}
    for object_id, rows in aircraft.groupby("object_id"):
        down = rows.loc[rows["alive"].eq(0).fillna(False), "step"]
        down_steps = {**down_steps, str(object_id): int(down.min()) if len(down) else None}
    return EpisodeFacts(str(frame["episode_id"].iloc[0]), last_step, teams, down_steps)


def validate_scenario(scenario: Scenario, facts: EpisodeFacts, limits: ScenarioLimitsConfig) -> None:
    """The loader's scenario rules, then episode id, steps within ``[0, last_step]``, known aircraft and teams."""
    check_scenario_rules(scenario, limits)  # e.g. no request or alarm before its target's first instruction
    if scenario.episode.episode_id != facts.episode_id:
        raise PipelineError("scenario episode_id does not match the loaded episode")
    steps = [item.issue_step for item in scenario.instructions]
    steps += [item.step for item in (*scenario.alarms, *scenario.requests)]
    if max(steps) > facts.last_step:
        raise PipelineError(f"scenario steps must lie in [0, {facts.last_step}] (last complete step)")
    targets = {item.target for item in scenario.instructions}
    targets |= {target for item in (*scenario.alarms, *scenario.requests) for target in item.targets}
    for target in targets:
        known = facts.down_steps if target.kind == TARGET_AIRCRAFT else facts.teams
        if target.target_id not in known:
            raise PipelineError(f"unknown {target.kind} id {target.target_id!r} for this episode")


def instructions_in_effect(scenario: Scenario, step: int) -> tuple[Instruction, ...]:
    """Each instructed target's current instruction at ``step`` (issued at or before it), canonical order."""
    current: dict[TargetRef, Instruction] = {}
    for item in sorted(scenario.instructions, key=lambda item: (item.issue_step, item.instruction_id)):
        if item.issue_step <= step:
            current = {**current, item.target: item}
    return tuple(current[target] for target in sorted(current))


def target_instructions(
    scenario: Scenario, facts: EpisodeFacts, target: TargetRef, step: int
) -> tuple[Instruction, ...]:
    """The target's own instruction in effect and, for an aircraft, its team's (canonical order)."""
    own = {target}
    if target.kind == TARGET_AIRCRAFT:
        own = {target, TargetRef(TARGET_TEAM, facts.team_of(target.target_id))}
    return tuple(item for item in instructions_in_effect(scenario, step) if item.target in own)


def _reaches(alarm: Alarm, target: TargetRef, facts: EpisodeFacts) -> bool:
    """An alarm reaches every target that shares an aircraft with one of its targets."""
    alarm_aircraft = {aircraft for item in alarm.targets for aircraft in facts.target_aircraft(item)}
    return bool(alarm_aircraft & set(facts.target_aircraft(target)))


def target_alarms(scenario: Scenario, facts: EpisodeFacts, target: TargetRef, step: int) -> tuple[Alarm, ...]:
    """Every alarm for ``target`` issued at or before ``step``, oldest first."""
    found = (item for item in scenario.alarms if item.step <= step and _reaches(item, target, facts))
    return tuple(sorted(found, key=lambda item: (item.step, item.alarm_id)))


def target_request(scenario: Scenario, target: TargetRef, step: int) -> RequestPoint | None:
    """The request for ``target`` at exactly ``step`` (at most one, a scenario rule)."""
    return next((item for item in scenario.requests if item.step == step and target in item.targets), None)
