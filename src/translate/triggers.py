"""When to translate: the two trigger sources, the deterministic schedule plan and the detector budget.

Trigger kinds are the event vocabulary: ``start`` (the start point at a target's first
instruction), ``change`` (the clef-flash detector's label) and ``request`` (a human request).
``online`` triggers come as the episode advances: the forced ``start`` and ``request``
from the scenario, and the detector on its tick grid. ``preset`` triggers are listed in the config
in advance, each step with its kind; a preset ``request`` point needs the scenario request it
answers, and every scenario request must be a preset point (a request is never dropped).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from translate.config import TRIGGER_SOURCE_ONLINE, DetectorConfig, PresetPoint, RunConfig
from translate.detector import (
    MAX_UTF8_BYTES_PER_CHAR,
    MESSAGE_ALARM,
    MESSAGE_QUESTION,
    DetectionContext,
    DetectorMessage,
    detector_state_bound,
)
from translate.events import (
    FIRST_INSTRUCTION_SOURCE,
    TRIGGER_REQUEST,
    TRIGGER_START,
    IntentTranslation,
    PreviousTranslation,
    Scenario,
    TargetRef,
    TriggerReason,
    sort_reasons,
)
from translate.lag.rendering import RenderedTrajectory
from translate.llm.system_one import SystemOneQuestion
from translate.scenario import EpisodeFacts, PipelineError, target_alarms, target_instructions, target_request

PRESET_SOURCE = "preset"

Triggers = dict[tuple[int, TargetRef], tuple[TriggerReason, ...]]


def _detector(config: RunConfig) -> DetectorConfig:
    if config.detector is None:
        raise PipelineError("this run has no detector (preset triggers)")
    return config.detector


def scenario_triggers(scenario: Scenario) -> Triggers:
    """Forced trigger reasons per (step, target): ``start`` at each target's first instruction, ``request``."""
    found: Triggers = {}

    def add(step: int, target: TargetRef, reason: TriggerReason) -> None:
        nonlocal found
        found = {**found, (step, target): (*found.get((step, target), ()), reason)}

    for target in sorted({item.target for item in scenario.instructions}):
        first = min((item for item in scenario.instructions if item.target == target), key=lambda i: i.issue_step)
        reason = TriggerReason(TRIGGER_START, FIRST_INSTRUCTION_SOURCE, first.instruction_id, None)
        add(first.issue_step, target, reason)
    for request in scenario.requests:
        for target in request.targets:
            add(request.step, target, TriggerReason(TRIGGER_REQUEST, request.source, request.request_id, None))
    return {key: sort_reasons(reasons) for key, reasons in found.items()}


def _add(found: Triggers, step: int, target: TargetRef, reason: TriggerReason) -> Triggers:
    return {**found, (step, target): (*found.get((step, target), ()), reason)}


def _first_instruction_steps(scenario: Scenario) -> dict[TargetRef, int]:
    targets = {item.target for item in scenario.instructions}
    return {t: min(i.issue_step for i in scenario.instructions if i.target == t) for t in targets}


def _preset_reason(scenario: Scenario, point: PresetPoint, target: TargetRef) -> TriggerReason:
    if point.kind != TRIGGER_REQUEST:
        return TriggerReason(point.kind, PRESET_SOURCE, f"preset-step-{point.step}", None)
    request = target_request(scenario, target, point.step)
    if request is None:
        raise PipelineError(f"preset request point at step {point.step} has no scenario request for {target.label}")
    return TriggerReason(TRIGGER_REQUEST, request.source, request.request_id, None)


def preset_triggers(scenario: Scenario, points: tuple[PresetPoint, ...]) -> Triggers:
    """Trigger reasons per (step, target) from the config's preset points."""
    first_steps = _first_instruction_steps(scenario)
    found: Triggers = {}
    for point in points:
        for target in point.targets or tuple(sorted(first_steps)):
            if target not in first_steps or first_steps[target] > point.step:
                raise PipelineError(f"preset point at step {point.step}: {target.label} has no instruction by then")
            found = _add(found, point.step, target, _preset_reason(scenario, point, target))
    for request in scenario.requests:
        for target in request.targets:
            kinds = {reason.kind for reason in found.get((request.step, target), ())}
            if TRIGGER_REQUEST not in kinds:
                raise PipelineError(f"scenario request {request.request_id} is not a preset request point")
    return {key: sort_reasons(reasons) for key, reasons in found.items()}


def check_preset_points(config: RunConfig, facts: EpisodeFacts) -> None:
    """Every preset point must lie within the episode (``[0, last complete step]``).

    The loop stops at the last complete step, so a later point would silently never fire.
    """
    late = sorted({point.step for point in config.triggers.points if point.step > facts.last_step})
    if late:
        raise PipelineError(f"preset points at steps {late} are after the last complete step {facts.last_step}")


def run_triggers(scenario: Scenario, config: RunConfig) -> Triggers:
    """The forced or preset trigger reasons of a run (the detector adds its own while running)."""
    if config.triggers.source == TRIGGER_SOURCE_ONLINE:
        return scenario_triggers(scenario)
    return preset_triggers(scenario, config.triggers.points)


def tick_interval(config: RunConfig) -> int | None:
    """The detector tick grid of an online run; ``None`` for preset triggers."""
    return config.triggers.tick_interval_steps if config.triggers.source == TRIGGER_SOURCE_ONLINE else None


def detector_messages(
    scenario: Scenario, facts: EpisodeFacts, target: TargetRef, step: int, lookback_steps: int
) -> tuple[DetectorMessage, ...]:
    """All alarms and questions for ``target`` issued in ``[step - lookback + 1, step]``, most recent last."""
    first = step - lookback_steps + 1
    alarms = [
        DetectorMessage(MESSAGE_ALARM, item.alarm_id, item.step, item.text)
        for item in target_alarms(scenario, facts, target, step)
        if item.step >= first
    ]
    questions = [
        DetectorMessage(MESSAGE_QUESTION, item.request_id, item.step, item.question)
        for item in scenario.requests
        if first <= item.step <= step and target in item.targets
    ]
    return tuple(sorted((*alarms, *questions), key=lambda item: (item.step, item.kind, item.message_id)))


def is_tick(step: int, tick_interval_steps: int) -> bool:
    """Fixed episode grid (D28): requests or instructions never move the ticks."""
    return step > 0 and step % tick_interval_steps == 0


# --- deterministic schedule (preview and preflight) -----------------------------------------------


@dataclass(frozen=True)
class TargetPlan:
    target: TargetRef
    first_instruction_step: int
    down_step: int | None
    mandatory_translation_steps: tuple[int, ...]  # scenario triggers while the target is up
    detector_tick_steps: tuple[int, ...]  # exact: every tick after the first translation while up
    suppressed_trigger_steps: tuple[int, ...]


@dataclass(frozen=True)
class SchedulePlan:
    targets: tuple[TargetPlan, ...]

    @property
    def mandatory_translations(self) -> int:
        return sum(len(plan.mandatory_translation_steps) for plan in self.targets)

    @property
    def detector_calls(self) -> int:
        return sum(len(plan.detector_tick_steps) for plan in self.targets)

    @property
    def translation_upper_bound(self) -> int:
        """Mandatory translations plus one per detector tick that has no scenario trigger."""
        return sum(len({*p.mandatory_translation_steps, *p.detector_tick_steps}) for p in self.targets)

    @property
    def suppressed_triggers(self) -> int:
        return sum(len(plan.suppressed_trigger_steps) for plan in self.targets)


def plan_schedule(
    scenario: Scenario,
    facts: EpisodeFacts,
    triggers: Mapping[tuple[int, TargetRef], tuple[TriggerReason, ...]],
    tick_interval_steps: int | None,
) -> SchedulePlan:
    """Calls a full run makes, without any model: detector calls never depend on model outputs.

    ``tick_interval_steps`` is ``None`` for preset triggers (no detector).
    """
    plans = []
    for target in sorted({item.target for item in scenario.instructions}):
        first = min(item.issue_step for item in scenario.instructions if item.target == target)
        down = facts.down_step(target)
        end = facts.last_step + 1 if down is None else max(first, down)
        trigger_steps = sorted(step for step, key in triggers if key == target)
        ticks = (
            ()
            if tick_interval_steps is None
            else tuple(step for step in range(first + 1, end) if is_tick(step, tick_interval_steps))
        )
        plans.append(
            TargetPlan(
                target=target,
                first_instruction_step=first,
                down_step=down,
                mandatory_translation_steps=tuple(step for step in trigger_steps if down is None or step < down),
                detector_tick_steps=ticks,
                suppressed_trigger_steps=tuple(step for step in trigger_steps if down is not None and step >= down),
            )
        )
    return SchedulePlan(targets=tuple(plans))


@dataclass(frozen=True)
class BudgetFit:
    input_token_budget: int
    worst_bound: int  # conservative tokens of the largest scheduled detector state (0 if none)
    worst_target: TargetRef | None
    worst_step: int | None
    checked_states: int
    intent_reserve_bytes: int

    @property
    def margin(self) -> int:
        return self.input_token_budget - self.worst_bound

    @property
    def fits(self) -> bool:
        return self.worst_bound < self.input_token_budget


def detector_context(
    frame: pd.DataFrame,
    *,
    scenario: Scenario,
    target: TargetRef,
    step: int,
    previous: PreviousTranslation,
    facts: EpisodeFacts,
    config: RunConfig,
    renderer: Callable[..., RenderedTrajectory],
) -> DetectionContext:
    """The compact window ``[max(0, step-W+1), step]`` of the target aircraft, then its opponents,
    plus the target's instructions in effect and its alarms and questions within the lookback."""
    detector = _detector(config)
    window = detector.window
    own, opponents = facts.target_aircraft(target), facts.opponents(target)
    trajectory = renderer(
        frame,
        start_step=max(0, step - window.past_steps + 1),
        end_step=step,
        anchor_step=step,
        object_ids=(*own, *opponents),
        fields=window.fields,
        float_precision=window.float_precision,
    )
    instructions = target_instructions(scenario, facts, target, step)
    messages = detector_messages(scenario, facts, target, step, detector.messages_lookback_steps)
    return DetectionContext(target, step, own, opponents, previous, instructions, messages, trajectory)


def check_detector_budget(
    frame: pd.DataFrame,
    *,
    ticks: tuple[tuple[TargetRef, int], ...],
    scenario: Scenario,
    facts: EpisodeFacts,
    config: RunConfig,
    questions: tuple[SystemOneQuestion, ...],
    renderer: Callable[..., RenderedTrajectory],
) -> BudgetFit:
    """Worst conservative bound over the given (target, tick) states: the exact state (instruction,
    alarm and question texts included) with a worst-case intent. Nothing is ever truncated."""
    budget, max_chars = _detector(config).input_token_budget, config.answer.limits.max_intent_chars
    worst: tuple[int, TargetRef | None, int | None] = (0, None, None)
    for target, step in ticks:
        placeholder = PreviousTranslation("", step, IntentTranslation("", ()))
        context = detector_context(
            frame,
            scenario=scenario,
            target=target,
            step=step,
            previous=placeholder,
            facts=facts,
            config=config,
            renderer=renderer,
        )
        bound = detector_state_bound(context, questions=questions, max_intent_chars=max_chars)
        worst = max(worst, (bound, target, step), key=lambda item: item[0])
    return BudgetFit(budget, worst[0], worst[1], worst[2], len(ticks), MAX_UTF8_BYTES_PER_CHAR * max_chars)
