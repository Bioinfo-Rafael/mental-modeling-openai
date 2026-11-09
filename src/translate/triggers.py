"""When to translate: the two trigger sources, the deterministic schedule plan and the detector budget.

Trigger kinds are the event vocabulary: ``start`` (the episode start, the first translatable step of
every target, with or without an instruction in effect), ``change`` (the clef-flash detector: the
previous intent no longer explains the behaviour because of an external factor or a human
instruction; progress along the plan is no change) and ``request`` (a human question, or a target's
first instruction arriving after the start). ``online`` triggers come as the episode advances: the
forced ``start`` and ``request`` from the scenario, and the detector on its tick grid. ``preset``
triggers are listed in the config in advance, each step with its kind; a preset ``request`` point
needs the scenario request it answers, and every scenario request must be a preset point (a request
is never dropped). A later instruction for a target that already has one only reaches the detector.
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
    TRIGGER_REQUEST,
    TRIGGER_START,
    Instruction,
    IntentTranslation,
    PreviousTranslation,
    Scenario,
    TargetRef,
    TriggerReason,
    sort_reasons,
)
from translate.lag.rendering import RenderedTrajectory
from translate.llm.system_one import SystemOneQuestion
from translate.scenario import (
    EpisodeFacts,
    PipelineError,
    target_alarms,
    target_instructions,
    target_request,
    translation_targets,
)

PRESET_SOURCE = "preset"
# Every input window clips at step 0 (``[max(0, t - past + 1), t]``), so step 0 is every target's first
# translatable step: the episode start.
EPISODE_START_STEP = 0
EPISODE_START_SOURCE, EPISODE_START_REFERENCE = "episode_start", "episode-start"
INSTRUCTION_SOURCE = "instruction"  # a target's first instruction arriving after the start: a request

Triggers = dict[tuple[int, TargetRef], tuple[TriggerReason, ...]]


def _detector(config: RunConfig) -> DetectorConfig:
    if config.detector is None:
        raise PipelineError("this run has no detector (preset triggers)")
    return config.detector


def _late_first_instruction(scenario: Scenario, target: TargetRef) -> Instruction | None:
    """The target's first instruction when it arrives after the episode start (a request point)."""
    own = [item for item in scenario.instructions if item.target == target]
    first = min(own, key=lambda item: (item.issue_step, item.instruction_id), default=None)
    return first if first is not None and first.issue_step > EPISODE_START_STEP else None


def scenario_triggers(scenario: Scenario) -> Triggers:
    """Forced trigger reasons per (step, target): ``start`` at the episode start for every target, and
    ``request`` at each question and at a target's first instruction when it arrives after the start."""
    found: Triggers = {}

    def add(step: int, target: TargetRef, reason: TriggerReason) -> None:
        nonlocal found
        found = {**found, (step, target): (*found.get((step, target), ()), reason)}

    for target in translation_targets(scenario):
        add(
            EPISODE_START_STEP,
            target,
            TriggerReason(TRIGGER_START, EPISODE_START_SOURCE, EPISODE_START_REFERENCE, None),
        )
        late = _late_first_instruction(scenario, target)
        if late is not None:
            add(late.issue_step, target, TriggerReason(TRIGGER_REQUEST, INSTRUCTION_SOURCE, late.instruction_id, None))
    for request in scenario.requests:
        for target in request.targets:
            add(request.step, target, TriggerReason(TRIGGER_REQUEST, request.source, request.request_id, None))
    return {key: sort_reasons(reasons) for key, reasons in found.items()}


def _add(found: Triggers, step: int, target: TargetRef, reason: TriggerReason) -> Triggers:
    return {**found, (step, target): (*found.get((step, target), ()), reason)}


def _preset_reason(scenario: Scenario, point: PresetPoint, target: TargetRef) -> TriggerReason:
    if point.kind != TRIGGER_REQUEST:
        return TriggerReason(point.kind, PRESET_SOURCE, f"preset-step-{point.step}", None)
    request = target_request(scenario, target, point.step)
    if request is not None:
        return TriggerReason(TRIGGER_REQUEST, request.source, request.request_id, None)
    late = _late_first_instruction(scenario, target)
    if late is not None and late.issue_step == point.step:
        return TriggerReason(TRIGGER_REQUEST, INSTRUCTION_SOURCE, late.instruction_id, None)
    raise PipelineError(
        f"preset request point at step {point.step} has no scenario request or late first instruction for {target.label}"
    )


def preset_triggers(scenario: Scenario, points: tuple[PresetPoint, ...]) -> Triggers:
    """Trigger reasons per (step, target) from the config's preset points (``targets: null``: every
    translation target); a target needs no instruction in effect to be translated."""
    targets = translation_targets(scenario)
    found: Triggers = {}
    for point in points:
        for target in point.targets or targets:
            if target not in targets:
                raise PipelineError(f"preset point at step {point.step}: {target.label} is not a translation target")
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
    first_translation_step: int | None  # the first trigger (None: the target has none)
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
    for target in translation_targets(scenario):
        trigger_steps = sorted(step for step, key in triggers if key == target)
        first = trigger_steps[0] if trigger_steps else None
        down = facts.down_step(target)
        end = facts.last_step + 1 if down is None else down
        ticks = (
            ()
            if tick_interval_steps is None or first is None
            else tuple(step for step in range(first + 1, end) if is_tick(step, tick_interval_steps))
        )
        plans.append(
            TargetPlan(
                target=target,
                first_translation_step=first,
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
