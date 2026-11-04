"""Chronological per-target Translator pipeline (online mode, D7/D9/D17/D21/D28/D35/D37/D50/D51).

For every step ``0 .. last_complete_step``:

1. apply the instructions issued at this step (a later one replaces the target's current one);
2. close every instructed target that is down (aircraft: ``alive == 0``; team: all its aircraft);
3. for each instructed target in canonical (kind, id) order: collect the forced triggers
   (``initial`` at the target's first instruction, ``request`` at a scenario request); a closed
   target only logs them as ``trigger_suppressed``; on a tick (``step > 0``, ``step % T == 0``) a
   target that already has a translation gets one detector call, which may add ``start`` or
   ``change``; if any trigger remains, the target gets exactly one translation, whose primary
   trigger follows ``TRIGGER_PRECEDENCE``. A later instruction triggers nothing by itself: it only
   reaches the detector state (D50).

Every model input holds only data issued at or before the step: the trajectory window ends at it,
and instructions, alarms and questions are filtered to ``step <= t``. Every call is written to the
sink before it is sent; its response line records ``latency_s``. Caps stop the run before call
cap+1. Ctrl-C ends the run like any other failure: a call in flight is journaled as ``uncertain``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Protocol

import pandas as pd

from mtd_rl_xavier.lag.dataset import last_complete_step
from mtd_rl_xavier.lag.rendering import RenderedTrajectory
from mtd_rl_xavier.llm.system_one import SystemOneQuestion

from .config import ScenarioLimitsConfig, TranslatorConfig, check_scenario_rules
from .detector import (
    MAX_UTF8_BYTES_PER_CHAR,
    MESSAGE_ALARM,
    MESSAGE_QUESTION,
    DetectionContext,
    DetectorLayer,
    DetectorMessage,
    detector_state_bound,
)
from .schema import (
    DETECTOR_SOURCE,
    DOWN_REASON,
    EVENT_SCHEMA_VERSION,
    INITIAL_SOURCE,
    RECORD_DETECTOR_CALL,
    RECORD_TARGET_CLOSED,
    RECORD_TRANSLATION_EVENT,
    RECORD_TRIGGER_SUPPRESSED,
    TARGET_AIRCRAFT,
    TARGET_TEAM,
    TRIGGER_INITIAL,
    TRIGGER_REQUEST,
    UNEXPECTED_ERROR_MESSAGE,
    Alarm,
    Instruction,
    IntentTranslation,
    ModelCallError,
    PreviousTranslation,
    RequestPoint,
    Scenario,
    TargetRef,
    TargetState,
    TriggerReason,
    canonical_json,
    detector_call_id,
    event_id,
    primary_trigger,
    redacted_error,
    sha256_text,
    sort_reasons,
    step_time_s,
    trajectory_json,
)
from .translation import (
    PROMPT_VERSION,
    IntentModelOutput,
    TranslationLayer,
    TranslationModelRequest,
    TranslationOutputError,
)

STATUS_COMPLETED = "completed"
STATUS_CAP_REACHED = "cap_reached"
STATUS_FAILED = "failed"
STATUS_UNCERTAIN = "uncertain"
LAYER_DETECTOR = "detector"
LAYER_TRANSLATION = "translation"
RESPONSE_OK = "ok"
RESPONSE_ERROR = "error"
RESPONSE_UNCERTAIN = "uncertain"
UNEXPECTED_ERROR_TYPE_PREFIX = "unexpected:"
TIMING_DECIMALS = 3  # latency_s and processing_s: monotonic seconds, rounded to milliseconds
# What ends a run with accurate counts. Ctrl-C during a call makes that call uncertain (exit 3).
INTERRUPTIONS = (Exception, KeyboardInterrupt)

Renderer = Callable[..., RenderedTrajectory]


class PipelineError(ValueError):
    """The scenario does not fit the episode, or the detector budget cannot hold a scheduled state."""


class EventSink(Protocol):
    def write_event(self, record: dict[str, Any]) -> None: ...  # events.jsonl
    def write_request(self, record: dict[str, Any]) -> None: ...  # requests.jsonl, before each send
    def write_response(self, record: dict[str, Any]) -> None: ...  # responses.jsonl


# --- episode facts and scenario checks ------------------------------------------------------------


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


def scenario_triggers(scenario: Scenario) -> dict[tuple[int, TargetRef], tuple[TriggerReason, ...]]:
    """Forced trigger reasons per (step, target): ``initial`` at each target's first instruction, ``request``."""
    found: dict[tuple[int, TargetRef], tuple[TriggerReason, ...]] = {}

    def add(step: int, target: TargetRef, reason: TriggerReason) -> None:
        nonlocal found
        found = {**found, (step, target): (*found.get((step, target), ()), reason)}

    for target in sorted({item.target for item in scenario.instructions}):
        first = min((item for item in scenario.instructions if item.target == target), key=lambda i: i.issue_step)
        add(first.issue_step, target, TriggerReason(TRIGGER_INITIAL, INITIAL_SOURCE, first.instruction_id, None))
    for request in scenario.requests:
        for target in request.targets:
            add(request.step, target, TriggerReason(TRIGGER_REQUEST, request.source, request.request_id, None))
    return {key: sort_reasons(reasons) for key, reasons in found.items()}


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


def plan_schedule(scenario: Scenario, facts: EpisodeFacts, tick_interval_steps: int) -> SchedulePlan:
    """Calls a full run makes, without any model: detector calls never depend on model outputs."""
    triggers = scenario_triggers(scenario)
    plans = []
    for target in sorted({item.target for item in scenario.instructions}):
        first = min(item.issue_step for item in scenario.instructions if item.target == target)
        down = facts.down_step(target)
        end = facts.last_step + 1 if down is None else max(first, down)
        trigger_steps = sorted(step for step, key in triggers if key == target)
        ticks = tuple(step for step in range(first + 1, end) if is_tick(step, tick_interval_steps))
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
    config: TranslatorConfig,
    renderer: Renderer,
) -> DetectionContext:
    """The compact window ``[max(0, step-W+1), step]`` of the target aircraft, then its opponents,
    plus the target's instructions in effect and its alarms and questions within the lookback."""
    window = config.detector.window
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
    messages = detector_messages(scenario, facts, target, step, config.detector.messages_lookback_steps)
    return DetectionContext(target, step, own, opponents, previous, instructions, messages, trajectory)


def check_detector_budget(
    frame: pd.DataFrame,
    *,
    ticks: tuple[tuple[TargetRef, int], ...],
    scenario: Scenario,
    facts: EpisodeFacts,
    config: TranslatorConfig,
    questions: tuple[SystemOneQuestion, ...],
    renderer: Renderer,
) -> BudgetFit:
    """Worst conservative bound over the given (target, tick) states: the exact state (instruction,
    alarm and question texts included) with a worst-case intent. Nothing is ever truncated."""
    budget, max_chars = config.detector.input_token_budget, config.translation.max_intent_chars
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


# --- run ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallCounts:
    attempted: int = 0  # requests written and handed to the model
    completed: int = 0  # answered (a later parse failure still counts as answered)
    failed: int = 0
    uncertain: int = 0


@dataclass(frozen=True)
class TranslatorSummary:
    status: str  # completed | cap_reached | failed | uncertain
    detector: CallCounts
    translation: CallCounts
    events_written: int
    unsent: dict[str, Any] | None  # cap_reached: the call that was not sent (layer, target, step)
    error: dict[str, Any] | None  # failed / uncertain: layer, call_id, type, message, uncertain


def _instructions_by_step(scenario: Scenario) -> dict[int, tuple[Instruction, ...]]:
    """Instructions per issue step, in instruction-id order (D21: applied before any target is processed)."""
    ordered = sorted(scenario.instructions, key=lambda item: item.instruction_id)
    return {step: tuple(i for i in ordered if i.issue_step == step) for step in {i.issue_step for i in ordered}}


class _Stop(Exception):
    def __init__(
        self, status: str, *, unsent: dict[str, Any] | None = None, error: dict[str, Any] | None = None
    ) -> None:
        super().__init__(status)
        self.status, self.unsent, self.error = status, unsent, error


class _Run:
    """Progress of one run. Local to ``Translator.run`` and never shared."""

    def __init__(self, frame: pd.DataFrame, scenario: Scenario, config: TranslatorConfig, sink: EventSink) -> None:
        self.frame, self.scenario, self.config, self.sink = frame, scenario, config, sink
        self.facts = episode_facts(frame)
        validate_scenario(scenario, self.facts, config.scenario_limits)
        self.triggers = scenario_triggers(scenario)
        self.instructions_at = _instructions_by_step(scenario)
        self.states: dict[TargetRef, TargetState] = {}
        self.counts = {LAYER_DETECTOR: CallCounts(), LAYER_TRANSLATION: CallCounts()}
        self.sequence = 0
        self.in_flight: tuple[str, str] | None = None  # (layer, call_id) sent but not yet journaled

    def bump(self, layer: str, field: str) -> None:
        counts = self.counts[layer]
        self.counts = {**self.counts, layer: replace(counts, **{field: getattr(counts, field) + 1})}

    def emit(
        self, record_type: str, step: int, target: TargetRef, body: dict[str, Any], *, started: float | None = None
    ) -> None:
        """Write one event; ``sequence`` advances only once the record is written (``events_written``).

        With ``started`` (translation events), ``processing_s`` is taken once the whole record is
        built, immediately before its own write.
        """
        sequence = self.sequence + 1
        common = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "sequence": sequence,
            "record_type": record_type,
            "scenario_id": self.scenario.scenario_id,
            "episode_id": self.scenario.episode.episode_id,
            "step": step,
            "time_s": step_time_s(step),
            "target": target.to_json(),
        }
        record = {**common, **body}
        if started is not None:
            record = {**record, "processing_s": _elapsed_s(started)}
        self.sink.write_event(record)
        self.sequence = sequence

    def check_cap(self, layer: str, target: TargetRef, step: int) -> None:
        limits = self.config.limits
        cap = limits.max_detector_calls if layer == LAYER_DETECTOR else limits.max_translation_calls
        if self.counts[layer].attempted >= cap:
            unsent = {"layer": layer, "target": target.to_json(), "step": step, "cap": cap}
            raise _Stop(STATUS_CAP_REACHED, unsent=unsent)

    def summary(self, status: str, stop: _Stop | None = None) -> TranslatorSummary:
        return TranslatorSummary(
            status=status,
            detector=self.counts[LAYER_DETECTOR],
            translation=self.counts[LAYER_TRANSLATION],
            events_written=self.sequence,
            unsent=stop.unsent if stop else None,
            error=stop.error if stop else None,
        )


class Translator:
    """Two layers, one chronological loop. Renderers are injectable (tests use cheap fakes)."""

    def __init__(
        self,
        *,
        detector: DetectorLayer,
        translation: TranslationLayer,
        detector_renderer: Renderer,
        translation_renderer: Renderer,
    ) -> None:
        self._detector = detector
        self._translation = translation
        self._detector_renderer = detector_renderer
        self._translation_renderer = translation_renderer

    def run(
        self, *, frame: pd.DataFrame, scenario: Scenario, config: TranslatorConfig, sink: EventSink
    ) -> TranslatorSummary:
        run = _Run(frame, scenario, config, sink)
        try:
            for step in range(run.facts.last_step + 1):
                self._step(run, step)
        except _Stop as stop:
            return run.summary(stop.status, stop)
        except INTERRUPTIONS as exc:  # a journal write failed or Ctrl-C: stop with accurate counts
            return run.summary(*_crash_status(run, exc))
        return run.summary(STATUS_COMPLETED)

    def _step(self, run: _Run, step: int) -> None:
        for instruction in run.instructions_at.get(step, ()):
            known = run.states.get(instruction.target)
            state = TargetState(
                target=instruction.target,
                instruction=instruction,
                previous=known.previous if known else None,
                closed=known.closed if known else False,
            )
            run.states = {**run.states, instruction.target: state}
        for target in sorted(run.states):
            down = run.facts.down_step(target)
            if not run.states[target].closed and down is not None and step >= down:
                self._close(run, step, target, down)
        for target in sorted(run.states):
            self._process_target(run, step, target)

    def _close(self, run: _Run, step: int, target: TargetRef, down: int) -> None:
        state = run.states[target]
        run.states = {**run.states, target: replace(state, closed=True)}
        aircraft = run.facts.target_aircraft(target)
        body = {
            "reason": DOWN_REASON,
            "down_step": down,
            "aircraft_down_steps": {a: run.facts.down_steps[a] for a in aircraft},
            "last_event_id": state.previous.event_id if state.previous else None,
        }
        run.emit(RECORD_TARGET_CLOSED, step, target, body)

    def _process_target(self, run: _Run, step: int, target: TargetRef) -> None:
        started = perf_counter()  # processing_s: from here (before any detector call) to the event record
        reasons = run.triggers.get((step, target), ())
        state = run.states[target]
        if state.closed:
            if reasons:
                body = {
                    "reason": DOWN_REASON,
                    "down_step": run.facts.down_step(target),
                    "primary_trigger": primary_trigger(reasons),
                    "trigger_reasons": [reason.to_json() for reason in reasons],
                }
                run.emit(RECORD_TRIGGER_SUPPRESSED, step, target, body)
            return
        if is_tick(step, run.config.pipeline.tick_interval_steps) and state.previous is not None:
            detected = self._detect(run, step, state)
            reasons = sort_reasons((*reasons, detected)) if detected else reasons
        if reasons:
            self._translate(run, step, state, reasons, started)

    def _send(self, run: _Run, layer: str, call_id: str, call: Callable[[], Any]) -> tuple[Any, float]:
        """Count and make one call; returns its result and ``latency_s``. A failure is logged and stops the run."""
        run.bump(layer, "attempted")
        run.in_flight = (layer, call_id)
        sent = perf_counter()
        try:
            result = call()
        except ModelCallError as exc:
            latency_s = _elapsed_s(sent)
            raise self._stopped(
                run, layer, call_id, latency_s, exc.error_type, str(exc), exc.uncertain, exc.raw
            ) from exc
        except INTERRUPTIONS as exc:  # anything else or Ctrl-C: the request may have reached the server
            latency_s = _elapsed_s(sent)
            error_type = UNEXPECTED_ERROR_TYPE_PREFIX + type(exc).__name__
            raise self._stopped(
                run, layer, call_id, latency_s, error_type, UNEXPECTED_ERROR_MESSAGE, True, None
            ) from exc
        return result, _elapsed_s(sent)

    def _answered(self, run: _Run, layer: str, call_id: str, latency_s: float, raw: Any) -> None:
        run.sink.write_response(_response_record(call_id, layer, RESPONSE_OK, latency_s, raw, None))
        run.in_flight = None
        run.bump(layer, "completed")

    def _stopped(
        self,
        run: _Run,
        layer: str,
        call_id: str,
        latency_s: float,
        error_type: str,
        message: str,
        uncertain: bool,
        raw: Any,
    ) -> _Stop:
        error = {"type": error_type, "message": message}
        status = RESPONSE_UNCERTAIN if uncertain else RESPONSE_ERROR
        run.sink.write_response(_response_record(call_id, layer, status, latency_s, raw, error))
        run.in_flight = None
        run.bump(layer, "uncertain" if uncertain else "failed")
        details = {"layer": layer, "call_id": call_id, **error, "uncertain": uncertain}
        return _Stop(STATUS_UNCERTAIN if uncertain else STATUS_FAILED, error=details)

    def _write_request(self, run: _Run, call_id: str, layer: str, step: int, target: TargetRef, payload: Any) -> str:
        """Journal the exact payload before sending; returns its ``request_sha256``."""
        record = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "call_id": call_id,
            "layer": layer,
            "scenario_id": run.scenario.scenario_id,
            "step": step,
            "target": target.to_json(),
            "payload": payload,
        }
        run.sink.write_request(record)
        return sha256_text(canonical_json(payload))

    def _detect(self, run: _Run, step: int, state: TargetState) -> TriggerReason | None:
        target, previous = state.target, state.previous
        if previous is None:
            return None
        run.check_cap(LAYER_DETECTOR, target, step)
        call_id = detector_call_id(run.scenario.scenario_id, target, step)
        context = detector_context(
            run.frame,
            scenario=run.scenario,
            target=target,
            step=step,
            previous=previous,
            facts=run.facts,
            config=run.config,
            renderer=self._detector_renderer,
        )
        state_text = self._detector.state_text(context)
        payload = self._detector.request_payload(state_text)
        request_sha256 = self._write_request(run, call_id, LAYER_DETECTOR, step, target, payload)
        result, latency_s = self._send(run, LAYER_DETECTOR, call_id, lambda: self._detector.detect(state_text))
        self._answered(run, LAYER_DETECTOR, call_id, latency_s, result.reply.raw)
        detector = run.config.detector
        body = {
            "call_id": call_id,
            "request_sha256": request_sha256,
            "input": {
                "trajectory": trajectory_json(context.trajectory),
                "previous_translation": {
                    "event_id": previous.event_id,
                    "step": previous.step,
                    "intent": previous.translation.intent,
                },
                "instructions_in_effect": [instruction.to_json() for instruction in context.instructions],
                "messages": [message.to_json() for message in context.messages],
                "state_text": state_text,
                "question": self._detector.question,
            },
            "output": result.output_json(),
            "raw_response": result.reply.raw,
            "model": {
                "backend": detector.backend,
                "model": detector.model,
                "revision": detector.revision,
                "served_model_name": detector.served_model_name,
                "response_model": result.reply.model,
            },
            "usage": result.reply.usage,
        }
        run.emit(RECORD_DETECTOR_CALL, step, target, body)
        if not result.triggered:
            return None
        return TriggerReason(result.label, DETECTOR_SOURCE, call_id, result.probability)

    def _translate(
        self, run: _Run, step: int, state: TargetState, reasons: tuple[TriggerReason, ...], started: float
    ) -> None:
        target = state.target
        run.check_cap(LAYER_TRANSLATION, target, step)
        current_id = event_id(run.scenario.scenario_id, target, step)
        window = run.config.translation.window
        trajectory = self._translation_renderer(
            run.frame,
            start_step=max(0, step - window.past_steps + 1),
            end_step=step,
            anchor_step=step,
            episode_alias=run.scenario.episode.alias,
        )
        in_effect = instructions_in_effect(run.scenario, step)
        request = TranslationModelRequest(
            target=target,
            target_aircraft=run.facts.target_aircraft(target),
            step=step,
            episode_alias=run.scenario.episode.alias,
            instructions_in_effect=in_effect,
            primary_instruction_id=state.instruction.instruction_id,
            trajectory=trajectory,
            previous=state.previous,
            alarms=target_alarms(run.scenario, run.facts, target, step),
            question=target_request(run.scenario, target, step),
        )
        prompt = self._translation.build_prompt(request)
        payload = self._translation.request_payload(prompt)
        request_sha256 = self._write_request(run, current_id, LAYER_TRANSLATION, step, target, payload)
        output: IntentModelOutput
        output, latency_s = self._send(run, LAYER_TRANSLATION, current_id, lambda: self._translation.generate(prompt))
        self._answered(run, LAYER_TRANSLATION, current_id, latency_s, output.raw)
        try:
            translation = self._translation.parse(output)
        except TranslationOutputError as exc:
            error = {"layer": LAYER_TRANSLATION, "call_id": current_id, "type": type(exc).__name__, "message": str(exc)}
            raise _Stop(STATUS_FAILED, error={**error, "uncertain": False}) from exc
        body = {
            **_translation_body(
                run, current_id, request, reasons, output, translation, self._translation.output_language
            ),
            "request_sha256": request_sha256,
        }
        run.emit(RECORD_TRANSLATION_EVENT, step, target, body, started=started)
        previous = PreviousTranslation(current_id, step, translation)
        run.states = {**run.states, target: replace(state, previous=previous)}


def _translation_body(
    run: _Run,
    current_id: str,
    request: TranslationModelRequest,
    reasons: tuple[TriggerReason, ...],
    output: IntentModelOutput,
    translation: IntentTranslation,
    output_language: str,
) -> dict[str, Any]:
    translation_config = run.config.translation
    episode = run.scenario.episode
    return {
        "event_id": current_id,
        "episode": {
            "episode_id": episode.episode_id,
            "alias": episode.alias,
            "csv_sha256": run.frame.attrs.get("csv_sha256"),
        },
        "primary_trigger": primary_trigger(reasons),
        "trigger_reasons": [reason.to_json() for reason in reasons],
        "input": {
            "prompt_version": PROMPT_VERSION,
            "target_aircraft": list(request.target_aircraft),
            "instructions_in_effect": [instruction.to_json() for instruction in request.instructions_in_effect],
            "primary_instruction_id": request.primary_instruction_id,
            "alarms": [alarm.to_json() for alarm in request.alarms],
            "question": _question_json(request.question),
            "trajectory": trajectory_json(request.trajectory),
            "previous_translation": request.previous.to_json() if request.previous else None,
            "output_language": output_language,
        },
        "output": translation.to_json(),
        "model": {
            "backend": translation_config.backend,
            "model": translation_config.model,
            "reasoning_effort": translation_config.reasoning_effort,
            "max_output_tokens": translation_config.max_output_tokens,
            "response_model": output.model,
            "response_id": output.response_id,
        },
        "usage": output.usage,
    }


def _question_json(question: RequestPoint | None) -> dict[str, Any] | None:
    if question is None:
        return None
    return {"request_id": question.request_id, "step": question.step, "text": question.question}


def _elapsed_s(since: float) -> float:
    """Monotonic seconds since ``since`` (a ``perf_counter`` reading), rounded to milliseconds."""
    return round(perf_counter() - since, TIMING_DECIMALS)


def _crash_status(run: _Run, exc: BaseException) -> tuple[str, _Stop]:
    """Status after an unexpected exception or Ctrl-C: ``uncertain`` while a sent call is not yet journaled."""
    error = redacted_error(exc)
    if run.in_flight is None:
        return STATUS_FAILED, _Stop(STATUS_FAILED, error={**error, "layer": None, "call_id": None, "uncertain": False})
    layer, call_id = run.in_flight
    run.bump(layer, "uncertain")
    return STATUS_UNCERTAIN, _Stop(
        STATUS_UNCERTAIN, error={**error, "layer": layer, "call_id": call_id, "uncertain": True}
    )


def _response_record(
    call_id: str, layer: str, status: str, latency_s: float, raw: Any, error: dict[str, str] | None
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "call_id": call_id,
        "layer": layer,
        "status": status,
        "latency_s": latency_s,
        "raw": raw,
        "error": error,
    }
