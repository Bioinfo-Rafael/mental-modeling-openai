"""The translation pipeline: one chronological loop, per slot: inputs -> prompt -> call -> answer -> event.

For every step ``0 .. last_complete_step``:

1. apply the instructions issued at this step (a later one replaces the target's current one);
2. close every instructed target that is down (aircraft: ``alive == 0``; team: all its aircraft);
3. for each instructed target in canonical (kind, id) order: take the run's trigger reasons for
   this (step, target) (online: the forced ``start`` and ``request``; preset: the configured
   points); a closed target only logs them as ``trigger_suppressed``; in an online run, on a tick
   (``step > 0``, ``step % T == 0``) a target that already has a translation gets one detector call,
   which may add ``change``; if any reason remains, the target gets one translation:
   the selected inputs become the prompt's values, the configured prompt is filled, the call is
   made, the answer is validated, and the event is written.

Every model input holds only data issued at or before the step. Every call is written to the sink
before it is sent; its response line records ``latency_s``. Caps stop the run before call cap+1.
An invalid answer stops the run (``answer.on_invalid: stop``) or is recorded with its problem
while the target keeps its last valid translation (``skip``). Ctrl-C ends the run like any other
failure: a call in flight is journaled as ``uncertain``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import pandas as pd

from translate.answers import INVALID_ANSWER_PREFIX
from translate.call_one_shot import CallResult, RetryJournal, SlotRef
from translate.call_session import SandboxRefused
from translate.config import ON_INVALID_STOP, RunConfig
from translate.detector import DetectorLayer
from translate.events import (
    DETECTOR_SOURCE,
    DOWN_REASON,
    EVENT_SCHEMA_VERSION,
    RECORD_DETECTOR_CALL,
    RECORD_TARGET_CLOSED,
    RECORD_TRANSLATION_EVENT,
    RECORD_TRIGGER_SUPPRESSED,
    UNEXPECTED_ERROR_MESSAGE,
    Instruction,
    IntentTranslation,
    ModelCallError,
    PreviousTranslation,
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
from translate.inputs import RunInputs, SlotState, event_input, slot_values, trajectory_window
from translate.lag.rendering import RenderedTrajectory, render_detector_compact, render_lag17_jsonl
from translate.policy.material import PolicyMaterial
from translate.scenario import episode_facts, validate_scenario
from translate.triggers import check_preset_points, detector_context, is_tick, run_triggers, tick_interval

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
INVALID_ANSWER_ERROR = "TranslationOutputError"
TIMING_DECIMALS = 3  # latency_s and processing_s: monotonic seconds, rounded to milliseconds
SESSIONS_DIR = "sessions"
# What ends a run with accurate counts. Ctrl-C during a call makes that call uncertain (exit 3).
INTERRUPTIONS = (Exception, KeyboardInterrupt)

Renderer = Callable[..., RenderedTrajectory]


class EventSink(Protocol):
    def write_event(self, record: dict[str, Any]) -> None: ...  # events.jsonl
    def write_request(self, record: dict[str, Any]) -> None: ...  # requests.jsonl, before each send
    def write_response(self, record: dict[str, Any]) -> None: ...  # responses.jsonl


class TranslationCall(Protocol):
    """``OneShotCall`` or ``SessionCall``: prepare (payload to journal), run, validate the answer."""

    def prepare(self, values: dict[str, object], slot: SlotRef) -> tuple[dict[str, Any], Any]: ...
    def run(self, prepared: Any) -> CallResult: ...
    def answer(self, result: CallResult) -> tuple[IntentTranslation | None, str | None]: ...


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

    def __init__(self, inputs: RunInputs, sink: EventSink, sessions_dir: Path | None) -> None:
        self.inputs, self.sink, self.sessions_dir = inputs, sink, sessions_dir
        self.frame, self.scenario, self.config, self.facts = inputs.frame, inputs.scenario, inputs.config, inputs.facts
        validate_scenario(self.scenario, self.facts, self.config.scenario_limits)
        check_preset_points(self.config, self.facts)
        self.triggers = run_triggers(self.scenario, self.config)
        self.tick = tick_interval(self.config)
        self.instructions_at = _instructions_by_step(self.scenario)
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
        """Write one event; ``sequence`` advances only once the record is written (``events_written``)."""
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
    """One loop for every configuration. Renderers are injectable (tests use cheap fakes)."""

    def __init__(
        self,
        *,
        call: TranslationCall,
        detector: DetectorLayer | None = None,
        detector_renderer: Renderer = render_detector_compact,
        translation_renderer: Renderer = render_lag17_jsonl,
        policy: PolicyMaterial | None = None,
        sessions_dir: Path | None = None,
        retries: RetryJournal | None = None,
    ) -> None:
        self._call = call
        self._detector = detector
        self._detector_renderer = detector_renderer
        self._translation_renderer = translation_renderer
        self._policy = policy
        self._sessions_dir = sessions_dir
        self._retries = retries  # failed attempts of a retried translation call (one-shot)

    def run(self, *, frame: pd.DataFrame, scenario: Scenario, config: RunConfig, sink: EventSink) -> TranslatorSummary:
        inputs = RunInputs(frame, scenario, episode_facts(frame), config, self._translation_renderer, self._policy)
        if config.online and self._detector is None:
            raise ValueError("an online run needs a detector layer")
        run = _Run(inputs, sink, self._sessions_dir)
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
        if run.tick is not None and is_tick(step, run.tick) and state.previous is not None:
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

    def _drain_retries(self, layer: str) -> tuple[dict[str, Any], ...]:
        if layer != LAYER_TRANSLATION or self._retries is None:
            return ()
        return self._retries.drain()

    def _answered(self, run: _Run, layer: str, call_id: str, latency_s: float, raw: Any) -> None:
        retries = self._drain_retries(layer)
        run.sink.write_response(_response_record(call_id, layer, RESPONSE_OK, latency_s, raw, None, retries))
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
        retries = self._drain_retries(layer)
        run.sink.write_response(_response_record(call_id, layer, status, latency_s, raw, error, retries))
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
        target, previous, detector = state.target, state.previous, self._detector
        if previous is None or detector is None or run.config.detector is None:
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
        state_text = detector.state_text(context)
        payload = detector.request_payload(state_text)
        request_sha256 = self._write_request(run, call_id, LAYER_DETECTOR, step, target, payload)
        result, latency_s = self._send(run, LAYER_DETECTOR, call_id, lambda: detector.detect(state_text))
        self._answered(run, LAYER_DETECTOR, call_id, latency_s, result.reply.raw)
        config = run.config.detector
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
                "question": detector.question,
            },
            "output": result.output_json(),
            "raw_response": result.reply.raw,
            "model": {
                "backend": config.backend,
                "model": config.model,
                "revision": config.revision,
                "served_model_name": config.served_model_name,
                "response_model": result.reply.model,
            },
            "usage": result.reply.usage,
        }
        run.emit(RECORD_DETECTOR_CALL, step, target, body)
        if not result.triggered:
            return None
        return TriggerReason(result.label, DETECTOR_SOURCE, call_id, result.probability)

    def _slot_ref(self, run: _Run, call_id: str, step: int, target: TargetRef) -> SlotRef:
        directory = None if run.sessions_dir is None else run.sessions_dir / f"s{step:04d}_{target.target_id}"
        return SlotRef(call_id, step, target, directory)

    def _prepare(self, run: _Run, values: dict[str, object], slot: SlotRef) -> tuple[dict[str, Any], Any]:
        """The call's request body and prepared state; a refused sandbox stops the run before any send."""
        try:
            return self._call.prepare(values, slot)
        except SandboxRefused as exc:
            error = {"layer": LAYER_TRANSLATION, "call_id": slot.call_id, "type": type(exc).__name__}
            raise _Stop(STATUS_FAILED, error={**error, "message": str(exc), "uncertain": False}) from exc

    def _translate(
        self, run: _Run, step: int, state: TargetState, reasons: tuple[TriggerReason, ...], started: float
    ) -> None:
        target = state.target
        run.check_cap(LAYER_TRANSLATION, target, step)
        current_id = event_id(run.scenario.scenario_id, target, step)
        slot_state = SlotState(step, target, state.instruction, state.previous)
        trajectory = trajectory_window(run.inputs, step)
        slot = self._slot_ref(run, current_id, step, target)
        payload, prepared = self._prepare(run, slot_values(run.inputs, slot_state, trajectory), slot)
        request_sha256 = self._write_request(run, current_id, LAYER_TRANSLATION, step, target, payload)
        result, latency_s = self._send(run, LAYER_TRANSLATION, current_id, lambda: self._call.run(prepared))
        self._answered(run, LAYER_TRANSLATION, current_id, latency_s, result.raw)
        translation, problem = self._call.answer(result)
        if translation is None and run.config.answer.on_invalid == ON_INVALID_STOP:
            message = (problem or "no answer").removeprefix(INVALID_ANSWER_PREFIX)
            error = {"layer": LAYER_TRANSLATION, "call_id": current_id, "type": INVALID_ANSWER_ERROR}
            raise _Stop(STATUS_FAILED, error={**error, "message": message, "uncertain": False})
        record_input = event_input(run.inputs, slot_state, trajectory)
        if result.observed_inputs is not None:  # code session: the bundle the sandbox read
            record_input = {**record_input, "policy": {**record_input.get("policy", {}), **result.observed_inputs}}
        body = _translation_body(run, current_id, reasons, result, translation, record_input)
        body = {**body, "request_sha256": request_sha256, **({"answer_problem": problem} if problem else {})}
        run.emit(RECORD_TRANSLATION_EVENT, step, target, body, started=started)
        if translation is not None:
            previous = PreviousTranslation(current_id, step, translation)
            run.states = {**run.states, target: replace(state, previous=previous)}


def _translation_body(
    run: _Run,
    current_id: str,
    reasons: tuple[TriggerReason, ...],
    result: CallResult,
    translation: IntentTranslation | None,
    record_input: dict[str, Any],
) -> dict[str, Any]:
    translation_config = run.config.translation
    episode = run.scenario.episode
    output = translation.to_json() if translation is not None else {"intent": "", "evidence": []}
    return {
        "event_id": current_id,
        "episode": {
            "episode_id": episode.episode_id,
            "alias": episode.alias,
            "csv_sha256": run.frame.attrs.get("csv_sha256"),
        },
        "primary_trigger": primary_trigger(reasons),
        "trigger_reasons": [reason.to_json() for reason in reasons],
        "input": record_input,
        "output": output,
        "model": {
            "backend": translation_config.backend,
            "protocol": run.config.prompt.protocol,
            "model": translation_config.model,
            "reasoning_effort": translation_config.reasoning_effort,
            "max_output_tokens": translation_config.max_output_tokens,
            "response_model": result.response_model,
            "response_id": result.response_id,
        },
        "usage": result.usage,
    }


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
    call_id: str,
    layer: str,
    status: str,
    latency_s: float,
    raw: Any,
    error: dict[str, str] | None,
    retries: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """One ``responses.jsonl`` line; ``retries`` (failed attempts before this outcome) only when there were any."""
    record = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "call_id": call_id,
        "layer": layer,
        "status": status,
        "latency_s": latency_s,
        "raw": raw,
        "error": error,
    }
    return {**record, "retries": list(retries)} if retries else record
