"""Translator evaluation: machine checks (no score; contribution is never judged by machine).

1. ``check_generation_inputs``: the time-range (leakage) check of every model call of both layers.
2. ``match_points``: a reference point's segment is ``[step, next start/change point of the same
   target)``, ending at the target's down step or the episode end; it matches the first same-kind
   detector event inside it (no window).
3. ``machine_results``: elements, delay, missed / duplicate / excess events, lead before the action
   onset, request ``processing_s`` against the deadline, question interval and utterance count. A
   ``null`` threshold gives the raw measurement and ``not_evaluated/<key>_unset``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mtd_rl_xavier.lag.rendering import SECONDS_DECIMALS, STEP_SECONDS

from .assess_data import (
    ASPECT_OVERALL,
    EVALUATOR_MACHINE,
    LAYER_OF_RECORD,
    POINT_CHANGE,
    POINT_REQUEST,
    POINT_START,
    REASON_FORCED_TRIGGER,
    RECORD_DETECTOR_CALL,
    RECORD_TRANSLATION_EVENT,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUATED,
    TRIGGER_INITIAL,
    Annotation,
    AssessConfig,
    ReferencePoint,
    TranslatorRun,
    canonical_json,
    is_int,
    point_kinds,
    sha256_text,
    strict_json_loads,
    target_end_step,
    target_key,
)
from .assess_store import event_subject, invalidated, overall_verdict, result_record, verdict

# --- 1. time-range check of the generation inputs --------------------------------------------------

CHECK_OK, CHECK_LEAK, CHECK_ERROR = "ok", "leak", "error"
LEAKAGE_FILE = "leakage.jsonl"
TRAJECTORY_LINE_PREFIX = '{"step":'  # both renderers' lines start so (compact JSON, "step" first)
ACTION_KEY_SUFFIXES = ("_action", "_action_id")
# Problems that prove information after t reached the call; any other problem is an error (the
# input could not be verified).
LEAK_CHECKS = frozenset(
    {"trajectory_future", "future_message", "previous_translation", "run_identifier", "action_at_t"}
)
INTERVAL_DECIMALS = 3


@dataclass(frozen=True)
class CallCheck:
    call_id: str
    layer: str
    target: dict[str, str]
    step: int
    visible_end_step: int | None  # latest trajectory step found in the payload
    request_sha256: str | None
    status: str  # ok | leak | error
    problems: tuple[dict[str, str], ...]  # {check, detail}

    def to_json(self, *, retrospective: bool, batch_id: str) -> dict[str, Any]:
        return {
            "batch_id": batch_id,
            "call_id": self.call_id,
            "layer": self.layer,
            "target": self.target,
            "step": self.step,
            "visible_end_step": self.visible_end_step,
            "request_sha256": self.request_sha256,
            "retrospective": retrospective,
            "status": self.status,
            "problems": list(self.problems),
        }


def _problem(check: str, detail: str) -> dict[str, str]:
    return {"check": check, "detail": detail}


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value, keys included, depth first (all text a payload carries)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in (str(key), *_strings(item))]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def record_call_id(record: dict[str, Any]) -> str:
    return record["event_id"] if record["record_type"] == RECORD_TRANSLATION_EVENT else record["call_id"]


def listed_messages(record: dict[str, Any]) -> list[tuple[str, int, str]]:
    """(label, step, text) of every instruction, alarm and question an event record says it showed."""
    payload = record["input"]
    items = [("instruction", item["issue_step"], item["text"]) for item in payload["instructions_in_effect"]]
    if record["record_type"] == RECORD_DETECTOR_CALL:
        return [*items, *((item["kind"], item["step"], item["text"]) for item in payload["messages"])]
    items += [("alarm", item["step"], item["text"]) for item in payload["alarms"]]
    question = payload["question"]
    return [*items, ("question", question["step"], question["text"])] if question else items


def _trajectory_lines(texts: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    lines, problems = [], []
    for line in (line.strip() for text in texts for line in text.split("\n")):
        if not line.startswith(TRAJECTORY_LINE_PREFIX):
            continue
        try:
            parsed = strict_json_loads(line)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and is_int(parsed.get("step")) and is_int(parsed.get("relative_step")):
            lines.append(parsed)
        else:
            problems.append(_problem("trajectory_line", "a trajectory line does not parse with step and relative_step"))
    return lines, problems


def _has_action(line: dict[str, Any]) -> bool:
    items = [*(line.get("entities") or []), *(line.get("objects") or [])]
    for item in (entry for entry in items if isinstance(entry, dict)):
        for key, value in item.items():
            if key == "action" and isinstance(value, list) and any(part is not None for part in value):
                return True
            if key.endswith(ACTION_KEY_SUFFIXES) and value is not None:
                return True
    return False


def _record_problems(request: dict[str, Any], record: dict[str, Any], joined: str) -> list[dict[str, str]]:
    """The request must be the one its event record describes, and show nothing issued after t."""
    problems = []
    if LAYER_OF_RECORD[record["record_type"]] != request["layer"]:
        problems.append(_problem("join", f"layer {request['layer']!r} does not match a {record['record_type']}"))
    if record["step"] != request["step"] or record["target"] != request["target"]:
        problems.append(_problem("join", "the request's step or target differs from its event record"))
    if sha256_text(canonical_json(request["payload"])) != record["request_sha256"]:
        problems.append(_problem("request_sha256", "the payload does not match the record's request_sha256"))
    trajectory = record["input"]["trajectory"]
    if sha256_text(trajectory["text"]) != trajectory["sha256"] or trajectory["text"] not in joined:
        problems.append(_problem("trajectory_record", "the recorded trajectory text is not the one in the payload"))
    for label, issued, _ in listed_messages(record):
        if issued > request["step"]:
            problems.append(_problem("future_message", f"{label} issued at step {issued} > t"))
    previous = record["input"]["previous_translation"]
    if previous is not None and previous["step"] >= request["step"]:
        problems.append(_problem("previous_translation", f"previous translation from step {previous['step']} >= t"))
    return problems


def _check_call(
    request: dict[str, Any],
    record: dict[str, Any] | None,
    counts: tuple[int, int],  # (event records, request lines) with this call id
    run: TranslatorRun,
    config: AssessConfig,
    known: frozenset[tuple[int, str]],
) -> CallCheck:
    step, texts = request["step"], _strings(request["payload"])
    joined = "\n".join(texts)
    problems: list[dict[str, str]] = []
    if counts != (1, 1):
        problems.append(_problem("join", f"{counts[0]} event records and {counts[1]} request lines; one each expected"))
    if record is not None:
        problems += _record_problems(request, record, joined)
    if run.future_steps is None:
        problems.append(_problem("visibility", "run.json declares no valid config.pipeline.visibility"))
    future = run.future_steps or 0
    lines, line_problems = _trajectory_lines(texts)
    problems += line_problems
    if not lines:
        problems.append(_problem("trajectory_line", "no trajectory line found in the payload"))
    for line in lines:
        if line["step"] > step + future or line["relative_step"] > future:
            problems.append(_problem("trajectory_future", f"trajectory step {line['step']} is after the visible end"))
        if not config.allow_action_at_t and line["step"] >= step and _has_action(line):
            problems.append(
                _problem("action_at_t", f"an action at step {line['step']} is shown (allow_action_at_t false)")
            )
    # A message text the run showed only later must not appear in this payload.
    allowed = {text for issued, text in known if issued <= step}
    for issued, text in sorted(known):
        if issued > step and text not in allowed and text in joined:
            problems.append(
                _problem("future_message", f"the text of a message issued at step {issued} > t appears in the payload")
            )
    for name, identifier in (("episode_id", run.episode_id), ("scenario_id", run.scenario_id)):
        if identifier in joined:
            problems.append(_problem("run_identifier", f"the run's own {name} appears in the payload"))
    leaked = any(problem["check"] in LEAK_CHECKS for problem in problems)
    return CallCheck(
        call_id=request["call_id"],
        layer=request["layer"],
        target=request["target"],
        step=step,
        visible_end_step=max((line["step"] for line in lines), default=None),
        request_sha256=record["request_sha256"] if record is not None else None,
        status=CHECK_LEAK if leaked else CHECK_ERROR if problems else CHECK_OK,
        problems=tuple(problems),
    )


def check_generation_inputs(run: TranslatorRun, config: AssessConfig) -> tuple[CallCheck, ...]:
    """One result per model call of both layers, plus one ``error`` per event record with no request.

    The generation input is the exact ``requests.jsonl`` payload. It must join exactly one event
    record whose ``request_sha256`` it matches; every trajectory line in it must lie at or before
    ``t`` (``t + future_steps`` for a declared retrospective run); every instruction, alarm and
    question shown must be issued at or before ``t``; the previous translation must be older than
    ``t``; and the run's own episode and scenario ids must not appear. ``run.json`` must declare
    the visibility.
    """
    calls = [r for r in run.records if r["record_type"] in LAYER_OF_RECORD]
    record_counts = Counter(record_call_id(record) for record in calls)
    by_id = {record_call_id(record): record for record in calls}  # used only for ids with one record
    request_counts = Counter(request["call_id"] for request in run.requests)
    # Every (step, text) any record of the run showed: the texts a call must not carry early.
    known = frozenset((step, text) for record in calls for _, step, text in listed_messages(record) if text.strip())
    checks = []
    for request in run.requests:
        call_id = request["call_id"]
        record = by_id[call_id] if record_counts[call_id] == 1 else None
        counts = (record_counts[call_id], request_counts[call_id])
        checks.append(_check_call(request, record, counts, run, config, known))
    orphans = [
        CallCheck(
            call_id=record_call_id(record),
            layer=LAYER_OF_RECORD[record["record_type"]],
            target=record["target"],
            step=record["step"],
            visible_end_step=None,
            request_sha256=record["request_sha256"],
            status=CHECK_ERROR,
            problems=(_problem("join", "no request line for this event record"),),
        )
        for record in calls
        if record_call_id(record) not in request_counts
    ]
    return (*checks, *orphans)


def invalid_targets(checks: tuple[CallCheck, ...]) -> frozenset[tuple[str, str]]:
    """Targets with a leak or error on any call: all their results are ``not_evaluated/generation_input_invalid``."""
    return frozenset(target_key(check.target) for check in checks if check.status != CHECK_OK)


# --- 2. reference points and segments --------------------------------------------------------------


@dataclass(frozen=True)
class PointMatch:
    point: ReferencePoint
    segment_end: int  # exclusive
    forced: bool  # the point lies at the target's forced initial step
    events: tuple[dict[str, Any], ...]  # same-kind detector events inside the segment, in step order

    @property
    def matched(self) -> dict[str, Any] | None:
        return self.events[0] if self.events else None


def kind_events(run: TranslatorRun, target: tuple[str, str], trigger_kind: str) -> list[dict[str, Any]]:
    """Translation events of the target with a reason of this trigger kind, in log order."""
    return [
        event
        for event in run.events
        if target_key(event["target"]) == target and any(r["kind"] == trigger_kind for r in event["trigger_reasons"])
    ]


def match_points(annotation: Annotation | None, run: TranslatorRun) -> tuple[PointMatch, ...]:
    if annotation is None:
        return ()
    points, matches = annotation.points, []
    for point in sorted(points, key=lambda item: (item.target, item.step)):
        later = [other.step for other in points if other.target == point.target and other.step > point.step]
        end = min([*later, target_end_step(run, point.target)])
        events = tuple(e for e in kind_events(run, point.target, point.kind) if point.step <= e["step"] < end)
        initial = kind_events(run, point.target, TRIGGER_INITIAL)
        forced = bool(initial) and initial[0]["step"] == point.step
        matches.append(PointMatch(point, end, forced, events))
    return tuple(matches)


def excess_events(run: TranslatorRun, matches: tuple[PointMatch, ...], target: tuple[str, str]) -> dict[str, int]:
    """Detector start/change events of an annotated target that lie in no segment of their own kind."""
    counts = {}
    for kind in (POINT_START, POINT_CHANGE):
        segments = [(m.point.step, m.segment_end) for m in matches if m.point.target == target and m.point.kind == kind]
        events = kind_events(run, target, kind)
        counts[kind] = sum(1 for e in events if not any(a <= e["step"] < b for a, b in segments))
    return counts


# --- 3. machine results ----------------------------------------------------------------------------

MACHINE_VERSION = "assess-machine-v1"
REASON_NO_MATCHED_EVENT = "no_matched_event"
REASON_NO_ACTION_ONSET = "no_action_onset"
REASON_FEWER_THAN_TWO_REQUESTS = "fewer_than_two_requests"


@dataclass(frozen=True)
class MachineContext:
    run: TranslatorRun
    config: AssessConfig
    batch_id: str
    invalid: frozenset[tuple[str, str]]


def _threshold(measurement: dict[str, Any], key: str, limit: float | None, passes: Callable[[Any], bool]) -> dict:
    """A verdict against a config threshold; ``None`` keeps the measurement as ``<key>_unset``."""
    if limit is None:
        return verdict(STATUS_NOT_EVALUATED, measurement=measurement, reason=f"{key}_unset")
    return verdict(STATUS_EVALUATED, measurement={**measurement, key: limit, "passed": bool(passes(limit))})


def _measured(measurement: dict[str, Any]) -> dict[str, Any]:
    return verdict(STATUS_EVALUATED, measurement=measurement)


def _machine_record(
    context: MachineContext,
    subject_key: str,
    subject: dict[str, Any],
    criterion_id: str,
    outcome: dict[str, Any],
    event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_key(subject["target"]) in context.invalid:
        outcome = invalidated(outcome)
    aspect = ASPECT_OVERALL if criterion_id == ASPECT_OVERALL else criterion_id.split(".", 1)[0]
    return result_record(
        result_id=f"{EVALUATOR_MACHINE}:{context.batch_id}:{subject_key}:{criterion_id}",
        evaluator={"kind": EVALUATOR_MACHINE, "version": MACHINE_VERSION},
        subject=subject,
        event=event,
        criterion=(criterion_id, aspect, None),
        outcome=outcome,
        config=context.config,
        retrospective=context.run.retrospective,
    )


def _subject(
    run: TranslatorRun,
    target: tuple[str, str],
    kinds: tuple[str, ...],
    *,
    step: int | None = None,
    event_id: str | None = None,
    point_id: str | None = None,
) -> dict[str, Any]:
    """The subject of a reference point (``step`` set) or of a whole target (no step)."""
    first = next((event for event in run.events if target_key(event["target"]) == target), None)
    return {
        "scenario_id": run.scenario_id,
        "episode_id": run.episode_id,
        "target": {"kind": target[0], "id": target[1]},
        "target_aircraft": list(first["input"]["target_aircraft"]) if first is not None else [],
        "step": step,
        "time_s": round(step * STEP_SECONDS, SECONDS_DECIMALS) if step is not None else None,
        "event_id": event_id,
        "point_kinds": list(kinds),
        "reference_point_id": point_id,
    }


def _event_results(context: MachineContext, event: dict[str, Any]) -> list[dict[str, Any]]:
    output, event_id = event["output"], event["event_id"]
    blank = sum(1 for item in output["evidence"] if not item.strip())
    elements = {
        "elements.intent_present": {
            "intent_chars": len(output["intent"].strip()),
            "passed": bool(output["intent"].strip()),
        },
        "elements.evidence_present": {
            "evidence_items": len(output["evidence"]),
            "blank_items": blank,
            "passed": bool(output["evidence"]) and blank == 0,
        },
    }
    subject = event_subject(event, context.run)
    records = [
        _machine_record(context, event_id, subject, criterion_id, _measured(measurement), event)
        for criterion_id, measurement in elements.items()
    ]
    if POINT_REQUEST in point_kinds(event):
        measurement = {"processing_s": event["processing_s"], "latency_s": context.run.latency_s.get(event_id)}
        outcome = _threshold(
            measurement,
            "request_deadline_s",
            context.config.timing.request_deadline_s,
            lambda limit: event["processing_s"] <= limit,
        )
        request_subject = event_subject(event, context.run, (POINT_REQUEST,))
        records.append(_machine_record(context, event_id, request_subject, "timing.request_latency", outcome, event))
    records.append(_machine_record(context, event_id, subject, ASPECT_OVERALL, overall_verdict(), event))
    return records


def _point_outcomes(config: AssessConfig, match: PointMatch) -> dict[str, dict[str, Any]]:
    point, event, timing = match.point, match.matched, config.timing
    delay = event["step"] - point.step if event is not None else None
    count = len(match.events)
    outcomes = {
        "timing.point_matched": _measured(
            {"matched": event is not None, "delay_steps": delay, "segment_end_step": match.segment_end}
        ),
        "timing.segment_events": _threshold(
            {"events_in_segment": count, "duplicates": max(0, count - 1)},
            "max_events_per_segment",
            timing.max_events_per_segment,
            lambda limit: count <= limit,
        ),
    }
    delay_key = "timing.change_delay" if point.kind == POINT_CHANGE else "timing.start_lead"
    if event is None:
        return {**outcomes, delay_key: verdict(STATUS_NOT_EVALUATED, reason=REASON_NO_MATCHED_EVENT)}
    if point.kind == POINT_CHANGE:
        limit = timing.change_max_delay_steps
        changed = _threshold({"delay_steps": delay}, "change_max_delay_steps", limit, lambda value: delay <= value)
        return {**outcomes, delay_key: changed}
    if point.action_onset_step is None:
        return {**outcomes, delay_key: verdict(STATUS_NOT_EVALUATED, reason=REASON_NO_ACTION_ONSET)}
    lead = point.action_onset_step - event["step"]
    limit = timing.start_min_lead_steps
    return {
        **outcomes,
        delay_key: _threshold({"lead_steps": lead}, "start_min_lead_steps", limit, lambda value: lead >= value),
    }


def _point_results(context: MachineContext, match: PointMatch) -> list[dict[str, Any]]:
    point, event = match.point, match.matched
    subject = _subject(
        context.run,
        point.target,
        (point.kind,),
        step=point.step,
        event_id=event["event_id"] if event is not None else None,
        point_id=point.point_id,
    )
    outcomes = _point_outcomes(context.config, match)
    if match.forced:
        outcomes = {key: verdict(STATUS_NOT_EVALUATED, reason=REASON_FORCED_TRIGGER) for key in outcomes}
    return [
        _machine_record(context, f"point:{point.point_id}", subject, criterion_id, outcome, event)
        for criterion_id, outcome in outcomes.items()
    ]


def _interval_outcome(config: AssessConfig, request_times: list[float]) -> dict[str, Any]:
    if len(request_times) < 2:
        measurement = {"requests": len(request_times)}
        return verdict(STATUS_NOT_EVALUATED, measurement=measurement, reason=REASON_FEWER_THAN_TWO_REQUESTS)
    intervals = [round(b - a, INTERVAL_DECIMALS) for a, b in zip(request_times[:-1], request_times[1:], strict=True)]
    limit = config.timing.min_question_interval_s
    measurement = {"requests": len(request_times), "intervals_s": intervals, "min_interval_s": min(intervals)}
    if limit is not None:
        measurement = {**measurement, "violations": sum(1 for interval in intervals if interval < limit)}
    return _threshold(measurement, "min_question_interval_s", limit, lambda value: min(intervals) >= value)


def _target_results(
    context: MachineContext, target: tuple[str, str], matches: tuple[PointMatch, ...]
) -> list[dict[str, Any]]:
    run, key = context.run, f"target:{target[0]}:{target[1]}"
    events = [event for event in run.events if target_key(event["target"]) == target]
    utterances = len(events)
    limit = context.config.timing.max_utterances_per_target
    counted = _threshold(
        {"utterances": utterances}, "max_utterances_per_target", limit, lambda value: utterances <= value
    )
    request_times = sorted(e["time_s"] for e in events if POINT_REQUEST in point_kinds(e))
    records = [
        _machine_record(context, key, _subject(run, target, ()), "timing.utterance_count", counted),
        _machine_record(
            context,
            key,
            _subject(run, target, (POINT_REQUEST,)),
            "timing.question_interval",
            _interval_outcome(context.config, request_times),
        ),
    ]
    if any(match.point.target == target for match in matches):
        counts = excess_events(run, matches, target)
        subject = _subject(run, target, (POINT_START, POINT_CHANGE))
        records.append(_machine_record(context, key, subject, "timing.excess_events", _measured(counts)))
    return records


def machine_results(
    run: TranslatorRun,
    config: AssessConfig,
    matches: tuple[PointMatch, ...],
    checks: tuple[CallCheck, ...],
    batch_id: str,
) -> list[dict[str, Any]]:
    """Every machine record of the run: per event, per reference point, per target."""
    context = MachineContext(run, config, batch_id, invalid_targets(checks))
    records = [record for event in run.events for record in _event_results(context, event)]
    records += [record for match in matches for record in _point_results(context, match)]
    targets = sorted({target_key(event["target"]) for event in run.events} | {m.point.target for m in matches})
    records += [record for target in targets for record in _target_results(context, target, matches)]
    return records
