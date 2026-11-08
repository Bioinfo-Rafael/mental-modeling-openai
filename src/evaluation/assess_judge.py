"""Translator evaluation: LLM as a Judge on the criteria humans rate (one packet, one call per event).

- Every word of the request comes from the config's prompt file (``config.prompt``,
  ``configs/evaluation/prompts/``); this module renders the material into its templates.
- The packet is the text humans read too. Every free text is one JSON string literal, so it cannot
  leave its envelope (the ``judge-text-v2`` injection fence).
- Start/change events get a ``FUTURE (evaluation only)`` block covering ``(t, end)``, ``end`` being
  the earliest of the target's next annotated start/change point, next instruction or alarm, down
  step, the episode end and ``t + 1 + future_evidence.max_steps``: behaviour after a later external
  factor never counts against a start explanation. Request-only events get none.
- ``run_judge_calls``: cap check -> request line (fsync) -> one call -> response line -> parse ->
  result rows. Any failure stops the run; nothing is repaired or retried.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

import pandas as pd

from translate.lag.dataset import AIRCRAFT
from translate.lag.rendering import render_detector_compact
from translate.llm.openai_responses import (
    OpenAIClientError,
    OpenAIRequest,
    OpenAIResponsesClient,
    build_responses_body,
    input_text,
)

from .assess_checks import listed_messages
from .assess_data import (
    EVALUATOR_JUDGE,
    JUDGEMENT_CANNOT_JUDGE,
    JUDGEMENT_NOT_APPLICABLE,
    LAYER_OF_RECORD,
    POINT_CHANGE,
    POINT_START,
    REFERENCE_FUTURE,
    STATUS_NOT_EVALUATED,
    Annotation,
    AssessConfig,
    AssessError,
    Criterion,
    EventCriteria,
    TranslatorRun,
    canonical_json,
    event_criteria,
    point_kind,
    point_kinds,
    sha256_text,
    strict_json_loads,
    target_end_step,
    target_key,
)
from .assess_store import (
    OVERALL,
    criterion_key,
    event_subject,
    judgement_verdict,
    overall_verdict,
    result_record,
    verdict,
)
from .prompt_files import JudgePrompt

OUTPUT_SCHEMA_NAME = "assess_verdict"
JUDGE_REQUESTS, JUDGE_RESPONSES = "judge_requests.jsonl", "judge_responses.jsonl"
JOURNAL_SCHEMA_VERSION = 1
LATENCY_DECIMALS = 3
UNICODE_LINE_BREAK_ESCAPES = {"\x85": "\\u0085", " ": "\\u2028", " ": "\\u2029"}
FUTURE_KINDS = frozenset({POINT_START, POINT_CHANGE})
REASON_CALL_FAILED, REASON_CALL_UNCERTAIN = "judge_call_failed", "judge_call_uncertain"
REASON_OUTPUT_INVALID = "judge_output_invalid"
STATUS_COMPLETED, STATUS_FAILED, STATUS_UNCERTAIN, STATUS_CAP_REACHED = (
    "completed",
    "failed",
    "uncertain",
    "cap_reached",
)
UNEXPECTED_MESSAGE = "unexpected exception; details withheld"
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
INTERRUPTIONS = (Exception, KeyboardInterrupt)  # Ctrl-C during a call makes that call uncertain


class JudgeInputError(AssessError):
    """A packet cannot be built (for example the FUTURE block needs the episode CSV)."""


class JudgeOutputError(AssessError):
    """The judge answered, but not with one valid verdict. Never repaired or retried."""


class JudgeCallError(Exception):
    """A judge call failed. ``uncertain``: it may have been billed with an unknown outcome; never resend."""

    def __init__(self, message: str, *, error_type: str, uncertain: bool, raw: Any = None, usage: Any = None) -> None:
        super().__init__(message)
        self.error_type, self.uncertain, self.raw, self.usage = error_type, uncertain, raw, usage


def quoted(text: str) -> str:
    """Untrusted free text as one JSON string literal on one line."""
    literal = json.dumps(text, ensure_ascii=False)
    for char, escape in UNICODE_LINE_BREAK_ESCAPES.items():
        literal = literal.replace(char, escape)
    return literal


# --- FUTURE window ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FutureWindow:
    end_step: int  # exclusive
    cut: str  # what ended it
    steps: tuple[int, ...]  # strided steps shown, all in (t, end_step)


def future_window(
    event: dict[str, Any], run: TranslatorRun, annotation: Annotation | None, config: AssessConfig
) -> FutureWindow:
    """The FUTURE bound of a start/change event (see the module docstring). Ties name the external factor."""
    step, target = event["step"], target_key(event["target"])
    # Instructions and alarms after t that any model call of this target recorded showing.
    calls = [r for r in run.records if r["record_type"] in LAYER_OF_RECORD and target_key(r["target"]) == target]
    later = [(label, issued) for record in calls for label, issued, _ in listed_messages(record) if issued > step]
    points = [p.step for p in annotation.points if p.target == target and p.step > step] if annotation else []
    candidates = [
        ("next_alarm", min((issued for label, issued in later if label == "alarm"), default=None)),
        ("next_instruction", min((issued for label, issued in later if label == "instruction"), default=None)),
        ("next_annotated_point", min(points, default=None)),
        ("target_down", target_end_step(run, target)),
        ("episode_end", run.last_step + 1),
        ("max_steps", step + 1 + config.future_evidence.max_steps),
    ]
    cut, end = min(((name, value) for name, value in candidates if value is not None), key=lambda item: item[1])
    return FutureWindow(end, cut, tuple(range(step + 1, end, config.future_evidence.stride_steps)))


def _future_lines(event: dict[str, Any], window: FutureWindow, frame: pd.DataFrame, config: AssessConfig) -> list[str]:
    if not window.steps:
        return []
    own = tuple(event["input"]["target_aircraft"])
    rendered = render_detector_compact(
        frame,
        start_step=event["step"],
        end_step=window.end_step - 1,
        anchor_step=event["step"],
        object_ids=(*own, *(aircraft for aircraft in AIRCRAFT if aircraft not in own)),
        fields=config.future_evidence.fields,
        float_precision=config.future_evidence.float_precision,
    )
    shown = set(window.steps)
    return [line for line in rendered.text.split("\n") if strict_json_loads(line)["step"] in shown]


# --- packet -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Packet:
    event_id: str
    point_kinds: tuple[str, ...]
    criteria: EventCriteria
    text: str
    sha256: str
    future_end_step: int | None  # exclusive end of the FUTURE block; None without one

    def to_json(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "point_kinds": list(self.point_kinds),
            "asked_criteria": [criterion.criterion_id for criterion in self.criteria.asked],
            "excluded_criteria": {criterion.criterion_id: reason for criterion, reason in self.criteria.excluded},
            "future_end_step": self.future_end_step,
            "sha256": self.sha256,
            "text": self.text,
        }


def _label(target: dict[str, str]) -> str:
    return f"{target['kind']} {target['id']}"


def _joined(prompt: JudgePrompt, block: str, texts: list[str]) -> str:
    return prompt.word(block, "joiner").join(texts)


def _trigger_reasons(event: dict[str, Any], config: AssessConfig) -> str:
    prompt = config.prompt
    reasons = [
        prompt.part(
            "trigger_reason", "item", kind=r["kind"], point_kind=point_kind(r, event, config), source=r["source"]
        )
        for r in event["trigger_reasons"]
    ]
    return _joined(prompt, "trigger_reason", reasons)


def _previous_translation(prompt: JudgePrompt, previous: dict[str, Any] | None) -> str:
    if previous is None:
        return prompt.word("previous_translation", "empty")
    evidence = prompt.word("previous_translation", "evidence_joiner").join(
        prompt.part("previous_translation", "evidence_item", item=quoted(item)) for item in previous.get("evidence", [])
    )
    intent = quoted(previous.get("intent", ""))
    return prompt.part("previous_translation", "present", step=previous["step"], intent=intent, evidence=evidence)


def _event_values(event: dict[str, Any], kinds: tuple[str, ...], config: AssessConfig) -> dict[str, Any]:
    """Header, the inputs the Translator was given (instructions, alarms, question, previous), the output."""
    payload, question, output = event["input"], event["input"]["question"], event["output"]
    instructions = sorted(payload["instructions_in_effect"], key=lambda i: (i["issue_step"], i["instruction_id"]))
    trajectory, prompt = payload["trajectory"], config.prompt
    return {
        "target": _label(event["target"]),
        "target_aircraft": ", ".join(payload["target_aircraft"]),
        "step": event["step"],
        "time_s": event["time_s"],
        "point_kinds": ", ".join(kinds),
        "trigger_reasons": _trigger_reasons(event, config),
        "instructions": prompt.items(
            "instructions",
            [
                {"issue_step": i["issue_step"], "target": _label(i["target"]), "text": quoted(i["text"])}
                for i in instructions
            ],
        ),
        "alarms": prompt.items("alarms", [{"step": a["step"], "text": quoted(a["text"])} for a in payload["alarms"]]),
        "question": prompt.optional(
            "question", {"step": question["step"], "text": quoted(question["text"])} if question else None
        ),
        "previous_translation": _previous_translation(prompt, payload["previous_translation"]),
        "window_start": trajectory["start_step"],
        "window_end": trajectory["end_step"],
        "renderer": trajectory["renderer"],
        "trajectory": trajectory["text"],
        "intent": quoted(output["intent"]),
        "evidence": prompt.items("evidence", [{"item": quoted(item)} for item in output["evidence"]]),
    }


def _timeline(items: list[dict[str, Any]], current: dict[str, Any], config: AssessConfig) -> str:
    prompt = config.prompt
    lines = [
        prompt.part(
            "timeline",
            "item",
            step=item["step"],
            time_s=item["time_s"],
            kinds=", ".join(r["kind"] for r in item["trigger_reasons"]),
            point_kinds=", ".join(point_kinds(item, config)),
            processing_s=item["processing_s"],
            mark=prompt.word("timeline", "this_event") if item["event_id"] == current["event_id"] else "",
        )
        for item in items
    ]
    return _joined(prompt, "timeline", lines)


def _future(
    event: dict[str, Any],
    own: list[dict[str, Any]],
    window: FutureWindow | None,
    frame: pd.DataFrame | None,
    config: AssessConfig,
) -> str:
    prompt = config.prompt
    if window is None:
        return prompt.word("future", "empty")
    if frame is None:
        raise JudgeInputError("the FUTURE block of a start/change event needs the episode CSV")
    later = [item for item in own if event["step"] < item["step"] < window.end_step]
    rows = _future_lines(event, window, frame, config)
    return prompt.part(
        "future",
        "present",
        last_step=window.end_step - 1,
        end_step=window.end_step,
        cut=window.cut,
        stride_steps=config.future_evidence.stride_steps,
        fields=", ".join(config.future_evidence.fields),
        rows=prompt.word("future", "row_joiner").join(rows) if rows else prompt.word("future", "no_rows"),
        later=_timeline(later, event, config) if later else prompt.word("future", "no_later"),
    )


def _criterion(prompt: JudgePrompt, criterion: Criterion, has_future: bool) -> str:
    """The criterion, its anchors from the top level down (D65), then its own judgement boundaries."""
    notes = [criterion.aspect]
    if criterion.conditional:
        notes.append(prompt.word("criterion", "conditional"))
    if criterion.reference:
        notes.append(prompt.part("criterion", "reference", reference=criterion.reference))
    if has_future:
        may = criterion.reference == REFERENCE_FUTURE or criterion.aspect == "timing"
        notes.append(prompt.word("criterion", "may_use_future" if may else "must_not_use_future"))
    anchors = "".join(
        prompt.part("criterion", "anchor", value=value, anchor=anchor)
        for value, anchor in reversed(criterion.levels.items())
    )
    boundaries = [
        (JUDGEMENT_CANNOT_JUDGE, criterion.cannot_judge),
        (JUDGEMENT_NOT_APPLICABLE, criterion.not_applicable),
    ]
    return prompt.part(
        "criterion",
        "item",
        criterion_id=criterion.criterion_id,
        notes=prompt.word("criterion", "note_joiner").join(notes),
        text=criterion.text,
        anchors=anchors,
        boundaries="".join(prompt.part("criterion", name, text=text) for name, text in boundaries if text is not None),
    )


def rating_rule(config: AssessConfig) -> str:
    return config.prompt.text("rating_rule", rationale_language=config.judge.rationale_language)


def _rubric_values(criteria: EventCriteria, has_future: bool, config: AssessConfig) -> dict[str, str]:
    prompt, scale = config.prompt, config.scale
    excluded = [
        prompt.part("not_rated", "item", criterion_id=criterion.criterion_id, reason=reason)
        for criterion, reason in criteria.excluded
    ]
    not_rated = (
        prompt.part("not_rated", "present", items=_joined(prompt, "not_rated", excluded))
        if excluded
        else prompt.word("not_rated", "empty")
    )
    return {
        "criteria": _joined(prompt, "criterion", [_criterion(prompt, c, has_future) for c in criteria.asked]),
        "not_rated": not_rated,
        "scale_values": _joined(
            prompt, "scale_value", [prompt.part("scale_value", "item", value=v) for v in scale.values]
        ),
        "scale_cannot_judge": scale.cannot_judge,
        "scale_not_applicable": scale.not_applicable,
        "rating_rule": rating_rule(config),
    }


def build_packet(
    event: dict[str, Any],
    *,
    run: TranslatorRun,
    config: AssessConfig,
    annotation: Annotation | None,
    frame: pd.DataFrame | None,
) -> Packet:
    """The prompt's ``user`` text: the event and its inputs, TIMELINE, FUTURE (start/change), the rubric."""
    kinds = point_kinds(event, config)
    criteria = event_criteria(event, config)
    window = future_window(event, run, annotation, config) if FUTURE_KINDS & set(kinds) else None
    own = [item for item in run.events if item["target"] == event["target"]]
    text = config.prompt.text(
        "user",
        **_event_values(event, kinds, config),
        timeline=_timeline([item for item in own if item["step"] <= event["step"]], event, config),
        future=_future(event, own, window, frame, config),
        **_rubric_values(criteria, window is not None, config),
    )
    future_end = window.end_step if window is not None else None
    return Packet(event["event_id"], kinds, criteria, text, sha256_text(text), future_end)


# --- request, output schema and parser -------------------------------------------------------------


def system_prompt(config: AssessConfig) -> str:
    return config.prompt.text(
        "system", rationale_language=config.judge.rationale_language, rating_rule=rating_rule(config)
    )


def output_schema(criteria: tuple[Criterion, ...], config: AssessConfig) -> dict[str, Any]:
    """Strict schema: exactly ``len(criteria)`` items; the rationale comes before the judgement."""
    judgements = [*config.scale.values, JUDGEMENT_CANNOT_JUDGE, JUDGEMENT_NOT_APPLICABLE]
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": {"type": "string", "enum": [criterion.criterion_id for criterion in criteria]},
            "rationale": {"type": "string"},
            "judgement": {"type": "string", "enum": judgements},
        },
        "required": ["criterion_id", "rationale", "judgement"],
    }
    count = len(criteria)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"criteria": {"type": "array", "minItems": count, "maxItems": count, "items": item}},
        "required": ["criteria"],
    }


def build_request(packet: Packet, config: AssessConfig) -> OpenAIRequest:
    return OpenAIRequest(
        model=config.judge.model,
        instructions=system_prompt(config),
        input_items=(input_text(packet.text),),
        schema_name=OUTPUT_SCHEMA_NAME,
        schema=output_schema(packet.criteria.asked, config),
        reasoning_effort=config.judge.reasoning_effort,
        max_output_tokens=config.judge.max_output_tokens,
    )


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def parse_verdict(
    output_text: str, criteria: tuple[Criterion, ...], config: AssessConfig
) -> dict[str, tuple[str, str]]:
    """criterion id -> (judgement, rationale): exactly the asked ids, each once; nothing repaired."""
    try:
        parsed = strict_json_loads(output_text)
    except (TypeError, ValueError) as exc:
        raise JudgeOutputError("judge output is not exactly one JSON object without repeated keys") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"criteria"} or not isinstance(parsed["criteria"], list):
        raise JudgeOutputError('judge output must be {"criteria": [...]}')
    by_id = {criterion.criterion_id: criterion for criterion in criteria}
    found: dict[str, tuple[str, str]] = {}
    for index, item in enumerate(parsed["criteria"]):
        name = f"criteria[{index}]"
        if not isinstance(item, dict) or set(item) != {"criterion_id", "rationale", "judgement"}:
            raise JudgeOutputError(f"{name} must have exactly criterion_id, rationale and judgement")
        criterion = by_id.get(item["criterion_id"]) if isinstance(item["criterion_id"], str) else None
        if criterion is None or criterion.criterion_id in found:
            raise JudgeOutputError(f"{name}.criterion_id is not an asked criterion, or repeats one")
        optional = (JUDGEMENT_NOT_APPLICABLE,) if criterion.conditional else ()
        allowed = (*config.scale.values, JUDGEMENT_CANNOT_JUDGE, *optional)
        if item["judgement"] not in allowed:
            raise JudgeOutputError(f"{name}.judgement must be one of {list(allowed)} for {criterion.criterion_id}")
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip() or not _is_utf8(rationale):
            raise JudgeOutputError(f"{name}.rationale must be a non-blank UTF-8 string")
        found[criterion.criterion_id] = (item["judgement"], rationale)
    missing = [criterion_id for criterion_id in by_id if criterion_id not in found]
    if missing:
        raise JudgeOutputError(f"criteria missing from the verdict: {missing}")
    return found


# --- OpenAI adapter --------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeReply:
    model: str
    response_id: str | None
    output_text: str
    usage: dict[str, int]
    raw: dict[str, Any]


class JudgeModel(Protocol):
    def score(self, request: OpenAIRequest) -> JudgeReply:
        """One call. Raises ``JudgeCallError`` on failure (refusal and incomplete included)."""
        ...


class OpenAIAssessJudge:
    """``JudgeModel`` over ``OpenAIResponsesClient`` (store false, no temperature, no retries)."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def score(self, request: OpenAIRequest) -> JudgeReply:
        try:
            result = self._client.create(request)
        except OpenAIClientError as exc:  # messages are sanitised by the LLM part
            raw = exc.result.raw if exc.result is not None else exc.raw
            usage = asdict(exc.result.usage) if exc.result is not None else None
            raise JudgeCallError(
                str(exc), error_type=type(exc).__name__, uncertain=exc.uncertain, raw=raw, usage=usage
            ) from exc
        return JudgeReply(result.model, result.response_id, result.output_text, asdict(result.usage), result.raw)


# --- the call loop ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannedCall:
    event: dict[str, Any]
    packet: Packet
    request: OpenAIRequest
    payload: dict[str, Any]  # the exact Responses body
    request_sha256: str


def plan_calls(
    events: list[dict[str, Any]],
    *,
    run: TranslatorRun,
    config: AssessConfig,
    annotation: Annotation | None,
    load_frame: Callable[[], pd.DataFrame],
) -> list[PlannedCall]:
    """One call per event, in order. The episode frame is loaded only when a FUTURE block needs it."""
    needs_frame = any(FUTURE_KINDS & set(point_kinds(event, config)) for event in events)
    frame = load_frame() if needs_frame else None
    calls = []
    for event in events:
        packet = build_packet(event, run=run, config=config, annotation=annotation, frame=frame)
        request = build_request(packet, config)
        payload = build_responses_body(request)
        calls.append(PlannedCall(event, packet, request, payload, sha256_text(canonical_json(payload))))
    return calls


class JudgeSink(Protocol):
    def write_request(self, record: dict[str, Any]) -> None: ...  # fsynced before the call is sent
    def write_response(self, record: dict[str, Any]) -> None: ...
    def write_results(self, records: list[dict[str, Any]]) -> None: ...


@dataclass(frozen=True)
class JudgeCounts:
    attempted: int = 0
    judged: int = 0
    failed: int = 0
    uncertain: int = 0


@dataclass(frozen=True)
class JudgeSummary:
    status: str
    counts: JudgeCounts
    usage: dict[str, int]
    unsent: dict[str, Any] | None
    error: dict[str, Any] | None


def judge_records(
    call: PlannedCall,
    *,
    run: TranslatorRun,
    config: AssessConfig,
    batch_id: str,
    reply: JudgeReply | None,
    judgements: dict[str, tuple[str, str]] | None,
    failure_reason: str | None,
) -> list[dict[str, Any]]:
    """One row per asked criterion (the verdict, or ``not_evaluated/<failure>``) plus the overall row."""
    event, criteria = call.event, call.packet.criteria
    evaluator = {
        "kind": EVALUATOR_JUDGE,
        "model": config.judge.model,
        "response_model": reply.model if reply is not None else None,
        "prompt_version": config.prompt.version,
        "prompt_sha256": sha256_text(system_prompt(config)),
        "reasoning_effort": config.judge.reasoning_effort,
        "max_output_tokens": config.judge.max_output_tokens,
        "request_sha256": call.request_sha256,
        "packet_sha256": call.packet.sha256,
        "response_id": reply.response_id if reply is not None else None,
    }
    rows = [
        (
            criterion_key(criterion),
            criteria.kinds_of[criterion.criterion_id],
            judgement_verdict(*judgements[criterion.criterion_id])
            if judgements is not None
            else verdict(STATUS_NOT_EVALUATED, reason=failure_reason),
        )
        for criterion in criteria.asked
    ]
    rows.append((OVERALL, point_kinds(event, config), overall_verdict()))
    return [
        result_record(
            result_id=f"{EVALUATOR_JUDGE}:{batch_id}:{event['event_id']}:{key[0]}",
            evaluator=dict(evaluator),  # one dict per row
            subject=event_subject(event, run, kinds),
            event=event,
            criterion=key,
            outcome=outcome,
            config=config,
            retrospective=run.retrospective,
            future_end_step=call.packet.future_end_step,
        )
        for key, kinds, outcome in rows
    ]


def _journal(call: PlannedCall, batch_id: str, **fields: Any) -> dict[str, Any]:
    return {"schema_version": JOURNAL_SCHEMA_VERSION, "batch_id": batch_id, "call_id": call.event["event_id"], **fields}


def _call_error(exc: BaseException) -> tuple[dict[str, str], bool, Any, Any]:
    """(error, uncertain, raw, usage) of a failed call; anything but ``JudgeCallError`` is uncertain."""
    if isinstance(exc, JudgeCallError):
        return {"type": exc.error_type, "message": str(exc)}, exc.uncertain, exc.raw, exc.usage
    return {"type": f"unexpected:{type(exc).__name__}", "message": UNEXPECTED_MESSAGE}, True, None, None


def _add_usage(usage: dict[str, int], extra: Any) -> dict[str, int]:
    if not isinstance(extra, dict):
        return usage
    return {key: usage[key] + int(extra.get(key, 0) or 0) for key in USAGE_KEYS}


def run_judge_calls(
    planned: tuple[PlannedCall, ...],
    *,
    model: JudgeModel,
    sink: JudgeSink,
    run: TranslatorRun,
    config: AssessConfig,
    batch_id: str,
) -> JudgeSummary:
    """Judge the planned calls in order. Stops before call ``judge.max_calls + 1`` (``cap_reached``)."""
    counts, usage = JudgeCounts(), dict.fromkeys(USAGE_KEYS, 0)
    in_flight: str | None = None  # the event whose request is journaled but whose response is not
    records = {"run": run, "config": config, "batch_id": batch_id}
    try:
        for call in planned:
            event_id = call.event["event_id"]
            if counts.attempted >= config.judge.max_calls:
                unsent = {"event_id": event_id, "cap": config.judge.max_calls}
                return JudgeSummary(STATUS_CAP_REACHED, counts, usage, unsent, None)
            sink.write_request(_journal(call, batch_id, request_sha256=call.request_sha256, payload=call.payload))
            counts = replace(counts, attempted=counts.attempted + 1)
            in_flight, started = event_id, time.perf_counter()
            try:
                reply = model.score(call.request)
            except INTERRUPTIONS as exc:  # never retried; an unknown outcome is uncertain
                error, uncertain, raw, call_usage = _call_error(exc)
                latency = round(time.perf_counter() - started, LATENCY_DECIMALS)
                status = STATUS_UNCERTAIN if uncertain else STATUS_FAILED
                sink.write_response(_journal(call, batch_id, status=status, raw=raw, error=error, latency_s=latency))
                in_flight, usage = None, _add_usage(usage, call_usage)
                if uncertain:
                    counts = replace(counts, uncertain=counts.uncertain + 1)
                else:
                    counts = replace(counts, failed=counts.failed + 1)
                reason = REASON_CALL_UNCERTAIN if uncertain else REASON_CALL_FAILED
                sink.write_results(judge_records(call, **records, reply=None, judgements=None, failure_reason=reason))
                return JudgeSummary(
                    status, counts, usage, None, {"event_id": event_id, **error, "uncertain": uncertain}
                )
            latency = round(time.perf_counter() - started, LATENCY_DECIMALS)
            sink.write_response(_journal(call, batch_id, status="ok", raw=reply.raw, error=None, latency_s=latency))
            in_flight, usage = None, _add_usage(usage, reply.usage)
            try:
                judgements = parse_verdict(reply.output_text, call.packet.criteria.asked, config)
            except JudgeOutputError as exc:
                counts = replace(counts, failed=counts.failed + 1)
                reason = REASON_OUTPUT_INVALID
                sink.write_results(judge_records(call, **records, reply=reply, judgements=None, failure_reason=reason))
                error = {"event_id": event_id, "type": type(exc).__name__, "message": str(exc), "uncertain": False}
                return JudgeSummary(STATUS_FAILED, counts, usage, None, error)
            sink.write_results(judge_records(call, **records, reply=reply, judgements=judgements, failure_reason=None))
            counts = replace(counts, judged=counts.judged + 1)
    except INTERRUPTIONS as exc:  # a journal write failed, or Ctrl-C outside a call
        uncertain = in_flight is not None or counts.uncertain > 0
        if in_flight is not None:
            counts = replace(counts, uncertain=counts.uncertain + 1)
        message = str(exc) if isinstance(exc, (OSError, AssessError)) else UNEXPECTED_MESSAGE
        error = {"event_id": in_flight, "type": type(exc).__name__, "message": message, "uncertain": uncertain}
        return JudgeSummary(STATUS_UNCERTAIN if uncertain else STATUS_FAILED, counts, usage, None, error)
    return JudgeSummary(STATUS_COMPLETED, counts, usage, None, None)
