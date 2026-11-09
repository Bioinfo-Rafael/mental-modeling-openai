"""One judge call per judged translation event (D20), journaled before it is sent.

- ``JudgeModel`` is the consumer-side Protocol; ``OpenAIJudgeModel`` adapts ``OpenAIResponsesClient``.
  The SDK client itself is built only in ``evaluation/cli.py`` after the paid-call gate.
- ``run_judge`` walks the translation events in log order. An event without key points becomes a
  ``not_judged`` result with no call (D36). For a judged event: cap check -> request line (fsync)
  -> one call -> raw response line (fsync, kept even when parsing fails) -> strict parse -> result.
- ``requests.jsonl`` holds the payload with every image replaced by its length and sha256
  (``redact_images``); ``request_sha256`` is over the real payload, images included.
- Any failure stops the run; nothing is repaired or retried. A call whose outcome is unknown
  (timeout, 5xx, unreadable response, unexpected exception) is ``uncertain`` and is never resent.
- Error records carry the exception class name and only messages this project or the LLM part
  controls; never tracebacks or response bodies (raw bodies live in ``responses.jsonl`` only).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

from translate.events import UNEXPECTED_ERROR_MESSAGE, canonical_json, redacted_error, sha256_text
from translate.llm.openai_responses import OpenAIClientError, OpenAIResponsesClient, build_responses_body

from .rendering import JudgeRequest, redact_images
from .schemas import JudgeSchemaError, JudgeVerdict, parse_judge_output

RECORD_SCHEMA_VERSION = 1
JUDGE_LAYER = "judge"
REQUESTS_FILE = "requests.jsonl"
RESPONSES_FILE = "responses.jsonl"
RESULTS_FILE = "results.jsonl"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_UNCERTAIN = "uncertain"
STATUS_CAP_REACHED = "cap_reached"
RESULT_JUDGED = "judged"
RESULT_NOT_JUDGED = "not_judged"
RESULT_FAILED = "failed"
RESULT_UNCERTAIN = "uncertain"
NOT_JUDGED_REASON = "no_key_points"
RESPONSE_OK = "ok"
RESPONSE_ERROR = "error"
RESPONSE_UNCERTAIN = "uncertain"
UNEXPECTED_ERROR_TYPE_PREFIX = "unexpected:"
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
# What ends a run with accurate counts. Ctrl-C during a call makes that call uncertain (exit 3).
INTERRUPTIONS = (Exception, KeyboardInterrupt)


class JudgeCallError(Exception):
    """A judge call failed. ``uncertain``: it may have been billed with an unknown outcome; never resend."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        uncertain: bool,
        raw: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.uncertain = uncertain
        self.raw = raw
        self.usage = usage


@dataclass(frozen=True)
class JudgeModelResponse:
    model: str  # the model name the API reported
    response_id: str | None
    output_text: str
    usage: dict[str, int]
    raw: dict[str, Any]


class JudgeModel(Protocol):
    def request_payload(self, request: JudgeRequest) -> dict[str, Any]:
        """The exact request body; journaled before the call is sent."""
        ...

    def score(self, request: JudgeRequest) -> JudgeModelResponse:
        """One call. Raises ``JudgeCallError`` on failure (refusal and incomplete included)."""
        ...


class OpenAIJudgeModel:
    """``JudgeModel`` over ``OpenAIResponsesClient`` (temperature is never sent; store is false)."""

    def __init__(self, client: OpenAIResponsesClient) -> None:
        self._client = client

    def request_payload(self, request: JudgeRequest) -> dict[str, Any]:
        return build_responses_body(request.openai_request)

    def score(self, request: JudgeRequest) -> JudgeModelResponse:
        try:
            result = self._client.create(request.openai_request)
        except OpenAIClientError as exc:  # refusal / incomplete keep .result; messages are sanitised
            raw = exc.result.raw if exc.result is not None else exc.raw
            usage = asdict(exc.result.usage) if exc.result is not None else None
            raise JudgeCallError(
                str(exc), error_type=type(exc).__name__, uncertain=exc.uncertain, raw=raw, usage=usage
            ) from exc
        return JudgeModelResponse(
            model=result.model,
            response_id=result.response_id,
            output_text=result.output_text,
            usage=asdict(result.usage),
            raw=result.raw,
        )


@dataclass(frozen=True)
class PlannedEvent:
    event: dict[str, Any]  # the translation_event record
    request: JudgeRequest | None  # None: no key points, so not judged


class JudgeSink(Protocol):
    def write_request(self, record: dict[str, Any]) -> None: ...
    def write_response(self, record: dict[str, Any]) -> None: ...
    def write_result(self, record: dict[str, Any]) -> None: ...


# --- JSONL sink -----------------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_json(value: Any) -> Any:
    """Standard JSON only: a non-finite float becomes the string "NaN" / "Infinity" / "-Infinity"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class JsonlJudgeSink:
    """Append-only JSONL files of one run; each line is flushed and fsynced before the run proceeds.

    A lone surrogate in a received body is written as its JSON escape (``backslashreplace``); the line
    stays valid JSON. So a fully received response is always journaled and a bad verdict fails
    instead of turning uncertain.
    """

    def __init__(self, run_dir: Path) -> None:
        names = (REQUESTS_FILE, RESPONSES_FILE, RESULTS_FILE)
        self._files: dict[str, TextIO] = {
            name: (run_dir / name).open("x", encoding="utf-8", errors="backslashreplace") for name in names
        }

    def _write(self, name: str, record: dict[str, Any]) -> None:
        handle = self._files[name]
        handle.write(json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_request(self, record: dict[str, Any]) -> None:
        self._write(REQUESTS_FILE, {**record, "written_at": utc_now()})

    def write_response(self, record: dict[str, Any]) -> None:
        self._write(RESPONSES_FILE, {**record, "written_at": utc_now()})

    def write_result(self, record: dict[str, Any]) -> None:
        self._write(RESULTS_FILE, record)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()

    def __enter__(self) -> JsonlJudgeSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --- run ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgeCounts:
    attempted: int = 0  # calls sent (or possibly sent)
    judged: int = 0
    not_judged: int = 0
    failed: int = 0
    uncertain: int = 0


@dataclass(frozen=True)
class JudgeSummary:
    status: str
    counts: JudgeCounts
    results_written: int
    usage: dict[str, int]  # summed over every response that reported usage
    unsent: dict[str, Any] | None  # cap_reached: the event that was not sent
    error: dict[str, Any] | None


class _Stop(Exception):
    def __init__(
        self, status: str, *, unsent: dict[str, Any] | None = None, error: dict[str, Any] | None = None
    ) -> None:
        super().__init__(status)
        self.status, self.unsent, self.error = status, unsent, error


class _Progress:
    """Progress of one ``run_judge`` call. Local to it and never shared."""

    def __init__(self, sink: JudgeSink) -> None:
        self.sink = sink
        self.counts = JudgeCounts()
        self.results_written = 0
        self.usage = dict.fromkeys(USAGE_KEYS, 0)
        self.in_flight: str | None = None  # event id sent but not yet journaled

    def bump(self, field: str) -> None:
        self.counts = JudgeCounts(**{**asdict(self.counts), field: getattr(self.counts, field) + 1})

    def add_usage(self, usage: dict[str, int] | None) -> None:
        if usage:
            self.usage = {key: self.usage[key] + int(usage.get(key, 0)) for key in USAGE_KEYS}

    def result(self, record: dict[str, Any]) -> None:
        self.sink.write_result(record)
        self.results_written += 1

    def summary(self, status: str, stop: _Stop | None = None) -> JudgeSummary:
        return JudgeSummary(
            status=status,
            counts=self.counts,
            results_written=self.results_written,
            usage=self.usage,
            unsent=stop.unsent if stop else None,
            error=stop.error if stop else None,
        )


def result_record(
    event: dict[str, Any],
    status: str,
    *,
    request_sha256: str | None = None,
    verdict: JudgeVerdict | None = None,
    response: JudgeModelResponse | None = None,
    usage: dict[str, int] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One ``results.jsonl`` line; every status has the same keys (null where not applicable)."""
    verdict_json = verdict.to_json() if verdict else None
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "event_id": event["event_id"],
        "status": status,
        "reason": NOT_JUDGED_REASON if status == RESULT_NOT_JUDGED else None,
        "step": event["step"],
        "time_s": event["time_s"],
        "target": event["target"],
        "primary_trigger": event["primary_trigger"],
        "key_points": verdict_json["key_points"] if verdict_json else None,
        "overall": verdict_json["overall"] if verdict_json else None,
        "request_sha256": request_sha256,
        "response_model": response.model if response else None,
        "response_id": response.response_id if response else None,
        "usage": response.usage if response else usage,
        "error": error,
    }


def _response_record(event_id: str, status: str, raw: Any, error: dict[str, str] | None) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "call_id": event_id,
        "layer": JUDGE_LAYER,
        "status": status,
        "raw": raw,
        "error": error,
    }


def _call_failed(progress: _Progress, event: dict[str, Any], request_sha256: str, exc: BaseException) -> _Stop:
    """Journal a failed call (response line + result line) and build the stop."""
    if isinstance(exc, JudgeCallError):
        error, uncertain, raw, usage = {"type": exc.error_type, "message": str(exc)}, exc.uncertain, exc.raw, exc.usage
    else:  # unknown outcome: the request may have reached the server
        error = {"type": UNEXPECTED_ERROR_TYPE_PREFIX + type(exc).__name__, "message": UNEXPECTED_ERROR_MESSAGE}
        uncertain, raw, usage = True, None, None
    event_id = event["event_id"]
    progress.sink.write_response(
        _response_record(event_id, RESPONSE_UNCERTAIN if uncertain else RESPONSE_ERROR, raw, error)
    )
    progress.in_flight = None
    progress.bump("uncertain" if uncertain else "failed")
    progress.add_usage(usage)
    status = RESULT_UNCERTAIN if uncertain else RESULT_FAILED
    progress.result(result_record(event, status, request_sha256=request_sha256, usage=usage, error=error))
    details = {"event_id": event_id, **error, "uncertain": uncertain}
    return _Stop(STATUS_UNCERTAIN if uncertain else STATUS_FAILED, error=details)


def _judge_one(progress: _Progress, event: dict[str, Any], request: JudgeRequest, model: JudgeModel) -> None:
    event_id = event["event_id"]
    payload = model.request_payload(request)
    request_sha256 = sha256_text(canonical_json(payload))  # over the real payload, images included
    progress.sink.write_request(  # the journal keeps each image as its length and sha256, not as base64
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "call_id": event_id,
            "layer": JUDGE_LAYER,
            "payload": redact_images(payload),
        }
    )
    progress.bump("attempted")
    progress.in_flight = event_id
    try:
        response = model.score(request)
    except INTERRUPTIONS as exc:  # JudgeCallError, anything else or Ctrl-C (uncertain); never retried
        raise _call_failed(progress, event, request_sha256, exc) from exc
    progress.sink.write_response(_response_record(event_id, RESPONSE_OK, response.raw, None))
    progress.in_flight = None
    progress.add_usage(response.usage)
    try:
        verdict = parse_judge_output(response.output_text, request.key_points)
    except JudgeSchemaError as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        progress.bump("failed")
        progress.result(
            result_record(event, RESULT_FAILED, request_sha256=request_sha256, response=response, error=error)
        )
        raise _Stop(STATUS_FAILED, error={"event_id": event_id, **error, "uncertain": False}) from exc
    progress.bump("judged")
    progress.result(
        result_record(event, RESULT_JUDGED, request_sha256=request_sha256, verdict=verdict, response=response)
    )


def _crash(progress: _Progress, exc: BaseException) -> JudgeSummary:
    """An unexpected failure outside a call (e.g. a journal write).

    ``uncertain`` while a sent call is not yet journaled, or once any call was uncertain;
    otherwise ``failed``. Only ``redacted_error`` messages are kept.
    """
    error = redacted_error(exc)
    event_id = progress.in_flight
    if event_id is not None:
        progress.in_flight = None
        progress.bump("uncertain")
    if progress.counts.uncertain:
        return progress.summary(
            STATUS_UNCERTAIN, _Stop(STATUS_UNCERTAIN, error={**error, "event_id": event_id, "uncertain": True})
        )
    return progress.summary(STATUS_FAILED, _Stop(STATUS_FAILED, error={**error, "event_id": None, "uncertain": False}))


def run_judge(
    planned: tuple[PlannedEvent, ...], *, model: JudgeModel | None, sink: JudgeSink, max_judged_events: int
) -> JudgeSummary:
    """Judge the planned events in order. Stops before call ``max_judged_events + 1`` (``cap_reached``)."""
    progress = _Progress(sink)
    try:
        for item in planned:
            if item.request is None:
                progress.bump("not_judged")
                progress.result(result_record(item.event, RESULT_NOT_JUDGED))
                continue
            if progress.counts.attempted >= max_judged_events:
                unsent = {"event_id": item.event["event_id"], "cap": max_judged_events}
                raise _Stop(STATUS_CAP_REACHED, unsent=unsent)
            if model is None:
                raise ValueError("a judged event needs a judge model")
            _judge_one(progress, item.event, item.request, model)
    except _Stop as stop:
        return progress.summary(stop.status, stop)
    except INTERRUPTIONS as exc:  # e.g. a journal write failed or Ctrl-C: stop with accurate counts
        return _crash(progress, exc)
    return progress.summary(STATUS_COMPLETED)
