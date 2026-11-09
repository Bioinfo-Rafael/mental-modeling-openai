"""Judge calls: one request per event, the guards that keep a paid request from being sent twice, the loop.

No-resend (design section 4.2): a paid run holds the ``O_EXCL`` marker ``<output>/.paid-judge.running``
for its whole duration; every ``request_sha256`` journaled anywhere under the output root (this
judge's ``requests.jsonl`` and the 2026-11-05 assess ``judge_requests.jsonl``) is never sent again;
an event already judged in the same directory with other bytes is refused; each request line is
fsynced before its call. Nothing is retried: a failed or uncertain call ends the run.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.config import Criterion, JudgeConfig, criteria_for
from evaluation.material import TranslationRun, event_kinds, read_jsonl, system_text, user_text
from evaluation.results import judge_rows
from translate.backends import BackendError, Prompt, input_text
from translate.backends.openai import OpenAIBackend
from translate.episode import Episode
from translate.journal import REPO_ROOT, append_jsonl, loads_unique, request_sha256

MARKER = ".paid-judge.running"
GUARD_ROOT = REPO_ROOT / "outputs" / "evaluation"  # always scanned and holds the marker, whatever `output` says
JOURNALS = ("requests.jsonl", "judge_requests.jsonl")  # this judge's name and the 2026-11-05 assess journal's
REQUESTS, RESPONSES = "requests.jsonl", "responses.jsonl"
CANNOT_JUDGE, NOT_APPLICABLE = "cannot_judge", "not_applicable"
LATENCY_DECIMALS = 3
WITHHELD = "details withheld"


class JudgeError(ValueError):
    """The judge cannot plan or run safely; nothing was sent for the event or run it names."""


@dataclass(frozen=True)
class Call:
    event: dict[str, Any]
    criteria: tuple[Criterion, ...]
    body: dict[str, Any]  # the exact Responses body, journaled before the send
    sha256: str

    @property
    def event_id(self) -> str:
        return self.event["event_id"]


def verdict_schema(criteria: tuple[Criterion, ...], config: JudgeConfig) -> dict[str, Any]:
    """Exactly one item per criterion; the rationale comes before the judgement."""
    judgements = [*config.scale.values, CANNOT_JUDGE, NOT_APPLICABLE]
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": {"type": "string", "enum": [criterion.id for criterion in criteria]},
            "rationale": {"type": "string"},
            "judgement": {"type": "string", "enum": judgements},
        },
        "required": ["criterion_id", "rationale", "judgement"],
    }
    items = {"type": "array", "minItems": len(criteria), "maxItems": len(criteria), "items": item}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"criteria": items},
        "required": ["criteria"],
    }


def plan_call(event: dict[str, Any], run: TranslationRun, frame: Episode, config: JudgeConfig) -> Call:
    """The event's request. ``rubric_version`` names the verdict schema, so a new rubric is a new request."""
    criteria = criteria_for(config, event_kinds(event))
    items = (input_text(user_text(event, run, frame, config)),)
    prompt = Prompt(system_text(config), items, f"verdict_{config.rubric_version}", verdict_schema(criteria, config))
    body = OpenAIBackend.request_body(config.model, prompt)
    return Call(event, criteria, body, request_sha256(body))


def parse_verdict(text: str, criteria: tuple[Criterion, ...], config: JudgeConfig) -> dict[str, tuple[str, str]]:
    """criterion id -> (judgement, rationale): exactly the asked criteria, each once; nothing is repaired."""
    try:
        parsed = loads_unique(text)
    except ValueError:
        raise JudgeError("the verdict is not one JSON object without repeated keys") from None
    if not isinstance(parsed, dict) or set(parsed) != {"criteria"} or not isinstance(parsed["criteria"], list):
        raise JudgeError('the verdict must be {"criteria": [...]}')
    by_id, found = {criterion.id: criterion for criterion in criteria}, {}
    for index, item in enumerate(parsed["criteria"]):
        if not isinstance(item, dict) or set(item) != {"criterion_id", "rationale", "judgement"}:
            raise JudgeError(f"criteria[{index}] must have exactly criterion_id, rationale and judgement")
        criterion = by_id.get(item["criterion_id"]) if isinstance(item["criterion_id"], str) else None
        if criterion is None or criterion.id in found:
            raise JudgeError(f"criteria[{index}].criterion_id is not an asked criterion, or repeats one")
        allowed = (*config.scale.values, CANNOT_JUDGE, *((NOT_APPLICABLE,) if criterion.conditional else ()))
        if item["judgement"] not in allowed:
            raise JudgeError(f"criteria[{index}].judgement must be one of {list(allowed)}")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise JudgeError(f"criteria[{index}].rationale must be a non-blank string")
        found = {**found, criterion.id: (item["judgement"], item["rationale"])}
    missing = [criterion_id for criterion_id in by_id if criterion_id not in found]
    if missing:
        raise JudgeError(f"criteria missing from the verdict: {missing}")
    return found


# --- the no-resend guards -------------------------------------------------------------------------


def _line_hashes(line: dict[str, Any]) -> frozenset[str]:
    """The line's embedded ``request_sha256`` and the hash of its body (``payload`` in the 2026-11-05 journal)."""
    body = line.get("body", line.get("payload"))
    embedded = {line["request_sha256"]} if isinstance(line.get("request_sha256"), str) else set()
    return frozenset(embedded | ({request_sha256(body)} if isinstance(body, dict) else set()))


def known_requests(*roots: Path) -> frozenset[str]:
    """Every request hash journaled under the roots. An unreadable line fails closed, naming the file."""
    found: frozenset[str] = frozenset()
    paths = {p for root in roots for name in JOURNALS for p in Path(root).rglob(name)}
    for path in sorted(paths):
        try:
            hashes = [_line_hashes(line) for line in read_jsonl(path)]
        except ValueError:
            hashes = [frozenset()]
        if not all(hashes):
            raise JudgeError(f"{path} has a line without a request hash: repair it by hand before judging")
        found = found.union(*hashes)
    return found


def journaled(directory: Path) -> dict[str, frozenset[str]]:
    """event id -> the request hashes this judgement directory already journaled for it."""
    path = directory / REQUESTS
    lines = read_jsonl(path) if path.is_file() else []
    events = {line["call_id"] for line in lines}
    return {event: frozenset().union(*(_line_hashes(x) for x in lines if x["call_id"] == event)) for event in events}


@dataclass(frozen=True)
class Plan:
    calls: tuple[Call, ...]
    to_send: tuple[Call, ...]
    already_sent: int
    refused: tuple[str, ...]  # events judged in this directory with other bytes

    def to_json(self) -> dict[str, Any]:
        counts = {"packets": len(self.calls), "already_sent": self.already_sent, "to_send": len(self.to_send)}
        return {**counts, "refused": list(self.refused)}


def make_plan(calls: list[Call], known: frozenset[str], journal: dict[str, frozenset[str]]) -> Plan:
    if len({call.sha256 for call in calls}) != len(calls):
        raise JudgeError("two planned events build the same request bytes: the run repeats an event")
    refused = tuple(c.event_id for c in calls if c.event_id in journal and c.sha256 not in journal[c.event_id])
    sent = [call for call in calls if call.sha256 in known]
    to_send = tuple(call for call in calls if call.sha256 not in known and call.event_id not in refused)
    return Plan(tuple(calls), to_send, len(sent), refused)


def missing_results(plan: Plan, directory: Path) -> list[str]:
    """Planned events (not refused) without a result line here; they are never sent again."""
    path = directory / "results.jsonl"
    done = {line["event_id"] for line in read_jsonl(path)} if path.is_file() else set()
    return [c.event_id for c in plan.calls if c.event_id not in plan.refused and c.event_id not in done]


@contextmanager
def paid_marker(root: Path) -> Iterator[None]:
    """One paid judge run at a time under ``root``: an ``O_EXCL`` file held for the whole run."""
    path = Path(root) / MARKER
    Path(root).mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise JudgeError(f"{path} exists: another paid judge run is active, or a crashed one left it") from None
    os.write(handle, json.dumps({"pid": os.getpid(), "started_at": utc_now()}).encode())
    os.close(handle)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# --- the loop -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    status: str  # completed | failed | uncertain | cap_reached
    attempted: int
    judged: int
    error: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _failure(exc: BaseException) -> tuple[bool, dict[str, str], Any]:
    """(uncertain, error, raw): a backend error keeps its sanitised message; anything else is uncertain."""
    if isinstance(exc, BackendError):
        return exc.uncertain, {"type": type(exc).__name__, "message": str(exc)}, exc.raw
    return True, {"type": type(exc).__name__, "message": WITHHELD}, None


def run_calls(plan: Plan, *, send: Callable[[Call], Any], directory: Path, config: JudgeConfig, batch: str) -> Outcome:
    """Send the planned calls in order, each once. ``send`` takes a call and returns a ``Reply``."""
    attempted, judged, in_flight = 0, 0, None
    try:
        for call in plan.to_send:
            if attempted >= config.limits.max_calls:
                return Outcome("cap_reached", attempted, judged, {"event_id": call.event_id, "cap": attempted})
            line = {"call_id": call.event_id, "layer": "judge", "batch": batch, "step": call.event["step"]}
            line = {**line, "target": call.event["target"], "request_sha256": call.sha256, "body": call.body}
            append_jsonl(directory / REQUESTS, {**line, "written_at": utc_now()})  # fsynced before the send
            attempted, in_flight, started = attempted + 1, call.event_id, time.perf_counter()
            try:
                reply = send(call)
            except (Exception, KeyboardInterrupt) as exc:  # never retried
                uncertain, error, raw = _failure(exc)
                status = "uncertain" if uncertain else "error"
                latency = round(time.perf_counter() - started, LATENCY_DECIMALS)
                response = {"call_id": call.event_id, "status": status, "latency_s": latency, "raw": raw}
                append_jsonl(directory / RESPONSES, {**response, "error": error, "written_at": utc_now()})
                return Outcome(
                    "uncertain" if uncertain else "failed", attempted, judged, {"event_id": in_flight, **error}
                )
            latency = round(time.perf_counter() - started, LATENCY_DECIMALS)
            response = {"call_id": call.event_id, "status": "ok", "latency_s": latency, "raw": reply.raw}
            append_jsonl(directory / RESPONSES, {**response, "error": None, "written_at": utc_now()})
            in_flight = None
            try:
                verdict = parse_verdict(reply.text, call.criteria, config)
            except JudgeError as exc:
                return Outcome("failed", attempted, judged, {"event_id": call.event_id, "message": str(exc)})
            append_jsonl(directory / "results.jsonl", *judge_rows(call, verdict, reply, config, batch))
            judged += 1
    except (Exception, KeyboardInterrupt) as exc:  # a journal write failed, or Ctrl-C between calls
        status = "uncertain" if in_flight is not None else "failed"
        return Outcome(status, attempted, judged, {"event_id": in_flight, "type": type(exc).__name__})
    return Outcome("completed", attempted, judged)
