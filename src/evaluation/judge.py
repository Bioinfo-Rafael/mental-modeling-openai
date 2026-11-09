"""Judge calls: one request per event, the guards that keep a paid request from being sent twice, the loop.

No-resend (design section 4.2): a paid run holds the ``O_EXCL`` marker ``<output>/.paid-judge.running``
for its whole duration; every ``request_sha256`` journaled anywhere under the output root (this
judge's ``requests.jsonl`` and the 2026-11-05 assess ``judge_requests.jsonl``) is never sent again;
an event already judged in the same directory with other bytes is refused; each request line is
fsynced before its call. Nothing is retried: a failed or uncertain call ends the run. A paid run
from a git worktree is refused (its ``outputs/`` holds none of the main checkout's journals).

The one exception is explicit: ``--only-event`` re-sends a single stuck event (sent from this
directory with the same bytes, no verdict here, journaled nowhere else) and marks the request line
as a user-requested re-send.
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

from evaluation.config import Criterion, JudgeConfig
from evaluation.material import read_jsonl
from evaluation.results import judge_rows
from translate.backends import BackendError
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


def _raise(exc: OSError) -> None:
    raise exc


def journal_paths(root: Path) -> set[Path]:
    """Every journal below ``root``, resolved, following directory symlinks (each directory once; a
    plain ``rglob`` skips them). An unreadable directory fails closed."""
    if not Path(root).is_dir():
        return set()
    found: set[Path] = set()
    seen: set[tuple[int, int]] = set()
    for directory, subdirs, files in os.walk(root, followlinks=True, onerror=_raise):
        info = os.stat(directory)
        if (info.st_dev, info.st_ino) in seen:
            subdirs.clear()  # os.walk prunes in place: a symlink loop or a second link to a seen directory
            continue
        seen.add((info.st_dev, info.st_ino))
        found |= {(Path(directory) / name).resolve() for name in files if name in JOURNALS}
    return found


def known_requests(*roots: Path, skip: Path | None = None) -> frozenset[str]:
    """Every request hash journaled under the roots, except in the journal ``skip``. An unreadable line
    fails closed, naming the file."""
    found: frozenset[str] = frozenset()
    paths = set().union(*(journal_paths(root) for root in roots))
    for path in sorted(paths - ({Path(skip).resolve()} if skip is not None else set())):
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


def _judged(directory: Path) -> set[str]:
    path = directory / "results.jsonl"
    return {line["event_id"] for line in read_jsonl(path)} if path.is_file() else set()


def missing_results(plan: Plan, directory: Path) -> list[str]:
    """Planned events (not refused) without a result line here; only ``--only-event`` sends one again."""
    done = _judged(directory)
    return [c.event_id for c in plan.calls if c.event_id not in plan.refused and c.event_id not in done]


def resend_plan(calls: list[Call], event_id: str, directory: Path, roots: tuple[Path, ...]) -> tuple[Plan, dict]:
    """``--only-event``: a plan of the one stuck event and the note its request line carries. Stuck means
    sent from this directory with these exact bytes, no verdict here, and journaled under no other
    journal (where its verdict may live). Anything else is refused."""
    call = next((c for c in calls if c.event_id == event_id), None)
    if call is None:
        raise JudgeError(f"{event_id} is not a planned event of this run (unknown id, or its target leaked)")
    path = directory / REQUESTS
    if path.is_symlink():
        raise JudgeError(f"{path} is a symlink: a re-send needs this directory's own journal")
    lines = read_jsonl(path) if path.is_file() else []
    sent = [x for x in lines if x["call_id"] == event_id]
    if not any(call.sha256 in _line_hashes(x) for x in sent):
        raise JudgeError(f"{event_id} was never sent from {directory} with these bytes: judge without --only-event")
    if any(call.sha256 in _line_hashes(x) for x in lines if x["call_id"] != event_id):
        raise JudgeError(f"the request of {event_id} is journaled here under another event id")
    if event_id in _judged(directory):
        raise JudgeError(f"{event_id} already has results in {directory}: it is not stuck")
    if call.sha256 in known_requests(*roots, skip=path):
        raise JudgeError(f"the request of {event_id} is journaled outside {directory}: its verdict may live there")
    note = {"requested_by": "user", "option": "--only-event", "previous_sends": len(sent)}  # each re-send is asked
    return Plan((call,), (call,), 1, ()), note


def main_checkout_only(root: Path) -> None:
    """Refuse a paid run from a git worktree (``.git`` is a file): it would scan only the worktree's
    ``outputs/evaluation`` and miss the paid journals of the main checkout."""
    dot_git = Path(root) / ".git"
    if not dot_git.is_file():
        return
    gitdir = (Path(root) / dot_git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()).resolve()
    in_worktrees = gitdir.parent.name == "worktrees" and len(gitdir.parents) > 2  # <main>/.git/worktrees/<name>
    main = gitdir.parents[2] if in_worktrees else "the main checkout"
    raise JudgeError(
        f"{root} is a git worktree: a paid judge run scans only its outputs/evaluation, not the main "
        f"checkout's paid journals. Run it from {main}"
    )


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


def run_calls(
    plan: Plan,
    *,
    send: Callable[[Call], Any],
    directory: Path,
    config: JudgeConfig,
    batch: str,
    note: dict | None = None,
) -> Outcome:
    """Send the planned calls in order, each once. ``send`` takes a call and returns a ``Reply``; ``note``
    (an ``--only-event`` re-send) is written into the request line as ``resend``."""
    attempted, judged, in_flight = 0, 0, None
    try:
        for call in plan.to_send:
            if attempted >= config.limits.max_calls:
                return Outcome("cap_reached", attempted, judged, {"event_id": call.event_id, "cap": attempted})
            line = {"call_id": call.event_id, "layer": "judge", "batch": batch, "step": call.event["step"]}
            line = {**line, "target": call.event["target"], "request_sha256": call.sha256, "body": call.body}
            line = {**line, "resend": note} if note else line
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
