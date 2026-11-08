"""Translator evaluation: result records, append-only stores and the human-rating import.

Machine, human and judge results share one record schema and live in three separate append-only
files; nothing is merged or rewritten. ``status`` is ``evaluated`` or ``not_evaluated`` with a
``reason`` (``cannot_judge`` and ``not_applicable`` are reasons, never a score of 0). A newer
result names the record it replaces in ``supersedes``; every record is kept (D58).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from translate.output_safety import derived_path

from .assess_data import (
    ASPECT_OVERALL,
    EVALUATOR_HUMAN,
    EVALUATOR_MACHINE,
    JUDGEMENT_CANNOT_JUDGE,
    JUDGEMENT_NOT_APPLICABLE,
    REASON_AGGREGATION_UNSET,
    REASON_INPUT_INVALID,
    RESULT_SCHEMA_VERSION,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUATED,
    AssessConfig,
    AssessError,
    Criterion,
    RatingsError,
    TranslatorRun,
    event_criteria,
    exact_keys,
    is_safe_id,
    point_kinds,
    read_jsonl,
    sha256_file,
    target_key,
)

# --- result records (machine, human and judge share one schema) -----------------------------------

OVERALL = (ASPECT_OVERALL, ASPECT_OVERALL, None)  # (criterion id, aspect, reference) of an overall row


def criterion_key(criterion: Criterion) -> tuple[str, str, str | None]:
    return (criterion.criterion_id, criterion.aspect, criterion.reference)


def event_subject(event: dict[str, Any], run: TranslatorRun, kinds: tuple[str, ...]) -> dict[str, Any]:
    return {
        "scenario_id": run.scenario_id,
        "episode_id": run.episode_id,
        "target": event["target"],
        "target_aircraft": list(event["input"]["target_aircraft"]),
        "step": event["step"],
        "time_s": event["time_s"],
        "event_id": event["event_id"],
        "point_kinds": list(kinds),
        "reference_point_id": None,
    }


def verdict(
    status: str,
    *,
    score: int | None = None,
    judgement: str | None = None,
    measurement: dict[str, Any] | None = None,
    rationale: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """The outcome part of a record."""
    return {
        "status": status,
        "score": score,
        "judgement": judgement,
        "measurement": measurement,
        "rationale": rationale,
        "reason": reason,
    }


def judgement_verdict(judgement: str, rationale: str) -> dict[str, Any]:
    """``"1".."5"`` -> evaluated with an integer score; ``cannot_judge`` / ``not_applicable`` -> not_evaluated."""
    if judgement in (JUDGEMENT_CANNOT_JUDGE, JUDGEMENT_NOT_APPLICABLE):
        return verdict(STATUS_NOT_EVALUATED, judgement=judgement, rationale=rationale, reason=judgement)
    return verdict(STATUS_EVALUATED, score=int(judgement), judgement=judgement, rationale=rationale)


def invalidated(outcome: dict[str, Any]) -> dict[str, Any]:
    """The same verdict as ``not_evaluated/generation_input_invalid``; judgement and measurement are kept."""
    return {**outcome, "status": STATUS_NOT_EVALUATED, "score": None, "reason": REASON_INPUT_INVALID}


def overall_verdict() -> dict[str, Any]:
    """D54: no aggregation of the four aspects until a method is decided."""
    return verdict(STATUS_NOT_EVALUATED, reason=REASON_AGGREGATION_UNSET)


def result_record(
    *,
    result_id: str,
    evaluator: dict[str, Any],
    subject: dict[str, Any],
    event: dict[str, Any] | None,
    criterion: tuple[str, str, str | None],
    outcome: dict[str, Any],
    config: AssessConfig,
    retrospective: bool,
    future_end_step: int | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """One stored result; every record has the same keys. ``criterion`` is (id, aspect, reference)."""
    criterion_id, aspect, reference = criterion
    input_range = output = None
    if event is not None:
        trajectory = event["input"]["trajectory"]
        input_range = {
            "start_step": trajectory["start_step"],
            "end_step": trajectory["end_step"],
            "request_sha256": event["request_sha256"],
            "future_end_step": future_end_step,  # exclusive end of the FUTURE block (judge only)
        }
        output = {"intent": event["output"]["intent"], "evidence": list(event["output"]["evidence"])}
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_id": result_id,
        "evaluator": evaluator,
        "subject": subject,
        "input_range": input_range,
        "translator_output": output,
        "aspect": aspect,
        "criterion_id": criterion_id,
        "reference": reference,
        **outcome,
        "rubric_version": config.rubric_version,
        "config_sha256": config.sha256,
        "retrospective": retrospective,
        "supersedes": supersedes,
    }


# --- append-only stores ---------------------------------------------------------------------------


def read_results(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path, AssessError) if Path(path).is_file() else []


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> int:
    """Append records with one fsync; earlier bytes are never rewritten. Returns the count."""
    lines = [json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records]
    if lines:
        with derived_path(path).open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
    return len(lines)


def active_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records no later record supersedes. Superseded ones stay in the file (D58: keep every rating)."""
    superseded = {record["supersedes"] for record in records if record.get("supersedes")}
    return [record for record in records if record["result_id"] not in superseded]


def result_slot(record: dict[str, Any]) -> tuple[Any, ...]:
    """What a newer result replaces: same evaluator (and rater), rubric (not machine), subject, criterion."""
    evaluator, subject = record["evaluator"], record["subject"]
    rubric = record["rubric_version"] if evaluator["kind"] != EVALUATOR_MACHINE else None
    where = (subject["event_id"], subject["reference_point_id"], target_key(subject["target"]))
    return (evaluator["kind"], evaluator.get("rater_id"), rubric, *where, record["criterion_id"])


def with_supersedes(new: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """New machine/judge records in order, each superseding the record active for its slot so far."""
    known = {record["result_id"] for record in existing}
    active = {result_slot(record): record["result_id"] for record in active_results(existing)}
    stored = []
    for record in new:  # ``known`` and ``active`` are local indexes, updated in place
        if record["result_id"] in known:
            raise AssessError("a result id is already stored or repeated; every invocation needs a new batch id")
        slot = result_slot(record)
        stored.append({**record, "supersedes": active.get(slot)})
        known.add(record["result_id"])
        active[slot] = record["result_id"]
    return stored


# --- human ratings --------------------------------------------------------------------------------

RATING_KEYS = {"rating_id", "rater_id", "event_id", "criterion_id", "judgement", "rationale", "rubric_version"}
RATING_KEYS |= {"rated_at", "supersedes"}


def load_ratings(path: Path) -> tuple[list[dict[str, Any]], str]:
    """The rating lines (exact keys) and the file's SHA-256. Content checks: ``human_results``."""
    lines = read_jsonl(path, RatingsError)
    for index, line in enumerate(lines):
        exact_keys(line, RATING_KEYS, f"ratings line {index + 1}", RatingsError)
    return lines, sha256_file(path)


def _is_aware_time(value: object) -> bool:
    try:
        return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _rated(line: dict[str, Any], events: dict[str, dict[str, Any]], config: AssessConfig) -> tuple[dict, Criterion]:
    """The event and criterion a rating line names, after every content check (``RatingsError`` otherwise)."""
    if not is_safe_id(line["rating_id"]) or not is_safe_id(line["rater_id"]):
        raise RatingsError("rating_id and rater_id must be safe ids (pseudonymous rater ids)")
    event = events.get(line["event_id"]) if isinstance(line["event_id"], str) else None
    if event is None:
        raise RatingsError(f"event_id {line['event_id']!r} is not a translation event of the run")
    if line["rubric_version"] != config.rubric_version:
        raise RatingsError(f"rubric_version {line['rubric_version']!r} is not the config's {config.rubric_version!r}")
    if not _is_aware_time(line["rated_at"]) or not isinstance(line["rationale"], str):
        raise RatingsError("rated_at must be ISO 8601 with a UTC offset, and rationale a string")
    criteria = event_criteria(event, config)
    excluded = {criterion.criterion_id: reason for criterion, reason in criteria.excluded}
    if line["criterion_id"] in excluded:
        raise RatingsError(f"{line['criterion_id']} is not rated for this event ({excluded[line['criterion_id']]})")
    criterion = next((c for c in criteria.asked if c.criterion_id == line["criterion_id"]), None)
    if criterion is None:
        raise RatingsError(f"{line['criterion_id']!r} is not a criterion of this event's point kinds")
    optional = (JUDGEMENT_NOT_APPLICABLE,) if criterion.conditional else ()
    if line["judgement"] not in (*config.scale.values, JUDGEMENT_CANNOT_JUDGE, *optional):
        raise RatingsError(f"judgement {line['judgement']!r} is not allowed for {criterion.criterion_id}")
    return event, criterion


def human_results(
    lines: list[dict[str, Any]],
    *,
    run: TranslatorRun,
    config: AssessConfig,
    invalid_targets: frozenset[tuple[str, str]],
    existing: list[dict[str, Any]],
    import_sha256: str,
) -> list[dict[str, Any]]:
    """Validate every line, then return the records; a ``RatingsError`` means nothing is appended.

    A re-rating names the rating it replaces in ``supersedes``; the same (rater, event, criterion,
    rubric version) twice without it is an error. Ratings of a target whose generation input is
    invalid are kept (judgement and rationale) as ``not_evaluated/generation_input_invalid``.
    One overall row (``aggregation_unset``) is added per rater and event.
    """
    events = {event["event_id"]: event for event in run.events}
    # Local indexes of this call, updated in place as lines are accepted. Rating ids are local to their rater.
    by_rating = {
        (r["evaluator"]["rater_id"], r["evaluator"]["rating_id"]): r for r in existing if "rating_id" in r["evaluator"]
    }
    active = {
        (r["evaluator"]["rater_id"], r["subject"]["event_id"], r["criterion_id"], r["rubric_version"]): r["result_id"]
        for r in active_results(existing)
        if r["aspect"] != ASPECT_OVERALL
    }
    known_ids = {r["result_id"] for r in existing}
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        where = f"ratings line {index + 1}"
        try:
            event, criterion = _rated(line, events, config)
        except RatingsError as exc:
            raise RatingsError(f"{where}: {exc}") from None
        rater = line["rater_id"]
        slot = (rater, line["event_id"], line["criterion_id"], line["rubric_version"])
        replaced = by_rating.get((rater, line["supersedes"])) if isinstance(line["supersedes"], str) else None
        if (rater, line["rating_id"]) in by_rating:
            raise RatingsError(f"{where}: this rater's rating_id is already stored or repeated")
        if line["supersedes"] is not None and (replaced is None or replaced["result_id"] != active.get(slot)):
            raise RatingsError(
                f"{where}: supersedes must name this rater's active rating of the same event and criterion"
            )
        if line["supersedes"] is None and active.get(slot) is not None:
            raise RatingsError(f"{where}: this rater already rated this criterion; set supersedes to re-rate")
        outcome = judgement_verdict(line["judgement"], line["rationale"])
        evaluator = {"kind": EVALUATOR_HUMAN, "rater_id": rater, "import_sha256": import_sha256}
        common = {"event": event, "config": config, "retrospective": run.retrospective}
        record = result_record(
            result_id=f"{EVALUATOR_HUMAN}:{rater}:{line['rating_id']}",
            evaluator={**evaluator, "rating_id": line["rating_id"], "rated_at": line["rated_at"]},
            subject=event_subject(event, run, event_criteria(event, config).kinds_of[criterion.criterion_id]),
            criterion=criterion_key(criterion),
            outcome=invalidated(outcome) if target_key(event["target"]) in invalid_targets else outcome,
            supersedes=replaced["result_id"] if replaced is not None else None,
            **common,
        )
        overall_id = f"{EVALUATOR_HUMAN}:{rater}:{event['event_id']}:{ASPECT_OVERALL}"
        overall = result_record(
            result_id=overall_id,
            evaluator=evaluator,
            subject=event_subject(event, run, point_kinds(event, config)),
            criterion=OVERALL,
            outcome=overall_verdict(),
            **common,
        )
        records += [record] if overall_id in known_ids else [record, overall]
        by_rating[(rater, line["rating_id"])] = record
        active[slot] = record["result_id"]
        known_ids.update((record["result_id"], overall_id))
    return records
