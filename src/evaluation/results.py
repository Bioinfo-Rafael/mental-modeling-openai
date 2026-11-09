"""Judge results, imported human ratings and the report (cells, agreement, the D55 request deadline).

``results.jsonl`` (judge) and ``ratings.jsonl`` (humans) are append-only, one line per (event,
criterion). A re-rating names the rating it replaces in ``supersedes``; every line is kept (D58).
``cannot_judge`` and ``not_applicable`` are recorded without a score and never count as 0.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from evaluation.config import JudgeConfig, criteria_for
from evaluation.material import TranslationRun, event_kinds, target_key

STAT_DECIMALS = 4
MIN_PAIRS = 2
RATING_KEYS = {"rating_id", "rater_id", "event_id", "criterion_id", "judgement", "rationale", "rubric_version"}
RATING_KEYS |= {"rated_at", "supersedes"}


class RatingsError(ValueError):
    """A ratings file does not match the run or the rubric; nothing was imported."""


def append_jsonl(path: Path, *records: dict[str, Any]) -> None:
    """Append records with one fsync; earlier bytes are never rewritten."""
    with Path(path).open("a", encoding="utf-8", errors="backslashreplace") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records)
        handle.flush()
        os.fsync(handle.fileno())


def _score(judgement: str) -> int | None:
    return int(judgement) if judgement.isdigit() else None


def judge_rows(call: Any, verdict: dict[str, tuple[str, str]], reply: Any, config: JudgeConfig, batch: str) -> list:
    event, kinds = call.event, list(event_kinds(call.event))
    common = {"event_id": event["event_id"], "target": event["target"], "step": event["step"], "kinds": kinds}
    common = {**common, "evaluator": "judge", "rubric_version": config.rubric_version, "batch": batch}
    common = {**common, "request_sha256": call.sha256, "response_id": reply.response_id}
    return [
        {
            **common,
            "criterion_id": c.id,
            "aspect": c.aspect,
            "judgement": verdict[c.id][0],
            "score": _score(verdict[c.id][0]),
            "rationale": verdict[c.id][1],
        }  # fmt: skip
        for c in call.criteria
    ]


def _rated_at(value: object) -> bool:
    try:
        return isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False


def _checked_rating(line: dict[str, Any], events: dict[str, dict], config: JudgeConfig) -> dict[str, Any]:
    if not isinstance(line, dict) or set(line) != RATING_KEYS:
        raise RatingsError(f"a rating line must have exactly {sorted(RATING_KEYS)}")
    event = events.get(line["event_id"])
    if event is None:
        raise RatingsError(f"event_id {line['event_id']!r} is not a translation event of the run")
    criterion = next((c for c in criteria_for(config, event_kinds(event)) if c.id == line["criterion_id"]), None)
    if criterion is None:
        raise RatingsError(f"{line['criterion_id']!r} is not a criterion of this event's point kinds")
    allowed = (*config.scale.values, "cannot_judge", *(("not_applicable",) if criterion.conditional else ()))
    if line["judgement"] not in allowed:
        raise RatingsError(f"judgement {line['judgement']!r} is not allowed for {criterion.id}")
    if line["rubric_version"] != config.rubric_version or not _rated_at(line["rated_at"]):
        raise RatingsError("rubric_version must be the config's, and rated_at ISO 8601 with a UTC offset")
    for key in ("rating_id", "rater_id"):
        if not isinstance(line[key], str) or not line[key].replace("-", "").replace("_", "").isalnum():
            raise RatingsError(f"{key} must be a pseudonymous id (letters, digits, - and _)")
    common = {"target": event["target"], "step": event["step"], "kinds": list(event_kinds(event))}
    return {**line, **common, "aspect": criterion.aspect, "score": _score(line["judgement"]), "evaluator": "human"}


def import_ratings(lines: list[dict], run: TranslationRun, config: JudgeConfig, existing: list[dict]) -> list[dict]:
    """Validated rating records; a re-rating must name the rater's active rating of the same slot."""
    events = {event["event_id"]: event for event in run.events}
    stored = {(r["rater_id"], r["rating_id"]): r for r in existing}
    active = {(r["rater_id"], r["event_id"], r["criterion_id"]): r["rating_id"] for r in active_ratings(existing)}
    records: list[dict] = []
    for index, line in enumerate(lines, start=1):
        try:
            record = _checked_rating(line, events, config)
        except (RatingsError, KeyError, TypeError) as exc:
            raise RatingsError(f"ratings line {index}: {exc}") from None
        slot = (record["rater_id"], record["event_id"], record["criterion_id"])
        if (record["rater_id"], record["rating_id"]) in stored:
            raise RatingsError(f"ratings line {index}: this rater's rating_id is already stored")
        if record["supersedes"] != active.get(slot):
            raise RatingsError(f"ratings line {index}: supersedes must name this rater's active rating of the slot")
        stored = {**stored, (record["rater_id"], record["rating_id"]): record}
        active = {**active, slot: record["rating_id"]}
        records = [*records, record]
    return records


def active_ratings(ratings: list[dict]) -> list[dict]:
    replaced = {(r["rater_id"], r["supersedes"]) for r in ratings if r.get("supersedes")}
    return [r for r in ratings if (r["rater_id"], r["rating_id"]) not in replaced]


# --- report ---------------------------------------------------------------------------------------


def quadratic_weighted_kappa(pairs: list[tuple[int, int]], categories: tuple[int, ...]) -> float | None:
    """Cohen's kappa with quadratic weights; ``None`` when the expected disagreement is 0."""
    count, span = len(pairs), categories[-1] - categories[0]
    if not count or span == 0:
        return None
    first = [sum(1 for a, _ in pairs if a == c) / count for c in categories]
    second = [sum(1 for _, b in pairs if b == c) / count for c in categories]
    weight = {(i, j): ((i - j) / span) ** 2 for i in categories for j in categories}
    observed = sum(weight[(a, b)] for a, b in pairs) / count
    expected = sum(
        weight[(i, j)] * first[x] * second[y] for x, i in enumerate(categories) for y, j in enumerate(categories)
    )
    return None if expected == 0 else round(1 - observed / expected, STAT_DECIMALS)


def _agreement(pairs: list[tuple[int, int]], categories: tuple[int, ...]) -> dict[str, Any]:
    if len(pairs) < MIN_PAIRS:
        return {"pairs": len(pairs), "exact_agreement": None, "weighted_kappa": None, "reason": "fewer_than_2_pairs"}
    kappa = quadratic_weighted_kappa(pairs, categories)
    exact = round(sum(1 for a, b in pairs if a == b) / len(pairs), STAT_DECIMALS)
    return {
        "pairs": len(pairs),
        "exact_agreement": exact,
        "weighted_kappa": kappa,
        "reason": None if kappa is not None else "no_variance",
    }


def _cell(records: list[dict]) -> dict[str, Any]:
    scores = [r["score"] for r in records if r["score"] is not None]
    reasons = Counter(r["judgement"] for r in records if r["score"] is None)
    mean = round(sum(scores) / len(scores), STAT_DECIMALS) if scores else None
    distribution = {str(value): scores.count(value) for value in sorted(set(scores))}
    return {
        "n": len(records),
        "n_scored": len(scores),
        "mean": mean,
        "distribution": distribution,
        "not_scored": dict(reasons),
    }


def _deadline(run: TranslationRun, config: JudgeConfig, excluded: set[str]) -> dict[str, Any]:
    """D55: a request event passes when its processing_s is at most the deadline."""
    times = [
        e.get("processing_s")
        for e in run.events
        if "request" in event_kinds(e) and target_key(e["target"]) not in excluded
    ]
    measured = [t for t in times if isinstance(t, (int, float))]
    passed = sum(1 for t in measured if t <= config.timing.request_deadline_s)
    rate = round(passed / len(measured), STAT_DECIMALS) if measured else None
    deadline = config.timing.request_deadline_s
    return {
        "deadline_s": deadline,
        "requests": len(times),
        "measured": len(measured),
        "passed": passed,
        "pass_rate": rate,
    }


def build_report(
    run: TranslationRun, config: JudgeConfig, results: list[dict], ratings: list[dict], leaks: set[str]
) -> dict[str, Any]:
    """Cells per kind, aspect and criterion for the judge and humans (this rubric only, leaking targets
    excluded), agreement per criterion, and the D55 deadline."""
    rows = [r for r in [*results, *active_ratings(ratings)] if r["rubric_version"] == config.rubric_version]
    rows = [r for r in rows if target_key(r["target"]) not in leaks]
    cells: dict[str, Any] = {}
    for kind in sorted({k for r in rows for k in r["kinds"]}):
        for evaluator in ("judge", "human"):
            for criterion_id in sorted({r["criterion_id"] for r in rows}):
                found = [
                    r
                    for r in rows
                    if kind in r["kinds"] and r["evaluator"] == evaluator and r["criterion_id"] == criterion_id
                ]
                if found:
                    aspect = found[0]["aspect"]
                    cells.setdefault(kind, {}).setdefault(aspect, {}).setdefault(criterion_id, {})[evaluator] = _cell(
                        found
                    )
    categories = tuple(int(value) for value in config.scale.values)
    scored = [r for r in rows if r["score"] is not None]
    agreement = {}
    for criterion_id in sorted({r["criterion_id"] for r in scored}):
        judge = {
            r["event_id"]: r["score"] for r in scored if r["evaluator"] == "judge" and r["criterion_id"] == criterion_id
        }
        humans = [r for r in scored if r["evaluator"] == "human" and r["criterion_id"] == criterion_id]
        human_judge = [(h["score"], judge[h["event_id"]]) for h in humans if h["event_id"] in judge]
        by_event = {}
        for h in humans:
            by_event.setdefault(h["event_id"], []).append(h["score"])
        human_human = [pair for scores in by_event.values() for pair in combinations(scores, 2)]
        agreement[criterion_id] = {
            "human_judge": _agreement(human_judge, categories),
            "human_human": _agreement(human_human, categories),
        }
    return {
        "translation_run": str(run.path),
        "rubric_version": config.rubric_version,
        "excluded_leaking_targets": sorted(leaks),
        "cells": cells,
        "agreement": agreement,
        "request_deadline": _deadline(run, config, leaks),
    }
