"""Translator evaluation: aggregation and report (one JSON file per report, never overwritten).

``cells[point_kind][aspect][criterion_id][evaluator]`` and the agreement statistics never count a
``not_evaluated`` result (missing rating, ``cannot_judge``, invalid generation input, ...) as 0;
too few pairs or no variance give ``null`` with a reason. Online and retrospective runs go to
separate sections (D46), and the overall result stays ``not_evaluated/aggregation_unset`` (D54).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, NamedTuple

from .assess_checks import CHECK_ERROR, CHECK_LEAK, CHECK_OK, CallCheck, invalid_targets
from .assess_data import (
    ASPECT_CORRECTNESS,
    ASPECT_OVERALL,
    ASPECT_TIMING,
    EVALUATOR_HUMAN,
    EVALUATOR_JUDGE,
    EVALUATOR_MACHINE,
    REASON_AGGREGATION_UNSET,
    REASON_INPUT_INVALID,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUATED,
    AssessConfig,
    TranslatorRun,
    event_criteria,
    target_key,
)
from .assess_store import active_results

REPORT_SCHEMA_VERSION = 1
MIN_PAIRS = 2
STAT_DECIMALS = 4
RATED_EVALUATORS = (EVALUATOR_HUMAN, EVALUATOR_JUDGE)
REASON_NO_RESULT = {EVALUATOR_HUMAN: "no_human_rating", EVALUATOR_JUDGE: "no_judge_result"}
TIMING_KEYS = ("n_records", "n_evaluated", "passed", "failed", "not_evaluated")

Index = dict[tuple[str, str], list[dict[str, Any]]]  # (event id, criterion id) -> records


class Entry(NamedTuple):
    status: str
    score: int | None
    reason: str | None
    passed: object  # a machine verdict: True / False; None otherwise


# --- cells ------------------------------------------------------------------------------------------


def _entry(record: dict[str, Any], invalid: frozenset[tuple[str, str]]) -> Entry:
    if target_key(record["subject"]["target"]) in invalid:
        return Entry(STATUS_NOT_EVALUATED, None, REASON_INPUT_INVALID, None)
    measurement = record["measurement"] if isinstance(record["measurement"], dict) else {}
    return Entry(record["status"], record["score"], record["reason"], measurement.get("passed"))


def _cell(entries: list[Entry], evaluator: str) -> dict[str, Any]:
    """Machine cells count verdicts (passed / failed); rated cells give the mean over evaluated scores
    only (``null`` when there is none) and the distribution. not_evaluated never counts as 0."""
    evaluated = [entry for entry in entries if entry.status == STATUS_EVALUATED]
    reasons = Counter(entry.reason or "unknown" for entry in entries if entry.status != STATUS_EVALUATED)
    counts = {"n_records": len(entries), "n_evaluated": len(evaluated), "not_evaluated": dict(reasons)}
    if evaluator == EVALUATOR_MACHINE:
        passed = sum(1 for entry in evaluated if entry.passed is True)
        return {**counts, "passed": passed, "failed": sum(1 for entry in evaluated if entry.passed is False)}
    scores = [entry.score for entry in evaluated if entry.score is not None]
    distribution = {str(value): scores.count(value) for value in sorted(set(scores))}
    mean = round(sum(scores) / len(scores), STAT_DECIMALS) if scores else None
    return {**counts, "mean": mean, "distribution": distribution}


def _cells(
    run: TranslatorRun,
    config: AssessConfig,
    machine: list[dict[str, Any]],
    rated: dict[str, Index],
    invalid: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    """Human and judge cells cover the expected grid (every criterion of every event, missing
    included); machine cells cover the stored machine records."""
    entries: dict[tuple[str, str, str, str], list[Entry]] = {}  # (kind, aspect, criterion id, evaluator)
    references: dict[str, str | None] = {}
    for event in run.events:
        criteria = event_criteria(event, config)
        excluded = {criterion.criterion_id: reason for criterion, reason in criteria.excluded}
        default = REASON_INPUT_INVALID if target_key(event["target"]) in invalid else None
        for criterion in (*criteria.asked, *(criterion for criterion, _ in criteria.excluded)):
            references[criterion.criterion_id] = criterion.reference
            for evaluator in RATED_EVALUATORS:
                missing = excluded.get(criterion.criterion_id) or default or REASON_NO_RESULT[evaluator]
                found = rated[evaluator].get((event["event_id"], criterion.criterion_id), [])
                items = [_entry(record, invalid) for record in found] or [
                    Entry(STATUS_NOT_EVALUATED, None, missing, None)
                ]
                for kind in criteria.kinds_of[criterion.criterion_id]:
                    entries.setdefault((kind, criterion.aspect, criterion.criterion_id, evaluator), []).extend(items)
    for record in machine:
        references[record["criterion_id"]] = record["reference"]
        for kind in record["subject"]["point_kinds"]:
            key = (kind, record["aspect"], record["criterion_id"], EVALUATOR_MACHINE)
            entries.setdefault(key, []).append(_entry(record, invalid))
    tree: dict[str, Any] = {}
    for (kind, aspect, criterion_id, evaluator), items in sorted(entries.items()):
        cell = tree.setdefault(kind, {}).setdefault(aspect, {}).setdefault(criterion_id, {})
        cell[evaluator] = _cell(items, evaluator)
        if aspect == ASPECT_CORRECTNESS:
            cell["reference"] = references[criterion_id]
    return tree


# --- timing and leakage summaries -------------------------------------------------------------------


def _subject_label(subject: dict[str, Any]) -> str:
    target = f"{subject['target']['kind']}:{subject['target']['id']}"
    if subject["reference_point_id"]:
        return f"point {subject['reference_point_id']} ({target})"
    return subject["event_id"] or f"target {target}"


def timing_summary(machine: list[dict[str, Any]], invalid: frozenset[tuple[str, str]]) -> dict[str, Any]:
    """Machine timing records per criterion: verdict counts, reasons and every raw measurement."""
    by_criterion: dict[str, list[dict[str, Any]]] = {}
    for record in machine:
        if record["aspect"] == ASPECT_TIMING:
            by_criterion.setdefault(record["criterion_id"], []).append(record)
    summary = {}
    for criterion_id, records in sorted(by_criterion.items()):
        cell = _cell([_entry(record, invalid) for record in records], EVALUATOR_MACHINE)
        measurements = [
            {"subject": _subject_label(record["subject"]), **record["measurement"]}
            for record in records
            if record["measurement"] is not None
        ]
        summary[criterion_id] = {**{key: cell[key] for key in TIMING_KEYS}, "measurements": measurements}
    return summary


def leakage_summary(checks: tuple[CallCheck, ...]) -> dict[str, Any]:
    statuses = [check.status for check in checks]
    return {
        "calls": len(checks),
        CHECK_OK: statuses.count(CHECK_OK),
        CHECK_LEAK: statuses.count(CHECK_LEAK),
        CHECK_ERROR: statuses.count(CHECK_ERROR),
        "invalid_targets": [f"{kind}:{target_id}" for kind, target_id in sorted(invalid_targets(checks))],
    }


# --- agreement ----------------------------------------------------------------------------------------


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
    if expected == 0:
        return None
    return round(1 - observed / expected, STAT_DECIMALS)


def agreement_stats(pairs: list[tuple[int, int]], categories: tuple[int, ...]) -> dict[str, Any]:
    if len(pairs) < MIN_PAIRS:
        return {"pairs": len(pairs), "exact_agreement": None, "weighted_kappa": None, "reason": "fewer_than_2_pairs"}
    exact = round(sum(1 for a, b in pairs if a == b) / len(pairs), STAT_DECIMALS)
    kappa = quadratic_weighted_kappa(pairs, categories)
    return {
        "pairs": len(pairs),
        "exact_agreement": exact,
        "weighted_kappa": kappa,
        "reason": "no_variance" if kappa is None else None,
    }


def _scored(records: list[dict[str, Any]], invalid: frozenset[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["status"] == STATUS_EVALUATED
        and record["score"] is not None
        and target_key(record["subject"]["target"]) not in invalid
    ]


def agreement(human: Index, judge: Index, categories: tuple[int, ...], invalid: frozenset[tuple[str, str]]) -> dict:
    """Per criterion: each rater's score against the judge's, and every pair of raters."""
    pairs: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for key in sorted(set(human) | set(judge)):  # key = (event id, criterion id)
        humans = sorted(_scored(human.get(key, []), invalid), key=lambda r: r["evaluator"]["rater_id"])
        judged = _scored(judge.get(key, []), invalid)[-1:]  # one active judge result per rubric version
        found = pairs.setdefault(key[1], {"human_judge": [], "human_human": []})
        found["human_judge"].extend((h["score"], j["score"]) for h in humans for j in judged)
        found["human_human"].extend((a["score"], b["score"]) for a, b in combinations(humans, 2))
    return {
        criterion_id: {name: agreement_stats(items, categories) for name, items in kinds.items()}
        for criterion_id, kinds in pairs.items()
    }


# --- report -------------------------------------------------------------------------------------------


def _index(records: list[dict[str, Any]]) -> Index:
    index: Index = {}
    for record in records:
        index.setdefault((record["subject"]["event_id"], record["criterion_id"]), []).append(record)
    return index


def _section(
    run: TranslatorRun,
    config: AssessConfig,
    checks: tuple[CallCheck, ...],
    stores: dict[str, list[dict[str, Any]]],
    rubric_version: str,
) -> dict[str, Any]:
    invalid = invalid_targets(checks)
    active = {
        kind: [r for r in active_results(records) if r["aspect"] != ASPECT_OVERALL] for kind, records in stores.items()
    }
    rated = {
        kind: _index([r for r in active[kind] if r["rubric_version"] == rubric_version]) for kind in RATED_EVALUATORS
    }
    excluded = {
        kind: sum(1 for r in active[kind] if r["rubric_version"] != rubric_version) for kind in RATED_EVALUATORS
    }
    categories = tuple(int(value) for value in config.scale.values)
    return {
        "leakage": leakage_summary(checks),
        "timing": timing_summary(active[EVALUATOR_MACHINE], invalid),
        "cells": _cells(run, config, active[EVALUATOR_MACHINE], rated, invalid),
        "agreement": agreement(rated[EVALUATOR_HUMAN], rated[EVALUATOR_JUDGE], categories, invalid),
        "excluded_other_rubric_versions": excluded,
    }


def build_report(
    *,
    run: TranslatorRun,
    config: AssessConfig,
    checks: tuple[CallCheck, ...],
    stores: dict[str, list[dict[str, Any]]],
    rubric_version: str,
    generated_at: str,
) -> dict[str, Any]:
    """The report of one assessment directory. ``stores`` maps evaluator kind -> every stored record."""
    section = _section(run, config, checks, stores, rubric_version)
    overall_rows = {
        kind: sum(1 for r in active_results(records) if r["aspect"] == ASPECT_OVERALL)
        for kind, records in stores.items()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "translator_run": {"path": str(run.run_dir), "files_sha256": run.file_sha256, "status": run.status},
        "config_sha256": config.sha256,
        "rubric_version": rubric_version,
        "visibility": {"future_steps": run.future_steps, "retrospective": run.retrospective},
        "sections": {
            "online": None if run.retrospective else section,
            "retrospective": section if run.retrospective else None,
        },
        "overall": {"status": STATUS_NOT_EVALUATED, "reason": REASON_AGGREGATION_UNSET, "rows": overall_rows},
        "open_items": config.open_items,
    }
