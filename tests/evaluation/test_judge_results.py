"""The causality audit, human ratings and the report (cells, agreement, the D55 request deadline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import cli
from evaluation.audit import audit_run, leaking_targets
from evaluation.config import load_config
from evaluation.material import load_run
from evaluation.results import RatingsError, build_report, import_ratings, quadratic_weighted_kappa

from .fakes import A0100, JUDGE_CONFIG, TEAM_A, Factory, event, judge_config, request_line, write_run

QUESTION = {"request_id": "q1", "step": 30, "text": "[TEST] Why turn left?"}
LATE = {"instruction_id": "i2", "issue_step": 90, "target": A0100, "text": "[TEST] the late instruction"}


def _rating(rating_id: str, event_id: str, criterion: str, judgement: str = "4", **fields: object) -> dict:
    line = {"rating_id": rating_id, "rater_id": "rater-01", "event_id": event_id, "criterion_id": criterion}
    line = {**line, "judgement": judgement, "rationale": "r", "rubric_version": "pilot-5"}
    return {**line, "rated_at": "2026-11-09T10:00:00+09:00", "supersedes": None, **fields}


def test_audit_flags_a_future_window_line_a_future_message_and_a_run_id(tmp_path: Path) -> None:
    records = [
        event(30, ("request",), question=QUESTION),
        event(90, ("start",), target=TEAM_A, instructions_in_effect=[LATE]),
    ]
    leaky = '{"step":31,"relative_step":1}\n[TEST] the late instruction\nmhi_v1.3_ic100601_win'
    requests = [request_line(records[0], leaky), request_line(records[1])]

    audit = audit_run(load_run(write_run(tmp_path / "run", records, requests)))

    assert [record["status"] for record in audit] == ["leak", "ok"]
    assert len(audit[0]["problems"]) == 3
    assert leaking_targets(audit) == {"aircraft:A0100"}


def test_a_leaking_target_is_excluded_from_judging(tmp_path: Path) -> None:
    records = [event(30, ("request",), question=QUESTION), event(45, ("start",), target=TEAM_A)]
    requests = [request_line(records[0], '{"step":40,"relative_step":10}'), request_line(records[1])]
    run_dir = write_run(tmp_path / "runs" / "trial", records, requests)

    code = cli.main(["judge", "--run", str(run_dir), "--config", str(judge_config(tmp_path))], client_factory=Factory())

    preview = json.loads((tmp_path / "evaluation" / "trial" / "judge" / "preview.json").read_text())
    assert code == 0 and [r["event_id"] for r in preview["requests"]] == [records[1]["event_id"]]


def test_ratings_import_checks_the_event_criterion_and_re_ratings(tmp_path: Path) -> None:
    record = event(30, ("request",), question=QUESTION)
    run, config = load_run(write_run(tmp_path / "run", [record])), load_config(JUDGE_CONFIG)
    first = import_ratings([_rating("r1", record["event_id"], "elements.what_now")], run, config, [])

    again = _rating("r2", record["event_id"], "elements.what_now", "5", supersedes="r1")
    assert import_ratings([again], run, config, first)[0]["score"] == 5
    for bad in (
        _rating("r2", record["event_id"], "elements.what_now"),  # a second rating without supersedes
        _rating("r2", record["event_id"], "correctness.start_plan"),  # not a criterion of a request
        _rating("r2", "nope", "elements.what_now"),
        _rating("r2", record["event_id"], "elements.why", "not_applicable"),  # elements.why is not conditional
        _rating("r2", record["event_id"], "elements.why", rated_at="2026-11-09T10:00:00"),
        _rating("r2", record["event_id"], "elements.why", rationale=" "),
    ):
        with pytest.raises(RatingsError):
            import_ratings([bad], run, config, first)


def test_the_report_counts_scores_agreement_and_the_request_deadline(tmp_path: Path) -> None:
    records = [event(30, ("request",), question=QUESTION), {**event(60, ("request",)), "processing_s": 4.2}]
    run, config = load_run(write_run(tmp_path / "run", records)), load_config(JUDGE_CONFIG)
    results = [
        {**_rating(f"j{i}", r["event_id"], "elements.what_now", str(score)), "evaluator": "judge"}
        for i, (r, score) in enumerate(zip(records, (4, 2), strict=True))
    ]
    results = [
        {**row, "target": A0100, "kinds": ["request"], "aspect": "elements", "score": int(row["judgement"])}
        for row in results
    ]
    ratings = import_ratings(
        [
            _rating(f"h{i}", r["event_id"], "elements.what_now", s)
            for i, (r, s) in enumerate(zip(records, ("4", "cannot_judge"), strict=True))
        ],
        run,
        config,
        [],
    )

    report = build_report(run, config, results, ratings, set())

    cell = report["cells"]["request"]["elements"]["elements.what_now"]
    assert cell["judge"]["mean"] == 3.0 and cell["human"]["not_scored"] == {"cannot_judge": 1}
    assert report["agreement"]["elements.what_now"]["human_judge"]["pairs"] == 1
    assert report["request_deadline"] == {
        "deadline_s": 3.0,
        "requests": 2,
        "measured": 2,
        "passed": 1,
        "pass_rate": 0.5,
    }


def test_quadratic_weighted_kappa_is_one_for_perfect_agreement_and_none_without_variance() -> None:
    assert quadratic_weighted_kappa([(1, 1), (5, 5), (3, 3)], (1, 2, 3, 4, 5)) == 1.0
    assert quadratic_weighted_kappa([(3, 3), (3, 3)], (1, 2, 3, 4, 5)) is None
