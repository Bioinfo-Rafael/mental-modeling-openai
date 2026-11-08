"""Stores and report of the 3-points x 4-aspects evaluation: separate append-only stores, human
ratings, not_evaluated vs 0, agreement, sections and the check/import/report CLI. Offline only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import assess_cli
from evaluation.assess_data import (
    DEFAULT_CONFIG,
    REASON_INPUT_INVALID,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUATED,
    AssessError,
    load_annotation,
    load_run,
)
from evaluation.assess_report import agreement_stats, quadratic_weighted_kappa
from evaluation.assess_store import active_results, with_supersedes
from tests.llm.llm_fakes import DumpableResponse, message_item, responses_body

from .assess_fakes import (
    E_INITIAL,
    E_LATE_REQUEST,
    E_REQUEST,
    E_START,
    EPISODE_ID,
    LEGACY_CONFIG,
    find,
    make_frame,
    make_run,
    rating,
    read_jsonl,
    verdict_text,
    write_jsonl,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "examples"
SCALE = (1, 2, 3, 4, 5)


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provenance calls git twice per CLI run; tests do not need it (and stay fast)."""
    monkeypatch.setattr(assess_cli, "_git_state", lambda: {"commit": None, "dirty": None})


class AutoClient:
    """SDK stand-in: every asked criterion gets ``judgement``."""

    def __init__(self, judgement: str = "4") -> None:
        self.responses, self.judgement, self.calls = self, judgement, 0

    def create(self, **body: Any) -> object:
        self.calls += 1
        schema = body["text"]["format"]["schema"]
        ids = schema["properties"]["criteria"]["items"]["properties"]["criterion_id"]["enum"]
        return DumpableResponse(responses_body(output=[message_item(verdict_text(ids, self.judgement))]))


def _cli(tmp_path: Path, command: str, run_dir: Path, *extra: str, client: object = None) -> int:
    argv = [command, "--translator-run", str(run_dir), "--output-root", str(tmp_path / "out"), *extra]
    argv += [] if "--config" in extra else ["--config", str(LEGACY_CONFIG)]
    factory = (lambda **kwargs: client) if client is not None else None
    options: dict[str, Any] = {"episode_loader": lambda path: make_frame()}
    if factory is not None:
        options = {**options, "openai_client_factory": factory}
    return assess_cli.main(argv, **options)


def _import(tmp_path: Path, run_dir: Path, lines: list[dict[str, Any]], name: str = "ratings.jsonl") -> int:
    return _cli(tmp_path, "import-ratings", run_dir, "--ratings", str(write_jsonl(tmp_path / name, lines)))


def _report(tmp_path: Path, run_dir: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    capsys.readouterr()
    assert _cli(tmp_path, "report", run_dir) == 0
    return json.loads(Path(json.loads(capsys.readouterr().out)["report"]).read_text(encoding="utf-8"))


def _cell(
    report: dict[str, Any], kind: str, criterion_id: str, evaluator: str, section: str = "online"
) -> dict[str, Any]:
    aspect = criterion_id.split(".", 1)[0]
    return report["sections"][section]["cells"][kind][aspect][criterion_id][evaluator]


# --- separate, append-only stores (request: 人評価とJudge評価が上書きされず別々に残る) ---------------------


def test_human_and_judge_results_are_separate_and_append_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, directory = make_run(tmp_path), tmp_path / "out" / "run"
    criterion = "contribution.question_resolved"
    assert _cli(tmp_path, "check", run_dir) == 0
    assert _cli(tmp_path, "judge", run_dir, "--execute", "--confirm-paid-api", client=AutoClient("4")) == 0
    assert _import(tmp_path, run_dir, [rating("r-1", E_REQUEST, criterion, "2")], "first.jsonl") == 0
    judge_bytes = (directory / "judge_results.jsonl").read_bytes()
    machine_bytes = (directory / "machine_results.jsonl").read_bytes()
    human_bytes = (directory / "human_results.jsonl").read_bytes()

    rerating = rating("r-2", E_REQUEST, criterion, "5", supersedes="r-1", rated_at="2026-11-06T09:00:00+00:00")
    assert _import(tmp_path, run_dir, [rerating], "second.jsonl") == 0

    human = read_jsonl(directory / "human_results.jsonl")
    assert (directory / "human_results.jsonl").read_bytes().startswith(human_bytes)
    assert (directory / "judge_results.jsonl").read_bytes() == judge_bytes
    assert (directory / "machine_results.jsonl").read_bytes() == machine_bytes
    first, second = find(human, criterion_id=criterion)
    assert (first["score"], second["score"], second["supersedes"]) == (2, 5, first["result_id"])
    assert find(active_results(human), criterion_id=criterion) == [second]
    assert {r["evaluator"]["kind"] for r in read_jsonl(directory / "judge_results.jsonl")} == {"judge"}
    report = _report(tmp_path, run_dir, capsys)
    assert _cell(report, "request", criterion, "human")["mean"] == 5.0
    assert _cell(report, "request", criterion, "judge")["mean"] == 4.0  # shown beside humans, never merged
    assert report["overall"]["status"] == "not_evaluated" and report["overall"]["reason"] == "aggregation_unset"


# --- not_evaluated is never 0 (request: 評価不能が0点と区別される) ----------------------------------------


def test_cannot_judge_is_not_a_score(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir, criterion = make_run(tmp_path), "correctness.records_to_t"
    lines = [rating("r-1", E_REQUEST, criterion, "cannot_judge"), rating("r-2", E_LATE_REQUEST, criterion, "3")]
    assert _import(tmp_path, run_dir, lines) == 0

    (stored, _) = find(read_jsonl(tmp_path / "out" / "run" / "human_results.jsonl"), criterion_id=criterion)
    cell = _cell(_report(tmp_path, run_dir, capsys), "request", criterion, "human")
    report = _report(tmp_path, run_dir, capsys)

    assert (stored["status"], stored["score"], stored["reason"], stored["judgement"]) == (
        STATUS_NOT_EVALUATED,
        None,
        "cannot_judge",
        "cannot_judge",
    )
    assert (cell["n_evaluated"], cell["mean"], cell["distribution"]) == (1, 3.0, {"3": 1})
    assert cell["not_evaluated"] == {"cannot_judge": 1}
    judge = _cell(report, "request", criterion, "judge")
    assert (judge["n_evaluated"], judge["mean"], judge["not_evaluated"]) == (0, None, {"no_judge_result": 2})
    pair = report["sections"]["online"]["agreement"][criterion]["human_judge"]
    assert pair == {"pairs": 0, "exact_agreement": None, "weighted_kappa": None, "reason": "fewer_than_2_pairs"}


def test_missing_ratings_are_no_human_rating_and_forced_timing_is_marked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = make_run(tmp_path)

    report = _report(tmp_path, run_dir, capsys)

    why = _cell(report, "start", "elements.why", "human")
    timing = _cell(report, "start", "timing.start_in_time", "judge")
    assert (why["mean"], why["not_evaluated"]) == (None, {"no_human_rating": 2})  # initial + detector start
    assert timing["not_evaluated"] == {"forced_trigger": 1, "no_judge_result": 1}
    assert report["sections"]["retrospective"] is None


# --- human-rating import ---------------------------------------------------------------------------------


def test_example_ratings_import_cleanly(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)

    ratings = str(EXAMPLES / "human-ratings.example.jsonl")  # the examples follow the default rubric
    assert _cli(tmp_path, "import-ratings", run_dir, "--ratings", ratings, "--config", str(DEFAULT_CONFIG)) == 0

    human = read_jsonl(tmp_path / "out" / "run" / "human_results.jsonl")
    reasons = {r["evaluator"].get("rating_id"): r["reason"] for r in human}
    assert reasons["r-0002"] == "cannot_judge" and reasons["r-0003"] == "not_applicable"
    assert {r["evaluator"]["rater_id"] for r in human} == {"rater-01"}
    assert len(find(human, criterion_id="overall")) == 2  # one per rater and event


def test_example_annotation_loads(tmp_path: Path) -> None:
    annotation = load_annotation(EXAMPLES / "assess-points.example.yaml", load_run(make_run(tmp_path)))

    assert [p.kind for p in annotation.points] == ["start", "change", "change"]


@pytest.mark.parametrize(
    "bad",
    [
        rating("r-2", "placeholder-episode-01:aircraft:A0100:000099", "elements.why", "3"),  # unknown event
        rating("r-2", E_REQUEST, "elements.why", "3", rubric_version="pilot-0"),
        rating("r-2", E_REQUEST, "elements.why", "not_applicable"),  # not a conditional element
        rating("r-2", E_REQUEST, "correctness.start_plan", "3"),  # not a request criterion
        rating("r-2", E_INITIAL, "timing.start_in_time", "3"),  # forced_trigger: not rated
        rating("r-2", E_REQUEST, "elements.why", "0"),
        rating("r-1", E_REQUEST, "elements.what_now", "3"),  # rating id repeated
        rating("r-2", E_START, "elements.why", "3", rated_at="2026-11-05"),  # no UTC offset
        {**rating("r-2", E_START, "elements.why", "3"), "extra": 1},
    ],
)
def test_import_validates_the_whole_file_before_appending(tmp_path: Path, bad: dict[str, Any]) -> None:
    run_dir = make_run(tmp_path)

    assert _import(tmp_path, run_dir, [rating("r-1", E_START, "elements.why", "4"), bad]) == 1

    assert not (tmp_path / "out" / "run" / "human_results.jsonl").exists()


def test_a_second_rating_of_the_same_criterion_needs_supersedes(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    assert _import(tmp_path, run_dir, [rating("r-1", E_START, "elements.why", "4")], "a.jsonl") == 0

    again = _import(tmp_path, run_dir, [rating("r-2", E_START, "elements.why", "2")], "b.jsonl")
    other_rater = _import(
        tmp_path, run_dir, [rating("r-3", E_START, "elements.why", "2", rater_id="rater-02")], "c.jsonl"
    )

    assert (again, other_rater) == (1, 0)
    assert len(find(read_jsonl(tmp_path / "out" / "run" / "human_results.jsonl"), criterion_id="elements.why")) == 2


def test_ratings_of_an_invalid_target_are_kept_as_generation_input_invalid(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, extra_text={E_START: EPISODE_ID})

    assert _import(tmp_path, run_dir, [rating("r-1", E_REQUEST, "elements.why", "4")]) == 0

    (record,) = find(read_jsonl(tmp_path / "out" / "run" / "human_results.jsonl"), criterion_id="elements.why")
    assert (record["status"], record["reason"], record["score"]) == (STATUS_NOT_EVALUATED, REASON_INPUT_INVALID, None)
    assert record["judgement"] == "4" and record["rationale"]


# --- agreement ---------------------------------------------------------------------------------------------


def test_agreement_statistics_are_null_with_a_reason_never_zero() -> None:
    assert quadratic_weighted_kappa([(1, 1), (3, 3), (5, 5)], SCALE) == 1.0
    assert quadratic_weighted_kappa([(1, 5), (5, 1)], SCALE) == -1.0
    assert agreement_stats([(4, 4)], SCALE) == {
        "pairs": 1,
        "exact_agreement": None,
        "weighted_kappa": None,
        "reason": "fewer_than_2_pairs",
    }
    assert agreement_stats([(4, 4), (4, 4)], SCALE) == {
        "pairs": 2,
        "exact_agreement": 1.0,
        "weighted_kappa": None,
        "reason": "no_variance",
    }
    partial = agreement_stats([(4, 4), (2, 3), (5, 4)], SCALE)
    assert partial["exact_agreement"] == round(1 / 3, 4) and 0 < partial["weighted_kappa"] < 1


def test_report_agreement_pairs_humans_with_the_judge_and_with_each_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, criterion = make_run(tmp_path), "contribution.question_resolved"
    assert _cli(tmp_path, "judge", run_dir, "--execute", "--confirm-paid-api", client=AutoClient("4")) == 0
    lines = [
        rating("r-1", E_REQUEST, criterion, "4"),
        rating("r-2", E_LATE_REQUEST, criterion, "2"),
        rating("r-3", E_REQUEST, criterion, "4", rater_id="rater-02"),
        rating("r-4", E_LATE_REQUEST, criterion, "3", rater_id="rater-02"),
    ]
    assert _import(tmp_path, run_dir, lines) == 0

    agreement = _report(tmp_path, run_dir, capsys)["sections"]["online"]["agreement"][criterion]

    assert agreement["human_judge"]["pairs"] == 4 and agreement["human_judge"]["exact_agreement"] == 0.5
    assert agreement["human_human"]["pairs"] == 2 and agreement["human_human"]["exact_agreement"] == 0.5


# --- report sections and the check CLI ------------------------------------------------------------------


def test_retrospective_run_is_reported_in_its_own_section(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = make_run(tmp_path, visibility={"mode": "retrospective", "future_steps": 2}, rendered_future=2)
    assert _cli(tmp_path, "check", run_dir) == 0

    report = _report(tmp_path, run_dir, capsys)

    assert report["sections"]["online"] is None and report["visibility"]["retrospective"] is True
    assert report["sections"]["retrospective"]["leakage"]["ok"] == 17
    machine = read_jsonl(tmp_path / "out" / "run" / "machine_results.jsonl")
    assert machine and all(record["retrospective"] for record in machine)


def test_check_reruns_append_and_supersede(tmp_path: Path) -> None:
    run_dir, directory = make_run(tmp_path), tmp_path / "out" / "run"
    assert _cli(tmp_path, "check", run_dir) == 0
    first = (directory / "machine_results.jsonl").read_bytes()

    assert _cli(tmp_path, "check", run_dir) == 0

    records = read_jsonl(directory / "machine_results.jsonl")
    assert (directory / "machine_results.jsonl").read_bytes().startswith(first)
    batches = {record["result_id"].split(":")[1] for record in records}
    active = active_results(records)
    assert len(batches) == 2 and len(active) == len(records) // 2
    assert all(record["supersedes"] for record in active)
    assert len(read_jsonl(directory / "leakage.jsonl")) == 2 * 17
    assert oct(directory.stat().st_mode & 0o777) == "0o700"


def test_check_exits_1_on_leaks_but_keeps_its_results(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, extra_text={E_REQUEST: '\n{"step":11,"relative_step":1,"entities":[]}'})

    assert _cli(tmp_path, "check", run_dir) == 1

    leakage = read_jsonl(tmp_path / "out" / "run" / "leakage.jsonl")
    assert [line["status"] for line in leakage if line["call_id"] == E_REQUEST] == ["leak"]
    machine = read_jsonl(tmp_path / "out" / "run" / "machine_results.jsonl")
    assert {r["reason"] for r in machine if r["aspect"] != "overall"} == {REASON_INPUT_INVALID}


def test_an_assessment_directory_belongs_to_one_translator_run(tmp_path: Path) -> None:
    first = make_run(tmp_path, "first")
    second = make_run(tmp_path, "second", status="failed")
    assert _cli(tmp_path, "check", first, "--name", "shared") == 0

    assert _cli(tmp_path, "check", second, "--name", "shared") == 1

    assert {r["status"] for r in read_jsonl(tmp_path / "out" / "shared" / "machine_results.jsonl")} >= {
        STATUS_EVALUATED
    }


def test_rating_ids_are_local_to_their_rater(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    lines = [
        rating("r-1", E_START, "elements.why", "4"),
        rating("r-1", E_START, "elements.why", "2", rater_id="rater-02"),
        rating("r-2", E_START, "elements.why", "3", rater_id="rater-02", supersedes="r-1"),
    ]

    assert _import(tmp_path, run_dir, lines) == 0

    human = find(read_jsonl(tmp_path / "out" / "run" / "human_results.jsonl"), criterion_id="elements.why")
    active = {r["evaluator"]["rater_id"]: r["score"] for r in active_results(human)}
    assert active == {"rater-01": 4, "rater-02": 3}
    assert human[2]["supersedes"] == "human:rater-02:r-1"


def test_supersedes_chains_inside_one_batch_and_rejects_repeated_ids() -> None:
    def record(result_id: str) -> dict[str, Any]:
        subject = {"event_id": "e", "reference_point_id": None, "target": {"kind": "aircraft", "id": "A0100"}}
        evaluator = {"kind": "machine"}
        return {
            "result_id": result_id,
            "evaluator": evaluator,
            "subject": subject,
            "criterion_id": "c",
            "rubric_version": "p",
            "supersedes": None,
        }

    stored = with_supersedes([record("b"), record("c")], [record("a")])

    assert [r["supersedes"] for r in stored] == ["a", "b"]
    assert [r["result_id"] for r in active_results([record("a"), *stored])] == ["c"]
    with pytest.raises(AssessError):
        with_supersedes([record("b"), record("b")], [])
