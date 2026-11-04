"""Machine side of the 3-points x 4-aspects evaluation: config, leakage check, points, metrics.

Offline: a hand-written v2 Translator run (``assess_fakes``); nothing imports the Translator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.assess_checks import (
    CHECK_ERROR,
    CHECK_LEAK,
    CHECK_OK,
    check_generation_inputs,
    machine_results,
    match_points,
)
from evaluation.assess_data import (
    REASON_INPUT_INVALID,
    STATUS_EVALUATED,
    STATUS_NOT_EVALUATED,
    AnnotationError,
    ConfigError,
    load_annotation,
    load_config,
    load_run,
    parse_config,
)

from .assess_fakes import (
    E_CHANGE,
    E_INITIAL,
    E_LATE_REQUEST,
    E_REQUEST,
    E_START,
    EPISODE_ID,
    INSTRUCTIONS,
    LAST_STEP,
    SCENARIO_ID,
    build_run,
    canonical,
    config_raw,
    detector_id,
    find,
    make_config,
    make_run,
    point,
    question,
    sha256_text,
    with_value,
    write_points,
    write_run,
)

TIMING_KEYS = (
    "request_deadline_s",
    "min_question_interval_s",
    "start_min_lead_steps",
    "change_max_delay_steps",
    "max_events_per_segment",
    "max_utterances_per_target",
)


def _machine(run_dir: Path, *, points: Path | None = None, **config_overrides: object) -> list[dict]:
    run, config = load_run(run_dir), make_config(**config_overrides)
    checks = check_generation_inputs(run, config)
    annotation = load_annotation(points, run) if points else None
    return machine_results(run, config, match_points(annotation, run), checks, "batch-1")


# --- config ---------------------------------------------------------------------------------------


def test_committed_config_loads_with_undecided_values_null() -> None:
    config = load_config()

    assert config.judge.model == "gpt-6.1-sol" and config.judge.max_calls == 100
    assert config.judge.rationale_language == "Japanese"
    assert (config.timing.request_deadline_s, config.timing.min_question_interval_s) == (3.0, 10.0)
    assert config.timing.start_min_lead_steps is None and config.timing.change_max_delay_steps is None
    assert config.timing.max_events_per_segment is None and config.timing.max_utterances_per_target is None
    assert set(config.open_items.values()) == {None}
    correctness = [c for items in config.criteria.values() for c in items if c.aspect == "correctness"]
    assert correctness and all(c.reference in ("observable_records", "future_behaviour") for c in correctness)


@pytest.mark.parametrize(
    ("dotted", "value", "message"),
    [
        ("timing.unknown_key", 1, "unknown"),
        ("judge.model", "gpt-6.1-terra", "gpt-6.1-sol"),
        ("open_items.tactic_vocabulary", ["attack"], "must stay null"),
        ("timing.request_deadline_s", 0, "null or a number > 0"),
        ("timing.max_events_per_segment", 0, "integer >= 1"),
        ("scale.values", ["1", "1"], "increasing"),
    ],
)
def test_config_rejects_invalid_values(dotted: str, value: object, message: str) -> None:
    raw = with_value(config_raw(), dotted, value)

    with pytest.raises(ConfigError, match=message):
        parse_config(raw, path=Path("assess.yaml"), sha256="x")


def test_correctness_criterion_without_observable_reference_is_rejected() -> None:
    raw = config_raw()
    internal = {"id": "correctness.internal_reason", "text": "RL の内的理由と一致するか", "levels": LEVELS}
    start = [*raw["criteria"]["start"], internal]

    with pytest.raises(ConfigError, match="reference .* is required for correctness criteria"):
        parse_config(with_value(raw, "criteria.start", start), path=Path("assess.yaml"), sha256="x")


# --- score anchors (D65) ----------------------------------------------------------------------------

LEVELS = {value: f"レベル{value}の条件" for value in ("1", "2", "3", "4", "5")}


def test_every_committed_criterion_has_one_non_blank_anchor_per_level() -> None:
    config = load_config()

    criteria = {c.criterion_id: c for items in config.criteria.values() for c in items}

    assert config.rubric_version == "pilot-2" and config.scale.values == ("1", "2", "3", "4", "5")
    assert len(criteria) == 24
    for criterion in criteria.values():
        assert list(criterion.levels) == ["1", "2", "3", "4", "5"], criterion.criterion_id
        assert all(anchor.strip() for anchor in criterion.levels.values()), criterion.criterion_id
        assert len(set(criterion.levels.values())) == len(LEVELS), criterion.criterion_id  # no copied level
        assert (criterion.not_applicable is not None) == criterion.conditional, criterion.criterion_id
    conditional = {c.criterion_id for c in criteria.values() if c.conditional}
    assert conditional == {
        "elements.target",
        "elements.wingman_role",
        "elements.instruction_relation",
        "elements.priority",
        "correctness.change_delta",  # no PREVIOUS (first explanation of the target) => not_applicable
        "correctness.consistent_with_latest",
    }
    for criterion_id in ("correctness.change_delta", "correctness.consistent_with_latest"):
        assert criteria[criterion_id].not_applicable == "PREVIOUS が無い（この対象の最初の説明）"
        assert criteria[criterion_id].cannot_judge is None  # no PREVIOUS is no longer cannot_judge


def _with_first_criterion(**changes: object) -> dict:
    raw = config_raw()
    first = {**raw["criteria"]["elements"][0], **changes}
    return with_value(raw, "criteria.elements", [first, *raw["criteria"]["elements"][1:]])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"levels": {k: v for k, v in LEVELS.items() if k != "3"}}, r"levels must have exactly"),
        ({"levels": {**LEVELS, "6": "x"}}, r"levels must have exactly"),
        ({"levels": {**LEVELS, "2": "  "}}, "anchors must be non-blank"),
        ({"levels": {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"}}, "levels"),  # unquoted YAML keys are integers
        ({"levels": None}, "levels"),
        ({"not_applicable": "場面に無い"}, "not_applicable .* required for conditional criteria, and only there"),
        ({"cannot_judge": " "}, "cannot_judge must be non-blank"),
    ],
)
def test_config_rejects_incomplete_or_misplaced_anchors(changes: dict, message: str) -> None:
    raw = _with_first_criterion(**changes)

    with pytest.raises(ConfigError, match=message):
        parse_config(raw, path=Path("assess.yaml"), sha256="x")


def test_config_rejects_a_criterion_without_levels_and_a_conditional_without_not_applicable() -> None:
    raw = config_raw()
    no_levels = {k: v for k, v in raw["criteria"]["elements"][0].items() if k != "levels"}
    target = {k: v for k, v in raw["criteria"]["elements"][2].items() if k != "not_applicable"}

    with pytest.raises(ConfigError, match="must have id, text and levels"):
        parse_config(with_value(raw, "criteria.elements", [no_levels]), path=Path("a.yaml"), sha256="x")
    with pytest.raises(ConfigError, match="not_applicable"):
        parse_config(with_value(raw, "criteria.elements", [target]), path=Path("a.yaml"), sha256="x")


def test_config_rejects_the_old_per_aspect_scale_levels() -> None:
    raw = with_value(config_raw(), "scale.levels", {"elements": LEVELS})

    with pytest.raises(ConfigError, match="scale"):
        parse_config(raw, path=Path("assess.yaml"), sha256="x")


def test_a_shared_criterion_with_different_anchors_is_rejected() -> None:
    raw = config_raw()
    request = raw["criteria"]["request"]
    delegation = next(c for c in request if c["id"] == "contribution.delegation")
    changed = [c if c is not delegation else {**c, "levels": LEVELS} for c in request]

    with pytest.raises(ConfigError, match="defined differently"):
        parse_config(with_value(raw, "criteria.request", changed), path=Path("assess.yaml"), sha256="x")


# --- leakage check (request: 未来情報の混入を検出できる) ---------------------------------------------


def test_check_reports_every_call_ok_for_a_clean_run(tmp_path: Path) -> None:
    run = load_run(make_run(tmp_path))

    checks = check_generation_inputs(run, make_config())

    assert len(checks) == len(run.requests) == len(run.events) + 12  # 5 translations + 12 detector ticks
    assert {check.status for check in checks} == {CHECK_OK}
    assert all(check.visible_end_step == check.step for check in checks)


FUTURE_LINE = '\n{"step":11,"time_s":2.2,"relative_step":1,"entities":[]}'
LEAK_CASES = {
    "trajectory_line_after_t": ({"extra_text": {E_REQUEST: FUTURE_LINE}}, E_REQUEST, CHECK_LEAK, "trajectory_future"),
    "future_alarm_text": (
        {"extra_text": {E_REQUEST: "\n[PLACEHOLDER] Missile warning."}},
        E_REQUEST,
        CHECK_LEAK,
        "future_message",
    ),
    "future_question_text": (
        {"extra_text": {E_START: "\n" + question(10)["text"]}},
        E_START,
        CHECK_LEAK,
        "future_message",
    ),
    "future_instruction_text": (
        {"extra_text": {detector_id(15): "\n" + INSTRUCTIONS[1]["text"]}},
        detector_id(15),
        CHECK_LEAK,
        "future_message",
    ),
    "listed_future_alarm": (
        {"input_changes": {E_REQUEST: {"alarms": [{"alarm_id": "a-x", "step": 12, "targets": [], "text": "x"}]}}},
        E_REQUEST,
        CHECK_LEAK,
        "future_message",
    ),
    "previous_translation_not_before_t": (
        {"input_changes": {E_REQUEST: {"previous_translation": {"event_id": E_REQUEST, "step": 10, "intent": "x"}}}},
        E_REQUEST,
        CHECK_LEAK,
        "previous_translation",
    ),
    "own_episode_id": ({"extra_text": {E_CHANGE: f"\n{EPISODE_ID}"}}, E_CHANGE, CHECK_LEAK, "run_identifier"),
    "own_scenario_id": ({"extra_text": {detector_id(20): SCENARIO_ID}}, detector_id(20), CHECK_LEAK, "run_identifier"),
    "undeclared_future_range": ({"rendered_future": 2}, E_START, CHECK_LEAK, "trajectory_future"),
    "visibility_not_declared": ({"visibility": None}, E_START, CHECK_ERROR, "visibility"),
    "payload_hash_mismatch": ({"after_hash_text": {E_REQUEST: " "}}, E_REQUEST, CHECK_ERROR, "request_sha256"),
}


@pytest.mark.parametrize("case", sorted(LEAK_CASES))
def test_check_detects_future_information(tmp_path: Path, case: str) -> None:
    options, call, status, problem = LEAK_CASES[case]
    run = load_run(make_run(tmp_path, **options))
    config = make_config()

    checks = {check.call_id: check for check in check_generation_inputs(run, config)}
    records = machine_results(run, config, (), tuple(checks.values()), "batch-1")

    assert checks[call].status == status
    assert problem in {item["check"] for item in checks[call].problems}
    scored = [r for r in records if r["aspect"] != "overall"]
    assert scored and all(r["status"] == STATUS_NOT_EVALUATED and r["reason"] == REASON_INPUT_INVALID for r in scored)


def test_declared_retrospective_run_is_ok_and_tagged(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, visibility={"mode": "retrospective", "future_steps": 2}, rendered_future=2)
    run = load_run(run_dir)

    checks = check_generation_inputs(run, make_config())
    records = machine_results(run, make_config(), (), checks, "batch-1")

    assert run.retrospective and {check.status for check in checks} == {CHECK_OK}
    assert all(check.to_json(retrospective=True, batch_id="b")["retrospective"] for check in checks)
    assert records and all(record["retrospective"] for record in records)


def test_future_beyond_the_declared_retrospective_range_is_a_leak(tmp_path: Path) -> None:
    run = load_run(make_run(tmp_path, visibility={"mode": "retrospective", "future_steps": 1}, rendered_future=2))

    checks = check_generation_inputs(run, make_config())

    assert {check.status for check in checks if check.step + 2 <= LAST_STEP} == {CHECK_LEAK}
    assert {check.status for check in checks if check.step + 1 >= LAST_STEP} == {CHECK_OK}  # nothing to show there


def test_action_at_t_is_a_leak_only_when_the_config_says_so(tmp_path: Path) -> None:
    run = load_run(make_run(tmp_path))

    allowed = check_generation_inputs(run, make_config())
    refused = check_generation_inputs(run, make_config(leakage__allow_action_at_t=False))

    assert {c.status for c in allowed} == {CHECK_OK}
    assert {c.status for c in refused if c.layer == "translation"} == {CHECK_LEAK}


def test_an_event_without_a_request_line_is_an_error(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    lines = (run_dir / "requests.jsonl").read_text(encoding="utf-8").split("\n")
    (run_dir / "requests.jsonl").write_text("\n".join(line for line in lines if E_CHANGE not in line), encoding="utf-8")

    checks = {c.call_id: c for c in check_generation_inputs(load_run(run_dir), make_config())}

    assert checks[E_CHANGE].status == CHECK_ERROR


# --- machine checks (必須要素, タイミング, 発言頻度) ----------------------------------------------------


def test_structural_elements_detect_blank_intent_or_empty_evidence(tmp_path: Path) -> None:
    changes = {E_START: {"intent": "   ", "evidence": []}, E_CHANGE: {"evidence": ["ok", " "]}}
    records = _machine(make_run(tmp_path, output_changes=changes))

    def passed(event: str, criterion: str) -> bool:
        (record,) = find(records, event_id=event, criterion_id=criterion)
        assert record["status"] == STATUS_EVALUATED and record["score"] is None
        return record["measurement"]["passed"]

    assert passed(E_START, "elements.intent_present") is False
    assert passed(E_START, "elements.evidence_present") is False
    assert passed(E_CHANGE, "elements.intent_present") is True
    assert passed(E_CHANGE, "elements.evidence_present") is False
    assert passed(E_REQUEST, "elements.evidence_present") is True


def test_timing_rules_differ_by_point_kind(tmp_path: Path) -> None:
    points = write_points(
        tmp_path / "points.yaml", [point("p-start", "start", 3, onset=8), point("p-change", "change", 13)]
    )
    records = _machine(
        make_run(tmp_path), points=points, timing__start_min_lead_steps=2, timing__change_max_delay_steps=1
    )

    def criteria(**where: object) -> set[str]:
        return {r["criterion_id"] for r in find(records, **where) if r["aspect"] == "timing"}

    assert criteria(reference_point_id="p-start") == {
        "timing.point_matched",
        "timing.segment_events",
        "timing.start_lead",
    }
    assert criteria(reference_point_id="p-change") == {
        "timing.point_matched",
        "timing.segment_events",
        "timing.change_delay",
    }
    assert criteria(event_id=E_REQUEST, reference_point_id=None) == {"timing.request_latency"}
    assert criteria(event_id=E_START, reference_point_id=None) == set()
    (lead,) = find(records, reference_point_id="p-start", criterion_id="timing.start_lead")
    (delay,) = find(records, reference_point_id="p-change", criterion_id="timing.change_delay")
    assert lead["measurement"] == {"lead_steps": 3, "start_min_lead_steps": 2, "passed": True}
    assert delay["measurement"] == {"delay_steps": 2, "change_max_delay_steps": 1, "passed": False}


def test_points_match_the_first_same_kind_event_and_count_missed_and_excess(tmp_path: Path) -> None:
    points = [point("p-init", "start", 0), point("p-change", "change", 6), point("p-start", "start", 20)]
    records = _machine(make_run(tmp_path), points=write_points(tmp_path / "points.yaml", points))

    (initial,) = find(records, reference_point_id="p-init", criterion_id="timing.point_matched")
    (change,) = find(records, reference_point_id="p-change", criterion_id="timing.point_matched")
    (missed,) = find(records, reference_point_id="p-start", criterion_id="timing.point_matched")
    (excess,) = find(records, criterion_id="timing.excess_events")
    assert (initial["status"], initial["reason"]) == (STATUS_NOT_EVALUATED, "forced_trigger")
    assert change["measurement"] == {"matched": True, "delay_steps": 9, "segment_end_step": 20}
    assert change["subject"]["event_id"] == E_CHANGE
    assert missed["measurement"]["matched"] is False and missed["subject"]["event_id"] is None
    assert excess["measurement"] == {"start": 0, "change": 0}  # the start at 5 lies in p-init's segment


def test_deadline_interval_and_utterance_limits_follow_config(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    loose = _machine(run_dir, timing__max_utterances_per_target=5)
    strict = _machine(
        run_dir,
        timing__request_deadline_s=2.0,
        timing__min_question_interval_s=5.0,
        timing__max_utterances_per_target=4,
    )

    def measurement(records: list[dict], criterion: str, event: str | None = None) -> dict:
        (record,) = find(records, criterion_id=criterion, **({"event_id": event} if event else {}))
        return record["measurement"]

    assert measurement(loose, "timing.request_latency", E_REQUEST)["passed"] is True  # 2.5 s <= 3.0
    assert measurement(strict, "timing.request_latency", E_REQUEST)["passed"] is False  # 2.5 s > 2.0
    assert measurement(loose, "timing.question_interval")["intervals_s"] == [6.0]
    assert measurement(loose, "timing.question_interval")["passed"] is False  # 6.0 s < 10.0
    assert measurement(strict, "timing.question_interval")["passed"] is True  # 6.0 s >= 5.0
    assert measurement(loose, "timing.utterance_count")["passed"] is True  # 5 <= 5
    assert measurement(strict, "timing.utterance_count")["passed"] is False  # 5 > 4


@pytest.mark.parametrize("key", TIMING_KEYS)
def test_null_thresholds_report_raw_values_without_verdict(tmp_path: Path, key: str) -> None:
    points = write_points(
        tmp_path / "points.yaml", [point("p-start", "start", 3, onset=8), point("p-change", "change", 13)]
    )
    decided = {
        "timing__request_deadline_s": 3.0,
        "timing__min_question_interval_s": 10.0,
        "timing__start_min_lead_steps": 1,
        "timing__change_max_delay_steps": 4,
        "timing__max_events_per_segment": 2,
        "timing__max_utterances_per_target": 9,
    }
    records = _machine(make_run(tmp_path), points=points, **{**decided, f"timing__{key}": None})

    unset = [r for r in records if r["reason"] == f"{key}_unset"]
    verdicts = [r for r in records if r["measurement"] and key in r["measurement"]]

    assert unset and not verdicts
    for record in unset:
        assert record["status"] == STATUS_NOT_EVALUATED and record["score"] is None
        assert record["measurement"] and "passed" not in record["measurement"]


# --- annotation -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        [point("p-1", "start", 3), point("p-1", "change", 9)],
        [point("p-1", "start", 3), point("p-2", "change", 3)],
        [{**point("p-1", "start", 3), "target": {"kind": "aircraft", "id": "B0200"}}],
        [point("p-1", "start", 3, onset=2)],
    ],
)
def test_annotation_rejects_ambiguous_or_unknown_points(tmp_path: Path, points: list[dict]) -> None:
    run = load_run(make_run(tmp_path))

    with pytest.raises(AnnotationError):
        load_annotation(write_points(tmp_path / "points.yaml", points), run)


def test_initial_event_is_the_forced_bootstrap(tmp_path: Path) -> None:
    run = load_run(make_run(tmp_path))
    (initial,) = (event for event in run.events if event["event_id"] == E_INITIAL)

    assert [reason["kind"] for reason in initial["trigger_reasons"]] == ["initial"]
    assert E_LATE_REQUEST in {event["event_id"] for event in run.events}


def test_own_ids_are_found_in_payload_keys_too(tmp_path: Path) -> None:
    files = build_run()
    (request,) = [r for r in files["requests"] if r["call_id"] == E_REQUEST]
    payload = {**request["payload"], EPISODE_ID: 1}
    requests = [{**r, "payload": payload} if r is request else r for r in files["requests"]]
    events = [
        {**e, "request_sha256": sha256_text(canonical(payload))} if e.get("event_id") == E_REQUEST else e
        for e in files["events"]
    ]

    run = load_run(write_run(tmp_path / "run", {**files, "requests": requests, "events": events}))
    checks = {check.call_id: check for check in check_generation_inputs(run, make_config())}

    assert checks[E_REQUEST].status == CHECK_LEAK
    assert [p["check"] for p in checks[E_REQUEST].problems] == ["run_identifier"]
