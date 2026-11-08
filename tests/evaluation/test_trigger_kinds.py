"""Trigger kinds per event schema: the explicit tables, the judge config's map, old and new runs.

The hand-written run in ``assess_fakes`` is event schema 2 (initial, detector start, request, detector
change); ``V3_PLAN`` is the same episode in schema 3 (2026-11-09: start = the episode start, change =
an external cause, request = a question or a first instruction after the start). The legacy config
rates schema 2 as it always did; the default config rates schema 3 only and refuses schema 2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.assess_checks import check_generation_inputs, machine_results, match_points
from evaluation.assess_data import (
    DEFAULT_CONFIG,
    ConfigError,
    RunFormatError,
    event_criteria,
    load_annotation,
    load_config,
    load_run,
    parse_config,
    point_kinds,
)
from evaluation.assess_judge import build_packet

from .assess_fakes import (
    E_CHANGE,
    E_INITIAL,
    E_REQUEST,
    E_START,
    V3_PLAN,
    build_run,
    config_raw,
    event_id,
    find,
    make_frame,
    make_run,
    point,
    with_value,
    write_points,
    write_run,
)

LEGACY = load_config(DEFAULT_CONFIG.parent / "assess-legacy.yaml")
DEFAULT = load_config(DEFAULT_CONFIG)
INSTRUCTION_REQUEST = {"kind": "request", "source": "instruction", "reference_id": "late-001", "probability": None}


def _events(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in load_run(run_dir).events}


def _rewrite(run_dir: Path, edit: Any) -> Path:
    path = run_dir / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(edit(record)) + "\n" for record in records), encoding="utf-8")
    return run_dir


def _v3_run(root: Path, **options: Any) -> Path:
    return make_run(root, plan=V3_PLAN, schema_version=3, **options)


def _packet(event: dict[str, Any], run_dir: Path, config: Any) -> str:
    run = load_run(run_dir)
    return build_packet(event, run=run, config=config, annotation=None, frame=make_frame()).text


# --- the config's map -------------------------------------------------------------------------------


def test_the_legacy_config_rates_the_detector_start_with_the_start_criteria(tmp_path: Path) -> None:
    events = _events(make_run(tmp_path))

    assert point_kinds(events[E_START], LEGACY) == ("start",)
    assert "correctness.start_future" in {c.criterion_id for c in event_criteria(events[E_START], LEGACY).asked}


def test_the_default_config_refuses_a_schema_2_run(tmp_path: Path) -> None:
    """A schema 2 detector start (a tactic begun along the plan) is no point now: never guessed into one."""
    events = _events(make_run(tmp_path))

    with pytest.raises(ConfigError, match=r"rates event schema versions \[3\] only, not 2"):
        point_kinds(events[E_START], DEFAULT)


def test_a_schema_3_run_loads_and_its_start_is_the_episode_start(tmp_path: Path) -> None:
    run = load_run(_v3_run(tmp_path))
    events = {event["event_id"]: event for event in run.events}

    assert run.event_schema_version == 3
    assert [point_kinds(events[e], DEFAULT) for e in (E_INITIAL, E_REQUEST, E_CHANGE)] == [
        ("start",),
        ("request",),
        ("change",),
    ]
    assert event_criteria(events[E_INITIAL], DEFAULT).excluded == ()  # no start timing criterion is left
    with pytest.raises(ConfigError, match=r"rates event schema versions \[2\] only, not 3"):
        point_kinds(events[E_CHANGE], LEGACY)


def test_a_start_with_no_instruction_in_effect_renders_none(tmp_path: Path) -> None:
    run_dir = _v3_run(tmp_path, input_changes={E_INITIAL: {"instructions_in_effect": []}})
    event = _events(run_dir)[E_INITIAL]

    text = _packet(event, run_dir, DEFAULT)

    assert "## INSTRUCTIONS IN EFFECT (issued at or before t)\n- none\n" in text
    assert "Point kinds: start.\nTrigger reasons: start (source episode_start)." in text
    assert "## QUESTION FROM A HUMAN\n- none\n" in text


def test_a_request_opened_by_an_instruction_asks_the_question_criteria_as_conditional(tmp_path: Path) -> None:
    run_dir = _v3_run(tmp_path)
    request = _events(run_dir)[E_REQUEST]
    event = {**request, "input": {**request["input"], "question": None}, "trigger_reasons": [INSTRUCTION_REQUEST]}

    text = _packet(event, run_dir, DEFAULT)

    assert "Trigger reasons: request (source instruction)." in text and "## QUESTION FROM A HUMAN\n- none\n" in text
    lines = {line.split("]")[0][3:]: line for line in text.split("\n") if line.startswith("- [")}
    for criterion_id in ("correctness.question_granularity", "contribution.question_resolved"):
        assert "conditional: not_applicable allowed" in lines[criterion_id]


def test_the_question_interval_ignores_a_request_opened_by_an_instruction(tmp_path: Path) -> None:
    files = build_run(plan=V3_PLAN, schema_version=3, input_changes={E_REQUEST: {"question": None}})
    events = [
        {**record, "trigger_reasons": [INSTRUCTION_REQUEST]} if record.get("event_id") == E_REQUEST else record
        for record in files["events"]
    ]
    run = load_run(write_run(tmp_path / "run", {**files, "events": events}))

    results = machine_results(run, DEFAULT, (), check_generation_inputs(run, DEFAULT), "batch-1")

    (interval,) = find(results, criterion_id="timing.question_interval")
    assert interval["measurement"]["requests"] == 1  # the question at 40 only; the instruction at 10 is none
    assert interval["reason"] == "fewer_than_two_requests"


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda r: {**r, "schema_version": 3}, "event schema 3 has only the trigger kinds"),  # initial in schema 3
        (lambda r: {**r, "schema_version": 4}, "schema_version"),
    ],
)
def test_a_reason_outside_its_schema_vocabulary_fails_to_load(tmp_path: Path, edit: Any, match: str) -> None:
    run_dir = _rewrite(make_run(tmp_path), edit)

    with pytest.raises(RunFormatError, match=match):
        load_run(run_dir)


def test_a_run_mixing_event_schemas_fails_to_load(tmp_path: Path) -> None:
    def detector_calls_as_schema_3(record: dict[str, Any]) -> dict[str, Any]:
        return {**record, "schema_version": 3} if record["record_type"] == "detector_call" else record

    with pytest.raises(RunFormatError, match="one event schema per run"):
        load_run(_rewrite(make_run(tmp_path), detector_calls_as_schema_3))


# --- annotation matching ------------------------------------------------------------------------------


def test_the_default_config_matches_an_annotated_change_to_the_detector_change(tmp_path: Path) -> None:
    run = load_run(_v3_run(tmp_path))
    points = write_points(tmp_path / "points.yaml", [point("p-start-000", "start", 0), point("p-change", "change", 14)])

    matches = {m.point.point_id: m for m in match_points(load_annotation(points, run), run, DEFAULT)}

    assert matches["p-start-000"].forced  # the episode start: its timing is fixed by definition
    assert matches["p-change"].matched["event_id"] == E_CHANGE == event_id(15)


# --- the trigger map is validated -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("triggers", "match"),
    [
        ([], "at least one event schema version"),
        ([{"event_schema_version": 1, "point_kinds": {}, "forced": []}], "event_schema_version must be one of"),
        (
            [{"event_schema_version": 3, "point_kinds": {"start": "start", "change": "change"}, "forced": []}],
            "point_kinds",
        ),
        (
            [
                {
                    "event_schema_version": 3,
                    "point_kinds": {"start": "start", "change": "change", "request": "question"},
                    "forced": [],
                }
            ],
            "must map each trigger kind",
        ),
        (
            [
                {
                    "event_schema_version": 3,
                    "point_kinds": {"start": "start", "change": "change", "request": "request"},
                    "forced": ["change"],
                }
            ],
            "forced must list distinct trigger kinds that map to 'start'",
        ),
    ],
)
def test_a_malformed_trigger_map_fails_at_load(tmp_path: Path, triggers: list[Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_config(with_value(config_raw(), "triggers", triggers), path=tmp_path / "a.yaml", sha256="x")


def test_a_trigger_map_may_not_list_a_version_twice(tmp_path: Path) -> None:
    triggers = config_raw()["triggers"] * 2

    with pytest.raises(ConfigError, match="lists an event schema version twice"):
        parse_config(with_value(config_raw(), "triggers", triggers), path=tmp_path / "a.yaml", sha256="x")
