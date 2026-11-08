"""Trigger kinds per event schema: the explicit tables, the judge config's map and old runs.

The hand-written run in ``assess_fakes`` is event schema 2 (initial, detector start, request,
detector change). The legacy config rates it as it always did; the default config maps its
detector start into change, the merge event schema 3 made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.assess_checks import match_points
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
    E_INITIAL,
    E_START,
    config_raw,
    make_frame,
    make_run,
    point,
    with_value,
    write_points,
)

LEGACY = load_config(DEFAULT_CONFIG.parent / "assess-legacy.yaml")
DEFAULT = load_config(DEFAULT_CONFIG)
# The explicit rename of the hand-written v2 run into event schema 3 (its detector start becomes change).
V2_TO_V3 = {"initial": "start", "start": "change", "change": "change", "request": "request"}


def _events(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {event["event_id"]: event for event in load_run(run_dir).events}


def _rewrite(run_dir: Path, edit: Any) -> Path:
    path = run_dir / "events.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps(edit(record)) + "\n" for record in records), encoding="utf-8")
    return run_dir


def _as_schema_3(record: dict[str, Any]) -> dict[str, Any]:
    reasons = [{**reason, "kind": V2_TO_V3[reason["kind"]]} for reason in record.get("trigger_reasons", [])]
    changed = {**record, "schema_version": 3}
    if "trigger_reasons" not in record:
        return changed
    return {**changed, "trigger_reasons": reasons, "primary_trigger": reasons[0]["kind"]}


# --- the config's map -------------------------------------------------------------------------------


def test_the_legacy_config_rates_the_detector_start_with_the_start_criteria(tmp_path: Path) -> None:
    events = _events(make_run(tmp_path))

    assert point_kinds(events[E_START], LEGACY) == ("start",)
    assert "correctness.start_future" in {c.criterion_id for c in event_criteria(events[E_START], LEGACY).asked}


def test_the_default_config_merges_a_schema_2_detector_start_into_change(tmp_path: Path) -> None:
    events = _events(make_run(tmp_path))

    asked = {c.criterion_id for c in event_criteria(events[E_START], DEFAULT).asked}

    assert point_kinds(events[E_START], DEFAULT) == ("change",)
    assert "correctness.change_cause" in asked and "correctness.start_future" not in asked
    assert point_kinds(events[E_INITIAL], DEFAULT) == ("start",)
    excluded = event_criteria(events[E_INITIAL], DEFAULT).excluded
    assert [(c.criterion_id, reason) for c, reason in excluded] == [("timing.start_in_time", "forced_trigger")]


def test_the_packet_shows_the_mapped_kinds_in_the_default_prompt_and_raw_ones_in_the_legacy(tmp_path: Path) -> None:
    run = load_run(make_run(tmp_path))
    event = next(item for item in run.events if item["event_id"] == E_START)

    def packet(config: Any) -> str:
        return build_packet(event, run=run, config=config, annotation=None, frame=make_frame()).text

    assert "Point kinds: change.\nTrigger reasons: change (source detector)." in packet(DEFAULT)
    assert "Point kinds: start.\nTrigger reasons: start (source detector)." in packet(LEGACY)


def test_a_schema_3_run_loads_and_its_start_is_the_forced_first_translation(tmp_path: Path) -> None:
    run_dir = _rewrite(make_run(tmp_path), _as_schema_3)
    run = load_run(run_dir)
    events = {event["event_id"]: event for event in run.events}

    assert run.event_schema_version == 3
    assert [point_kinds(events[e], DEFAULT) for e in (E_INITIAL, E_START)] == [("start",), ("change",)]
    excluded = event_criteria(events[E_INITIAL], DEFAULT).excluded
    assert [c.criterion_id for c, _ in excluded] == ["timing.start_in_time"]
    with pytest.raises(ConfigError, match=r"rates event schema versions \[2\] only, not 3"):
        point_kinds(events[E_START], LEGACY)


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
    seen: list[int] = []

    def second_line_schema_3(record: dict[str, Any]) -> dict[str, Any]:
        seen.append(1)
        return _as_schema_3(record) if len(seen) == 2 else record

    with pytest.raises(RunFormatError, match="one event schema per run"):
        load_run(_rewrite(make_run(tmp_path), second_line_schema_3))


# --- annotation matching ------------------------------------------------------------------------------


def test_the_default_config_matches_a_schema_2_detector_start_to_a_change_point(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    run = load_run(run_dir)
    points = write_points(tmp_path / "points.yaml", [point("p-start-000", "start", 0), point("p-change", "change", 4)])
    annotation = load_annotation(points, run)

    default = {m.point.point_id: m for m in match_points(annotation, run, DEFAULT)}
    legacy = {m.point.point_id: m for m in match_points(annotation, run, LEGACY)}

    assert default["p-start-000"].forced and legacy["p-start-000"].forced
    assert default["p-change"].matched["event_id"] == E_START  # the detector start at 5 is a change now
    assert legacy["p-change"].matched["event_id"] != E_START  # the legacy rubric waits for a detector change


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
