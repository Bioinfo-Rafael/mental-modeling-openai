"""Focused tests for cues projection and the event-input audit record."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from translate.events import RECORD_TRANSLATION_EVENT, RECORD_TRIGGER_SUPPRESSED
from translate.inputs import RunInputs, SlotState, event_input
from translate.run_files import cues_from_events, write_cues
from translate.scenario import EpisodeFacts

from .tr_fakes import fake_translation_renderer, instruction, make_config, make_scenario


def _translation_event(*, problem: str | None = None) -> dict:
    event = {
        "record_type": RECORD_TRANSLATION_EVENT,
        "step": 7,
        "time_s": 1.4,
        "target": {"kind": "aircraft", "id": "A0100"},
        "primary_trigger": "request",
        "model": {"model": "gpt-test", "reasoning_effort": "medium"},
        "event_id": "scenario:aircraft:A0100:000007",
        "output": {"intent": "turn left", "evidence": ["A0100 step 7"]},
    }
    return event if problem is None else {**event, "output": {"intent": "", "evidence": []}, "answer_problem": problem}


def test_cues_from_events_projects_valid_invalid_and_suppressed_slots() -> None:
    invalid_problem = 'invalid final answer: "intent" has 201 characters; the limit is 200'
    suppressed = {
        "record_type": RECORD_TRIGGER_SUPPRESSED,
        "step": 9,
        "time_s": 1.8,
        "target": {"kind": "aircraft", "id": "A0200"},
        "primary_trigger": "request",
        "down_step": 8,
    }

    cues = cues_from_events(
        [
            {"record_type": "detector_call"},
            _translation_event(),
            _translation_event(problem=invalid_problem),
            suppressed,
        ]
    )

    assert cues == [
        {
            "step": 7,
            "time_s": 1.4,
            "target": {"kind": "aircraft", "id": "A0100"},
            "aircraft": "A0100",
            "trigger": "request",
            "model": "gpt-test",
            "reasoning_effort": "medium",
            "event_id": "scenario:aircraft:A0100:000007",
            "intent": "turn left",
            "evidence": ["A0100 step 7"],
        },
        {
            "step": 7,
            "time_s": 1.4,
            "target": {"kind": "aircraft", "id": "A0100"},
            "aircraft": "A0100",
            "trigger": "request",
            "model": "gpt-test",
            "reasoning_effort": "medium",
            "event_id": "scenario:aircraft:A0100:000007",
            "skipped": invalid_problem,
        },
        {
            "step": 9,
            "time_s": 1.8,
            "target": {"kind": "aircraft", "id": "A0200"},
            "aircraft": "A0200",
            "trigger": "request",
            "skipped": "target_down: alive == 0 since step 8",
        },
    ]


def test_write_cues_with_events_rewrites_projection_file(tmp_path: Path) -> None:
    events = [_translation_event()]
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    path = write_cues(tmp_path)

    assert path == tmp_path / "cues.jsonl"
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == cues_from_events(events)


def test_event_input_without_trajectory_selected_still_records_trajectory_and_selected_inputs() -> None:
    scenario = make_scenario([instruction("a1", 0, "aircraft", "A0100")])
    config = replace(make_config(), inputs=("instructions",))
    facts = EpisodeFacts(
        scenario.episode.episode_id,
        30,
        {"A": ("A0100", "A0200"), "B": ("B0100", "B0200")},
        {"A0100": None, "A0200": None, "B0100": None, "B0200": None},
    )
    inputs = RunInputs(pd.DataFrame(), scenario, facts, config, fake_translation_renderer)
    slot = SlotState(0, scenario.instructions[0].target, scenario.instructions[0], None)
    trajectory = fake_translation_renderer(
        pd.DataFrame(),
        start_step=0,
        end_step=0,
        anchor_step=0,
        episode_alias=scenario.episode.alias,
    )

    record = event_input(inputs, slot, trajectory)

    assert record["translator_inputs"] == ["instructions"]
    assert record["trajectory"] == {
        "renderer": trajectory.renderer,
        "renderer_version": trajectory.renderer_version,
        "start_step": 0,
        "end_step": 0,
        "anchor_step": 0,
        "object_ids": ["A0100", "A0200", "B0100", "B0200"],
        "fields": [],
        "text": "FAKE-LAG17 0..0 anchor=0",
        "sha256": trajectory.sha256,
    }
