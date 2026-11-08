"""Focused preset-trigger contract tests."""

from __future__ import annotations

import pytest

from translate.config import PresetPoint
from translate.events import TargetRef
from translate.scenario import PipelineError
from translate.triggers import preset_triggers

from .tr_fakes import instruction, make_scenario, request

AIRCRAFT = TargetRef("aircraft", "A0100")
TEAM = TargetRef("team", "A")


def test_preset_triggers_with_points_projects_exact_targets_and_request_source() -> None:
    scenario = make_scenario(
        [instruction("a1", 0, "aircraft", "A0100"), instruction("t1", 0, "team", "A")],
        [request("r1", 10, ("aircraft", "A0100"), source="human")],
    )
    points = (
        PresetPoint(0, "start", (AIRCRAFT,)),
        PresetPoint(10, "request", (AIRCRAFT,)),
        PresetPoint(20, "change", None),
    )

    found = preset_triggers(scenario, points)

    assert set(found) == {(0, AIRCRAFT), (10, AIRCRAFT), (20, AIRCRAFT), (20, TEAM)}
    assert found[(10, AIRCRAFT)][0].to_json() == {
        "kind": "request",
        "source": "human",
        "reference_id": "r1",
        "probability": None,
    }
    assert found[(20, TEAM)][0].to_json() == {
        "kind": "change",
        "source": "preset",
        "reference_id": "preset-step-20",
        "probability": None,
    }


def test_preset_request_point_at_a_late_first_instruction_is_that_instruction() -> None:
    scenario = make_scenario([instruction("a1", 0, "aircraft", "A0100"), instruction("t1", 5, "team", "A")])

    found = preset_triggers(scenario, (PresetPoint(5, "request", (TEAM,)),))

    assert found[(5, TEAM)][0].to_json() == {
        "kind": "request",
        "source": "instruction",
        "reference_id": "t1",
        "probability": None,
    }


def test_preset_request_point_without_scenario_request_raises_pipeline_error() -> None:
    scenario = make_scenario([instruction("a1", 0, "aircraft", "A0100")])

    with pytest.raises(PipelineError, match="has no scenario request"):
        preset_triggers(scenario, (PresetPoint(10, "request", (AIRCRAFT,)),))


def test_preset_triggers_with_uncovered_scenario_request_raises_pipeline_error() -> None:
    scenario = make_scenario(
        [instruction("a1", 0, "aircraft", "A0100")],
        [request("r1", 10, ("aircraft", "A0100"))],
    )

    with pytest.raises(PipelineError, match="scenario request r1 is not a preset request point"):
        preset_triggers(scenario, (PresetPoint(0, "start", (AIRCRAFT,)),))
