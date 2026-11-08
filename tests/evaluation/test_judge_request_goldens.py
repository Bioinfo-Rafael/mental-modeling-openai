"""Frozen request bytes of both judges, case by case (prompt-file guard).

``tests/fixtures/judge-request-goldens.json`` holds the SHA-256 of the serialized Responses body of
every case below, computed before the judge prompts moved from code into config files. The cases
walk every branch of the rendering: each block on and off, each display option, empty lists, the
FUTURE block with rows, without rows and cut by an annotation, a criterion excluded for the forced
first translation, and the video block. The key-point judge renders from the frozen event-contract
v2 run in ``tests/fixtures/frozen/translator-run-v2``. A mismatch means a request byte changed: fix
the change, do not update the golden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.assess_data import AssessConfig, load_annotation, load_run
from evaluation.assess_judge import plan_calls
from evaluation.configuration import JudgeConfig
from evaluation.rendering import VideoContext, build_judge_request
from evaluation.schemas import EventKeyPoints, KeyPoint, load_translator_run
from translate import paths
from translate.events import canonical_json, sha256_text
from translate.llm.openai_responses import build_responses_body

from .assess_fakes import LAST_STEP, event_id, make_config, make_frame, make_run, point, write_points
from .eval_fakes import (
    A1_CHANGE,
    A1_REQUEST,
    EVENT_IDS,
    FakeFrameSource,
    make_judge_config,
    make_video_config,
)

GOLDENS = paths.ROOT / "tests" / "fixtures" / "judge-request-goldens.json"
FROZEN_RUN = paths.ROOT / "tests" / "fixtures" / "frozen" / "translator-run-v2"
KEY_POINTS = (KeyPoint("kp-001", "[PLACEHOLDER] judge key point one"), KeyPoint("kp-002", "[PLACEHOLDER] two"))
VIDEO_LAST_STEP = 340


def goldens() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text(encoding="utf-8"))


def body_sha256(request: Any) -> str:
    return sha256_text(canonical_json(build_responses_body(request)))


# --- key-point judge (D29) ------------------------------------------------------------------------


def key_point_configs() -> dict[str, JudgeConfig]:
    """Every display option away from its default once, plus each block removed."""
    return {
        "text": make_judge_config(),
        "text-plain-options": make_judge_config(
            inputs__instructions__show_ids=False,
            inputs__instructions__show_targets=False,
            inputs__instructions__show_issue_steps=False,
            inputs__timestamps__show_sim_seconds=False,
            inputs__timestamps__show_sources=False,
            inputs__trajectory__show_window_metadata=False,
        ),
        "text-some-options": make_judge_config(
            inputs__instructions__show_targets=False,
            inputs__timestamps__show_sources=False,
        ),
        "text-no-key-points": make_judge_config(inputs__key_points__include=False),
        "text-only-intent": make_judge_config(
            inputs__instructions__include=False,
            inputs__timestamps__include=False,
            inputs__trajectory__include=False,
            inputs__key_points__include=False,
        ),
    }


def key_point_hashes(run_dir: Path) -> dict[str, str]:
    events = load_translator_run(run_dir).translation_events
    hashes = {}
    for name, config in key_point_configs().items():
        for event in events:
            request = build_judge_request(
                event=event,
                event_log=events,
                key_points=EventKeyPoints(event["event_id"], KEY_POINTS),
                config=config,
            )
            hashes[f"{name}:{event['event_id']}"] = body_sha256(request.openai_request)
    by_id = {event["event_id"]: event for event in events}
    for event_id_ in (A1_REQUEST, A1_CHANGE):
        video = VideoContext(FakeFrameSource.for_committed_videos(), "mhi_v1.3_ic100601_win", VIDEO_LAST_STEP)
        request = build_judge_request(
            event=by_id[event_id_],
            event_log=events,
            key_points=EventKeyPoints(event_id_, KEY_POINTS),
            config=make_video_config(),
            video=video,
        )
        hashes[f"video:{event_id_}"] = body_sha256(request.openai_request)
    return hashes


# --- assess judge -----------------------------------------------------------------------------------

# (step, trigger kinds): the forced first translation, a FUTURE block that ends at the episode end
# with no row inside it, and a request without FUTURE.
LAST_STEP_PLAN = ((0, ("initial",)), (30, ("request",)), (LAST_STEP, ("start", "change")))
EMPTY_LISTS = {
    event_id(5): {"instructions_in_effect": []},
    event_id(15): {"alarms": []},
}
NO_EVIDENCE = {event_id(10): {"evidence": []}, event_id(15): {"evidence": ["[PLACEHOLDER] a", "[PLACEHOLDER] b"]}}
ANNOTATION = (
    point("p-start-000", "start", 0),
    point("p-start-001", "start", 4, 7),
    point("p-change-001", "change", 14),
)


def assess_cases(root: Path) -> dict[str, tuple[Path, AssessConfig, Path | None]]:
    """case -> (Translator run, config, annotation file or None)."""
    run = make_run(root, "plan")
    points = write_points(root / "points.yaml", list(ANNOTATION))
    return {
        "plan": (run, make_config(), None),
        "plan-annotated": (run, make_config(), points),
        "last-step": (make_run(root, "last-step", plan=LAST_STEP_PLAN), make_config(), None),
        "empty-lists": (
            make_run(root, "empty-lists", input_changes=EMPTY_LISTS, output_changes=NO_EVIDENCE),
            make_config(),
            None,
        ),
        "english-rationale": (run, make_config(judge__rationale_language="English"), None),
    }


def assess_hashes(root: Path) -> dict[str, list[str]]:
    hashes = {}
    for name, (run_dir, config, points) in assess_cases(root).items():
        run = load_run(run_dir)
        annotation = load_annotation(points, run) if points is not None else None
        calls = plan_calls(list(run.events), run=run, config=config, annotation=annotation, load_frame=make_frame)
        hashes[name] = [call.request_sha256 for call in calls]
    return hashes


# --- tests ------------------------------------------------------------------------------------------


# The example run's trigger kinds, event schema 2 -> 3 (it has no detector start, which schema 3 merged
# into change): the first-instruction trigger was renamed, nothing else changed.
V2_TO_V3_KINDS = {"initial": "start", "change": "change", "request": "request"}


def as_schema_3(event: dict[str, Any]) -> dict[str, Any]:
    reasons = [{**reason, "kind": V2_TO_V3_KINDS[reason["kind"]]} for reason in event["trigger_reasons"]]
    primary = V2_TO_V3_KINDS[event["primary_trigger"]]
    return {**event, "schema_version": 3, "primary_trigger": primary, "trigger_reasons": reasons}


def test_frozen_run_is_the_example_run_of_the_fixture(translator_run: Path) -> None:
    """The frozen v2 events are the example run's translation events, but for the trigger-kind rename."""
    generated = load_translator_run(translator_run).translation_events
    frozen = load_translator_run(FROZEN_RUN).translation_events

    def strip(event: dict[str, Any]) -> dict[str, Any]:  # re-sequenced; processing_s is measured wall time
        return {key: value for key, value in event.items() if key not in ("sequence", "processing_s")}

    assert [event["event_id"] for event in frozen] == list(EVENT_IDS)
    assert {event["schema_version"] for event in frozen} == {2} and {e["schema_version"] for e in generated} == {3}
    assert [strip(as_schema_3(event)) for event in frozen] == [strip(event) for event in generated]


def test_key_point_judge_request_bytes_are_unchanged() -> None:
    assert key_point_hashes(FROZEN_RUN) == goldens()["key_point_judge"]


def test_assess_judge_request_bytes_are_unchanged(tmp_path: Path) -> None:
    assert assess_hashes(tmp_path) == goldens()["assess_judge"]


@pytest.mark.parametrize("case", ["plan", "plan-annotated", "last-step", "empty-lists"])
def test_assess_golden_cases_walk_their_branch(tmp_path: Path, case: str) -> None:
    """Each assess case really renders the branch it is named for (so the golden covers it)."""
    run_dir, config, points = assess_cases(tmp_path)[case]
    run = load_run(run_dir)
    annotation = load_annotation(points, run) if points is not None else None
    texts = [
        call.packet.text
        for call in plan_calls(list(run.events), run=run, config=config, annotation=annotation, load_frame=make_frame)
    ]
    expected = {
        "plan": "Not rated for this event:",
        "plan-annotated": "(next_annotated_point)",
        "last-step": "(no step inside the window)",
        "empty-lists": "## INSTRUCTIONS IN EFFECT (issued at or before t)\n- none",
    }[case]
    assert any(expected in text for text in texts)


def test_key_point_golden_cases_cover_every_event() -> None:
    names = {key.split(":", 1)[0] for key in goldens()["key_point_judge"]}
    assert names == {*key_point_configs(), "video"}
