"""Pipeline: initial/tick/trigger rules, precedence, mid-episode targets, windows, death, caps, errors."""

from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from mtd_rl_xavier.translator import pipeline
from mtd_rl_xavier.translator.config import ScenarioError, load_scenario
from mtd_rl_xavier.translator.pipeline import (
    STATUS_CAP_REACHED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    TranslatorSummary,
    episode_facts,
    plan_schedule,
    validate_scenario,
)
from translate.events import (
    RECORD_TYPES,
    Alarm,
    ModelCallError,
    RequestPoint,
    Scenario,
    TargetRef,
)
from translate.lag.rendering import render_detector_compact, render_lag17_jsonl

from .tr_fakes import (
    CHANGE,
    EXAMPLE_SCENARIO,
    LOSS01,
    NONE,
    START,
    WIN01,
    Answer,
    FakeDetectionModel,
    FakeIntentModel,
    MemorySink,
    alarm,
    answers_at,
    instruction,
    make_config,
    make_scenario,
    make_translator,
    request,
)


def _run(
    frame: pd.DataFrame,
    scenario: Scenario,
    *,
    detection: FakeDetectionModel | None = None,
    intent: FakeIntentModel | None = None,
    log: list[tuple[str, str]] | None = None,
    **config_overrides: Any,
) -> tuple[TranslatorSummary, MemorySink]:
    config = make_config(**config_overrides)
    sink = MemorySink(log)
    translator = make_translator(detection, intent, config=config)
    summary = translator.run(frame=frame, scenario=scenario, config=config, sink=sink)
    return summary, sink


def _translations(sink: MemorySink) -> list[tuple[int, str, str]]:
    return [
        (e["step"], f"{e['target']['kind']} {e['target']['id']}", e["primary_trigger"])
        for e in sink.of_type("translation_event")
    ]


def _detector_steps(sink: MemorySink, label: str) -> list[int]:
    return [e["step"] for e in sink.of_type("detector_call") if f"{e['target']['kind']} {e['target']['id']}" == label]


def _answer_at(answers: dict[tuple[str, int], Answer]) -> FakeDetectionModel:
    return FakeDetectionModel(answers_at(answers))


A1 = ("aircraft", "A0100")


def test_first_instruction_gives_every_target_an_initial_translation_in_canonical_order(
    frames: dict[str, pd.DataFrame],
) -> None:
    scenario = make_scenario([instruction("t1", 0, "team", "A"), instruction("a1", 0, *A1)])

    summary, sink = _run(frames[WIN01], scenario)

    first, second = sink.of_type("translation_event")[:2]
    assert summary.status == STATUS_COMPLETED
    assert _translations(sink) == [(0, "aircraft A0100", "initial"), (0, "team A", "initial")]
    assert first["event_id"] == "test-scenario:aircraft:A0100:000000"
    assert first["schema_version"] == 2 and first["input"]["prompt_version"] == "translator-intent-v2"
    assert [i["instruction_id"] for i in first["input"]["instructions_in_effect"]] == ["a1", "t1"]
    assert first["input"]["primary_instruction_id"] == "a1" and second["input"]["primary_instruction_id"] == "t1"
    assert first["input"]["previous_translation"] is None
    assert (first["input"]["alarms"], first["input"]["question"]) == ([], None)
    assert first["trigger_reasons"] == [
        {"kind": "initial", "source": "first_instruction", "reference_id": "a1", "probability": None}
    ]
    assert first["time_s"] == 0.0 and first["episode"]["csv_sha256"] == frames[WIN01].attrs["csv_sha256"]


def test_detector_ticks_stay_on_the_fixed_grid(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 3, A1)])

    summary, sink = _run(frames[WIN01], scenario)

    assert _detector_steps(sink, "aircraft A0100") == list(range(5, 341, 5))
    assert _translations(sink) == [(0, "aircraft A0100", "initial"), (3, "aircraft A0100", "request")]
    assert summary.detector.completed == 68 and summary.translation.completed == 2


def test_tick_with_coinciding_triggers_gives_one_detector_call_and_one_translation(
    frames: dict[str, pd.DataFrame],
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 10, A1)])

    _, sink = _run(frames[WIN01], scenario, detection=_answer_at({("aircraft A0100", 10): CHANGE}))

    events = [e for e in sink.of_type("translation_event") if e["step"] == 10]
    assert _detector_steps(sink, "aircraft A0100").count(10) == 1
    assert len(events) == 1
    reasons = events[0]["trigger_reasons"]
    assert [r["kind"] for r in reasons] == ["change", "request"]
    assert reasons[0] == {
        "kind": "change",
        "source": "detector",
        "reference_id": "test-scenario:detector:aircraft:A0100:000010",
        "probability": 0.9,
    }
    assert events[0]["primary_trigger"] == "change"
    assert events[0]["input"]["question"] == {"request_id": "r1", "step": 10, "text": "[PLACEHOLDER] question r1"}


@pytest.mark.parametrize(
    ("instructions", "requests", "answer", "step", "expected"),
    [
        ([], [request("r1", 0, A1)], NONE, 0, ["initial", "request"]),
        ([], [request("r1", 20, A1)], START, 20, ["start", "request"]),
        ([], [request("r1", 20, A1)], CHANGE, 20, ["change", "request"]),
        ([], [request("r1", 20, A1)], NONE, 20, ["request"]),
        ([instruction("a2", 20, *A1)], [request("r1", 20, A1)], NONE, 20, ["request"]),  # D50: no instruction kind
        ([instruction("a2", 20, *A1)], [], CHANGE, 20, ["change"]),
    ],
)
def test_primary_trigger_precedence(
    frames: dict[str, pd.DataFrame],
    instructions: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    answer: Answer,
    step: int,
    expected: list[str],
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1), *instructions], requests)

    _, sink = _run(frames[WIN01], scenario, detection=_answer_at({("aircraft A0100", 20): answer}))

    (event,) = [e for e in sink.of_type("translation_event") if e["step"] == step]
    assert event["primary_trigger"] == expected[0]
    assert [reason["kind"] for reason in event["trigger_reasons"]] == expected


def test_later_instruction_only_feeds_the_detector_and_a_mid_episode_target_starts_with_initial(
    frames: dict[str, pd.DataFrame],
) -> None:
    instructions = [
        instruction("a1", 0, *A1),
        instruction("ta1", 0, "team", "A"),
        instruction("ta2", 20, "team", "A"),
        instruction("tb1", 23, "team", "B"),
    ]

    _, sink = _run(frames[WIN01], make_scenario(instructions))

    by_key = {(e["step"], e["target"]["id"]): e for e in sink.of_type("translation_event")}
    first_b = by_key[(23, "B")]
    assert sorted(by_key) == [(0, "A"), (0, "A0100"), (23, "B")]  # no translation for the replacement at 20
    assert first_b["primary_trigger"] == "initial" and first_b["input"]["previous_translation"] is None
    assert first_b["trigger_reasons"][0]["reference_id"] == "tb1"
    assert [i["instruction_id"] for i in first_b["input"]["instructions_in_effect"]] == ["a1", "ta2", "tb1"]
    assert _detector_steps(sink, "team B")[:2] == [25, 30]  # no detector before its first translation
    assert 20 in _detector_steps(sink, "team A")  # the replaced target is still checked at its tick
    team_a = {e["step"]: e for e in sink.of_type("detector_call") if e["target"]["id"] == "A"}
    assert [i["instruction_id"] for i in team_a[15]["input"]["instructions_in_effect"]] == ["ta1"]
    assert [i["instruction_id"] for i in team_a[20]["input"]["instructions_in_effect"]] == ["ta2"]
    assert "issued at step 20: [PLACEHOLDER] instruction ta2" in team_a[20]["input"]["state_text"]
    # D21: the replaced target keeps its previous translation
    assert team_a[20]["input"]["previous_translation"]["event_id"] == "test-scenario:team:A:000000"
    aircraft = next(e for e in sink.of_type("detector_call") if e["target"]["id"] == "A0100" and e["step"] == 20)
    assert [i["instruction_id"] for i in aircraft["input"]["instructions_in_effect"]] == ["a1", "ta2"]  # own + team


@pytest.mark.parametrize(("answer", "kind"), [(START, "start"), (CHANGE, "change")])
def test_detector_start_or_change_triggers_a_translation(
    frames: dict[str, pd.DataFrame], answer: Answer, kind: str
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)])

    _, sink = _run(frames[WIN01], scenario, detection=_answer_at({("aircraft A0100", 15): answer}))

    call = next(e for e in sink.of_type("detector_call") if e["step"] == 15)
    assert _translations(sink) == [(0, "aircraft A0100", "initial"), (15, "aircraft A0100", kind)]
    assert call["output"] == {
        "label": kind,
        "probabilities": {
            label: 0.9 if label == kind else pytest.approx(0.05) for label in ("none", "start", "change")
        },
        "threshold": 0.65,
        "triggered": True,
    }
    event = sink.of_type("translation_event")[1]
    assert event["trigger_reasons"] == [
        {"kind": kind, "source": "detector", "reference_id": call["call_id"], "probability": 0.9}
    ]
    assert event["input"]["previous_translation"]["event_id"] == "test-scenario:aircraft:A0100:000000"


@pytest.mark.parametrize("answer", [("start", 0.64), ("change", 0.6499), ("none", 1.0), ("none", 0.4)])
def test_none_or_a_label_below_the_threshold_triggers_nothing(frames: dict[str, pd.DataFrame], answer: Answer) -> None:
    _, sink = _run(frames[WIN01], make_scenario([instruction("a1", 0, *A1)]), detection=FakeDetectionModel(answer))

    assert _translations(sink) == [(0, "aircraft A0100", "initial")]
    assert all(
        e["output"]["triggered"] is False and e["output"]["label"] == answer[0] for e in sink.of_type("detector_call")
    )


def test_windows_clip_at_step_zero_and_end_at_the_current_step(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 10, A1), request("r2", 340, A1)])

    _, sink = _run(frames[WIN01], scenario)

    windows = {e["step"]: e["input"]["trajectory"] for e in sink.of_type("translation_event")}
    detector = {e["step"]: e["input"]["trajectory"] for e in sink.of_type("detector_call")}
    assert (windows[10]["start_step"], windows[10]["end_step"], windows[10]["anchor_step"]) == (0, 10, 10)
    assert (windows[340]["start_step"], windows[340]["end_step"]) == (281, 340)
    assert (detector[5]["start_step"], detector[5]["end_step"]) == (0, 5)
    assert (detector[15]["start_step"], detector[15]["end_step"]) == (6, 15)
    assert max(detector) == 340  # the last complete step is a tick


def test_detector_input_uses_only_the_targets_own_previous_translation(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1), instruction("ta1", 0, "team", "A")])
    intent = FakeIntentModel()

    _, sink = _run(frames[WIN01], scenario, intent=intent)

    team_call = next(e for e in sink.of_type("detector_call") if e["target"]["id"] == "A")
    assert team_call["input"]["previous_translation"]["event_id"] == "test-scenario:team:A:000000"
    assert "[FAKE] intent 2" in team_call["input"]["state_text"]
    assert "[FAKE] intent 1" not in team_call["input"]["state_text"]
    assert team_call["input"]["trajectory"]["object_ids"] == ["A0100", "A0200", "B0100", "B0200"]
    aircraft_call = next(e for e in sink.of_type("detector_call") if e["target"]["id"] == "A0100")
    assert aircraft_call["input"]["trajectory"]["object_ids"] == ["A0100", "B0100", "B0200"]
    assert aircraft_call["input"]["question"]["id"] == "point_kind"
    assert aircraft_call["raw_response"]["answers"]["point_kind"]["choice"] == "none"


def test_death_closes_targets_and_suppresses_later_triggers(frames: dict[str, pd.DataFrame]) -> None:
    instructions = [instruction(f"i-{a}", 0, "aircraft", a) for a in ("A0100", "A0200", "B0200")]
    instructions.append(instruction("tb", 0, "team", "B"))
    requests = [
        request("r0", 140, ("aircraft", "B0200")),
        request("r1", 150, ("aircraft", "B0200")),
        request("r2", 250, ("aircraft", "A0200"), A1),
    ]

    summary, sink = _run(frames[LOSS01], make_scenario(instructions, requests, episode_id=LOSS01))

    closed = {e["target"]["id"]: e for e in sink.of_type("target_closed")}
    assert set(closed) == {"B0200", "A0200"}  # team B keeps B0100 alive
    assert (closed["B0200"]["step"], closed["B0200"]["down_step"]) == (137, 137)
    assert (closed["A0200"]["step"], closed["A0200"]["down_step"]) == (210, 210)
    assert closed["B0200"]["last_event_id"] == "test-scenario:aircraft:B0200:000000"
    suppressed = [(e["step"], e["target"]["id"], e["primary_trigger"]) for e in sink.of_type("trigger_suppressed")]
    assert suppressed == [(140, "B0200", "request"), (150, "B0200", "request"), (250, "A0200", "request")]
    assert max(_detector_steps(sink, "aircraft B0200")) == 135
    assert max(_detector_steps(sink, "aircraft A0200")) == 205
    assert (250, "aircraft A0100", "request") in _translations(sink)
    assert all(e["target"]["id"] not in ("B0200", "A0200") or e["step"] == 0 for e in sink.of_type("translation_event"))
    assert summary.status == STATUS_COMPLETED


def test_team_closes_when_its_last_aircraft_is_down(frames: dict[str, pd.DataFrame]) -> None:
    _, sink = _run(frames[WIN01], make_scenario([instruction("tb", 0, "team", "B")]))

    (closed,) = sink.of_type("target_closed")
    assert (closed["step"], closed["down_step"]) == (335, 335)
    assert closed["aircraft_down_steps"] == {"B0100": 335, "B0200": 136}
    assert max(_detector_steps(sink, "team B")) == 330


def test_instruction_for_an_already_down_target_is_suppressed(frames: dict[str, pd.DataFrame]) -> None:
    instructions = [instruction("a1", 0, *A1), instruction("late", 150, "aircraft", "B0200")]

    _, sink = _run(frames[LOSS01], make_scenario(instructions, episode_id=LOSS01))

    (closed,) = sink.of_type("target_closed")
    (suppressed,) = sink.of_type("trigger_suppressed")
    assert (closed["step"], closed["down_step"], closed["last_event_id"]) == (150, 137, None)
    assert (suppressed["step"], suppressed["primary_trigger"]) == (150, "initial")
    assert all(e["target"]["id"] != "B0200" for e in sink.of_type("translation_event"))


@pytest.mark.parametrize(
    ("cap", "status", "sent"), [(67, STATUS_CAP_REACHED, 67), (68, STATUS_COMPLETED, 68), (69, STATUS_COMPLETED, 68)]
)
def test_detector_cap_at_cap_minus_one_cap_and_cap_plus_one(
    frames: dict[str, pd.DataFrame], cap: int, status: str, sent: int
) -> None:
    detection = FakeDetectionModel(NONE)

    summary, sink = _run(
        frames[WIN01], make_scenario([instruction("a1", 0, *A1)]), detection=detection, limits__max_detector_calls=cap
    )

    assert summary.status == status
    assert len(detection.states) == sent == summary.detector.attempted == len(sink.of_type("detector_call"))
    if status == STATUS_CAP_REACHED:
        assert summary.unsent == {
            "layer": "detector",
            "target": {"kind": "aircraft", "id": "A0100"},
            "step": 340,
            "cap": 67,
        }
    assert {e["record_type"] for e in sink.events} <= set(RECORD_TYPES)


@pytest.mark.parametrize(
    ("cap", "status", "sent"), [(68, STATUS_CAP_REACHED, 68), (69, STATUS_COMPLETED, 69), (70, STATUS_COMPLETED, 69)]
)
def test_translation_cap_at_cap_minus_one_cap_and_cap_plus_one(
    frames: dict[str, pd.DataFrame], cap: int, status: str, sent: int
) -> None:
    intent = FakeIntentModel()

    summary, sink = _run(
        frames[WIN01],
        make_scenario([instruction("a1", 0, *A1)]),
        detection=FakeDetectionModel(("start", 1.0)),
        intent=intent,
        limits__max_translation_calls=cap,
    )

    assert summary.status == status
    assert (
        len(intent.prompts) == sent == summary.translation.attempted == len(sink.requests) - summary.detector.attempted
    )
    if status == STATUS_CAP_REACHED:
        assert summary.unsent == {
            "layer": "translation",
            "target": {"kind": "aircraft", "id": "A0100"},
            "step": 340,
            "cap": 68,
        }
    assert {e["record_type"] for e in sink.events} <= set(RECORD_TYPES)


def test_requests_are_written_before_each_send(frames: dict[str, pd.DataFrame]) -> None:
    log: list[tuple[str, str]] = []
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 7, A1)])

    _run(
        frames[WIN01],
        scenario,
        detection=FakeDetectionModel(("change", 0.7), log=log),
        intent=FakeIntentModel(log=log),
        log=log,
    )

    sends = [i for i, entry in enumerate(log) if entry[0] == "send"]
    assert sends and all(log[i - 1] == ("request", log[i][1]) and log[i + 1] == ("response", log[i][1]) for i in sends)


@pytest.mark.parametrize(
    ("error", "status", "response_status"),
    [
        (ModelCallError("timed out", error_type="OpenAIClientError", uncertain=True), STATUS_UNCERTAIN, "uncertain"),
        (
            ModelCallError("HTTP 400", error_type="OpenAIClientError", uncertain=False, raw={"e": 1}),
            STATUS_FAILED,
            "error",
        ),
        (RuntimeError("surprise"), STATUS_UNCERTAIN, "uncertain"),
    ],
)
def test_translation_call_failure_stops_the_run_and_is_logged(
    frames: dict[str, pd.DataFrame], error: Exception, status: str, response_status: str
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 7, A1), request("r2", 9, A1)])

    summary, sink = _run(frames[WIN01], scenario, intent=FakeIntentModel(fail=(2, error)))

    assert summary.status == status
    assert [e["step"] for e in sink.of_type("translation_event")] == [0]
    assert sink.requests[-1]["call_id"] == "test-scenario:aircraft:A0100:000007"
    assert sink.responses[-1]["status"] == response_status
    assert summary.translation == type(summary.translation)(
        attempted=2, completed=1, failed=int(status == STATUS_FAILED), uncertain=int(status == STATUS_UNCERTAIN)
    )
    expected_type = "unexpected:RuntimeError" if isinstance(error, RuntimeError) else "OpenAIClientError"
    assert summary.error["type"] == expected_type and summary.error["call_id"] == sink.requests[-1]["call_id"]


def test_detector_failure_stops_the_run(frames: dict[str, pd.DataFrame]) -> None:
    error = ModelCallError("HTTP 500", error_type="SystemOneServerError", uncertain=False)

    summary, sink = _run(
        frames[WIN01], make_scenario([instruction("a1", 0, *A1)]), detection=FakeDetectionModel(NONE, fail=(3, error))
    )

    assert summary.status == STATUS_FAILED and summary.error["layer"] == "detector"
    assert summary.detector.attempted == 3 and summary.detector.failed == 1
    assert len(sink.of_type("detector_call")) == 2


def test_invalid_translation_output_fails_after_logging_the_raw_response(frames: dict[str, pd.DataFrame]) -> None:
    summary, sink = _run(
        frames[WIN01], make_scenario([instruction("a1", 0, *A1)]), intent=FakeIntentModel(['{"intent": ""}'])
    )

    assert summary.status == STATUS_FAILED and summary.error["type"] == "TranslationOutputError"
    assert sink.responses[0]["status"] == "ok" and sink.responses[0]["raw"]["text"] == '{"intent": ""}'
    assert sink.of_type("translation_event") == []


@pytest.mark.parametrize("answer", [NONE, ("start", 1.0), ("change", 1.0)])
def test_plan_schedule_matches_the_run(frames: dict[str, pd.DataFrame], answer: Answer) -> None:
    instructions = [instruction("a2", 0, "aircraft", "A0200"), instruction("tb", 30, "team", "B")]
    requests = [request("r1", 7, ("aircraft", "A0200")), request("r2", 250, ("aircraft", "A0200"))]
    scenario = make_scenario(instructions, requests, episode_id=LOSS01)
    plan = plan_schedule(scenario, episode_facts(frames[LOSS01]), 5)

    summary, sink = _run(frames[LOSS01], scenario, detection=FakeDetectionModel(answer))

    expected_translations = plan.translation_upper_bound if answer != NONE else plan.mandatory_translations
    assert summary.detector.completed == plan.detector_calls
    assert summary.translation.completed == expected_translations
    assert len(sink.of_type("trigger_suppressed")) == plan.suppressed_triggers == 1
    assert [e["sequence"] for e in sink.events] == list(range(1, len(sink.events) + 1))


class FailingResponseSink(MemorySink):
    """Journals requests, then fails to journal the first response (e.g. a full disk)."""

    def write_response(self, record: dict[str, Any]) -> None:
        raise OSError("disk full")


def test_journal_failure_after_a_send_is_uncertain_with_accurate_counts(frames: dict[str, pd.DataFrame]) -> None:
    config = make_config()
    sink = FailingResponseSink()
    intent = FakeIntentModel()
    translator = make_translator(FakeDetectionModel(0.0), intent, config=config)

    summary = translator.run(
        frame=frames[WIN01], scenario=make_scenario([instruction("a1", 0, *A1)]), config=config, sink=sink
    )

    assert summary.status == STATUS_UNCERTAIN and len(intent.prompts) == 1
    assert summary.translation == type(summary.translation)(attempted=1, completed=0, failed=0, uncertain=1)
    assert summary.error == {
        "type": "OSError",
        "message": "disk full",
        "layer": "translation",
        "call_id": "test-scenario:aircraft:A0100:000000",
        "uncertain": True,
    }


def test_unexpected_exception_text_is_never_recorded(frames: dict[str, pd.DataFrame]) -> None:
    secret = "sk-SECRET-should-not-appear"
    scenario = make_scenario([instruction("a1", 0, *A1)])

    summary, sink = _run(frames[WIN01], scenario, intent=FakeIntentModel(fail=(1, RuntimeError(secret))))

    assert summary.status == STATUS_UNCERTAIN
    assert secret not in json.dumps(summary.error) + json.dumps(sink.responses)
    assert summary.error["type"] == "unexpected:RuntimeError"


# --- review fixes (D38) ---------------------------------------------------------------------------


def test_scenario_built_in_code_obeys_the_loader_rules(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1), instruction("t1", 5, "team", "A")])
    team_a = (TargetRef("team", "A"),)
    early_request = replace(scenario, requests=(RequestPoint("r1", 2, team_a, "human", "[PLACEHOLDER] q"),))
    early_alarm = replace(scenario, alarms=(Alarm("c1", 3, team_a, "[PLACEHOLDER] warning"),))
    long_alarm = replace(scenario, alarms=(Alarm("c2", 6, team_a, "x" * 201),))
    limits = make_config().scenario_limits

    for invalid, match in (
        (early_request, "no instruction at or before"),
        (early_alarm, "no instruction at or before"),
        (long_alarm, "the limit is 200"),
    ):
        with pytest.raises(ScenarioError, match=match):
            validate_scenario(invalid, episode_facts(frames[WIN01]), limits)
        with pytest.raises(ScenarioError, match=match):  # the run refuses it too, so plan_schedule never disagrees
            make_translator().run(frame=frames[WIN01], scenario=invalid, config=make_config(), sink=MemorySink())


class FailingSecondEventSink(MemorySink):
    def write_event(self, record: dict[str, Any]) -> None:
        if len(self.events) == 1:
            raise OSError("disk full")
        super().write_event(record)


def test_events_written_counts_only_events_that_reached_the_sink(frames: dict[str, pd.DataFrame]) -> None:
    config = make_config()
    sink = FailingSecondEventSink()
    scenario = make_scenario([instruction("a1", 0, *A1), instruction("t1", 0, "team", "A")])

    summary = make_translator(config=config).run(frame=frames[WIN01], scenario=scenario, config=config, sink=sink)

    assert summary.status == STATUS_FAILED and summary.error["type"] == "OSError"
    assert summary.events_written == len(sink.events) == 1


def test_ctrl_c_during_a_call_is_uncertain_and_journaled(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 7, A1)])

    summary, sink = _run(frames[WIN01], scenario, intent=FakeIntentModel(fail=(2, KeyboardInterrupt())))

    assert summary.status == STATUS_UNCERTAIN and summary.error["type"] == "unexpected:KeyboardInterrupt"
    assert summary.translation.uncertain == 1 and sink.responses[-1]["status"] == "uncertain"
    assert summary.error["call_id"] == sink.requests[-1]["call_id"] == "test-scenario:aircraft:A0100:000007"


class InterruptingEventSink(MemorySink):
    def write_event(self, record: dict[str, Any]) -> None:
        raise KeyboardInterrupt


def test_ctrl_c_between_calls_stops_the_run_as_failed(frames: dict[str, pd.DataFrame]) -> None:
    config = make_config()
    sink = InterruptingEventSink()
    intent = FakeIntentModel()

    summary = make_translator(intent=intent, config=config).run(
        frame=frames[WIN01], scenario=make_scenario([instruction("a1", 0, *A1)]), config=config, sink=sink
    )

    assert summary.status == STATUS_FAILED and len(intent.prompts) == 1
    assert summary.error == {
        "type": "KeyboardInterrupt",
        "message": "unexpected exception; details withheld",
        "layer": None,
        "call_id": None,
        "uncertain": False,
    }
    assert summary.translation.completed == 1 and sink.responses[0]["status"] == "ok"


# --- contract v2: inputs at or before t, messages, timing -----------------------------------------

STEP_MENTIONS = (
    re.compile(r"issued at step (\d+)"),  # instructions (both layers) and alarms (translation)
    re.compile(r"(?:alarm|question) at step (\d+)"),  # detector messages
    re.compile(r"Question from a human \(step (\d+)\)"),
)
PREVIOUS_MENTIONS = (re.compile(r"\(from step (\d+)\):"), re.compile(r"^From step (\d+):$", re.MULTILINE))


def _payload_text(value: Any) -> str:
    """Every string inside a request payload, one per line."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_payload_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_payload_text(item) for item in value)
    return ""


def _trajectory_lines(text: str) -> list[dict[str, Any]]:
    lines = []
    for line in text.split("\n"):
        if not line.startswith('{"step":'):
            continue
        lines.append(json.loads(line))
    return lines


def test_no_payload_contains_information_after_its_step(short_frame: pd.DataFrame) -> None:
    """User minimum test 1: real renderers, real 13_LAG rows, the v2 example scenario, both layers."""
    config = make_config()
    scenario = load_scenario(EXAMPLE_SCENARIO, limits=config.scenario_limits)
    answers = {("aircraft A0100", 5): START, ("aircraft A0100", 15): CHANGE, ("team A", 20): CHANGE}
    translator = make_translator(
        FakeDetectionModel(answers_at(answers)),
        config=config,
        detector_renderer=render_detector_compact,
        translation_renderer=render_lag17_jsonl,
    )
    sink = MemorySink()

    summary = translator.run(frame=short_frame, scenario=scenario, config=config, sink=sink)

    later_texts = [
        *((item.issue_step, item.text) for item in scenario.instructions),
        *((item.step, item.text) for item in scenario.alarms),
        *((item.step, item.question) for item in scenario.requests),
    ]
    assert summary.status == STATUS_COMPLETED and len(sink.requests) == 12 + 6
    for record in sink.requests:
        step, text = record["step"], _payload_text(record["payload"])
        trajectory = _trajectory_lines(text)
        assert trajectory and all(line["step"] <= step and line["relative_step"] <= 0 for line in trajectory)
        mentioned = [int(m) for pattern in STEP_MENTIONS for m in pattern.findall(text)]
        assert mentioned and all(mentioned_step <= step for mentioned_step in mentioned), record["call_id"]
        previous = [int(m) for pattern in PREVIOUS_MENTIONS for m in pattern.findall(text)]
        assert all(previous_step < step for previous_step in previous), record["call_id"]
        assert all(item not in text for issued, item in later_texts if issued > step), record["call_id"]
        assert scenario.episode.episode_id not in text and scenario.scenario_id not in text
    shown = "\n".join(_payload_text(record["payload"]) for record in sink.requests)
    assert all(item in shown for _, item in later_texts)  # every scenario text reaches some model input


def _message_ids(sink: MemorySink, label: str, step: int) -> list[str]:
    (call,) = [e for e in sink.of_type("detector_call") if e["step"] == step and _label(e) == label]
    return [message["id"] for message in call["input"]["messages"]]


def _label(event: dict[str, Any]) -> str:
    return f"{event['target']['kind']} {event['target']['id']}"


def test_messages_reach_the_detector_within_the_lookback_and_alarms_reach_every_sharing_target(
    frames: dict[str, pd.DataFrame],
) -> None:
    instructions = [
        instruction("a1", 0, *A1),
        instruction("a2", 0, "aircraft", "A0200"),
        instruction("ta", 0, "team", "A"),
        instruction("tb", 0, "team", "B"),
    ]
    alarms = [alarm("w1", 12, A1), alarm("w2", 11, ("team", "B")), alarm("w3", 16, ("team", "A"))]
    scenario = make_scenario(instructions, [request("r1", 10, A1)], alarms)

    _, sink = _run(frames[WIN01], scenario, detection=_answer_at({("team A", 20): CHANGE}))

    assert _message_ids(sink, "aircraft A0100", 10) == ["r1"]
    assert _message_ids(sink, "aircraft A0100", 15) == ["r1", "w1"]  # most recent last
    assert _message_ids(sink, "aircraft A0100", 30) == ["r1", "w1", "w3"]  # [6, 30]
    assert _message_ids(sink, "aircraft A0100", 35) == ["w1", "w3"]  # [11, 35]: lookback 25
    assert _message_ids(sink, "aircraft A0100", 40) == ["w3"]
    assert _message_ids(sink, "aircraft A0100", 45) == []
    assert _message_ids(sink, "team A", 15) == ["w1"]  # shares A0100; the question was for the aircraft only
    assert _message_ids(sink, "aircraft A0200", 15) == [] and _message_ids(sink, "aircraft A0200", 20) == ["w3"]
    assert _message_ids(sink, "team B", 15) == ["w2"]
    call = next(e for e in sink.of_type("detector_call") if e["step"] == 15 and _label(e) == "aircraft A0100")
    assert call["input"]["messages"][1] == {"kind": "alarm", "id": "w1", "step": 12, "text": "[PLACEHOLDER] alarm w1"}
    assert "- alarm at step 12: [PLACEHOLDER] alarm w1" in call["input"]["state_text"]
    events = {(e["step"], _label(e)): e for e in sink.of_type("translation_event")}
    assert [a["alarm_id"] for a in events[(20, "team A")]["input"]["alarms"]] == ["w1", "w3"]
    assert events[(10, "aircraft A0100")]["input"]["alarms"] == []  # w1 is issued later
    assert events[(10, "aircraft A0100")]["input"]["question"]["request_id"] == "r1"
    assert events[(20, "team A")]["input"]["question"] is None


def test_every_message_reaches_at_least_one_detector_state_of_every_target_it_is_for(
    frames: dict[str, pd.DataFrame],
) -> None:
    instructions = [instruction("a1", 0, *A1), instruction("ta", 0, "team", "A")]
    steps = (1, 4, 5, 6, 9, 333, 339, 340)  # around ticks, and the last ticks of the episode
    alarms = [alarm(f"w{step}", step, A1) for step in steps]
    requests = [request(f"r{step}", step, ("team", "A")) for step in steps]
    scenario = make_scenario(instructions, requests, alarms)

    _, sink = _run(frames[WIN01], scenario)

    seen = {(_label(e), m["id"]) for e in sink.of_type("detector_call") for m in e["input"]["messages"]}
    expected = {(label, f"w{step}") for label in ("aircraft A0100", "team A") for step in steps}
    expected |= {("team A", f"r{step}") for step in steps}
    assert expected <= seen
    assert not {key for key in seen if key[0] == "aircraft A0100" and key[1].startswith("r")}


def _counting_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = itertools.count()
    monkeypatch.setattr(pipeline, "perf_counter", lambda: float(next(ticks)))


def test_latency_and_processing_seconds_follow_the_monotonic_clock(
    frames: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    _counting_clock(monkeypatch)  # every reading advances by exactly 1 s
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 5, A1)])

    _, sink = _run(frames[WIN01], scenario)

    events = {e["step"]: e for e in sink.of_type("translation_event")}
    assert all(response["latency_s"] == 1.0 for response in sink.responses)
    assert events[0]["processing_s"] == 3.0  # start, send, answer, record complete
    assert events[5]["processing_s"] == 5.0  # the detector call at the tick is part of it


def test_processing_seconds_end_after_the_whole_record_and_timings_round_to_milliseconds(
    frames: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(pipeline, "perf_counter", lambda: clock["now"])
    step_time_s = pipeline.step_time_s

    def slow_step_time_s(step: int) -> float:  # built into every record's common fields
        clock["now"] += 0.12345
        return step_time_s(step)

    def slow_detector(state: str) -> Answer:
        clock["now"] += 1.23456
        return NONE

    monkeypatch.setattr(pipeline, "step_time_s", slow_step_time_s)
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 5, A1)])

    _, sink = _run(frames[WIN01], scenario, detection=FakeDetectionModel(slow_detector))

    events = {e["step"]: e for e in sink.of_type("translation_event")}
    detector_latencies = {r["latency_s"] for r in sink.responses if r["layer"] == "detector"}
    assert detector_latencies == {1.235}  # 1.23456 rounded to 3 decimals
    assert events[0]["processing_s"] == 0.123  # its own record's fields are inside the measure
    assert events[5]["processing_s"] == 1.481  # detector call + detector_call record + its own record


@pytest.mark.parametrize(
    ("layer", "error", "status"),
    [
        ("translation", None, "ok"),
        ("translation", ModelCallError("HTTP 400", error_type="OpenAIClientError", uncertain=False), "error"),
        ("translation", ModelCallError("timed out", error_type="OpenAIClientError", uncertain=True), "uncertain"),
        ("translation", KeyboardInterrupt(), "uncertain"),
        ("detector", ModelCallError("HTTP 500", error_type="SystemOneServerError", uncertain=False), "error"),
        ("detector", ModelCallError("timed out", error_type="SystemOneTransportError", uncertain=True), "uncertain"),
    ],
)
def test_latency_is_recorded_finite_and_non_negative_on_every_attempted_call(
    frames: dict[str, pd.DataFrame], layer: str, error: BaseException | None, status: str
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 7, A1)])
    fail = None if error is None else (2, error)
    intent = FakeIntentModel(fail=fail if layer == "translation" else None)
    detection = FakeDetectionModel(NONE, fail=fail if layer == "detector" else None)

    summary, sink = _run(frames[WIN01], scenario, detection=detection, intent=intent)

    attempted = summary.detector.attempted + summary.translation.attempted
    assert len(sink.responses) == attempted and sink.responses[-1]["status"] == status
    assert error is None or sink.responses[-1]["layer"] == layer  # the failing call is the last one
    assert all(
        isinstance(r["latency_s"], float) and math.isfinite(r["latency_s"]) and r["latency_s"] >= 0
        for r in sink.responses
    )
    processing = [e["processing_s"] for e in sink.of_type("translation_event")]
    assert processing and all(isinstance(value, float) and math.isfinite(value) and value >= 0 for value in processing)
