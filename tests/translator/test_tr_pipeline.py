"""Pipeline: start/tick/trigger rules, precedence, mid-episode targets, windows, death, caps, errors."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from mtd_rl_xavier.translator.config import ScenarioError
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
from mtd_rl_xavier.translator.schema import (
    RECORD_TYPES,
    ExternalChangePoint,
    ModelCallError,
    RequestPoint,
    Scenario,
    TargetRef,
)

from .tr_fakes import (
    LOSS01,
    WIN01,
    FakeDetectionModel,
    FakeIntentModel,
    MemorySink,
    change,
    instruction,
    make_config,
    make_scenario,
    make_translator,
    request,
    state_key,
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


def _positive_at(*keys: tuple[str, int]) -> FakeDetectionModel:
    return FakeDetectionModel(lambda state: 0.9 if state_key(state) in keys else 0.1)


A1 = ("aircraft", "A0100")


def test_step_zero_starts_every_instructed_target_in_canonical_order(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("t1", 0, "team", "A"), instruction("a1", 0, *A1)])

    summary, sink = _run(frames[WIN01], scenario)

    first, second = sink.of_type("translation_event")[:2]
    assert summary.status == STATUS_COMPLETED
    assert _translations(sink) == [(0, "aircraft A0100", "start"), (0, "team A", "start")]
    assert first["event_id"] == "test-scenario:aircraft:A0100:000000"
    assert [i["instruction_id"] for i in first["input"]["instructions_in_effect"]] == ["a1", "t1"]
    assert first["input"]["primary_instruction_id"] == "a1" and second["input"]["primary_instruction_id"] == "t1"
    assert first["input"]["previous_translation"] is None
    assert first["trigger_reasons"] == [
        {"kind": "start", "source": "scenario", "reference_id": "a1", "probability": None}
    ]
    assert first["time_s"] == 0.0 and first["episode"]["csv_sha256"] == frames[WIN01].attrs["csv_sha256"]


def test_detector_ticks_stay_on_the_fixed_grid(frames: dict[str, pd.DataFrame]) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 3, A1)])

    summary, sink = _run(frames[WIN01], scenario)

    assert _detector_steps(sink, "aircraft A0100") == list(range(5, 341, 5))
    assert _translations(sink) == [(0, "aircraft A0100", "start"), (3, "aircraft A0100", "request")]
    assert summary.detector.completed == 68 and summary.translation.completed == 2


def test_tick_with_coinciding_triggers_gives_one_detector_call_and_one_translation(
    frames: dict[str, pd.DataFrame],
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1)], [request("r1", 10, A1)], [change("c1", 10, *A1)])

    _, sink = _run(frames[WIN01], scenario, detection=_positive_at(("aircraft A0100", 10)))

    events = [e for e in sink.of_type("translation_event") if e["step"] == 10]
    assert _detector_steps(sink, "aircraft A0100").count(10) == 1
    assert len(events) == 1
    reasons = events[0]["trigger_reasons"]
    assert [r["kind"] for r in reasons] == ["external_change", "detector_change", "request"]
    assert reasons[1] == {
        "kind": "detector_change",
        "source": "detector",
        "reference_id": "test-scenario:detector:aircraft:A0100:000010",
        "probability": 0.9,
    }
    assert events[0]["primary_trigger"] == "external_change"


@pytest.mark.parametrize(
    ("instructions", "requests", "changes", "positive", "expected"),
    [
        ([instruction("a2", 20, *A1)], [request("r1", 20, A1)], [change("c1", 20, *A1)], True, "instruction"),
        ([], [request("r1", 20, A1)], [change("c1", 20, *A1)], True, "external_change"),
        ([], [request("r1", 20, A1)], [], True, "detector_change"),
        ([], [request("r1", 20, A1)], [], False, "request"),
        ([], [request("r1", 0, A1)], [change("c1", 0, *A1)], False, "start"),
    ],
)
def test_primary_trigger_precedence(
    frames: dict[str, pd.DataFrame],
    instructions: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    positive: bool,
    expected: str,
) -> None:
    scenario = make_scenario([instruction("a1", 0, *A1), *instructions], requests, changes)
    detection = _positive_at(("aircraft A0100", 20)) if positive else FakeDetectionModel(0.0)

    _, sink = _run(frames[WIN01], scenario, detection=detection)

    step = 0 if expected == "start" else 20
    (event,) = [e for e in sink.of_type("translation_event") if e["step"] == step]
    assert event["primary_trigger"] == expected
    assert len(event["trigger_reasons"]) == len(requests) + len(changes) + len(instructions) + positive + (step == 0)


def test_target_first_instructed_mid_episode_differs_from_a_replacement(frames: dict[str, pd.DataFrame]) -> None:
    instructions = [
        instruction("a1", 0, *A1),
        instruction("ta1", 0, "team", "A"),
        instruction("ta2", 20, "team", "A"),
        instruction("tb1", 23, "team", "B"),
    ]

    _, sink = _run(frames[WIN01], make_scenario(instructions))

    by_key = {(e["step"], e["target"]["id"]): e for e in sink.of_type("translation_event")}
    replacement, first_b = by_key[(20, "A")], by_key[(23, "B")]
    assert replacement["primary_trigger"] == "instruction"
    assert replacement["input"]["previous_translation"]["event_id"] == "test-scenario:team:A:000000"
    assert first_b["primary_trigger"] == "instruction" and first_b["input"]["previous_translation"] is None
    assert [i["instruction_id"] for i in first_b["input"]["instructions_in_effect"]] == ["a1", "ta2", "tb1"]
    assert _detector_steps(sink, "team B")[:2] == [25, 30]  # no detector before its first translation
    assert 20 in _detector_steps(sink, "team A")  # the replaced target is still checked at its tick


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
    assert aircraft_call["input"]["question"]["id"] == "unexplained_change"
    assert aircraft_call["raw_response"]["answers"]["unexplained_change"]["noul"] == 0.0


def test_death_closes_targets_and_suppresses_later_triggers(frames: dict[str, pd.DataFrame]) -> None:
    instructions = [instruction(f"i-{a}", 0, "aircraft", a) for a in ("A0100", "A0200", "B0200")]
    instructions.append(instruction("tb", 0, "team", "B"))
    requests = [request("r1", 150, ("aircraft", "B0200")), request("r2", 250, ("aircraft", "A0200"), A1)]
    changes = [change("c1", 140, "aircraft", "B0200")]

    summary, sink = _run(frames[LOSS01], make_scenario(instructions, requests, changes, episode_id=LOSS01))

    closed = {e["target"]["id"]: e for e in sink.of_type("target_closed")}
    assert set(closed) == {"B0200", "A0200"}  # team B keeps B0100 alive
    assert (closed["B0200"]["step"], closed["B0200"]["down_step"]) == (137, 137)
    assert (closed["A0200"]["step"], closed["A0200"]["down_step"]) == (210, 210)
    assert closed["B0200"]["last_event_id"] == "test-scenario:aircraft:B0200:000000"
    suppressed = [(e["step"], e["target"]["id"], e["primary_trigger"]) for e in sink.of_type("trigger_suppressed")]
    assert suppressed == [(140, "B0200", "external_change"), (150, "B0200", "request"), (250, "A0200", "request")]
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
    assert (suppressed["step"], suppressed["primary_trigger"]) == (150, "instruction")
    assert all(e["target"]["id"] != "B0200" for e in sink.of_type("translation_event"))


@pytest.mark.parametrize(
    ("cap", "status", "sent"), [(67, STATUS_CAP_REACHED, 67), (68, STATUS_COMPLETED, 68), (69, STATUS_COMPLETED, 68)]
)
def test_detector_cap_at_cap_minus_one_cap_and_cap_plus_one(
    frames: dict[str, pd.DataFrame], cap: int, status: str, sent: int
) -> None:
    detection = FakeDetectionModel(0.0)

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
        detection=FakeDetectionModel(1.0),
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

    _run(frames[WIN01], scenario, detection=FakeDetectionModel(0.7, log=log), intent=FakeIntentModel(log=log), log=log)

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
        frames[WIN01], make_scenario([instruction("a1", 0, *A1)]), detection=FakeDetectionModel(0.0, fail=(3, error))
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


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_plan_schedule_matches_the_run(frames: dict[str, pd.DataFrame], probability: float) -> None:
    instructions = [instruction("a2", 0, "aircraft", "A0200"), instruction("tb", 30, "team", "B")]
    requests = [request("r1", 7, ("aircraft", "A0200")), request("r2", 250, ("aircraft", "A0200"))]
    scenario = make_scenario(instructions, requests, episode_id=LOSS01)
    plan = plan_schedule(scenario, episode_facts(frames[LOSS01]), 5)

    summary, sink = _run(frames[LOSS01], scenario, detection=FakeDetectionModel(probability))

    expected_translations = plan.translation_upper_bound if probability else plan.mandatory_translations
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
    early_request = replace(scenario, requests=(RequestPoint("r1", 2, (TargetRef("team", "A"),), "human"),))
    early_change = replace(
        scenario, external_change_points=(ExternalChangePoint("c1", 3, TargetRef("team", "A"), "x"),)
    )

    for invalid in (early_request, early_change):
        with pytest.raises(ScenarioError, match="no instruction at or before"):
            validate_scenario(invalid, episode_facts(frames[WIN01]))
        with pytest.raises(ScenarioError):  # the run refuses it too, so plan_schedule never disagrees
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
