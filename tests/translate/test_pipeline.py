"""The causal loop: the three trigger kinds, the change sources, down targets, failures and the journal."""

from __future__ import annotations

from pathlib import Path

from translate.backends import BackendError

from .fakes import INSTRUCTION, QUESTION, FakeBackend, FakeDetector, clef, config, events, lines, run, scenario


def _user(record_or_body: dict) -> str:
    return record_or_body["input"][0]["content"][0]["text"]


def test_a_bare_start_is_translated_with_no_instruction_in_effect(tmp_path: Path) -> None:
    backend = FakeBackend()

    code = run(config(tmp_path, scenario(tmp_path)), backend=backend)

    (start,) = events(tmp_path)
    assert code == 0 and (start["step"], start["kinds"]) == (0, ["start"])
    assert start["trigger_reasons"] == [
        {"kind": "start", "source": "episode_start", "reference_id": "episode-start", "probability": None}
    ]
    assert "I: (no instruction in effect)" in _user(backend.bodies[0])
    assert start["input"]["instructions_in_effect"] == []


def test_a_start_with_an_instruction_in_effect_shows_it(tmp_path: Path) -> None:
    backend = FakeBackend()

    run(config(tmp_path, scenario(tmp_path, [INSTRUCTION])), backend=backend)

    assert 'I: - aircraft A0100, issued at step 0: "[TEST] hold the line" (this target)' in _user(backend.bodies[0])
    assert events(tmp_path)[0]["input"]["instructions_in_effect"][0]["instruction_id"] == "i1"


def test_a_question_is_a_request_point_and_an_instruction_is_not(tmp_path: Path) -> None:
    later_instruction = {**INSTRUCTION, "id": "i2", "step": 20, "to": ["team:A"], "text": "[TEST] switch"}
    backend = FakeBackend()

    run(
        config(tmp_path, scenario(tmp_path, [QUESTION, later_instruction], ["aircraft:A0100", "team:A"])),
        backend=backend,
    )

    found = [(e["step"], e["target"]["id"], e["kinds"]) for e in events(tmp_path)]
    assert found == [(0, "A0100", ["start"]), (0, "A", ["start"]), (10, "A0100", ["request"])]
    request = events(tmp_path)[2]
    assert request["input"]["question"] == {"request_id": "q1", "step": 10, "text": "[TEST] why turn?"}
    assert 'Q: "[TEST] why turn?"' in _user(backend.bodies[2]) and 'P: from step 0: "intent 1"' in _user(
        backend.bodies[2]
    )


def test_preset_change_steps_translate_every_live_target_and_coinciding_triggers_give_one_translation(
    tmp_path: Path,
) -> None:
    change = {"source": "preset", "steps": [10, 145]}

    run(config(tmp_path, scenario(tmp_path, [QUESTION], ["aircraft:A0100", "aircraft:B0200"]), change))

    found = [(e["step"], e["target"]["id"], e["kinds"]) for e in events(tmp_path)]
    assert found == [
        (0, "A0100", ["start"]),
        (0, "B0200", ["start"]),
        (10, "A0100", ["change", "request"]),
        (10, "B0200", ["change"]),
        (145, "A0100", ["change"]),
    ]
    coinciding = events(tmp_path)[2]["trigger_reasons"]
    assert [(r["kind"], r["source"], r["reference_id"]) for r in coinciding] == [
        ("change", "preset", "preset-step-10"),
        ("request", "question", "q1"),
    ]
    (closed,) = events(tmp_path, "target_closed")
    (suppressed,) = events(tmp_path, "trigger_suppressed")
    assert (closed["step"], closed["target"]["id"]) == (136, "B0200")
    assert (suppressed["step"], suppressed["kinds"]) == (145, ["change"])


def test_clef_flash_asks_on_every_tick_and_translates_only_on_change(tmp_path: Path) -> None:
    detector = FakeDetector({("aircraft:A0100", 10): 0.9, ("aircraft:A0100", 15): 0.6})
    path = config(tmp_path, scenario(tmp_path, [QUESTION]), clef(threshold=0.65))

    code = run(path, backend=FakeBackend(), detector=detector)

    calls = events(tmp_path, "detector_call")
    assert code == 0 and detector.healthy == 1 and len(calls) == 340 // 5
    assert [c["step"] for c in calls[:3]] == [5, 10, 15]
    assert calls[2]["output"] == {
        "label": "change",
        "probabilities": {"none": 0.4, "change": 0.6},
        "threshold": 0.65,
        "triggered": False,
    }
    found = [(e["step"], e["kinds"]) for e in events(tmp_path)]
    assert found == [(0, ["start"]), (10, ["change", "request"])]
    assert events(tmp_path)[1]["trigger_reasons"][0]["probability"] == 0.9
    assert 'Previous intent translation: from step 0: "intent 1"' in detector.bodies[0]["state"]


def test_an_invalid_answer_stops_the_run_or_is_skipped(tmp_path: Path) -> None:
    stop_dir, skip_dir = tmp_path / "stop", tmp_path / "skip"
    stop = run(
        config(stop_dir, scenario(stop_dir), {"source": "preset", "steps": [5]}),
        backend=FakeBackend(['{"intent": ""}']),
    )
    skip_config = config(
        skip_dir, scenario(skip_dir), {"source": "preset", "steps": [5]}, answer={"on_invalid": "skip"}
    )
    skip = run(skip_config, backend=FakeBackend(['{"intent": "x"}']))

    assert (stop, events(stop_dir), len(lines(stop_dir / "out" / "run" / "requests.jsonl"))) == (1, [], 1)
    skipped, answered = events(skip_dir)
    assert skip == 0 and skipped["output"] == {"intent": "", "evidence": []} and "answer_problem" in skipped
    assert answered["input"]["previous_translation"] is None  # a skipped answer is no previous translation


def test_a_failed_call_is_journaled_before_the_send_and_never_retried(tmp_path: Path) -> None:
    path = config(tmp_path, scenario(tmp_path), {"source": "preset", "steps": [5, 9]})
    backend = FakeBackend(
        [None, BackendError("timed out", uncertain=True)], journal=tmp_path / "out" / "run" / "requests.jsonl"
    )

    code = run(path, backend=backend)

    responses = lines(tmp_path / "out" / "run" / "responses.jsonl")
    assert code == 3 and len(backend.bodies) == 2 and backend.journal_lines_at_send == [1, 2]
    assert [r["status"] for r in responses] == ["ok", "uncertain"]


def test_a_cap_stops_before_call_cap_plus_one(tmp_path: Path) -> None:
    backend = FakeBackend()

    code = run(
        config(tmp_path, scenario(tmp_path), {"source": "preset", "steps": [5, 9]}, limits={"max_translations": 2}),
        backend=backend,
    )

    assert (code, len(backend.bodies)) == (4, 2)


def test_poison_two_runs_equal_through_t_send_identical_requests_through_t(tmp_path: Path) -> None:
    """The run-level poison test: a message after t changes nothing that was sent through t."""
    future = {"id": "w9", "step": 30, "kind": "alarm", "to": ["aircraft:A0100"], "text": "[TEST] FUTURE"}
    bodies = []
    for name, messages in (("clean", [QUESTION]), ("poisoned", [QUESTION, future])):
        directory = tmp_path / name
        run(config(directory, scenario(directory, messages), {"source": "preset", "steps": [5, 29, 30]}))
        bodies.append(
            [line["body"] for line in lines(directory / "out" / "run" / "requests.jsonl") if line["step"] <= 29]
        )

    assert bodies[0] == bodies[1] and len(bodies[0]) == 4  # steps 0, 5, 10 and 29
