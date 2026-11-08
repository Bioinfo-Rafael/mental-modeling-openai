"""The leak check of the generation inputs is method-independent (decision 2026-11-08).

A translator that did not read the trajectory (a policy input) still records the trajectory window
as the judge's material; its payload needs no window, the recorded window must match its hash, and
its activation window must end at or before t. Events without ``translator_inputs`` (runs before
2026-11-08) keep the original rule: the window must be in the payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.assess_checks import CHECK_ERROR, CHECK_LEAK, CHECK_OK, check_generation_inputs, reads_trajectory
from evaluation.assess_data import load_run

from .assess_fakes import E_REQUEST, build_run, canonical, make_config, sha256_text, write_run

POLICY_INPUTS = ["instructions", "previous_translation", "policy"]


def _payload(step: int) -> dict[str, Any]:
    """A policy-input request body: no trajectory window in it."""
    text = f"Policy at decision step {step}: weights W and activations A are in the sandbox."
    return {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}]}


def _policy_run(root: Path, *, window_end_offset: int = 0, translator_inputs: list[str] | None = POLICY_INPUTS,
                break_hash: bool = False) -> Path:  # fmt: skip
    """The fake run with the request event turned into a policy-input event."""
    files = build_run()
    events, requests = files["events"], files["requests"]
    index = next(i for i, event in enumerate(events) if event.get("event_id") == E_REQUEST)
    record = events[index]
    step = record["step"]
    payload = _payload(step)
    record_input = {**record["input"], "policy": {"window": [max(0, step - 59), step + window_end_offset]}}
    if translator_inputs is not None:
        record_input = {**record_input, "translator_inputs": translator_inputs}
    if break_hash:
        record_input = {**record_input, "trajectory": {**record_input["trajectory"], "sha256": "0" * 64}}
    events[index] = {**record, "input": record_input, "request_sha256": sha256_text(canonical(payload))}
    request_index = next(i for i, request in enumerate(requests) if request["call_id"] == E_REQUEST)
    requests[request_index] = {**requests[request_index], "payload": payload}
    return write_run(root / "run", files)


def _check(run_dir: Path) -> Any:
    checks = {check.call_id: check for check in check_generation_inputs(load_run(run_dir), make_config())}
    return checks[E_REQUEST]


def test_reads_trajectory_defaults_to_true_for_events_without_translator_inputs() -> None:
    assert reads_trajectory({"input": {}}) is True
    assert reads_trajectory({"input": {"translator_inputs": ["trajectory", "instructions"]}}) is True
    assert reads_trajectory({"input": {"translator_inputs": POLICY_INPUTS}}) is False


def test_policy_input_event_without_window_in_payload_is_ok(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path))

    assert check.status == CHECK_OK and check.problems == ()
    assert check.visible_end_step == check.step


def test_policy_input_event_with_window_after_t_is_a_leak(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, window_end_offset=1))

    assert check.status == CHECK_LEAK
    assert "trajectory_future" in {problem["check"] for problem in check.problems}


def test_policy_input_event_with_a_broken_window_hash_is_an_error(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, break_hash=True))

    assert check.status == CHECK_ERROR
    assert "trajectory_record" in {problem["check"] for problem in check.problems}


def test_event_without_translator_inputs_still_needs_the_window_in_its_payload(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, translator_inputs=None))

    assert check.status != CHECK_OK
    assert {"trajectory_record", "trajectory_line"} <= {problem["check"] for problem in check.problems}


def test_event_that_read_the_trajectory_still_needs_the_window_in_its_payload(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, translator_inputs=["instructions", "trajectory"]))

    assert check.status != CHECK_OK
    assert {"trajectory_record", "trajectory_line"} <= {problem["check"] for problem in check.problems}
