"""The leak check of the generation inputs is method-independent (decision 2026-11-08).

A translator that did not read the trajectory (a policy input) still records the trajectory window
as the judge's material; its payload needs no window, and the recorded window must match its hash.
Whenever the policy input is selected, the event must record the bundle the sandbox read
(``input.policy``: session directory, manifest SHA-256 and activation window); the check verifies
it against ``sessions/<dir>/inputs/manifest.json`` and requires the window to end at or before t
(and before t when ``allow_action_at_t`` is false). Events without ``translator_inputs`` (runs before
2026-11-08) keep the original rule: the window must be in the payload.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.assess_checks import CHECK_ERROR, CHECK_LEAK, CHECK_OK, check_generation_inputs, reads_trajectory
from evaluation.assess_data import load_run

from .assess_fakes import E_REQUEST, build_run, canonical, make_config, sha256_text, write_run

POLICY_INPUTS = ["instructions", "previous_translation", "policy"]
SESSION_DIR = "s0010_A0100"
WINDOW_STEPS = 60


def _payload(step: int) -> dict[str, Any]:
    """A policy-input request body: no trajectory window in it."""
    text = f"Policy at decision step {step}: weights W and activations A are in the sandbox."
    return {"model": "gpt-6.1-sol", "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}]}


def _manifest_bytes(step: int, window: list[int]) -> bytes:
    manifest = {"kind": "intent-b-session-inputs", "aircraft": "A0100", "step": step, "window": window}
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _policy_run(
    root: Path,
    *,
    window_end_offset: int = 0,
    translator_inputs: list[str] | None = POLICY_INPUTS,
    keep_trajectory_payload: bool = False,
    break_hash: bool = False,
    policy: dict[str, Any] | None | str = "bundle",
    manifest: bytes | None = None,
) -> Path:
    """The fake run with the request event turned into a policy-input event and its session bundle."""
    files = build_run()
    events, requests = files["events"], files["requests"]
    index = next(i for i, event in enumerate(events) if event.get("event_id") == E_REQUEST)
    record = events[index]
    step = record["step"]
    window = [max(0, step - WINDOW_STEPS + 1), step + window_end_offset]
    written = _manifest_bytes(step, window) if manifest is None else manifest
    record_input = dict(record["input"])
    if policy == "bundle":
        sha = hashlib.sha256(_manifest_bytes(step, window)).hexdigest()
        record_input["policy"] = {"session_dir": SESSION_DIR, "inputs_manifest_sha256": sha, "window": window}
    elif policy is not None:
        record_input["policy"] = policy
    if translator_inputs is not None:
        record_input["translator_inputs"] = translator_inputs
    if break_hash:
        record_input["trajectory"] = {**record_input["trajectory"], "sha256": "0" * 64}
    request_index = next(i for i, request in enumerate(requests) if request["call_id"] == E_REQUEST)
    payload = requests[request_index]["payload"] if keep_trajectory_payload else _payload(step)
    events[index] = {**record, "input": record_input, "request_sha256": sha256_text(canonical(payload))}
    requests[request_index] = {**requests[request_index], "payload": payload}
    run_dir = write_run(root / "run", files)
    inputs_dir = run_dir / "sessions" / SESSION_DIR / "inputs"
    inputs_dir.mkdir(parents=True)
    (inputs_dir / "manifest.json").write_bytes(written)
    return run_dir


def _check(run_dir: Path, **config: object) -> Any:
    checks = {check.call_id: check for check in check_generation_inputs(load_run(run_dir), make_config(**config))}
    return checks[E_REQUEST]


def _problems(check: Any) -> set[str]:
    return {problem["check"] for problem in check.problems}


def test_reads_trajectory_defaults_to_true_for_events_without_translator_inputs() -> None:
    assert reads_trajectory({"input": {}}) is True
    assert reads_trajectory({"input": {"translator_inputs": ["trajectory", "instructions"]}}) is True
    assert reads_trajectory({"input": {"translator_inputs": POLICY_INPUTS}}) is False


def test_policy_input_event_with_its_bundle_and_no_window_in_payload_is_ok(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path))

    assert check.status == CHECK_OK and check.problems == ()
    assert check.visible_end_step == check.step


def test_policy_input_event_with_window_after_t_is_a_leak(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, window_end_offset=1))

    assert check.status == CHECK_LEAK and "trajectory_future" in _problems(check)


def test_policy_and_trajectory_inputs_still_check_the_policy_window(tmp_path: Path) -> None:
    run_dir = _policy_run(
        tmp_path,
        window_end_offset=1,
        translator_inputs=["instructions", "trajectory", "policy"],
        keep_trajectory_payload=True,
    )

    check = _check(run_dir)

    assert check.status == CHECK_LEAK and "trajectory_future" in _problems(check)


def test_policy_window_ending_at_t_is_a_leak_when_action_at_t_is_not_allowed(tmp_path: Path) -> None:
    run_dir = _policy_run(tmp_path)

    allowed, refused = _check(run_dir), _check(run_dir, leakage__allow_action_at_t=False)

    assert allowed.status == CHECK_OK
    assert refused.status == CHECK_LEAK and "action_at_t" in _problems(refused)


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        (None, "records no input.policy"),
        ({"session_dir": SESSION_DIR, "window": [0, 10]}, "inputs_manifest_sha256"),
        ({"session_dir": "../escape", "inputs_manifest_sha256": "0" * 64, "window": [0, 10]}, "session_dir"),
        ({"session_dir": SESSION_DIR, "inputs_manifest_sha256": "0" * 64, "window": [0]}, "window"),
    ],
)
def test_policy_input_event_without_a_verifiable_bundle_record_is_an_error(
    tmp_path: Path, policy: dict[str, Any] | None, match: str
) -> None:
    check = _check(_policy_run(tmp_path, policy=policy))

    assert check.status == CHECK_ERROR
    assert any(match in problem["detail"] for problem in check.problems if problem["check"] == "policy_record")


def test_policy_bundle_manifest_changed_after_the_run_is_an_error(tmp_path: Path) -> None:
    tampered = _manifest_bytes(10, [0, 9])

    check = _check(_policy_run(tmp_path, manifest=tampered))

    assert check.status == CHECK_ERROR and "policy_record" in _problems(check)


def test_policy_bundle_whose_window_differs_from_the_event_is_an_error(tmp_path: Path) -> None:
    other = _manifest_bytes(10, [0, 9])
    record = {
        "session_dir": SESSION_DIR,
        "inputs_manifest_sha256": hashlib.sha256(other).hexdigest(),
        "window": [0, 10],
    }

    check = _check(_policy_run(tmp_path, policy=record, manifest=other))

    assert check.status == CHECK_ERROR and "policy_record" in _problems(check)


def test_policy_input_event_with_a_broken_window_hash_is_an_error(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, break_hash=True))

    assert check.status == CHECK_ERROR and "trajectory_record" in _problems(check)


def test_event_without_translator_inputs_still_needs_the_window_in_its_payload(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, translator_inputs=None, policy=None))

    assert check.status != CHECK_OK
    assert {"trajectory_record", "trajectory_line"} <= _problems(check)


def test_event_that_read_the_trajectory_still_needs_the_window_in_its_payload(tmp_path: Path) -> None:
    check = _check(_policy_run(tmp_path, translator_inputs=["instructions", "trajectory"], policy=None))

    assert check.status != CHECK_OK
    assert {"trajectory_record", "trajectory_line"} <= _problems(check)
