"""Leak check of a policy input: the bundle the sandbox read must be recorded, verifiable and causal.

A translation that selected the ``policy`` input (``input.translator_inputs``) records in
``input.policy`` the session directory, the SHA-256 of the bundle manifest the sandbox read and the
activation window that manifest states. The check re-reads
``<run>/sessions/<session_dir>/inputs/manifest.json``, compares its SHA-256, window and step with the
event, and requires the window to end at or before t (before t when ``allow_action_at_t`` is false:
the last row of ``A`` holds the logits and the action chosen at that step). A missing or
unverifiable record is an error; a window after t is a leak. This holds whatever else was selected.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .assess_data import is_int, is_safe_id

POLICY_INPUT = "policy"
SESSIONS_DIR = "sessions"
INPUTS_MANIFEST = Path("inputs") / "manifest.json"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
WINDOW_LENGTH = 2  # [start, end], inclusive steps


def _problem(check: str, detail: str) -> dict[str, str]:
    return {"check": check, "detail": detail}


def _record_error(detail: str) -> dict[str, str]:
    return _problem("policy_record", detail)


def uses_policy(record: dict[str, Any]) -> bool:
    """Whether the translator read a policy input (or the event records one)."""
    payload = record["input"]
    return POLICY_INPUT in payload.get("translator_inputs", []) or POLICY_INPUT in payload


def _is_window(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == WINDOW_LENGTH
        and all(is_int(item) for item in value)
        and value[0] <= value[1]
    )


def _record_problem(policy: object) -> dict[str, str] | None:
    """What is malformed in ``input.policy`` (None = well-formed)."""
    if not isinstance(policy, dict):
        return _record_error("the event selected the policy input but records no input.policy")
    if not is_safe_id(policy.get("session_dir")):
        return _record_error("input.policy.session_dir must be a session directory name")
    digest = policy.get("inputs_manifest_sha256")
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        return _record_error("input.policy.inputs_manifest_sha256 must be a SHA-256 hex digest")
    if not _is_window(policy.get("window")):
        return _record_error("input.policy.window must be [start, end] steps")
    return None


def _bundle_problem(policy: dict[str, Any], run_dir: Path, step: int) -> dict[str, str] | None:
    """Whether the recorded bundle is the one on disk, with the recorded window and step."""
    path = run_dir / SESSIONS_DIR / policy["session_dir"] / INPUTS_MANIFEST
    if not path.is_file():
        return _record_error(f"the session inputs manifest {policy['session_dir']}/{INPUTS_MANIFEST} is missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != policy["inputs_manifest_sha256"]:
        return _record_error("the session inputs manifest does not match input.policy.inputs_manifest_sha256")
    try:
        manifest = json.loads(data)
    except ValueError:
        return _record_error("the session inputs manifest is not valid JSON")
    if not isinstance(manifest, dict) or manifest.get("window") != policy["window"] or manifest.get("step") != step:
        return _record_error("the session inputs manifest states another window or step than the event")
    return None


def policy_problems(
    record: dict[str, Any], run_dir: Path, step: int, future_steps: int, allow_action_at_t: bool
) -> tuple[int | None, list[dict[str, str]]]:
    """(the verified window's end step or None, the problems) of one translation record."""
    if not uses_policy(record):
        return None, []
    policy = record["input"].get(POLICY_INPUT)
    problem = _record_problem(policy)
    if problem is not None:
        return None, [problem]
    problems = []
    bundle_problem = _bundle_problem(policy, run_dir, record["step"])
    if bundle_problem is not None:
        problems.append(bundle_problem)
    end = policy["window"][1]
    if end > step + future_steps:
        problems.append(_problem("trajectory_future", f"the policy window ends at step {end} > t"))
    elif not allow_action_at_t and end >= step:
        detail = "the policy window holds the logits and the action chosen at t (allow_action_at_t false)"
        problems.append(_problem("action_at_t", detail))
    return end, problems
