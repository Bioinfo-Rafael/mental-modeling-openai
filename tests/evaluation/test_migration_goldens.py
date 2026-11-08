"""Frozen request bytes of both judges (restructure guard).

Each test rebuilds serialized Responses request bodies from committed fixtures and compares their
SHA-256 with ``tests/fixtures/prompt-request-goldens.json``. The already-paid assess request
(``paid_assess_request_sha256``) needs the gitignored outputs of a real run, so it is checked by
``scripts/check_assess_resume.py`` instead. A mismatch means a prompt or request byte changed: fix the
change, do not update the golden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.assess_data import load_run
from evaluation.assess_judge import plan_calls
from evaluation.rendering import build_judge_request
from evaluation.schemas import EventKeyPoints, KeyPoint, load_translator_run
from translate.events import canonical_json, sha256_text
from translate.llm.openai_responses import build_responses_body

from .assess_fakes import make_config, make_frame, make_run
from .eval_fakes import A1_REQUEST, make_judge_config

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures" / "prompt-request-goldens.json"
KEY_POINTS = (KeyPoint("kp-001", "[PLACEHOLDER] judge key point one"), KeyPoint("kp-002", "[PLACEHOLDER] two"))
# initial + start + change/request: every point kind, and a FUTURE block from the synthetic frame.
ASSESS_PLAN = ((0, ("initial",)), (5, ("start",)), (15, ("change", "request")))


def goldens() -> dict[str, Any]:
    return json.loads(GOLDENS.read_text(encoding="utf-8"))


def d29_request_sha256(translator_run: Path) -> str:
    """The D29 key-point judge request of one request-triggered event, default text config."""
    events = load_translator_run(translator_run).translation_events
    event = next(item for item in events if item["event_id"] == A1_REQUEST)
    request = build_judge_request(
        event=event,
        event_log=events,
        key_points=EventKeyPoints(A1_REQUEST, KEY_POINTS),
        config=make_judge_config(),
    )
    return sha256_text(canonical_json(build_responses_body(request.openai_request)))


def assess_request_sha256s(root: Path) -> list[str]:
    """``request_sha256`` of every assess call planned over a synthetic Translator run."""
    run = load_run(make_run(root, plan=ASSESS_PLAN))
    calls = plan_calls(list(run.events), run=run, config=make_config(), annotation=None, load_frame=make_frame)
    return [call.request_sha256 for call in calls]


def test_d29_keypoint_judge_request_bytes_are_unchanged(translator_run: Path) -> None:
    assert d29_request_sha256(translator_run) == goldens()["d29_keypoint_judge_request_sha256"]


def test_assess_judge_request_bytes_are_unchanged(tmp_path: Path) -> None:
    assert assess_request_sha256s(tmp_path) == goldens()["assess_judge_request_sha256"]
