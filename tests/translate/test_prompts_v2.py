"""Focused tests for prompt-file loading, blocks, and configured follow-up turns."""

from __future__ import annotations

from pathlib import Path

import pytest

from translate.prompts import ConfiguredTurns, PromptError, list_block, load_prompt, optional_block

from .tr_fakes import write_yaml


def _one_shot_prompt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": "test-v1",
        "protocol": "one_shot",
        "output_schema": "intent_evidence",
        "system": "System {language}",
        "user": "Step {step}",
        "blocks": {},
        "turns": None,
    }


@pytest.mark.parametrize("case", ["invalid_utf8", "unknown_key", "bad_schema"])
def test_load_prompt_with_malformed_file_raises_prompt_error(tmp_path: Path, case: str) -> None:
    path = tmp_path / "prompt.yaml"
    if case == "invalid_utf8":
        path.write_bytes(b"\xff")
    else:
        raw = _one_shot_prompt()
        raw = {**raw, "extra": "x"} if case == "unknown_key" else {**raw, "schema_version": 2}
        write_yaml(path, raw)

    with pytest.raises(PromptError):
        load_prompt(path, {"language": "English"})


def test_load_prompt_block_with_unknown_item_field_raises_prompt_error(tmp_path: Path) -> None:
    raw = _one_shot_prompt()
    raw["blocks"] = {
        "question": {
            "present": "step {step}: {unknown}",
            "empty": "none",
        }
    }
    path = write_yaml(tmp_path / "prompt.yaml", raw)

    with pytest.raises(PromptError, match=r"question.present.*unknown"):
        load_prompt(path, {"language": "English"})


def test_prompt_list_and_optional_blocks_render_present_and_empty_values() -> None:
    listed = {"item": "{step}:{text}", "joiner": " | ", "empty": "none"}
    optional = {"present": "Q{step}:{text}", "empty": "no question"}

    assert list_block(listed, []) == "none"
    assert list_block(listed, [{"step": 1, "text": "a"}, {"step": 2, "text": "b"}]) == "1:a | 2:b"
    assert optional_block(optional, None) == "no question"
    assert optional_block(optional, {"step": 3, "text": "why"}) == "Q3:why"


def _turns() -> dict[str, object]:
    return {
        "run_result": {
            "header": "run {run_index}/{max_runs}: {status}",
            "status_exit": "exit {exit_code}",
            "status_timed_out": "timed out",
            "merged_note": "merged {blocks}",
            "truncated_note": "truncated",
            "stderr_header": "stderr",
            "runs_remaining": "{remaining} left",
            "no_runs_remaining": "none left",
        },
        "protocol_error": {
            "text": "problems:\n{problems}\n{remaining}",
            "problem_item": "- {problem}",
            "runs_remaining": "{remaining} left",
            "no_runs_remaining": "none left",
        },
        "budget_exhausted": "answer now",
    }


def test_configured_turns_render_run_protocol_and_budget_messages() -> None:
    turns = ConfiguredTurns(_turns())

    assert turns.run_result(1, 2, "out", "err", 0, False, True, 2) == (
        "run 1/2: exit 0\n```text\nout\n```\nmerged 2\ntruncated\nstderr\n```text\nerr\n```\n1 left"
    )
    assert turns.run_result(2, 2, "", "", None, True, False) == ("run 2/2: timed out\n```text\n\n```\nnone left")
    assert turns.protocol_error(("bad one", "bad two"), 0) == ("problems:\n- bad one\n- bad two\nnone left")
    assert turns.budget_exhausted() == "answer now"
