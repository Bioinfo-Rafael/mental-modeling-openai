"""Focused tests for prompt-file loading and blocks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from translate.prompts import PromptError, list_block, load_prompt, optional_block

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


def test_load_prompt_with_duplicate_key_raises_prompt_error(tmp_path: Path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_text(
        yaml.safe_dump(_one_shot_prompt(), sort_keys=False) + "user: a second user template\n", encoding="utf-8"
    )

    with pytest.raises(PromptError, match="duplicate key 'user'"):
        load_prompt(path, {"language": "English"})
