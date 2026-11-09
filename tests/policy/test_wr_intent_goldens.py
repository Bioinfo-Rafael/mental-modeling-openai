"""Frozen policy-prompt bytes through the unified config-driven pipeline."""

from __future__ import annotations

from pathlib import Path

from tests.translate.test_request_goldens import body_sha256, goldens, policy_values
from translate import paths
from translate.cli import _request_body
from translate.config import load_config
from translate.events import sha256_text
from translate.prompts import ConfiguredTurns


def test_intent_b_v1_messages_are_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    values = policy_values(tmp_path, small_files)
    found = {name: sha256_text(config.prompt.render(slot_values)[1]) for name, (config, slot_values) in values.items()}

    assert found == goldens("intent-b-v1-goldens.json")["messages_sha256"]


def test_intent_b_v1_first_request_body_is_unchanged(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    config, values = policy_values(tmp_path, small_files)["en/first"]

    assert (
        body_sha256(_request_body(config, values)) == goldens("intent-b-v1-goldens.json")["first_request_body_sha256"]
    )


def test_intent_b_v1_follow_up_turns_are_unchanged() -> None:
    config = load_config(paths.CONFIGS / "translate" / "seed100601-policy.yaml")
    turns = ConfiguredTurns(config.prompt.turns)
    found = {
        "run_result": turns.run_result(2, 12, "out", "", 0, False, False, 1),
        "run_result_all_notes": turns.run_result(12, 12, "out", "err tail", None, True, True, 3),
        "protocol_error": turns.protocol_error(("missing a block",), 5),
        "protocol_error_no_runs": turns.protocol_error(("a", "b"), 0),
        "budget_exhausted": turns.budget_exhausted(),
    }

    assert {name: sha256_text(text) for name, text in found.items()} == goldens("intent-b-v1-goldens.json")[
        "turns_sha256"
    ]
