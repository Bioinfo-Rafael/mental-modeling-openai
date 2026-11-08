"""Policy-input units: activation replay, prompt hygiene, and answer validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.translate.test_request_goldens import policy_values
from translate import paths
from translate.answers import MAX_INTENT_CHARS
from translate.policy.reference import build_policy
from translate.policy.spec import load_spec
from translate.policy.torch_reader import read_checkpoint
from translate.policy.trace import (
    TraceError,
    activation_docs,
    concatenated_logits,
    trace_episode,
    validate_trace,
    window_arrays,
    window_bounds,
)
from translate.sandbox.protocol import (
    KIND_CODE,
    KIND_ERROR,
    KIND_FINAL,
    IntentAnswer,
    IntentAnswerError,
    parse_answer,
    parse_reply,
)

MAPPO_SPEC = paths.CONFIGS / "translate" / "policies" / "lag_mappo_2v2_33.spec.json"
ROLLOUT = paths.OUTPUTS / "weightreading" / "rollouts" / "mhi-v1.3-20261104T010525Z" / "episode_seed100601.npz"
REAL_LOGITS_TOLERANCE = 1e-5


def _small_policy(small_files: tuple[Path, Path]):
    checkpoint, spec = small_files
    return build_policy(load_spec(spec), read_checkpoint(checkpoint).tensors)


def _answer_text(payload: object) -> str:
    return "Done.\n```json\n" + json.dumps(payload) + "\n```"


def test_trace_episode_matches_reference_rollout_and_docs(small_files) -> None:
    policy = _small_policy(small_files)
    obs = np.random.default_rng(1).standard_normal((6, 2, policy.input_size))

    trace = trace_episode(policy, obs)
    final = policy.rollout(obs, policy.initial_hidden(2))

    np.testing.assert_allclose(trace["gru.h_new"][-1], final.hidden, atol=1e-12)
    np.testing.assert_allclose(concatenated_logits(trace, policy)[-1], np.concatenate(final.logits, -1), atol=1e-12)
    np.testing.assert_array_equal(trace["h_prev"][0], 0.0)
    np.testing.assert_array_equal(trace["h_prev"][1:], trace["gru.h_new"][:-1])
    assert [doc.key for doc in activation_docs(policy)] == ["step", *trace]
    for doc in activation_docs(policy)[1:]:
        assert trace[doc.key].shape == (6, 2, *doc.shape_tail), doc.key


def test_validate_trace_flags_a_changed_recording(small_files) -> None:
    policy = _small_policy(small_files)
    trace = trace_episode(policy, np.random.default_rng(2).standard_normal((4, 2, policy.input_size)))
    logits, hidden, actions = concatenated_logits(trace, policy), trace["gru.h_new"], trace["action"]

    assert validate_trace(trace, policy, logits, hidden, actions)["ok"] is True
    shifted = validate_trace(trace, policy, logits + 1e-3, hidden, actions)
    assert shifted["ok"] is False and shifted["max_abs_logits_diff"] == pytest.approx(1e-3)
    flipped = actions.copy()
    flipped[0, 0, 0] += 1
    assert validate_trace(trace, policy, logits, hidden, flipped)["action_mismatches"] == 1


@pytest.mark.parametrize(
    ("step", "expected"), [(0, (0, 0)), (45, (0, 45)), (59, (0, 59)), (60, (1, 60)), (315, (256, 315))]
)
def test_window_bounds_clamps_at_episode_start(step: int, expected: tuple[int, int]) -> None:
    assert window_bounds(step, 60) == expected


def test_window_arrays_slices_one_row_with_dtypes(small_files) -> None:
    policy = _small_policy(small_files)
    trace = trace_episode(policy, np.random.default_rng(3).standard_normal((8, 2, policy.input_size)))

    arrays = window_arrays(trace, 1, 2, 5)

    np.testing.assert_array_equal(arrays["step"], [2, 3, 4, 5])
    assert arrays["x"].dtype == np.float32 and arrays["action"].dtype == np.int64
    np.testing.assert_allclose(arrays["gru.r"], trace["gru.r"][2:6, 1].astype(np.float32))
    with pytest.raises(TraceError):
        window_arrays(trace, 0, 5, 8)


def test_real_recording_replay_matches_torch(mhi_model_dir: Path) -> None:
    if not ROLLOUT.exists():
        pytest.skip("rollout recording not available")
    checkpoint = read_checkpoint(mhi_model_dir / "v1.3" / "files" / "actor_latest.pt")
    policy = build_policy(load_spec(MAPPO_SPEC), checkpoint.tensors)
    data = np.load(ROLLOUT)

    trace = trace_episode(policy, data["obs"])
    result = validate_trace(trace, policy, data["logits"], data["gru_out"], data["actions"], data["rnn_in"])

    assert result["ok"] and result["finite"] and result["checked_state_in"] and result["action_agreement"] == 1.0
    assert (
        result["max_abs_logits_diff"] < REAL_LOGITS_TOLERANCE and result["max_abs_hidden_diff"] < REAL_LOGITS_TOLERANCE
    )


def test_intent_message_hides_spec_evaluation_notes_and_provenance(
    tmp_path: Path, small_files: tuple[Path, Path]
) -> None:
    config, values = policy_values(tmp_path, small_files)["en/first"]
    message = config.prompt.render(values)[1]

    for hidden in ("evaluation", "secret note", "mtd-rl-lag", "multiplecombat_task", "7cdf5ebc"):
        assert hidden not in message
    assert "Low-level aircraft flight controller test network." in message


def test_intent_message_has_no_world_trajectory_and_chains_previous(
    tmp_path: Path, small_files: tuple[Path, Path]
) -> None:
    config, values = policy_values(tmp_path, small_files)["en/chained"]
    message = config.prompt.render(values)[1]

    assert "From step 3:" in message and "[PLACEHOLDER] climb and turn left" in message
    for world in ("lon_deg", "lat_deg", "geodetic", "position_neu", "velocity_neu", "missiles_left"):
        assert world not in message


def test_intent_message_asks_for_japanese_only_when_requested(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    found = policy_values(tmp_path, small_files)
    english = found["en/first"][0].prompt.render(found["en/first"][1])[1]
    japanese = found["ja/first"][0].prompt.render(found["ja/first"][1])[1]

    assert "Write both fields in English." in english and "natural Japanese" not in english
    assert "Write both fields in natural Japanese" in japanese
    en_instruction = found["en/first"][0].prompt.vars["language_instruction"]
    ja_instruction = found["ja/first"][0].prompt.vars["language_instruction"]
    assert english.replace(en_instruction, ja_instruction) == japanese


def test_parse_intent_reply_kinds() -> None:
    valid = _answer_text({"intent": "Pursue.", "evidence": ["a", "b"]})

    assert parse_reply("```python\nprint(1)\n```").kind == KIND_CODE
    assert parse_reply(valid).kind == KIND_FINAL
    assert parse_answer(valid) == IntentAnswer("Pursue.", ("a", "b"))
    assert parse_reply("I am done.").kind == KIND_ERROR


@pytest.mark.parametrize(
    "payload",
    [
        {"intent": "x"},
        {"intent": "x", "evidence": [], "extra": 1},
        {"intent": " ", "evidence": ["a"]},
        {"intent": "x" * (MAX_INTENT_CHARS + 1), "evidence": ["a"]},
        {"intent": "x", "evidence": []},
        {"intent": "x", "evidence": ["ok", ""]},
        {"intent": "x", "evidence": "not a list"},
        ["intent", "evidence"],
    ],
)
def test_parse_answer_rejects_schema_violations_without_a_retry(payload: object) -> None:
    text = _answer_text(payload)

    assert parse_reply(text).kind == KIND_FINAL
    with pytest.raises(IntentAnswerError):
        parse_answer(text)


def test_parse_answer_counts_japanese_characters_not_bytes() -> None:
    at_limit = _answer_text({"intent": "あ" * MAX_INTENT_CHARS, "evidence": ["根拠"]})
    assert parse_answer(at_limit).intent == "あ" * MAX_INTENT_CHARS
    with pytest.raises(IntentAnswerError):
        parse_answer(_answer_text({"intent": "あ" * (MAX_INTENT_CHARS + 1), "evidence": ["根拠"]}))


def test_parse_answer_accepts_the_length_limit_and_rejects_two_blocks_duplicates_surrogates() -> None:
    at_limit = _answer_text({"intent": "x" * MAX_INTENT_CHARS, "evidence": ["a"]})
    assert parse_answer(at_limit).intent == "x" * MAX_INTENT_CHARS
    for bad in (
        at_limit + "\n" + at_limit,
        '```json\n{"intent": "a", "intent": "b", "evidence": ["e"]}\n```',
        '```json\n{"intent": "\\ud800", "evidence": ["e"]}\n```',
    ):
        with pytest.raises(IntentAnswerError):
            parse_answer(bad)
