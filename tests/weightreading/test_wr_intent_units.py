"""Method B units: activation trace, window slicing, prompt hygiene, answer schema, cues, config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mtd_rl_xavier import paths
from mtd_rl_xavier.intent_compare import CompareConfigError, check_run_id, parse_compare
from weightreading.architecture import expected_params
from weightreading.intent_cues import CueContext, CueWriter, skipped_cue, translated_cue
from weightreading.intent_prompt import IntentRequest, PreviousTranslation, intent_message
from weightreading.intent_protocol import (
    MAX_INTENT_CHARS,
    AnswerLimits,
    IntentAnswer,
    IntentAnswerError,
    parse_answer,
    parse_intent_reply,
)
from weightreading.intent_trace import (
    TraceError,
    activation_docs,
    concatenated_logits,
    trace_episode,
    validate_trace,
    window_arrays,
    window_bounds,
)
from weightreading.prompt import ExecutorLimits
from weightreading.protocol import KIND_CODE, KIND_ERROR, KIND_FINAL
from weightreading.reference import build_policy
from weightreading.spec import Spec, load_spec, parse_spec
from weightreading.tensors import Tensor
from weightreading.torch_reader import read_checkpoint

MAPPO_SPEC = paths.CONFIGS / "weightreading" / "lag_mappo_2v2_33.spec.json"
LIMITS = ExecutorLimits(max_runs=5, timeout_s=10, stdout_cap_chars=100, stderr_tail_chars=50)
ROLLOUT = paths.OUTPUTS / "weightreading" / "rollouts" / "mhi-v1.3-20261104T010525Z" / "episode_seed100601.npz"
REAL_LOGITS_TOLERANCE = 1e-5


def _small_policy(small_files: tuple[Path, Path]):
    checkpoint, spec = small_files
    return build_policy(load_spec(spec), read_checkpoint(checkpoint).tensors)


def _answer_text(payload: object) -> str:
    return "Done.\n```json\n" + json.dumps(payload) + "\n```"


# --- trace -------------------------------------------------------------------------------------


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


# --- prompt ------------------------------------------------------------------------------------


def _zero_tensors(spec) -> tuple[Tensor, ...]:
    shapes = {key: shape for block in spec.blocks for key, shape in expected_params(block).items()}
    return tuple(Tensor(key, "float32", shape, bytes(4 * int(np.prod(shape)))) for key, shape in shapes.items())


def _mappo_message(
    previous: PreviousTranslation | None,
    spec: Spec | None = None,
    language: str | None = None,
    limits: AnswerLimits | None = None,
) -> str:
    spec = spec or load_spec(MAPPO_SPEC)
    tensors = _zero_tensors(spec)
    policy = build_policy(spec, tensors)
    extra = {} if language is None else {"output_language": language}
    extra = extra if limits is None else {**extra, "limits": limits}
    request = IntentRequest("A0100", 145, 29.0, (86, 145), 0.2, "No operator instruction.", previous, **extra)
    return intent_message(spec, tensors, activation_docs(policy), request, LIMITS)


def test_intent_message_hides_spec_evaluation_notes_and_provenance() -> None:
    raw = json.loads(MAPPO_SPEC.read_text(encoding="utf-8"))
    message = _mappo_message(None)
    stripped = {**raw, "evaluation": None, "notes": [], "provenance": {}}
    stripped = {key: value for key, value in stripped.items() if value is not None}

    # The message is a function of the whitelisted fields only: removing the hidden ones changes nothing.
    assert _mappo_message(None, parse_spec(stripped)) == message
    for note in [*raw["notes"], *raw["evaluation"]["notes"]]:
        assert note not in message
    assert "evaluation" not in message.lower()
    assert raw["description"] in message
    assert "Window: steps 86 through 145 inclusive (60 decisions" in message
    assert "`A['gru.h_new']` shape [60, 128]" in message
    assert "first translation for this aircraft" in message


def test_intent_message_has_no_world_trajectory_and_chains_previous() -> None:
    previous = PreviousTranslation(45, IntentAnswer("Climb and close on B0100.", ("logit gap at step 40",)))
    message = _mappo_message(previous)

    assert "From step 45:" in message and "Climb and close on B0100." in message
    for world in ("lon_deg", "lat_deg", "geodetic", "position_neu", "velocity_neu", "missiles_left"):
        assert world not in message


def test_intent_message_asks_for_japanese_only_when_requested() -> None:
    english = _mappo_message(None)
    explicit_en = _mappo_message(None, language="en")
    japanese = _mappo_message(None, language="ja")

    assert english == explicit_en and "Write both fields in English." in english and "Japanese" not in english
    assert "Write both fields in natural Japanese" in japanese and "English" not in japanese.split("# Final answer")[1]
    assert japanese.split("# Final answer")[0] == english.split("# Final answer")[0]  # rest of the prompt unchanged
    assert f"at most {MAX_INTENT_CHARS} characters" in japanese  # characters, not bytes


def test_intent_message_with_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="output_language"):
        _mappo_message(None, language="fr")


# --- protocol ----------------------------------------------------------------------------------


def test_parse_intent_reply_kinds() -> None:
    valid = _answer_text({"intent": "Pursue.", "evidence": ["a", "b"]})

    assert parse_intent_reply("```python\nprint(1)\n```").kind == KIND_CODE
    assert parse_intent_reply(valid).kind == KIND_FINAL
    assert parse_answer(valid) == IntentAnswer("Pursue.", ("a", "b"))
    assert parse_intent_reply("I am done.").kind == KIND_ERROR


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

    assert parse_intent_reply(text).kind == KIND_FINAL  # a json attempt ends the session, as in method A
    with pytest.raises(IntentAnswerError):
        parse_answer(text)


def test_parse_answer_counts_japanese_characters_not_bytes() -> None:
    at_limit = _answer_text({"intent": "あ" * MAX_INTENT_CHARS, "evidence": ["根拠"]})
    assert parse_answer(at_limit).intent == "あ" * MAX_INTENT_CHARS  # 3000 UTF-8 bytes, 1000 characters
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


# --- cues and config ---------------------------------------------------------------------------


def test_cue_writer_writes_contract_lines_and_refuses_existing(tmp_path: Path) -> None:
    context = CueContext(100601, "gpt-6.1-sol", "high", 0.2)
    path = tmp_path / "run" / "cues_B.jsonl"
    writer = CueWriter(path)

    writer.append(translated_cue(context, 145, "A0100", "runs/x", IntentAnswer("Pursue.", ("e",))))
    writer.append(skipped_cue(context, 315, "A0200", "runs/x", "A0200 is not alive at step 315"))

    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert set(first) == {"seed", "step", "t_s", "aircraft", "method", "model", "reasoning_effort", "intent",
                          "evidence", "source_run"}  # fmt: skip
    assert first["t_s"] == 29.0 and first["method"] == "B" and first["evidence"] == ["e"]
    assert second["skipped"].startswith("A0200") and "intent" not in second and "evidence" not in second
    with pytest.raises(FileExistsError):
        CueWriter(path)


def _compare_raw() -> dict:
    return {
        "seed": 1,
        "rollout_dir": "r",
        "steps": [45, 145],
        "targets": ["A0100"],
        "window_steps": 60,
        "step_seconds": 0.2,
        "model": "m",
        "reasoning_effort": "high",
        "instruction": "i",
        "output_root": "o",
        "clip_half_width_s": 30,
    }


def test_parse_compare_keeps_increasing_steps_and_ignores_unused_keys(tmp_path: Path) -> None:
    compare = parse_compare({**_compare_raw(), "unused": 1}, tmp_path / "c.yaml", "sha", tmp_path)

    assert compare.steps == (45, 145) and compare.rollout_dir == tmp_path / "r"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("steps", [45, 45]),
        ("steps", [145, 45]),  # B used to sort these; the shared parser rejects them like method A
        ("window_steps", 0),
        ("reasoning_effort", "turbo"),
        ("instruction", " "),
    ],
)
def test_parse_compare_rejects_bad_values(tmp_path: Path, key: str, value: object) -> None:
    with pytest.raises(CompareConfigError):
        parse_compare({**_compare_raw(), key: value}, tmp_path / "c.yaml", "sha", tmp_path)


def test_parse_compare_output_language_defaults_to_en_and_accepts_ja(tmp_path: Path) -> None:
    parse = lambda raw: parse_compare(raw, tmp_path / "c.yaml", "sha", tmp_path)  # noqa: E731

    assert parse(_compare_raw()).output_language == "en"
    assert parse({**_compare_raw(), "output_language": "en"}).output_language == "en"
    assert parse({**_compare_raw(), "output_language": "ja"}).output_language == "ja"


@pytest.mark.parametrize("language", ["fr", "", "JA", None, 1])
def test_parse_compare_rejects_unknown_output_language(tmp_path: Path, language: object) -> None:
    with pytest.raises(CompareConfigError, match="output_language"):
        parse_compare({**_compare_raw(), "output_language": language}, tmp_path / "c.yaml", "sha", tmp_path)


@pytest.mark.parametrize("run_id", ["", "../x", "a/b", ".hidden", "a..b", "run1\n", "run1\r\n"])
def test_check_run_id_rejects_unsafe_names(run_id: str) -> None:
    with pytest.raises(CompareConfigError):
        check_run_id(run_id)
