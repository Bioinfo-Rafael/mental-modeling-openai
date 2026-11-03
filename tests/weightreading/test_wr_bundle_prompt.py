"""Bundle determinism and safety, prompt rendering (shown/blind), reply protocol, config, estimate."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from mtd_rl_xavier import paths
from weightreading.bundle import BundleError, verify_bundle, write_bundle
from weightreading.config import ConfigError, load_config, parse_config
from weightreading.estimate import preview_estimate
from weightreading.prompt import ExecutorLimits, first_message, run_result_message
from weightreading.protocol import KIND_CODE, KIND_ERROR, KIND_FINAL, parse_reply
from weightreading.spec import parse_spec
from weightreading.tensors import Tensor

from .wr_helpers import small_arrays

RUN_CONFIG = paths.CONFIGS / "weightreading" / "run.yaml"
LIMITS = ExecutorLimits(max_runs=7, timeout_s=60, stdout_cap_chars=1000, stderr_tail_chars=200)


def _tensors() -> tuple[Tensor, ...]:
    return tuple(Tensor(k, "float32", v.shape, v.astype("<f4").tobytes()) for k, v in small_arrays().items())


def test_write_bundle_twice_gives_identical_manifest(tmp_path: Path) -> None:
    first = write_bundle(_tensors(), "net-0123abcd", "abc", tmp_path)

    second = write_bundle(_tensors(), "net-0123abcd", "abc", tmp_path)

    assert first.manifest_sha256 == second.manifest_sha256
    assert sorted(p.name for p in tmp_path.iterdir()) == ["net-0123abcd-abc"]


def test_write_bundle_npy_files_round_trip(tmp_path: Path) -> None:
    bundle = write_bundle(_tensors(), "net-0123abcd", "abc", tmp_path)

    keys = json.loads((bundle.weights / "index.json").read_text())
    loaded = np.load(io.BytesIO((bundle.weights / f"{keys[0]}.npy").read_bytes()))

    np.testing.assert_array_equal(loaded, small_arrays()[keys[0]])
    assert keys == list(small_arrays())


def test_write_bundle_over_different_existing_bundle_refuses(tmp_path: Path) -> None:
    write_bundle(_tensors(), "net-0123abcd", "abc", tmp_path)

    with pytest.raises(BundleError, match="differs"):
        write_bundle(_tensors()[:-1], "net-0123abcd", "abc", tmp_path)


def test_verify_bundle_after_tampering_raises(tmp_path: Path) -> None:
    bundle = write_bundle(_tensors(), "net-0123abcd", "abc", tmp_path)
    target = next(bundle.weights.glob("*.npy"))
    target.write_bytes(target.read_bytes()[:-4] + b"\0\0\0\0")

    with pytest.raises(BundleError, match="changed"):
        verify_bundle(bundle.path)


@pytest.mark.parametrize("protected", [paths.DATA, paths.MENTALMODELING])
def test_write_bundle_under_protected_directory_refuses(protected: Path) -> None:
    with pytest.raises(ValueError, match="read-only source"):
        write_bundle(_tensors(), "net-0123abcd", "abc", protected / "wr-test")
    assert not (protected / "wr-test").exists()


def test_first_message_shown_contains_semantics_equations_and_rules(small_spec_dict: dict) -> None:
    message = first_message(parse_spec(small_spec_dict), _tensors(), "shown", LIMITS)

    for expected in ["net-0123abcd", "aircraft flight controller", "roll_sin", "aileron", "throttle",
                     "h' = (1 - z) * n + z * h", "LayerNorm comes AFTER the ReLU", "must NOT run the network",
                     "at most 7 code blocks", "clipped to [-10, 10]", '"claims"']:  # fmt: skip
        assert expected in message


def test_first_message_never_contains_provenance_or_notes(small_spec_dict: dict) -> None:
    for mode in ("shown", "blind"):
        message = first_message(parse_spec(small_spec_dict), _tensors(), mode, LIMITS)

        assert "mtd-rl-lag" not in message
        assert "baseline" not in message.lower()
        assert "secret note" not in message


def test_first_message_blind_withholds_task_and_names(small_spec_dict: dict) -> None:
    message = first_message(parse_spec(small_spec_dict), _tensors(), "blind", LIMITS)

    for withheld in ["aircraft", "flight", "roll", "pitch", "airspeed", "aileron", "throttle"]:
        assert withheld not in message.lower()
    assert "x0, x1, x2" in message
    assert "head_0, head_1" in message
    assert "Task hypothesis" in message


def test_run_result_message_reports_timeout_and_remaining_runs() -> None:
    text = run_result_message(3, 5, "out", "err tail", None, True, True)

    assert "run 3 of at most 5 (TIMED OUT)" in text
    assert "(stdout was truncated)" in text
    assert "Runs remaining: 2." in text


WELL_FORMED_FINAL = (
    '## Observations\nx\n## Inputs\nx\n## Heads\nx\n## Recurrence\nx\n## Summary\nx\n```json\n{"claims": []}\n```'
)


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("Let me look.\n```python\nprint(W.keys())\n```", KIND_CODE),
        ("```py\nprint(1)\n```", KIND_CODE),
        (WELL_FORMED_FINAL, KIND_FINAL),
        ('## Summary\nDone.\n```json\n{"claims": []}\n```', KIND_ERROR),
        ("```python\na=1\n```\n```python\nb=2\n```", KIND_CODE),
        ("", KIND_ERROR),
    ],
)
def test_parse_reply_classifies_reply(reply: str, kind: str) -> None:
    assert parse_reply(reply).kind == kind


def test_parse_reply_returns_code_body() -> None:
    assert parse_reply("```python\nx = 1\nprint(x)\n```").code == "x = 1\nprint(x)\n"


def test_load_config_repository_file_is_valid() -> None:
    config = load_config(RUN_CONFIG, paths.ROOT)

    assert config.model
    assert config.price() is not None
    assert config.sandbox.python == paths.ROOT / ".venv/bin/python"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(model=None), "model must be set"),
        (lambda c: c.update(model="  "), "model must be set"),
        (lambda c: c.update(extra=1), "unknown config keys"),
        (lambda c: c.update(task_mode="hidden"), "task_mode"),
        (lambda c: c["limits"].update(max_runs=0), "positive integer"),
        (lambda c: c["sandbox"].pop("timeout_s"), "exactly"),
        (lambda c: c["pricing"].update(m={"input": 1}), "pricing.m"),
    ],
)
def test_parse_config_with_invalid_value_raises(mutate, message: str) -> None:
    import yaml

    raw = yaml.safe_load(RUN_CONFIG.read_text())
    mutate(raw)

    with pytest.raises(ConfigError, match=message):
        parse_config(raw, paths.ROOT)


def test_preview_estimate_worst_case_is_bounded_and_above_typical() -> None:
    config = load_config(RUN_CONFIG, paths.ROOT)

    estimate = preview_estimate(8000, config)

    worst = estimate["worst_case"]
    assert worst["input_tokens"] + worst["output_tokens"] <= 3 * config.limits.max_total_tokens
    assert worst["cost_usd"] > estimate["typical"]["cost_usd"] > 0
    assert estimate["first_message_tokens_approx"] == 2000
