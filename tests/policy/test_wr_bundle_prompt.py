"""Bundle determinism and safety plus the config-driven code-session turns."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from translate import paths
from translate.config import load_config
from translate.policy.bundle import BundleError, verify_bundle, write_bundle
from translate.policy.render import spec_sections
from translate.policy.spec import parse_spec
from translate.policy.tensors import Tensor
from translate.prompts import ConfiguredTurns
from translate.sandbox.protocol import KIND_CODE, KIND_ERROR, KIND_FINAL, parse_reply

from .wr_helpers import small_arrays

RUN_CONFIG = paths.CONFIGS / "translate" / "seed100601-policy.yaml"


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


@pytest.mark.parametrize("protected", [paths.DATA])
def test_write_bundle_under_protected_directory_refuses(protected: Path) -> None:
    with pytest.raises(ValueError, match="read-only source"):
        write_bundle(_tensors(), "net-0123abcd", "abc", protected / "wr-test")
    assert not (protected / "wr-test").exists()


def test_first_message_never_contains_provenance_or_notes(small_spec_dict: dict) -> None:
    message = "\n".join(spec_sections(parse_spec(small_spec_dict), _tensors()).values())

    assert "mtd-rl-lag" not in message
    assert "baseline" not in message.lower()
    assert "secret note" not in message


def test_run_result_message_reports_timeout_and_remaining_runs() -> None:
    turns = ConfiguredTurns(load_config(RUN_CONFIG).prompt.turns)
    text = turns.run_result(3, 5, "out", "err tail", None, True, True)

    assert "run 3 of at most 5 (TIMED OUT)" in text
    assert "(stdout was truncated)" in text
    assert "Runs remaining: 2." in text


WELL_FORMED_FINAL = '```json\n{"intent": "Pursue.", "evidence": ["A changed"]}\n```'


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("Let me look.\n```python\nprint(W.keys())\n```", KIND_CODE),
        ("```py\nprint(1)\n```", KIND_CODE),
        (WELL_FORMED_FINAL, KIND_FINAL),
        ('## Summary\nDone.\n```json\n{"intent": "x", "evidence": []}\n```', KIND_FINAL),
        ("```python\na=1\n```\n```python\nb=2\n```", KIND_CODE),
        ("", KIND_ERROR),
    ],
)
def test_parse_reply_classifies_reply(reply: str, kind: str) -> None:
    assert parse_reply(reply).kind == kind


def test_parse_reply_returns_code_body() -> None:
    assert parse_reply("```python\nx = 1\nprint(x)\n```").code == "x = 1\nprint(x)\n"


def test_load_config_repository_file_is_valid() -> None:
    config = load_config(RUN_CONFIG)

    assert config.translation.model
    assert config.price(config.translation.model) is not None
    assert config.session.sandbox.python == paths.ROOT / ".venv/bin/python"
