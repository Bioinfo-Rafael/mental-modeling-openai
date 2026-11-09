"""Focused schema-v2 configuration consistency tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from translate import paths
from translate.config import ConfigError, parse_config
from translate.llm.retry import RetryPolicy

from .tr_fakes import RUN_CONFIG, run_config_raw, with_value, write_yaml

TRANSLATE_CONFIGS = paths.CONFIGS / "translate"
POLICY_CONFIG = TRANSLATE_CONFIGS / "seed100601-policy.yaml"


def _parse(raw: object, path: Path = RUN_CONFIG):
    return parse_config(raw, path=path, sha256="test")


def _prompt_raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_config_prompt_with_unknown_placeholder_raises_config_error(tmp_path: Path) -> None:
    raw = run_config_raw()
    prompt_path = paths.ROOT / raw["prompt"]["file"]
    prompt = _prompt_raw(prompt_path)
    custom = write_yaml(tmp_path / "prompt.yaml", {**prompt, "system": prompt["system"] + "\n{unknown_value}"})
    raw = with_value(raw, "prompt.file", str(custom))

    with pytest.raises(ConfigError, match=r"unknown_value.*no selected input"):
        _parse(raw)


def test_config_selected_but_unused_input_raises_config_error(tmp_path: Path) -> None:
    raw = run_config_raw()
    prompt_path = paths.ROOT / raw["prompt"]["file"]
    prompt = _prompt_raw(prompt_path)
    custom = write_yaml(tmp_path / "prompt.yaml", {**prompt, "user": prompt["user"].replace("{question}", "")})
    raw = with_value(raw, "prompt.file", str(custom))

    with pytest.raises(ConfigError, match=r"question.*selected but prompt"):
        _parse(raw)


@pytest.mark.parametrize(
    ("triggers", "detector", "match"),
    [
        ({"source": "online", "tick_interval_steps": 5, "points": None}, None, "detector is required"),
        (
            {
                "source": "preset",
                "tick_interval_steps": None,
                "points": [{"step": 0, "kind": "start", "targets": None}],
            },
            "keep",
            "must be null for preset",
        ),
    ],
)
def test_config_trigger_source_and_detector_mismatch_raises_config_error(
    triggers: dict[str, object], detector: object, match: str
) -> None:
    raw = run_config_raw()
    raw = with_value(raw, "triggers", triggers)
    if detector is None:
        raw = with_value(raw, "detector", None)

    with pytest.raises(ConfigError, match=match):
        _parse(raw)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_policy", "policy is required"),
        ("missing_session", "code-session prompt needs"),
        ("one_shot_with_policy_session", "code-session prompt needs"),
        ("one_shot_with_session", "code-session prompt needs"),
    ],
)
def test_config_policy_session_and_prompt_protocol_mismatch_raises_config_error(mutation: str, match: str) -> None:
    policy_raw = yaml.safe_load(POLICY_CONFIG.read_text(encoding="utf-8"))
    if mutation == "missing_policy":
        raw = {**policy_raw, "policy": None}
    elif mutation == "missing_session":
        raw = {**policy_raw, "session": None}
    elif mutation == "one_shot_with_policy_session":
        raw = {
            **policy_raw,
            "prompt": run_config_raw()["prompt"],
        }
    else:
        online = run_config_raw()
        raw = {**online, "session": policy_raw["session"]}

    with pytest.raises(ConfigError, match=match):
        _parse(raw, POLICY_CONFIG)


def test_config_retry_none_is_a_zero_retry_policy() -> None:
    config = _parse(with_value(run_config_raw(), "translation.retry", "none"))

    assert config.translation.retry == RetryPolicy(max_retries=0)


def test_config_retry_default_is_the_shared_policy() -> None:
    config = _parse(with_value(run_config_raw(), "translation.retry", "default"))

    assert config.translation.retry == RetryPolicy()
