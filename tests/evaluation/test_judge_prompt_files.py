"""Judge prompt files: strict loading, placeholder checks, and the shipped default and legacy configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evaluation import configuration
from evaluation.assess_data import ConfigError, load_config, load_run, parse_config
from evaluation.assess_judge import build_packet, system_prompt
from evaluation.prompt_files import ASSESS_PROMPT_SPEC, KEY_POINT_PROMPT_SPEC, JudgePromptError, load_judge_prompt
from translate import paths

from .assess_fakes import E_CHANGE, V3_PLAN, config_raw, make_frame, make_run

CONFIGS = paths.ROOT / "configs" / "evaluation"
PROMPTS = CONFIGS / "prompts"
LEGACY_HEADING = "## TRAJECTORY THE TRANSLATOR SAW"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _legacy_assess_text() -> str:
    return (PROMPTS / "assess-judge-v2.yaml").read_text(encoding="utf-8")


# --- shipped files ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "version"),
    [("assess.yaml", "assess-judge-v3"), ("assess-legacy.yaml", "assess-judge-v2")],
)
def test_each_assess_config_loads_its_prompt(config: str, version: str) -> None:
    assert load_config(CONFIGS / config).prompt.version == version


@pytest.mark.parametrize(
    ("config", "version"),
    [
        ("judge-text.yaml", "judge-text-v4"),
        ("judge-text-video.yaml", "judge-text-v4"),
        ("judge-text-legacy.yaml", "judge-text-v3"),
    ],
)
def test_each_key_point_config_loads_its_prompt(config: str, version: str) -> None:
    assert configuration.load_config(CONFIGS / config).prompt.version == version


def test_the_default_assess_prompt_calls_the_trajectory_human_visible(tmp_path: Path) -> None:
    """The change at 15 of the same episode: schema 2 under the legacy config, schema 3 under the default."""
    default, legacy = load_config(CONFIGS / "assess.yaml"), load_config(CONFIGS / "assess-legacy.yaml")
    runs = {
        "default": load_run(make_run(tmp_path, "v3", plan=V3_PLAN, schema_version=3)),
        "legacy": load_run(make_run(tmp_path, "v2")),
    }
    texts = {}
    for name, config in (("default", default), ("legacy", legacy)):
        run = runs[name]
        event = next(item for item in run.events if item["event_id"] == E_CHANGE)
        texts[name] = build_packet(event, run=run, config=config, annotation=None, frame=make_frame()).text

    assert LEGACY_HEADING in texts["legacy"] and LEGACY_HEADING not in texts["default"]
    assert "## TRAJECTORY (human-visible telemetry up to t)" in texts["default"]
    assert "The Translator read the recorded" in system_prompt(legacy)
    assert "The Translator read the recorded" not in system_prompt(default)
    assert "human-visible telemetry" in system_prompt(default)
    material = (event["input"]["trajectory"]["text"], event["output"]["intent"], "## FUTURE (evaluation only")
    assert all(part in texts["default"] and part in texts["legacy"] for part in material)  # same material


# --- strict loading ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("judge: assess", "judge: key_point", "not 'assess'"),
        ("schema_version: 1", "schema_version: 2", "schema_version must be 1"),
        ("version: assess-judge-v2", "version: ''", "version must be a non-blank string"),
        ("    {alarms}\n", "    {alarmz}\n", "unknown fields ['alarmz']"),
        ("{timeline}{future}", "{timeline}", "must use ['future']"),
        ('    item: "- step {step}: {text}"', '    item: "- step {stepp}: {text}"', "blocks.alarms.item uses unknown"),
        ('    item: "- step {step}: {text}"', '    item: "- step {}: {text}"', "name every value"),
        ('    item: "- step {step}: {text}"', '    item: "- step {step: {text}"', "malformed template"),
        ("    {trajectory}\n", "    {trajectory:.1}\n", "must be a bare name"),  # would cut the material
        ("Intent: {intent}", "Intent: {intent[0]}", "must be a bare name"),
        ("Intent: {intent}", "Intent: {intent!r}", "must be a bare name"),
        ('    item: "- step {step}: {text}"', '    item: "- step {step.real}: {text}"', "must be a bare name"),
        ('    no_later: "- none"\n', "", "blocks.future must have exactly"),
        ('    empty: "- none"\n', '    empty: "- none"\n    extra: ""\n', "must have exactly"),
        ("blocks:\n", "blocks:\n  extra_block: {item: x}\n", "blocks must have exactly"),
        ("texts:\n", "texts:\n  user: duplicate\n", "not valid YAML"),
    ],
)
def test_a_malformed_assess_prompt_fails_at_load(tmp_path: Path, old: str, new: str, message: str) -> None:
    text = _legacy_assess_text()
    assert old in text

    with pytest.raises(JudgePromptError) as exc:
        load_judge_prompt(_write(tmp_path / "prompt.yaml", text.replace(old, new, 1)), ASSESS_PROMPT_SPEC)

    assert message in str(exc.value)


def test_a_prompt_that_is_not_utf8_fails_at_load(tmp_path: Path) -> None:
    path = tmp_path / "prompt.yaml"
    path.write_bytes(_legacy_assess_text().encode("utf-8") + b"\xff")

    with pytest.raises(JudgePromptError, match="not valid UTF-8"):
        load_judge_prompt(path, ASSESS_PROMPT_SPEC)


def test_a_key_point_prompt_that_drops_material_fails_at_load(tmp_path: Path) -> None:
    text = (PROMPTS / "judge-text-v3.yaml").read_text(encoding="utf-8")
    old = "    Evidence the Translator cited:{evidence}"
    assert old in text

    with pytest.raises(JudgePromptError, match=r"texts.translated_intent must use \['evidence'\]"):
        load_judge_prompt(_write(tmp_path / "p.yaml", text.replace(old, "    Evidence:")), KEY_POINT_PROMPT_SPEC)


def test_a_config_names_a_missing_prompt_file_as_a_config_error(tmp_path: Path) -> None:
    raw = {**config_raw(), "prompt": "configs/evaluation/prompts/no-such-prompt.yaml"}

    with pytest.raises(ConfigError, match="cannot read prompt file"):
        parse_config(raw, path=tmp_path / "assess.yaml", sha256="x")


def test_a_key_point_config_rejects_an_assess_prompt() -> None:
    raw = yaml.safe_load((CONFIGS / "judge-text.yaml").read_text(encoding="utf-8"))
    raw = {**raw, "prompt": "configs/evaluation/prompts/assess-judge-v2.yaml"}

    with pytest.raises(configuration.JudgeConfigError, match="is a prompt for judge 'assess'"):
        configuration.parse_config(raw)
