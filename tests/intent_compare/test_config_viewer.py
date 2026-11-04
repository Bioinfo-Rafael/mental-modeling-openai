"""Offline tests of the shared config helpers and viewer escaping."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from intent_compare_config import DEFAULT_CONFIG, CompareConfigError, clip_bounds, load_config, parse_config
from intent_compare_viewer import cue_cell, load_cues

SEED = 100601


def test_load_config_shared_file_has_fixed_decisions() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert (config.seed, config.steps, config.targets) == (SEED, (45, 145, 245, 315), ("A0100", "A0200"))
    assert (config.window_steps, config.step_seconds, config.clip_half_width_s) == (60, 0.2, 30.0)
    assert config.window_start(45) == 0 and config.window_start(145) == 86
    assert config.t_s(315) == 63.0


def test_parse_config_missing_key_raises() -> None:
    with pytest.raises(CompareConfigError, match="missing"):
        parse_config({"seed": 1}, Path("x.yaml"))


@pytest.mark.parametrize(
    ("center", "end", "expected"),
    [(9.0, 71.2, (0.0, 39.0)), (49.0, 71.2, (19.0, 71.2)), (35.0, 71.2, (5.0, 65.0))],
)
def test_clip_bounds_clamps_to_episode(center: float, end: float, expected: tuple[float, float]) -> None:
    assert clip_bounds(center, 30.0, end) == expected


def test_clip_bounds_center_after_end_raises() -> None:
    with pytest.raises(CompareConfigError):
        clip_bounds(100.0, 30.0, 50.0)


def test_cue_cell_escapes_model_text(tmp_path: Path) -> None:
    path = tmp_path / "cues_A.jsonl"
    record = {"seed": SEED, "step": 45, "aircraft": "A0100", "intent": "<script>alert(1)</script>",
              "evidence": ['"><img src=x onerror=alert(1)>'], "model": "m", "reasoning_effort": "high",
              "source_run": "r"}  # fmt: skip
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    cell = cue_cell(load_cues(path, SEED), 45, "A0100")

    assert "<script>" not in cell and "<img" not in cell
    assert "&lt;script&gt;" in cell


def test_cue_cell_missing_file_and_skip_are_labelled(tmp_path: Path) -> None:
    path = tmp_path / "cues_B.jsonl"
    path.write_text(json.dumps({"seed": SEED, "step": 315, "aircraft": "A0200", "skipped": "target_down"}) + "\n")

    assert "not run yet" in cue_cell(load_cues(tmp_path / "absent.jsonl", SEED), 45, "A0100")
    assert "skipped: target_down" in cue_cell(load_cues(path, SEED), 315, "A0200")
    assert "no translation" in cue_cell(load_cues(path, SEED), 45, "A0100")


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", ".hidden"])
def test_run_dir_with_unsafe_run_id_raises(run_id: str) -> None:
    with pytest.raises(CompareConfigError):
        load_config(DEFAULT_CONFIG).run_dir(run_id)
