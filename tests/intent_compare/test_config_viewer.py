"""Offline tests of the shared config helpers and viewer escaping."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
import yaml
from intent_compare_config import (
    DEFAULT_CONFIG,
    OUTPUT_LANGUAGES,
    CompareConfigError,
    clip_bounds,
    load_config,
    parse_config,
)
from intent_compare_labels import EN, JA, LABELS
from intent_compare_viewer import build_html, cue_cell, load_cues, script

SEED = 100601
JA_CONFIG = DEFAULT_CONFIG.with_name("compare.ja.yaml")


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


def test_output_language_absent_defaults_to_en_and_ja_file_differs_only_in_language() -> None:
    english, japanese = load_config(DEFAULT_CONFIG), load_config(JA_CONFIG)

    assert english.output_language == "en" and japanese.output_language == "ja"
    same = dataclasses.replace(japanese, output_language="en", source_path=english.source_path)
    assert same == english


def test_ja_config_file_adds_only_output_language_to_the_shared_keys() -> None:
    shared = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    japanese = yaml.safe_load(JA_CONFIG.read_text(encoding="utf-8"))

    assert japanese == {**shared, "output_language": "ja"}


@pytest.mark.parametrize("language", ["fr", "", "EN", None, 1, ["ja"]])
def test_parse_config_invalid_output_language_raises(language: object) -> None:
    raw = {**yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")), "output_language": language}

    with pytest.raises(CompareConfigError, match="output_language"):
        parse_config(raw, Path("x.yaml"))


def test_label_tables_cover_exactly_the_supported_languages() -> None:
    assert tuple(LABELS) == OUTPUT_LANGUAGES
    assert {name: type(value) for name, value in dataclasses.asdict(EN).items()} == {
        name: str for name in dataclasses.asdict(JA)
    }


def test_cue_cell_english_labels_are_the_original_strings(tmp_path: Path) -> None:
    path = tmp_path / "cues_B.jsonl"
    path.write_text(json.dumps({"seed": SEED, "step": 315, "aircraft": "A0200", "skipped": "target_down"}) + "\n")

    assert cue_cell(None, 45, "A0100") == '<td class="missing">not run yet</td>'
    assert cue_cell(load_cues(path, SEED), 315, "A0200") == '<td class="missing">skipped: target_down</td>'
    assert cue_cell(load_cues(path, SEED), 45, "A0100") == '<td class="missing">no translation for this step</td>'


def test_cue_cell_japanese_labels_and_text_are_escaped(tmp_path: Path) -> None:
    path = tmp_path / "cues_A.jsonl"
    records = [
        {"seed": SEED, "step": 45, "aircraft": "A0100", "intent": "敵機<script>へ接近して旋回する", "model": "m",
         "evidence": ["ステップ45で A['x'] が \"上昇\" を示す & 速度増加"], "reasoning_effort": "high", "source_run": "r"},
        {"seed": SEED, "step": 315, "aircraft": "A0200", "skipped": "target_down"},
    ]  # fmt: skip
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    cues = load_cues(path, SEED)

    cell = cue_cell(cues, 45, "A0100", JA)

    assert "敵機&lt;script&gt;へ接近して旋回する" in cell and "<script>" not in cell
    assert "A[&#x27;x&#x27;]" in cell and "&quot;上昇&quot;" in cell and "&amp; 速度増加" in cell
    assert cue_cell(None, 45, "A0100", JA) == '<td class="missing">未実行</td>'
    assert cue_cell(cues, 315, "A0200", JA) == '<td class="missing">スキップ：target_down</td>'
    assert "このステップの翻訳なし" in cue_cell(cues, 145, "A0100", JA)


def _empty_run_config(language: str, tmp_path: Path):
    """A real config redirected to an empty output root: no clips, no cues, no human annotation."""
    return dataclasses.replace(load_config(DEFAULT_CONFIG), output_language=language, output_root=tmp_path)


def test_build_html_japanese_page_has_japanese_labels_and_no_english_ui(tmp_path: Path) -> None:
    page = build_html(_empty_run_config("ja", tmp_path), "seed100601-ja", {}, None)

    assert '<html lang="ja">' in page
    for expected in ("意図翻訳の比較", "A：トラジェクトリ", "B：重み＋中間状態＋観測", "未実行", "機体",
                     "クリップがありません", "翻訳時刻へ移動", "翻訳ウィンドウ", "ステップ 45（t = 9.0 秒）"):  # fmt: skip
        assert expected in page
    for english in ("not run yet", "Seek to translation", "Blue:", "aircraft</th>", "clip missing", "Step 45"):
        assert english not in page
    assert "default-src 'none'; media-src data:" in page
    assert "シミュレーション時刻 t = " not in page.split("<script>")[0]  # clock text lives in the script only


def test_build_html_english_page_keeps_original_labels(tmp_path: Path) -> None:
    page = build_html(_empty_run_config("en", tmp_path), "seed100601", {}, None)

    assert '<html lang="en">' in page
    for expected in ("Intent translation comparison: method A vs method B", "<th>A</th><th>B</th>", "not run yet",
                     "Step 45 (t = 9.0 s); translation window steps 0-45 (0.0-9.0 s); clip 0.0-9.0 s",
                     "Seek to translation instant (t = 9.0 s)", "clip missing (run scripts/intent_compare_video.sbatch)",
                     "clock.textContent = 'sim t = ' + sim.toFixed(1) + ' s (step ' + Math.round(sim / dt) + ')';"):  # fmt: skip
        assert expected in page
    assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", page) is None


def test_script_japanese_clock_text_is_a_safe_ascii_js_literal() -> None:
    clock = [line for line in script(JA).splitlines() if "clock.textContent" in line][0]

    assert clock.isascii() and "__CLOCK" not in clock
    assert "\\u30b7" in clock  # non-ASCII label is \u-escaped
    assert "__CLOCK" not in script(EN)
