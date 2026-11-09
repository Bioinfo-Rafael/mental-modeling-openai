"""Offline coverage for the N-column intent translation viewer."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from evaluation.clips import clip_bounds
from evaluation.labels import EN, JA, LABELS
from evaluation.viewer import build_html, cue_cell, load_cues, script


def _write_cues(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _cue(step: int, aircraft: str, intent: str) -> dict[str, object]:
    return {
        "step": step,
        "aircraft": aircraft,
        "intent": intent,
        "evidence": ["observed <turn> & climb"],
        "model": "fake-model",
        "reasoning_effort": "high",
        "event_id": f"event-{step}-{aircraft}",
    }


def test_cue_cell_with_model_text_escapes_html(tmp_path: Path) -> None:
    cues_path = _write_cues(tmp_path / "cues.jsonl", [_cue(45, "A0100", "<script>alert(1)</script>")])

    cell = cue_cell(load_cues(cues_path), 45, "A0100")

    assert "<script>" not in cell
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in cell
    assert "observed &lt;turn&gt; &amp; climb" in cell


def test_cue_cell_with_missing_file_reports_not_run_yet(tmp_path: Path) -> None:
    cues = load_cues(tmp_path / "missing.jsonl")

    assert cue_cell(cues, 45, "A0100") == '<td class="missing">not run yet</td>'


def test_cue_cell_with_skipped_cue_reports_reason(tmp_path: Path) -> None:
    path = _write_cues(
        tmp_path / "cues.jsonl",
        [{"step": 315, "aircraft": "A0200", "skipped": "target_down"}],
    )

    cues = load_cues(path)

    assert cue_cell(cues, 315, "A0200") == '<td class="missing">skipped: target_down</td>'
    assert cue_cell(cues, 45, "A0100") == '<td class="missing">no translation for this step</td>'


@pytest.mark.parametrize(
    ("center", "end", "expected"),
    [(9.0, 71.2, (0.0, 39.0)), (49.0, 71.2, (19.0, 71.2)), (35.0, 71.2, (5.0, 65.0))],
)
def test_clip_bounds_with_episode_edges_clamps_to_episode(
    center: float,
    end: float,
    expected: tuple[float, float],
) -> None:
    assert clip_bounds(center, 30.0, end) == expected


def test_clip_bounds_with_center_after_episode_raises() -> None:
    with pytest.raises(ValueError, match="empty clip"):
        clip_bounds(100.0, 30.0, 50.0)


def test_build_html_with_legacy_and_new_cue_files_renders_all_columns(tmp_path: Path) -> None:
    legacy_a = _write_cues(tmp_path / "cues_A.jsonl", [_cue(45, "A0100", "legacy A")])
    legacy_b = _write_cues(tmp_path / "cues_B.jsonl", [_cue(45, "A0100", "legacy B")])
    new_run = tmp_path / "new-run"
    _write_cues(new_run / "cues.jsonl", [_cue(45, "A0100", "new pipeline")])

    page = build_html(
        {"trajectory": legacy_a, "policy-legacy": legacy_b, "policy-new": new_run},
        clips_dir=None,
        language="en",
        title="comparison",
    )

    assert "legacy A" in page
    assert "legacy B" in page
    assert "new pipeline" in page
    assert "cues_A.jsonl" in page
    assert "cues_B.jsonl" in page
    assert "new-run/cues.jsonl" in page
    assert "<th>trajectory</th><th>policy-legacy</th><th>policy-new</th>" in page


def test_build_html_sets_restrictive_content_security_policy(tmp_path: Path) -> None:
    cues_path = _write_cues(tmp_path / "cues.jsonl", [_cue(45, "A0100", "intent")])

    page = build_html({"run": cues_path}, clips_dir=None, language="en", title="safe")

    assert (
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; media-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'\">"
        in page
    )


def test_label_tables_cover_exactly_supported_languages() -> None:
    assert tuple(LABELS) == ("en", "ja")
    assert {name: type(value) for name, value in dataclasses.asdict(EN).items()} == {
        name: str for name in dataclasses.asdict(JA)
    }


def test_cue_cell_with_japanese_labels_escapes_text(tmp_path: Path) -> None:
    path = _write_cues(
        tmp_path / "cues.jsonl",
        [
            _cue(45, "A0100", "敵機<script>へ接近"),
            {"step": 315, "aircraft": "A0200", "skipped": "target_down"},
        ],
    )

    cues = load_cues(path)
    cell = cue_cell(cues, 45, "A0100", JA)

    assert "敵機&lt;script&gt;へ接近" in cell
    assert "<script>" not in cell
    assert cue_cell(None, 45, "A0100", JA) == '<td class="missing">未実行</td>'
    assert cue_cell(cues, 315, "A0200", JA) == '<td class="missing">スキップ：target_down</td>'
    assert cue_cell(cues, 145, "A0100", JA) == '<td class="missing">このステップの翻訳なし</td>'


def test_build_html_with_japanese_language_has_no_english_ui(tmp_path: Path) -> None:
    cues_path = _write_cues(tmp_path / "cues.jsonl", [_cue(45, "A0100", "旋回する")])

    page = build_html(
        {"軌道": cues_path, "未実行列": tmp_path / "missing.jsonl"},
        clips_dir=None,
        language="ja",
        title="比較",
    )

    assert '<html lang="ja">' in page
    for expected in ("意図翻訳の比較", "未実行", "機体", "クリップがありません", "翻訳時刻へ移動"):
        assert expected in page
    for english in ("not run yet", "Seek to translation", "aircraft</th>", "clip missing", "Step 45"):
        assert english not in page


def test_build_html_with_english_language_has_no_japanese_ui(tmp_path: Path) -> None:
    cues_path = _write_cues(tmp_path / "cues.jsonl", [_cue(45, "A0100", "turn")])

    page = build_html({"trajectory": cues_path}, clips_dir=None, language="en", title="comparison")

    assert '<html lang="en">' in page
    assert "Step 45 (t = 9.0 s)" in page
    assert "Seek to translation instant (t = 9.0 s)" in page
    assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", page) is None


def test_script_with_japanese_labels_uses_safe_ascii_literal() -> None:
    clock = [line for line in script(JA).splitlines() if "clock.textContent" in line][0]

    assert clock.isascii()
    assert "__CLOCK" not in clock
    assert "\\u30b7" in clock
