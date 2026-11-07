"""User-visible strings of the comparison viewer, one table per output language.

``en`` reproduces the original English page exactly; ``ja`` is the Japanese page. Strings with
``{name}`` slots are filled with ``str.format`` by the viewer. Everything here is static text:
model-written cue text never passes through this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    html_lang: str
    page_title: str  # {seed} {run_id}
    heading: str
    header_seed: str  # {seed} {steps} {targets}
    header_model: str  # {model} {effort} {window}
    header_instruction: str  # {instruction}
    header_cues: str  # {method} {source}
    not_run_suffix: str  # appended to a cues source that does not exist
    drift_note: str
    annotation_given: str  # {path}
    annotation_absent: str
    section_title: str  # {step} {t_s} {win_a} {win_b} {win_start_s} {clip_start} {clip_end}
    clip_missing: str
    timeline_title: str
    seek_button: str  # {t_s}
    legend: str
    aircraft_column: str
    method_a: str
    method_b: str
    human: str
    not_run: str
    no_cue: str
    skipped: str  # {reason}
    clock_prefix: str  # JS clock: <prefix><t>< mid ><step><suffix>
    clock_mid: str
    clock_suffix: str

    def column_name(self, key: str) -> str:
        """Header text of a viewer column keyed ``A``, ``B`` or ``human``."""
        return {"A": self.method_a, "B": self.method_b, "human": self.human}.get(key, key)


EN = Labels(
    html_lang="en",
    page_title="Intent comparison seed {seed} ({run_id})",
    heading="Intent translation comparison: method A vs method B",
    header_seed="seed {seed}; steps {steps}; targets {targets}",
    header_model="model {model}; reasoning_effort {effort}; window {window} steps",
    header_instruction="instruction (identical for A and B): {instruction}",
    header_cues="cues {method}: {source}",
    not_run_suffix=" (not run yet)",
    drift_note="Note: this rollout drifts numerically from the 13_LAG episode mhi_v1.3_ic100601_win used for "
    "the experiment-17 human annotation (same initial condition and launches; positions diverge later).",
    annotation_given="human annotation: {path}",
    annotation_absent="human annotation (experiment 17): no local copy (only a Google Sheet link in the "
    "workspace project's 17_commentary_LAG/README.md, not fetched); expected at {path}; column omitted.",
    section_title="Step {step} (t = {t_s} s); translation window steps {win_a}-{step} "
    "({win_start_s}-{t_s} s); clip {clip_start}-{clip_end} s",
    clip_missing="clip missing (run scripts/intent_compare_video.sbatch)",
    timeline_title="click to seek",
    seek_button="Seek to translation instant (t = {t_s} s)",
    legend="Blue: translation window (the input both methods see). Red: translation instant. "
    "Text is dimmed before the instant (online translation).",
    aircraft_column="aircraft",
    method_a="A",
    method_b="B",
    human="human",
    not_run="not run yet",
    no_cue="no translation for this step",
    skipped="skipped: {reason}",
    clock_prefix="sim t = ",
    clock_mid=" s (step ",
    clock_suffix=")",
)

JA = Labels(
    html_lang="ja",
    page_title="意図翻訳の比較 seed {seed}（{run_id}）",
    heading="意図翻訳の比較：手法A と 手法B",
    header_seed="seed {seed}；ステップ {steps}；対象機 {targets}",
    header_model="モデル {model}；reasoning_effort {effort}；ウィンドウ {window} ステップ",
    header_instruction="指示文（AとBで共通）：{instruction}",
    header_cues="{method} のキュー：{source}",
    not_run_suffix="（未実行）",
    drift_note="注意：このロールアウトは、実験17 の人手アノテーションに使われた 13_LAG のエピソード "
    "mhi_v1.3_ic100601_win と数値的にずれています（初期条件と発射は同じで、位置は後半で乖離します）。",
    annotation_given="人手アノテーション：{path}",
    annotation_absent="人手アノテーション（実験17）：ローカルコピーはありません"
    "（workspace プロジェクトの 17_commentary_LAG/README.md に Google Sheet へのリンクがあるのみで、取得していません）。"
    "置き場所は {path} です。この列は省略しました。",
    section_title="ステップ {step}（t = {t_s} 秒）；翻訳ウィンドウ ステップ {win_a}～{step}"
    "（{win_start_s}～{t_s} 秒）；クリップ {clip_start}～{clip_end} 秒",
    clip_missing="クリップがありません（scripts/intent_compare_video.sbatch を実行してください）",
    timeline_title="クリックで移動",
    seek_button="翻訳時刻へ移動（t = {t_s} 秒）",
    legend="青：翻訳ウィンドウ（両手法が見る入力）。赤：翻訳時刻。"
    "翻訳時刻より前は文章を薄く表示します（オンライン翻訳）。",
    aircraft_column="機体",
    method_a="A：トラジェクトリ",
    method_b="B：重み＋中間状態＋観測",
    human="人手",
    not_run="未実行",
    no_cue="このステップの翻訳なし",
    skipped="スキップ：{reason}",
    clock_prefix="シミュレーション時刻 t = ",
    clock_mid=" 秒（ステップ ",
    clock_suffix="）",
)

LABELS = {"en": EN, "ja": JA}  # keys == mtd_rl_xavier.intent_compare.OUTPUT_LANGUAGES (checked by a test)
