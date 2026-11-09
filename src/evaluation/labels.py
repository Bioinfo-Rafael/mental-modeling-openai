"""User-visible strings of the comparison viewer, one table per page language."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Labels:
    html_lang: str
    page_title: str  # {title}
    heading: str
    header_steps: str  # {steps} {targets}
    header_column: str  # {name} {source}
    not_run_suffix: str
    section_title: str  # {step} {t_s} {win_a} {win_start_s} {clip_start} {clip_end}
    clip_missing: str
    timeline_title: str
    seek_button: str  # {t_s}
    legend: str
    aircraft_column: str
    not_run: str
    no_cue: str
    skipped: str  # {reason}
    clock_prefix: str  # JS clock: <prefix><t>< mid ><step><suffix>
    clock_mid: str
    clock_suffix: str


EN = Labels(
    html_lang="en",
    page_title="Intent translation comparison ({title})",
    heading="Intent translation comparison",
    header_steps="steps {steps}; targets {targets}",
    header_column="column {name}: {source}",
    not_run_suffix=" (not run yet)",
    section_title="Step {step} (t = {t_s} s); trajectory window steps {win_a}-{step} "
    "({win_start_s}-{t_s} s); clip {clip_start}-{clip_end} s",
    clip_missing="clip missing (run slurm/episode_prepare.sbatch)",
    timeline_title="click to seek",
    seek_button="Seek to translation instant (t = {t_s} s)",
    legend="Blue: the trajectory window up to the instant. Red: translation instant. "
    "Text is dimmed before the instant (online translation).",
    aircraft_column="aircraft",
    not_run="not run yet",
    no_cue="no translation for this step",
    skipped="skipped: {reason}",
    clock_prefix="sim t = ",
    clock_mid=" s (step ",
    clock_suffix=")",
)

JA = Labels(
    html_lang="ja",
    page_title="意図翻訳の比較（{title}）",
    heading="意図翻訳の比較",
    header_steps="ステップ {steps}；対象機 {targets}",
    header_column="列 {name}：{source}",
    not_run_suffix="（未実行）",
    section_title="ステップ {step}（t = {t_s} 秒）；トラジェクトリのウィンドウ ステップ {win_a}～{step}"
    "（{win_start_s}～{t_s} 秒）；クリップ {clip_start}～{clip_end} 秒",
    clip_missing="クリップがありません（slurm/episode_prepare.sbatch を実行してください）",
    timeline_title="クリックで移動",
    seek_button="翻訳時刻へ移動（t = {t_s} 秒）",
    legend="青：翻訳時刻までのトラジェクトリのウィンドウ。赤：翻訳時刻。"
    "翻訳時刻より前は文章を薄く表示します（オンライン翻訳）。",
    aircraft_column="機体",
    not_run="未実行",
    no_cue="このステップの翻訳なし",
    skipped="スキップ：{reason}",
    clock_prefix="シミュレーション時刻 t = ",
    clock_mid=" 秒（ステップ ",
    clock_suffix="）",
)

LABELS = {"en": EN, "ja": JA}
