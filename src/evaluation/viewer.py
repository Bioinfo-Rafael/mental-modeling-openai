#!/usr/bin/env python3
"""Build one self-contained HTML page comparing intent translations side by side, one column per run.

A column is a translate run directory (its ``cues.jsonl``) or a cue JSONL file (also the former
``cues_A.jsonl`` / ``cues_B.jsonl``). Per translation step: the clip as a base64 ``<video>``, a
timeline with the trajectory window ``[t - W + 1, t]`` shaded and a marker at the translation
instant (click to seek), and per aircraft every column's translation (intent + evidence),
highlighted once playback reaches the instant. A missing cue file shows "not run yet". Every text
value is HTML-escaped; no cue text reaches JavaScript. Page text comes from ``evaluation.labels``.

    python -m evaluation.viewer --column trajectory=outputs/translate/<run> --column policy=<run or cues file> \
        --clips outputs/intent_compare/episode/seed100601/clips --out outputs/translate/<run>/viewer.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.clips import CLIPS_MANIFEST
from evaluation.labels import EN, LABELS, Labels
from translate.lag.rendering import STEP_SECONDS

CUES_FILE = "cues.jsonl"
TIME_DECIMALS = 1
DEFAULT_WINDOW_STEPS = 60  # the shipped trajectory window

STYLE = """
body{font-family:system-ui,sans-serif;margin:1.5rem;max-width:1400px;color:#222}
header p{margin:.2rem 0}.note{color:#555;font-size:.9rem}
section{border-top:2px solid #ccc;padding:1rem 0}
.layout{display:grid;grid-template-columns:minmax(420px,1fr) 2fr;gap:1rem;align-items:start}
video{width:100%;background:#000}
.timeline{position:relative;height:22px;background:#eee;cursor:pointer;margin-top:.3rem}
.window{position:absolute;top:0;bottom:0;background:#9ecbff}
.marker{position:absolute;top:-3px;bottom:-3px;width:3px;background:#d00}
.playhead{position:absolute;top:0;bottom:0;width:2px;background:#000}
.clock{font-variant-numeric:tabular-nums;font-size:.9rem;margin:.3rem 0}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:.4rem;vertical-align:top}
th{background:#f4f4f4}td.missing{color:#888;font-style:italic}
.cues{opacity:.45;transition:opacity .3s}.cues.live{opacity:1}
.cues.now{outline:3px solid #d00}
ul{margin:.3rem 0 0 1rem;padding:0}button{margin-top:.3rem}
"""

SCRIPT = """
document.querySelectorAll('section.timing').forEach(function (sec) {
  var video = sec.querySelector('video'), bar = sec.querySelector('.timeline');
  var head = sec.querySelector('.playhead'), clock = sec.querySelector('.clock');
  var cues = sec.querySelector('.cues');
  var c0 = parseFloat(sec.dataset.clipStart), c1 = parseFloat(sec.dataset.clipEnd);
  var tr = parseFloat(sec.dataset.translation), dt = parseFloat(sec.dataset.stepSeconds);
  var span = c1 - c0, wasLive = false;
  function pct(t) { return (100 * (t - c0) / span) + '%'; }
  sec.querySelector('.window').style.left = pct(parseFloat(sec.dataset.windowStart));
  sec.querySelector('.window').style.width = (100 * (tr - parseFloat(sec.dataset.windowStart)) / span) + '%';
  sec.querySelector('.marker').style.left = pct(tr);
  function update() {
    var sim = c0 + video.currentTime;
    head.style.left = pct(Math.min(sim, c1));
    clock.textContent = '__CLOCK_PREFIX__' + sim.toFixed(1) + '__CLOCK_MID__' + Math.round(sim / dt) + '__CLOCK_SUFFIX__';
    var live = sim >= tr - dt / 2;
    cues.classList.toggle('live', live);
    cues.classList.toggle('now', Math.abs(sim - tr) < 1.0);
    if (live && !wasLive && !video.paused) { cues.scrollIntoView({behavior: 'smooth', block: 'nearest'}); }
    wasLive = live;
  }
  video.addEventListener('timeupdate', update);
  video.addEventListener('seeked', update);
  video.addEventListener('loadedmetadata', update);
  bar.addEventListener('click', function (e) {
    var r = bar.getBoundingClientRect();
    video.currentTime = Math.max(0, ((e.clientX - r.left) / r.width) * span);
  });
  sec.querySelector('button.seek').addEventListener('click', function () {
    video.currentTime = Math.max(0, tr - c0); video.pause();
  });
  update();
});
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def js_text(text: str) -> str:
    """Static label text as the inside of a single-quoted JS string (ASCII-only, quote-safe)."""
    return json.dumps(text)[1:-1].replace("'", "\\u0027").replace("<", "\\u003c")


def script(labels: Labels) -> str:
    return (
        SCRIPT.replace("__CLOCK_PREFIX__", js_text(labels.clock_prefix))
        .replace("__CLOCK_MID__", js_text(labels.clock_mid))
        .replace("__CLOCK_SUFFIX__", js_text(labels.clock_suffix))
    )


def video_tag(clips_dir: Path, clip: dict[str, Any] | None, labels: Labels = EN) -> str:
    if clip is None:
        return f'<p class="missing">{esc(labels.clip_missing)}</p>'
    data = base64.b64encode((clips_dir / clip["file"]).read_bytes()).decode("ascii")
    return f'<video controls preload="metadata" src="data:video/mp4;base64,{data}"></video>'


def cue_file(source: Path) -> Path:
    """A run directory's ``cues.jsonl``, or the given cue file."""
    return source / CUES_FILE if source.is_dir() else source


def load_cues(path: Path) -> dict[tuple[int, str], dict[str, Any]] | None:
    """(step, aircraft) -> cue; ``None`` when the file does not exist."""
    if not path.is_file():
        return None
    cues: dict[tuple[int, str], dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or "step" not in record or "aircraft" not in record:
            raise ValueError(f"{path}:{number}: a cue needs step and aircraft")
        cues = {**cues, (int(record["step"]), str(record["aircraft"])): record}
    return cues


def cue_cell(cues: dict[tuple[int, str], dict[str, Any]] | None, step: int, aircraft: str, labels: Labels = EN) -> str:
    if cues is None:
        return f'<td class="missing">{esc(labels.not_run)}</td>'
    record = cues.get((step, aircraft))
    if record is None:
        return f'<td class="missing">{esc(labels.no_cue)}</td>'
    if record.get("skipped"):
        return f'<td class="missing">{esc(labels.skipped.format(reason=record["skipped"]))}</td>'
    evidence = record.get("evidence") or []
    items = "".join(f"<li>{esc(item)}</li>" for item in evidence)
    source = record.get("event_id") or record.get("source_run") or "?"
    meta = f"{record.get('model', '?')} / {record.get('reasoning_effort', '?')} / {source}"
    return f'<td><p>{esc(record.get("intent", ""))}</p><ul>{items}</ul><p class="note">{esc(meta)}</p></td>'


def cue_table(targets: list[str], step: int, columns: dict[str, Any], labels: Labels = EN) -> str:
    head = "".join(f"<th>{esc(name)}</th>" for name in columns)
    rows = "".join(
        f"<tr><th>{esc(aircraft)}</th>{''.join(cue_cell(c, step, aircraft, labels) for c in columns.values())}</tr>"
        for aircraft in targets
    )
    return f'<table class="cues"><tr><th>{esc(labels.aircraft_column)}</th>{head}</tr>{rows}</table>'


def t_s(step: int) -> float:
    return round(step * STEP_SECONDS, TIME_DECIMALS)


def section(
    view: dict[str, Any], step: int, clip: dict[str, Any] | None, columns: dict[str, Any], labels: Labels = EN
) -> str:
    instant = t_s(step)
    window_start = max(0, step - view["window_steps"] + 1)
    start_s = clip["clip_start_s"] if clip else 0.0
    end_s = clip["clip_end_s"] if clip else instant
    attrs = {
        "data-clip-start": start_s,
        "data-clip-end": end_s,
        "data-translation": instant,
        "data-window-start": t_s(window_start),
        "data-step-seconds": STEP_SECONDS,
    }
    attr_text = " ".join(f'{name}="{esc(value)}"' for name, value in attrs.items())
    title = labels.section_title.format(
        step=step, t_s=instant, win_a=window_start, win_start_s=t_s(window_start), clip_start=start_s, clip_end=end_s
    )
    return (
        f'<section class="timing" {attr_text}><h2>{esc(title)}</h2><div class="layout"><div>'
        f"{video_tag(view['clips_dir'], clip, labels)}"
        f'<div class="timeline" title="{esc(labels.timeline_title)}"><div class="window"></div>'
        '<div class="marker"></div><div class="playhead"></div></div><div class="clock"></div>'
        f'<button class="seek">{esc(labels.seek_button.format(t_s=instant))}</button>'
        f'<p class="note">{esc(labels.legend)}</p>'
        f"</div><div>{cue_table(view['targets'], step, columns, labels)}</div></div></section>"
    )


def header(steps: list[int], targets: list[str], sources: dict[str, str], labels: Labels = EN) -> str:
    rows = [
        labels.header_steps.format(steps=steps, targets=targets),
        *(labels.header_column.format(name=name, source=source) for name, source in sources.items()),
    ]
    body = "".join(f"<p>{esc(row)}</p>" for row in rows)
    return f"<header><h1>{esc(labels.heading)}</h1>{body}</header>"


def build_html(
    columns_in: dict[str, Path],
    clips_dir: Path | None,
    language: str,
    title: str,
    window_steps: int = DEFAULT_WINDOW_STEPS,
) -> str:
    labels = LABELS[language]
    manifest = clips_dir / CLIPS_MANIFEST if clips_dir is not None else None
    clips = json.loads(manifest.read_text(encoding="utf-8"))["clips"] if manifest and manifest.is_file() else []
    clip_by_step = {clip["step"]: clip for clip in clips}
    columns: dict[str, Any] = {name: load_cues(cue_file(path)) for name, path in columns_in.items()}
    sources = {
        name: f"{cue_file(path)}" + ("" if columns[name] is not None else labels.not_run_suffix)
        for name, path in columns_in.items()
    }
    keys = {key for cues in columns.values() if cues for key in cues}
    steps = sorted({step for step, _ in keys} | set(clip_by_step))
    targets = sorted({aircraft for _, aircraft in keys})
    view = {"window_steps": window_steps, "clips_dir": clips_dir, "targets": targets}
    body = "".join(section(view, step, clip_by_step.get(step), columns, labels) for step in steps)
    csp = "default-src 'none'; media-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"
    return (
        f'<!DOCTYPE html><html lang="{esc(labels.html_lang)}"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<title>{esc(labels.page_title.format(title=title))}</title><style>{STYLE}</style></head>"
        f"<body>{header(steps, targets, sources, labels)}{body}<script>{script(labels)}</script></body></html>\n"
    )


def _column(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("a column is NAME=PATH (a run directory or a cue JSONL file)")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--column", type=_column, action="append", required=True, help="NAME=PATH, repeatable")
    parser.add_argument("--clips", type=Path, help="clips directory with clips.json (python -m evaluation.clips)")
    parser.add_argument("--language", choices=sorted(LABELS), default="en", help="page language")
    parser.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS, help="trajectory window shaded")
    parser.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    args = parser.parse_args(argv)
    columns = dict(args.column)
    if len(columns) != len(args.column):
        parser.error("column names must be unique")
    page = build_html(columns, args.clips, args.language, args.out.parent.name, args.window_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(json.dumps({"viewer": str(args.out), "bytes": args.out.stat().st_size}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
