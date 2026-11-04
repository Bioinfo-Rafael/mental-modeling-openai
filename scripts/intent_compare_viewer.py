#!/usr/bin/env python3
"""Build one self-contained HTML page comparing method A and method B intent translations.

Per translation step: the clip as a base64 ``<video>``, a timeline with the translation window
``[t-59, t]`` shaded and a marker at the translation instant (click to seek), and per aircraft the
A and B translations (intent + evidence) side by side, highlighted once playback reaches the
translation instant. A missing ``cues_<method>.jsonl`` shows "not run yet". Every text value is
HTML-escaped; no cue text reaches JavaScript. Output: ``<output_root>/<run_id>/viewer.html``.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any

from intent_compare_config import CLIPS_MANIFEST, DEFAULT_CONFIG, METHODS, REPO_ROOT, CompareConfig, load_config

HUMAN = "human"
# Places a local copy of the workspace/17 human annotation would live; none exists today (the
# annotation is a Google Sheet linked from workspace/17_commentary_LAG/README.md, never fetched).
HUMAN_ANNOTATION_CANDIDATES = (
    REPO_ROOT / "workspace" / "17_commentary_LAG" / "annotation" / "human_annotation.jsonl",
    REPO_ROOT / "private" / "intent_compare" / "human_annotation.jsonl",
)
NOT_RUN = "not run yet"
NO_CUE = "no translation for this step"

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
    clock.textContent = 'sim t = ' + sim.toFixed(1) + ' s (step ' + Math.round(sim / dt) + ')';
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


def load_cues(path: Path | None, seed: int) -> dict[tuple[int, str], dict[str, Any]] | None:
    """(step, aircraft) -> cue for this seed; ``None`` when the file does not exist."""
    if path is None or not path.is_file():
        return None
    cues: dict[tuple[int, str], dict[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or "step" not in record or "aircraft" not in record:
            raise ValueError(f"{path}:{number}: a cue needs step and aircraft")
        if record.get("seed") != seed:
            continue
        cues = {**cues, (int(record["step"]), str(record["aircraft"])): record}
    return cues


def cue_cell(cues: dict[tuple[int, str], dict[str, Any]] | None, step: int, aircraft: str) -> str:
    if cues is None:
        return f'<td class="missing">{esc(NOT_RUN)}</td>'
    record = cues.get((step, aircraft))
    if record is None:
        return f'<td class="missing">{esc(NO_CUE)}</td>'
    if record.get("skipped"):
        return f'<td class="missing">skipped: {esc(record["skipped"])}</td>'
    evidence = record.get("evidence") or []
    items = "".join(f"<li>{esc(item)}</li>" for item in evidence)
    meta = f"{record.get('model', '?')} / {record.get('reasoning_effort', '?')} / {record.get('source_run', '?')}"
    return f'<td><p>{esc(record.get("intent", ""))}</p><ul>{items}</ul><p class="note">{esc(meta)}</p></td>'


def cue_table(config: CompareConfig, step: int, columns: dict[str, Any]) -> str:
    head = "".join(f"<th>{esc(name)}</th>" for name in columns)
    rows = "".join(
        f"<tr><th>{esc(aircraft)}</th>{''.join(cue_cell(c, step, aircraft) for c in columns.values())}</tr>"
        for aircraft in config.targets
    )
    return f'<table class="cues"><tr><th>aircraft</th>{head}</tr>{rows}</table>'


def video_tag(clips_dir: Path, clip: dict[str, Any] | None) -> str:
    if clip is None:
        return '<p class="missing">clip missing (run scripts/intent_compare_video.sbatch)</p>'
    data = base64.b64encode((clips_dir / clip["file"]).read_bytes()).decode("ascii")
    return f'<video controls preload="metadata" src="data:video/mp4;base64,{data}"></video>'


def section(config: CompareConfig, step: int, clip: dict[str, Any] | None, columns: dict[str, Any]) -> str:
    t_s = config.t_s(step)
    window_start = config.window_start(step)
    start_s = clip["clip_start_s"] if clip else 0.0
    end_s = clip["clip_end_s"] if clip else t_s
    attrs = {
        "data-clip-start": start_s,
        "data-clip-end": end_s,
        "data-translation": t_s,
        "data-window-start": config.t_s(window_start),
        "data-step-seconds": config.step_seconds,
    }
    attr_text = " ".join(f'{name}="{esc(value)}"' for name, value in attrs.items())
    title = (
        f"Step {step} (t = {t_s} s); translation window steps {window_start}-{step} "
        f"({config.t_s(window_start)}-{t_s} s); clip {start_s}-{end_s} s"
    )
    return (
        f'<section class="timing" {attr_text}><h2>{esc(title)}</h2><div class="layout"><div>'
        f"{video_tag(config.clips_dir, clip)}"
        '<div class="timeline" title="click to seek"><div class="window"></div><div class="marker"></div>'
        '<div class="playhead"></div></div><div class="clock"></div>'
        f'<button class="seek">Seek to translation instant (t = {esc(t_s)} s)</button>'
        '<p class="note">Blue: translation window (the input both methods see). Red: translation instant. '
        "Text is dimmed before the instant (online translation).</p>"
        f"</div><div>{cue_table(config, step, columns)}</div></div></section>"
    )


def header(config: CompareConfig, sources: dict[str, str], annotation_note: str) -> str:
    rows = [
        f"seed {config.seed}; steps {list(config.steps)}; targets {list(config.targets)}",
        f"model {config.model}; reasoning_effort {config.reasoning_effort}; window {config.window_steps} steps",
        f"instruction (identical for A and B): {config.instruction}",
        *(f"cues {method}: {source}" for method, source in sources.items()),
        annotation_note,
        "Note: this rollout drifts numerically from the 13_LAG episode mhi_v1.3_ic100601_win used for the "
        "workspace/17 human annotation (same initial condition and launches; positions diverge later).",
    ]
    body = "".join(f"<p>{esc(row)}</p>" for row in rows)
    return f"<header><h1>Intent translation comparison: method A vs method B</h1>{body}</header>"


def human_annotation(explicit: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        return explicit, f"human annotation: {explicit}"
    for candidate in HUMAN_ANNOTATION_CANDIDATES:
        if candidate.is_file():
            return candidate, f"human annotation: {candidate.relative_to(REPO_ROOT)}"
    return None, (
        "human annotation (workspace/17): no local copy in this repository (only a Google Sheet link in "
        "workspace/17_commentary_LAG/README.md, not fetched); column omitted."
    )


def build_html(config: CompareConfig, run_id: str, overrides: dict[str, Path | None], annotation: Path | None) -> str:
    manifest_path = config.clips_dir / CLIPS_MANIFEST
    clips = json.loads(manifest_path.read_text(encoding="utf-8"))["clips"] if manifest_path.is_file() else []
    clip_by_step = {clip["step"]: clip for clip in clips}
    paths = {m: overrides.get(m) or config.cues_path(run_id, m) for m in METHODS}
    columns: dict[str, Any] = {m: load_cues(paths[m], config.seed) for m in METHODS}
    sources = {m: f"{paths[m]}" + ("" if columns[m] is not None else f" ({NOT_RUN})") for m in METHODS}
    human_path, annotation_note = human_annotation(annotation)
    if human_path is not None:
        columns = {**columns, HUMAN: load_cues(human_path, config.seed)}
    body = "".join(section(config, step, clip_by_step.get(step), columns) for step in config.steps)
    csp = "default-src 'none'; media-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<title>{esc(f'Intent comparison seed {config.seed} ({run_id})')}</title><style>{STYLE}</style></head>"
        f"<body>{header(config, sources, annotation_note)}{body}<script>{SCRIPT}</script></body></html>\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="output sub-directory (default: seed<seed>)")
    parser.add_argument("--cues-a", type=Path, help="override path of cues_A.jsonl")
    parser.add_argument("--cues-b", type=Path, help="override path of cues_B.jsonl")
    parser.add_argument("--annotation", type=Path, help="local human annotation in the cue JSONL format")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run_id = args.run_id or f"seed{config.seed}"
    page = build_html(config, run_id, {"A": args.cues_a, "B": args.cues_b}, args.annotation)
    out = config.run_dir(run_id) / "viewer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(json.dumps({"viewer": str(out), "bytes": out.stat().st_size}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
