#!/usr/bin/env python3
"""Trim the full-episode mp4 into one clip per translation step and record exact clip timing.

Clip ``k`` covers sim time ``[t_k - half, t_k + half]`` clamped to ``[0, full video duration]``
(``t_k = step_k * step_seconds``). Because ``acmi_to_mp4.py --speed 1`` emits one frame per ACMI
frame (0.2 s), video time equals sim time, so ``sim_time = clip_start_s + video.currentTime``.
Needs an ``ffmpeg`` binary on PATH (the sbatch provides the imageio-ffmpeg one). Writes
``clips.json`` with the measured frame counts and durations of every clip.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from intent_compare_config import CLIPS_MANIFEST, DEFAULT_CONFIG, clip_bounds, load_config

FRAME_PATTERN = re.compile(r"frame=\s*(\d+)")
FPS_PATTERN = re.compile(r"(\d+(?:\.\d+)?) fps")
ENCODE_ARGS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an")
FFMPEG_TIMEOUT_S = 600
DURATION_TOLERANCE_S = 0.25  # about one frame at 5 fps


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise SystemExit("ffmpeg is not on PATH")
    return exe


def probe(path: Path) -> tuple[int, float]:
    """(decoded frame count, stream fps) by decoding the whole file to the null muxer."""
    run = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=FFMPEG_TIMEOUT_S,
        check=True,
    )
    frames = FRAME_PATTERN.findall(run.stderr)
    fps = FPS_PATTERN.search(run.stderr)
    if not frames or fps is None:
        raise RuntimeError(f"cannot read frame count / fps of {path}")
    return int(frames[-1]), float(fps.group(1))


def trim(full: Path, out: Path, start_s: float, end_s: float) -> None:
    subprocess.run(
        [ffmpeg_exe(), "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(full)]
        + list(ENCODE_ARGS)
        + [str(out)],
        timeout=FFMPEG_TIMEOUT_S,
        check=True,
    )


def make_clips(full: Path, clips_dir: Path, steps: tuple[int, ...], config: Any) -> dict[str, Any]:
    frames, fps = probe(full)
    episode_end_s = round(frames / fps, 3)
    clips: list[dict[str, Any]] = []
    for step in steps:
        center = config.t_s(step)
        start, end = clip_bounds(center, config.clip_half_width_s, episode_end_s)
        out = clips_dir / f"clip_step{step:03d}.mp4"
        trim(full, out, start, end)
        clip_frames, clip_fps = probe(out)
        duration = round(clip_frames / clip_fps, 3)
        clips.append(
            {
                "step": step,
                "t_s": center,
                "window_start_step": config.window_start(step),
                "clip_start_s": start,
                "clip_end_s": end,
                "expected_duration_s": round(end - start, 3),
                "measured_frames": clip_frames,
                "measured_fps": clip_fps,
                "measured_duration_s": duration,
                "duration_ok": abs(duration - (end - start)) <= DURATION_TOLERANCE_S,
                "file": out.name,
                "bytes": out.stat().st_size,
            }
        )
    return {
        "seed": config.seed,
        "full_video": full.name,
        "full_frames": frames,
        "full_fps": fps,
        "episode_end_s": episode_end_s,
        "time_mapping": "sim_time_s = clip_start_s + video.currentTime (speed 1, 0.2 s per frame)",
        "clips": clips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--full", type=Path, required=True, help="full-episode mp4 from acmi_to_mp4.py")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config.clips_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_clips(args.full, config.clips_dir, config.steps, config)
    (config.clips_dir / CLIPS_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [(c["step"], c["clip_start_s"], c["clip_end_s"], c["measured_duration_s"]) for c in manifest["clips"]]
        )
    )
    return 0 if all(c["duration_ok"] for c in manifest["clips"]) else 2


if __name__ == "__main__":
    sys.exit(main())
