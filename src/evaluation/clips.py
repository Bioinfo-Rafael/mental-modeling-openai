"""Trim the full-episode mp4 into one clip per translation step and record exact clip timing.

    python -m evaluation.clips --full <episode>/full.mp4 --out <episode>/clips --steps 45 145 245 315

Each clip spans ``[t - half, t + half]`` seconds around the step's instant, clamped to the episode;
``clips.json`` records the requested and the measured timing (the viewer maps video time to
simulation time with it). ffmpeg must be on PATH (slurm/episode_prepare.sbatch provides one).
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

from translate.lag.rendering import STEP_SECONDS

FRAME_PATTERN = re.compile(r"frame=\s*(\d+)")
FPS_PATTERN = re.compile(r"(\d+(?:\.\d+)?) fps")
ENCODE_ARGS = ("-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an")
FFMPEG_TIMEOUT_S = 600
DURATION_TOLERANCE_S = 0.25  # about one frame at 5 fps
CLIPS_MANIFEST = "clips.json"
TIME_DECIMALS = 1
DEFAULT_HALF_WIDTH_S = 30.0
DEFAULT_WINDOW_STEPS = 60


def t_s(step: int) -> float:
    return round(step * STEP_SECONDS, TIME_DECIMALS)


def clip_bounds(center_s: float, half_width_s: float, episode_end_s: float) -> tuple[float, float]:
    """``[center - half, center + half]`` clamped to ``[0, episode_end_s]``."""
    if episode_end_s <= 0:
        raise ValueError("episode_end_s must be positive")
    start = max(0.0, center_s - half_width_s)
    end = min(episode_end_s, center_s + half_width_s)
    if start >= end:
        raise ValueError(f"empty clip around {center_s} s")
    return round(start, TIME_DECIMALS), round(end, TIME_DECIMALS)


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


def make_clips(
    full: Path, clips_dir: Path, steps: tuple[int, ...], half_width_s: float, window_steps: int
) -> dict[str, Any]:
    frames, fps = probe(full)
    episode_end_s = round(frames / fps, 3)
    clips: list[dict[str, Any]] = []
    for step in steps:
        center = t_s(step)
        start, end = clip_bounds(center, half_width_s, episode_end_s)
        out = clips_dir / f"clip_step{step:03d}.mp4"
        trim(full, out, start, end)
        clip_frames, clip_fps = probe(out)
        duration = round(clip_frames / clip_fps, 3)
        clips.append(
            {
                "step": step,
                "t_s": center,
                "window_start_step": max(0, step - window_steps + 1),
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
        "full_video": full.name,
        "full_frames": frames,
        "full_fps": fps,
        "episode_end_s": episode_end_s,
        "time_mapping": "sim_time_s = clip_start_s + video.currentTime (speed 1, 0.2 s per frame)",
        "clips": clips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--full", type=Path, required=True, help="full-episode mp4 from acmi_to_mp4.py")
    parser.add_argument("--out", type=Path, required=True, help="clips directory (clips.json goes here)")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="translation steps")
    parser.add_argument("--half-width-s", type=float, default=DEFAULT_HALF_WIDTH_S)
    parser.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = make_clips(args.full, args.out, tuple(args.steps), args.half_width_s, args.window_steps)
    (args.out / CLIPS_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [(c["step"], c["clip_start_s"], c["clip_end_s"], c["measured_duration_s"]) for c in manifest["clips"]]
        )
    )
    return 0 if all(c["duration_ok"] for c in manifest["clips"]) else 2


if __name__ == "__main__":
    sys.exit(main())
