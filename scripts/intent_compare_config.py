"""Shared configuration and paths for the intent-translation comparison (method A, video, viewer).

``configs/intent_compare/compare.yaml`` is the single source of the fixed decisions (seed, steps,
targets, window, model, reasoning effort, instruction, clip width, output root) plus the optional
``output_language`` (``en`` default, ``ja``) and the optional answer-length limits ``max_intent_chars``
(default 1000), ``max_evidence_items`` and ``max_evidence_chars`` (default unlimited). Method B reads the
same file. This module only parses it and derives paths; it never writes anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "intent_compare" / "compare.yaml"
REQUIRED_KEYS = (
    "seed",
    "rollout_dir",
    "steps",
    "targets",
    "window_steps",
    "step_seconds",
    "model",
    "reasoning_effort",
    "instruction",
    "clip_half_width_s",
    "output_root",
)
DEFAULT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ("en", "ja")
DEFAULT_MAX_INTENT_CHARS = 1000
METHODS = ("A", "B")
EPISODE_DIR_NAME = "episode"
CLIPS_DIR_NAME = "clips"
CLIPS_MANIFEST = "clips.json"
EPISODE_CSV_TEMPLATE = "seed{seed}.csv"
TIME_DECIMALS = 1
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # used with fullmatch only


class CompareConfigError(ValueError):
    """``compare.yaml`` is missing a key or holds an invalid value."""


@dataclass(frozen=True)
class CompareConfig:
    seed: int
    rollout_dir: Path
    steps: tuple[int, ...]
    targets: tuple[str, ...]
    window_steps: int
    step_seconds: float
    model: str
    reasoning_effort: str
    instruction: str
    clip_half_width_s: float
    output_root: Path
    source_path: Path
    output_language: str = DEFAULT_LANGUAGE
    max_intent_chars: int = DEFAULT_MAX_INTENT_CHARS
    max_evidence_items: int | None = None
    max_evidence_chars: int | None = None

    @property
    def npz_path(self) -> Path:
        return self.rollout_dir / f"episode_seed{self.seed}.npz"

    @property
    def episode_dir(self) -> Path:
        """Run-independent derived episode artifacts (ACMI copy, CSV, clips)."""
        return self.output_root / EPISODE_DIR_NAME / f"seed{self.seed}"

    @property
    def clips_dir(self) -> Path:
        return self.episode_dir / CLIPS_DIR_NAME

    @property
    def episode_csv(self) -> Path:
        return self.episode_dir / EPISODE_CSV_TEMPLATE.format(seed=self.seed)

    def run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
            raise CompareConfigError(f"run id {run_id!r} must match {RUN_ID_PATTERN.pattern} without '..'")
        return self.output_root / run_id

    def cues_path(self, run_id: str, method: str) -> Path:
        if method not in METHODS:
            raise CompareConfigError(f"method must be one of {METHODS}")
        return self.run_dir(run_id) / f"cues_{method}.jsonl"

    def t_s(self, step: int) -> float:
        return round(step * self.step_seconds, TIME_DECIMALS)

    def window_start(self, step: int) -> int:
        """First step of the translation window ``[max(0, step - window_steps + 1), step]``."""
        return max(0, step - self.window_steps + 1)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _require_int_list(raw: Any, name: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw or not all(type(x) is int and x >= 0 for x in raw):
        raise CompareConfigError(f"{name} must be a non-empty list of nonnegative integers")
    if list(raw) != sorted(set(raw)):
        raise CompareConfigError(f"{name} must be strictly increasing")
    return tuple(raw)


def _output_language(raw: dict[str, Any]) -> str:
    """Optional ``output_language`` (``en`` when absent, so existing configs and runs are unchanged)."""
    language = raw.get("output_language", DEFAULT_LANGUAGE)
    if language not in OUTPUT_LANGUAGES:
        raise CompareConfigError(f"output_language must be one of {list(OUTPUT_LANGUAGES)}")
    return language


def _optional_positive(raw: dict[str, Any], key: str, default: int | None) -> int | None:
    """Optional positive integer key; absent means ``default`` (today's behaviour)."""
    if key not in raw:
        return default
    value = raw[key]
    if type(value) is not int or value < 1:
        raise CompareConfigError(f"{key} must be a positive integer")
    return value


def parse_config(raw: dict[str, Any], source_path: Path) -> CompareConfig:
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise CompareConfigError(f"missing keys: {missing}")
    targets = raw["targets"]
    if not isinstance(targets, list) or not targets or not all(isinstance(t, str) and t for t in targets):
        raise CompareConfigError("targets must be a non-empty list of aircraft ids")
    for key in ("model", "reasoning_effort", "instruction"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise CompareConfigError(f"{key} must be a non-empty string")
    if type(raw["window_steps"]) is not int or raw["window_steps"] < 1:
        raise CompareConfigError("window_steps must be a positive integer")
    if float(raw["step_seconds"]) <= 0 or float(raw["clip_half_width_s"]) <= 0:
        raise CompareConfigError("step_seconds and clip_half_width_s must be positive")
    return CompareConfig(
        seed=int(raw["seed"]),
        rollout_dir=_resolve(str(raw["rollout_dir"])),
        steps=_require_int_list(raw["steps"], "steps"),
        targets=tuple(targets),
        window_steps=raw["window_steps"],
        step_seconds=float(raw["step_seconds"]),
        model=raw["model"],
        reasoning_effort=raw["reasoning_effort"],
        instruction=raw["instruction"],
        clip_half_width_s=float(raw["clip_half_width_s"]),
        output_root=_resolve(str(raw["output_root"])),
        source_path=source_path,
        output_language=_output_language(raw),
        max_intent_chars=_optional_positive(raw, "max_intent_chars", DEFAULT_MAX_INTENT_CHARS),
        max_evidence_items=_optional_positive(raw, "max_evidence_items", None),
        max_evidence_chars=_optional_positive(raw, "max_evidence_chars", None),
    )


def load_config(path: Path = DEFAULT_CONFIG) -> CompareConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CompareConfigError("compare.yaml must be a mapping")
    return parse_config(raw, path.resolve())


def clip_bounds(center_s: float, half_width_s: float, episode_end_s: float) -> tuple[float, float]:
    """``[center - half, center + half]`` clamped to ``[0, episode_end_s]``."""
    if episode_end_s <= 0:
        raise CompareConfigError("episode_end_s must be positive")
    start = max(0.0, center_s - half_width_s)
    end = min(episode_end_s, center_s + half_width_s)
    if start >= end:
        raise CompareConfigError(f"empty clip around {center_s} s")
    return round(start, TIME_DECIMALS), round(end, TIME_DECIMALS)
