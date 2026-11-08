"""The intent-translation comparison's shared contract: ``compare.yaml``, run ids and answer limits.

One parser and one dataclass for ``configs/intent_compare/*.yaml``, used by method A (the
trajectory-based Translator, ``mtd_rl_xavier.intent_compare.method_a_cli``), method B (weights and activations,
``weightreading intent-run``), the clip cutter and the viewer. Rules (the strictest of the two
former parsers):

- required keys: ``seed`` (int >= 0, no bool / str / float), ``rollout_dir``, ``steps`` (non-empty,
  strictly increasing ints >= 0), ``targets`` (non-empty, unique ids), ``window_steps`` (int >= 1),
  ``step_seconds`` (> 0), ``model``, ``reasoning_effort`` (one of the shared
  ``openai_responses.REASONING_EFFORTS``), ``instruction`` (stripped, non-blank),
  ``clip_half_width_s`` (> 0), ``output_root``; other keys are ignored;
- optional ``output_language`` (``en`` default, ``ja``) and the answer limits ``max_intent_chars``
  (default ``MAX_INTENT_CHARS``), ``max_evidence_items`` and ``max_evidence_chars`` (default
  unlimited); absent keys keep the earlier behaviour;
- relative paths are resolved against the repository root; a run id is one safe path component
  matched with ``fullmatch`` (a trailing newline is rejected).

The answer keys come from the Translator's schema (``INTENT_OUTPUT_KEYS``); ``payload_problem`` and
``limits_sentence`` are the one validation and the one prompt wording of the length limits for both
methods. This module only parses and derives paths; it never writes anything.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from translate import paths
from translate.events import INTENT_OUTPUT_KEYS
from translate.llm.openai_responses import REASONING_EFFORTS

DEFAULT_CONFIG = paths.CONFIGS / "intent_compare" / "compare.yaml"
MAX_INTENT_CHARS = 1000  # default intent cap of the comparison (characters)
DEFAULT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ("en", "ja")
METHODS = ("A", "B")
EPISODE_DIR_NAME = "episode"
CLIPS_DIR_NAME = "clips"
CLIPS_MANIFEST = "clips.json"
EPISODE_CSV_TEMPLATE = "seed{seed}.csv"
TIME_DECIMALS = 1
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # used with fullmatch only
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


class CompareConfigError(ValueError):
    """``compare.yaml`` is missing a key or holds an invalid value, or a run id is unsafe."""


# --- answer limits (shared by methods A and B) ---------------------------------------------------


@dataclass(frozen=True)
class AnswerLimits:
    """Length limits of a final answer; ``None`` means unlimited (the default keeps today's rules)."""

    max_intent_chars: int = MAX_INTENT_CHARS
    max_evidence_items: int | None = None
    max_evidence_chars: int | None = None


DEFAULT_LIMITS = AnswerLimits()


def _evidence_limit_text(limits: AnswerLimits) -> str:
    items, chars = limits.max_evidence_items, limits.max_evidence_chars
    if items is not None and chars is not None:
        return f"at most {items} evidence items, each at most {chars} characters"
    if items is not None:
        return f"at most {items} evidence items"
    if chars is not None:
        return f"each evidence item at most {chars} characters"
    return ""


def limits_sentence(limits: AnswerLimits) -> str:
    """The one prompt sentence that states the limits; ``""`` when they equal the defaults.

    Single source of the wording: method B puts it into its final-answer section and method A appends
    it to the Translator's instructions, so the two prompts cannot drift apart.
    """
    if limits == DEFAULT_LIMITS:
        return ""
    parts = [f"intent at most {limits.max_intent_chars} characters", _evidence_limit_text(limits)]
    stated = "; ".join(part for part in parts if part)
    return f"Length limits (counted in characters, not bytes; a longer answer is rejected): {stated}."


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _evidence_length_problem(evidence: list, limits: AnswerLimits) -> str | None:
    if limits.max_evidence_items is not None and len(evidence) > limits.max_evidence_items:
        return f'"evidence" has {len(evidence)} items; the limit is {limits.max_evidence_items}'
    if limits.max_evidence_chars is None:
        return None
    too_long = [len(item) for item in evidence if len(item) > limits.max_evidence_chars]
    if too_long:
        return f'an "evidence" item has {max(too_long)} characters; the limit is {limits.max_evidence_chars}'
    return None


def payload_problem(payload: object, limits: AnswerLimits) -> str | None:
    """What is wrong with a parsed final-answer object (None = valid); shared by methods A and B."""
    if not isinstance(payload, dict) or set(payload) != INTENT_OUTPUT_KEYS:
        return f"the ```json block must be an object with exactly the keys {sorted(INTENT_OUTPUT_KEYS)}"
    intent, evidence = payload["intent"], payload["evidence"]
    if not isinstance(intent, str) or not intent.strip():
        return '"intent" must be a non-blank string'
    if len(intent) > limits.max_intent_chars:
        return f'"intent" has {len(intent)} characters; the limit is {limits.max_intent_chars}'
    if not isinstance(evidence, list) or not evidence:
        return '"evidence" must be a non-empty list of strings'
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        return 'every "evidence" item must be a non-blank string'
    length_problem = _evidence_length_problem(evidence, limits)
    if length_problem:
        return length_problem
    if not all(_is_utf8(text) for text in (intent, *evidence)):
        return "the answer contains text that is not valid UTF-8 (lone surrogate)"
    return None


# --- run ids and the config ----------------------------------------------------------------------


def check_run_id(run_id: object) -> str:
    """A run id is one safe path component (shared by methods A and B and the viewer)."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise CompareConfigError(f"run id {run_id!r} must match {RUN_ID_PATTERN.pattern} and contain no '..'")
    return run_id


@dataclass(frozen=True)
class CompareConfig:
    path: Path  # as given to ``load_compare`` (recorded by method B)
    sha256: str  # of the file's bytes
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
    output_language: str = DEFAULT_LANGUAGE
    answer_limits: AnswerLimits = DEFAULT_LIMITS

    @property
    def source_path(self) -> Path:
        """Absolute path of the config file (recorded by method A)."""
        return self.path.resolve()

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
        return self.output_root / check_run_id(run_id)

    def cues_path(self, run_id: str, method: str) -> Path:
        if method not in METHODS:
            raise CompareConfigError(f"method must be one of {METHODS}")
        return self.run_dir(run_id) / f"cues_{method}.jsonl"

    def t_s(self, step: int) -> float:
        return round(step * self.step_seconds, TIME_DECIMALS)

    def window_start(self, step: int) -> int:
        """First step of the translation window ``[max(0, step - window_steps + 1), step]``."""
        return max(0, step - self.window_steps + 1)


def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _int(raw: dict[str, Any], key: str, minimum: int) -> int:
    value = raw[key]
    if not _is_strict_int(value) or value < minimum:
        raise CompareConfigError(f"compare.{key} must be an integer >= {minimum}")
    return value


def _positive_number(raw: dict[str, Any], key: str) -> float:
    value = raw[key]
    if not isinstance(value, int | float) or isinstance(value, bool) or not value > 0:
        raise CompareConfigError(f"compare.{key} must be a positive number")
    return float(value)


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise CompareConfigError(f"compare.{key} must be a non-blank string")
    return value.strip()


def _path(raw: dict[str, Any], key: str, root: Path) -> Path:
    return root / _text(raw, key)  # an absolute value stays absolute


def _steps(raw: dict[str, Any]) -> tuple[int, ...]:
    value = raw["steps"]
    if not isinstance(value, list) or not value or not all(_is_strict_int(s) and s >= 0 for s in value):
        raise CompareConfigError("compare.steps must be a non-empty list of integers >= 0")
    if value != sorted(set(value)):
        raise CompareConfigError("compare.steps must be strictly increasing")
    return tuple(value)


def _targets(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw["targets"]
    if not isinstance(value, list) or not value or not all(isinstance(t, str) and t for t in value):
        raise CompareConfigError("compare.targets must be a non-empty list of aircraft ids")
    if len(set(value)) != len(value):
        raise CompareConfigError("compare.targets must not repeat an aircraft")
    return tuple(value)


def _reasoning_effort(raw: dict[str, Any]) -> str:
    value = raw["reasoning_effort"]
    if not isinstance(value, str) or value not in REASONING_EFFORTS:
        raise CompareConfigError(f"compare.reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    return value


def _output_language(raw: dict[str, Any]) -> str:
    """Optional ``output_language``; absent means ``en`` (existing configs and their hashes stay valid)."""
    value = raw.get("output_language", DEFAULT_LANGUAGE)
    if not isinstance(value, str) or value not in OUTPUT_LANGUAGES:
        raise CompareConfigError(f"compare.output_language must be one of {list(OUTPUT_LANGUAGES)}")
    return value


def _optional_positive(raw: dict[str, Any], key: str, default: int | None) -> int | None:
    """Optional positive integer key; absent means ``default`` (today's behaviour)."""
    if key not in raw:
        return default
    return _int(raw, key, 1)


def _answer_limits(raw: dict[str, Any]) -> AnswerLimits:
    return AnswerLimits(
        max_intent_chars=_optional_positive(raw, "max_intent_chars", MAX_INTENT_CHARS),
        max_evidence_items=_optional_positive(raw, "max_evidence_items", None),
        max_evidence_chars=_optional_positive(raw, "max_evidence_chars", None),
    )


def parse_compare(raw: object, path: Path, sha256: str, root: Path = paths.ROOT) -> CompareConfig:
    """Validate a decoded compare.yaml mapping; relative paths are resolved against ``root``."""
    if not isinstance(raw, dict):
        raise CompareConfigError("compare config must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise CompareConfigError(f"compare config is missing keys: {missing}")
    return CompareConfig(
        path=path,
        sha256=sha256,
        seed=_int(raw, "seed", 0),
        rollout_dir=_path(raw, "rollout_dir", root),
        steps=_steps(raw),
        targets=_targets(raw),
        window_steps=_int(raw, "window_steps", 1),
        step_seconds=_positive_number(raw, "step_seconds"),
        model=_text(raw, "model"),
        reasoning_effort=_reasoning_effort(raw),
        instruction=_text(raw, "instruction"),
        clip_half_width_s=_positive_number(raw, "clip_half_width_s"),
        output_root=_path(raw, "output_root", root),
        output_language=_output_language(raw),
        answer_limits=_answer_limits(raw),
    )


def load_compare(path: Path = DEFAULT_CONFIG, root: Path = paths.ROOT) -> CompareConfig:
    data = path.read_bytes()
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CompareConfigError(f"compare config is not valid UTF-8 YAML: {exc}") from exc
    return parse_compare(raw, path, hashlib.sha256(data).hexdigest(), root)


def clip_bounds(center_s: float, half_width_s: float, episode_end_s: float) -> tuple[float, float]:
    """``[center - half, center + half]`` clamped to ``[0, episode_end_s]``."""
    if episode_end_s <= 0:
        raise CompareConfigError("episode_end_s must be positive")
    start = max(0.0, center_s - half_width_s)
    end = min(episode_end_s, center_s + half_width_s)
    if start >= end:
        raise CompareConfigError(f"empty clip around {center_s} s")
    return round(start, TIME_DECIMALS), round(end, TIME_DECIMALS)
