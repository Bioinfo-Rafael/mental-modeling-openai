"""The shared comparison config (``configs/intent_compare/compare.yaml``, owned by method A's session).

Only the keys method B needs are read and validated; other keys (e.g. ``clip_half_width_s`` for the
viewer) are ignored, so the shared file can grow without breaking B. The optional ``output_language``
(``en`` default, ``ja``) selects the language of the final answer. The file's sha256 goes into the
run record. The optional ``max_intent_chars`` (default 1000), ``max_evidence_items`` and
``max_evidence_chars`` (default unlimited) shorten the answers; absent keys change nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from weightreading.config import REASONING_EFFORTS, ConfigError
from weightreading.intent_protocol import MAX_INTENT_CHARS, AnswerLimits
from weightreading.provenance import sha256_file

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_LANGUAGE = "en"
OUTPUT_LANGUAGES = ("en", "ja")


@dataclass(frozen=True)
class CompareConfig:
    path: Path
    sha256: str
    seed: int
    rollout_dir: Path
    steps: tuple[int, ...]
    targets: tuple[str, ...]
    window_steps: int
    step_seconds: float
    model: str
    reasoning_effort: str
    instruction: str
    output_root: Path
    output_language: str = DEFAULT_LANGUAGE
    answer_limits: AnswerLimits = AnswerLimits()


def _int(raw: dict, key: str, minimum: int) -> int:
    value = raw.get(key)
    if type(value) is not int or value < minimum:
        raise ConfigError(f"compare.{key} must be an integer >= {minimum}")
    return value


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"compare.{key} must be a non-blank string")
    return value.strip()


def _steps(raw: dict) -> tuple[int, ...]:
    value = raw.get("steps")
    if not isinstance(value, list) or not value or not all(type(s) is int and s >= 0 for s in value):
        raise ConfigError("compare.steps must be a non-empty list of integers >= 0")
    if len(set(value)) != len(value):
        raise ConfigError("compare.steps must not repeat a step")
    return tuple(sorted(value))


def _targets(raw: dict) -> tuple[str, ...]:
    value = raw.get("targets")
    if not isinstance(value, list) or not value or not all(isinstance(t, str) and t for t in value):
        raise ConfigError("compare.targets must be a non-empty list of aircraft ids")
    if len(set(value)) != len(value):
        raise ConfigError("compare.targets must not repeat an aircraft")
    return tuple(value)


def _step_seconds(raw: dict) -> float:
    value = raw.get("step_seconds")
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ConfigError("compare.step_seconds must be a positive number")
    return float(value)


def _output_language(raw: dict) -> str:
    """Optional ``output_language``; absent means ``en`` (existing configs and their hashes stay valid)."""
    value = raw.get("output_language", DEFAULT_LANGUAGE)
    if value not in OUTPUT_LANGUAGES:
        raise ConfigError(f"compare.output_language must be one of {list(OUTPUT_LANGUAGES)}")
    return value


def _optional_positive(raw: dict, key: str, default: int | None) -> int | None:
    """Optional positive integer key; absent means ``default`` (today's behaviour)."""
    if key not in raw:
        return default
    return _int(raw, key, 1)


def _answer_limits(raw: dict) -> AnswerLimits:
    return AnswerLimits(
        max_intent_chars=_optional_positive(raw, "max_intent_chars", MAX_INTENT_CHARS),
        max_evidence_items=_optional_positive(raw, "max_evidence_items", None),
        max_evidence_chars=_optional_positive(raw, "max_evidence_chars", None),
    )


def parse_compare(raw: object, path: Path, sha256: str, root: Path) -> CompareConfig:
    """Validate a decoded compare.yaml mapping; relative paths are resolved against ``root``."""
    if not isinstance(raw, dict):
        raise ConfigError("compare config must be a mapping")
    effort = _text(raw, "reasoning_effort")
    if effort not in REASONING_EFFORTS:
        raise ConfigError(f"compare.reasoning_effort must be one of {list(REASONING_EFFORTS)}")
    return CompareConfig(
        path=path,
        sha256=sha256,
        seed=_int(raw, "seed", 0),
        rollout_dir=root / _text(raw, "rollout_dir"),
        steps=_steps(raw),
        targets=_targets(raw),
        window_steps=_int(raw, "window_steps", 1),
        step_seconds=_step_seconds(raw),
        model=_text(raw, "model"),
        reasoning_effort=effort,
        instruction=_text(raw, "instruction"),
        output_root=root / _text(raw, "output_root"),
        output_language=_output_language(raw),
        answer_limits=_answer_limits(raw),
    )


def load_compare(path: Path, root: Path) -> CompareConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"compare config is not valid YAML: {exc}") from exc
    return parse_compare(raw, path, sha256_file(path), root)


def check_run_id(run_id: str) -> str:
    """A run id is one safe path component (shared by methods A and B)."""
    if not RUN_ID_RE.match(run_id) or run_id in {".", ".."} or ".." in run_id:
        raise ConfigError(f"run id {run_id!r} must match {RUN_ID_RE.pattern} and contain no '..'")
    return run_id
