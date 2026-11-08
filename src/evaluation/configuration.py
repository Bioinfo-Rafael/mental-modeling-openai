"""Strict judge YAML config (D15): model, limits, the six input blocks, output root and pricing.

Exact key sets everywhere: unknown, missing or wrong-typed keys fail, booleans never count as
numbers, and duplicate YAML keys fail. The model is fixed to ``gpt-6.1-sol`` (D8); there is no
default model in code. The ``video`` block (L3, D30) maps each episode id to its video and sets
the frame window, sampling, resize, JPEG quality, image ``detail``, the label format and the
(unverified) image-token estimate; every key is required, so there are no hidden defaults. Named
``configuration.py`` so it does not collide with the ``evaluation/config/`` directory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from translate import paths
from translate.events import SAFE_ID_PATTERN
from translate.llm.openai_responses import IMAGE_DETAILS, REASONING_EFFORTS

CONFIG_SCHEMA_VERSION = 1
JUDGE_MODEL = "gpt-6.1-sol"  # D8: the only accepted judge model

BLOCK_INSTRUCTIONS = "instructions"
BLOCK_VIDEO = "video"
BLOCK_TIMESTAMPS = "timestamps"
BLOCK_TRAJECTORY = "trajectory"
BLOCK_TRANSLATED_INTENT = "translated_intent"
BLOCK_KEY_POINTS = "key_points"
# Exactly these six blocks (D15), in request order.
INPUT_BLOCKS = (
    BLOCK_INSTRUCTIONS,
    BLOCK_VIDEO,
    BLOCK_TIMESTAMPS,
    BLOCK_TRAJECTORY,
    BLOCK_TRANSLATED_INTENT,
    BLOCK_KEY_POINTS,
)

RENDERER_CHRONOLOGICAL = "chronological"
RENDERER_SAMPLED_JPEG = "sampled-jpeg"
RENDERER_VERBATIM = "translation_input_verbatim"
RENDERER_INTENT_EVIDENCE = "intent-and-evidence"
RENDERER_RUBRIC = "numbered-with-fixed-rubric"
VIDEO_SOURCE = "scenario_action_video"
VIDEO_STRATEGY = "event_window"
TIMESTAMP_SCOPE = "through_event"
MAX_JPEG_QUALITY = 100
LABEL_FIELD_STEP = "step"
LABEL_FIELD_TIME = "time_s"
LABEL_FIELD_RELATIVE = "relative_step"
LABEL_FIELDS = (LABEL_FIELD_STEP, LABEL_FIELD_TIME, LABEL_FIELD_RELATIVE)
LABEL_FIELDS_TEXT = ", ".join("{" + field + "}" for field in LABEL_FIELDS)
LABEL_REQUIRED_FIELDS = (LABEL_FIELD_STEP, LABEL_FIELD_TIME)
TOKEN_STATUSES = ("unverified", "verified")

TOP_KEYS = {
    "schema_version",
    "model",
    "reasoning_effort",
    "max_output_tokens",
    "timeout_s",
    "limits",
    "inputs",
    "output",
    "pricing",
}
LIMIT_KEYS = {"max_judged_events"}
INSTRUCTIONS_KEYS = {"include", "renderer", "show_ids", "show_targets", "show_issue_steps"}
VIDEO_KEYS = {
    "include",
    "renderer",
    "source",
    "videos",
    "sampling",
    "resize",
    "detail",
    "label_format",
    "image_tokens",
}
VIDEO_EPISODE_KEYS = {"path", "fps", "frame_offset", "frames_per_step", "step_seconds"}
IMAGE_TOKEN_KEYS = {"per_image", "status"}
SAMPLING_KEYS = {"strategy", "before_steps", "after_steps", "stride_steps", "max_frames"}
RESIZE_KEYS = {"max_width", "max_height", "jpeg_quality"}
TIMESTAMPS_KEYS = {"include", "renderer", "scope", "show_sim_seconds", "show_sources"}
TRAJECTORY_KEYS = {"include", "renderer", "show_window_metadata"}
RENDERER_ONLY_KEYS = {"include", "renderer"}
OUTPUT_KEYS = {"root"}
PRICE_KEYS = {
    "input",
    "cached_input",
    "output",
    "long_context_threshold_tokens",
    "long_input_multiplier",
    "long_output_multiplier",
}


class JudgeConfigError(ValueError):
    """The judge configuration is malformed."""


@dataclass(frozen=True)
class InstructionsInput:
    include: bool
    renderer: str
    show_ids: bool
    show_targets: bool
    show_issue_steps: bool


@dataclass(frozen=True)
class VideoEpisode:
    """One episode's video: frame ``frame_offset + step * frames_per_step`` shows simulation step ``step``."""

    path: Path
    fps: float  # the video's own frame rate; checked against the file when it is opened
    frame_offset: int
    frames_per_step: int
    step_seconds: float  # simulation seconds per step, for the time shown in a label


@dataclass(frozen=True)
class VideoSampling:
    strategy: str
    before_steps: int
    after_steps: int
    stride_steps: int
    max_frames: int


@dataclass(frozen=True)
class VideoResize:
    max_width: int
    max_height: int
    jpeg_quality: int


@dataclass(frozen=True)
class ImageTokens:
    """Preview-only estimate of the input tokens one image costs. ``status`` says whether it was checked."""

    per_image: int
    status: str


@dataclass(frozen=True)
class VideoInput:
    include: bool
    renderer: str
    source: str
    videos: dict[str, VideoEpisode]  # episode id -> video; never mutated after parsing
    sampling: VideoSampling
    resize: VideoResize
    detail: str
    label_format: str  # text item put right before each image; fields {step} {time_s} {relative_step}
    image_tokens: ImageTokens


@dataclass(frozen=True)
class TimestampsInput:
    include: bool
    renderer: str
    scope: str
    show_sim_seconds: bool
    show_sources: bool


@dataclass(frozen=True)
class TrajectoryInput:
    include: bool
    renderer: str
    show_window_metadata: bool


@dataclass(frozen=True)
class TranslatedIntentInput:
    include: bool
    renderer: str


@dataclass(frozen=True)
class KeyPointsInput:
    include: bool
    renderer: str


@dataclass(frozen=True)
class JudgeInputs:
    instructions: InstructionsInput
    video: VideoInput
    timestamps: TimestampsInput
    trajectory: TrajectoryInput
    translated_intent: TranslatedIntentInput
    key_points: KeyPointsInput

    def included(self) -> tuple[str, ...]:
        """Names of the included blocks, in request order."""
        return tuple(name for name in INPUT_BLOCKS if getattr(self, name).include)


@dataclass(frozen=True)
class JudgePrice:
    """USD per 1M tokens (list price; verify before paid runs). Preview and usage estimates only."""

    input: float
    cached_input: float
    output: float
    long_context_threshold_tokens: int  # a request above this many input tokens is billed at the multipliers
    long_input_multiplier: float
    long_output_multiplier: float


@dataclass(frozen=True)
class JudgeConfig:
    schema_version: int
    model: str
    reasoning_effort: str
    max_output_tokens: int
    timeout_s: float
    max_judged_events: int
    inputs: JudgeInputs
    output_root: Path
    price: JudgePrice


# --- strict YAML ----------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys (PyYAML keeps the last one silently)."""


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError:
            raise yaml.constructor.ConstructorError(None, None, "unhashable mapping key", key_node.start_mark) from None
        if duplicate:
            raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_strict_yaml(path: Path, error: type[ValueError]) -> object:
    """Decode a YAML file with the safe loader; duplicate keys and unreadable files raise ``error``."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"cannot read {path}: {type(exc).__name__}") from exc
    try:
        return yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass: no arbitrary objects
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None)
        raise error(f"{path} is not valid YAML" + (f": {problem}" if problem else "")) from exc


# --- primitive checks -----------------------------------------------------------------------------


def _mapping(raw: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise JudgeConfigError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise JudgeConfigError(f"{name} keys must be strings")
    if set(raw) != keys:
        missing, unknown = sorted(keys - set(raw)), sorted(set(raw) - keys)
        raise JudgeConfigError(f"{name} must have exactly {sorted(keys)} (missing {missing}, unknown {unknown})")
    return raw


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_at_least(value: object, minimum: int, name: str) -> int:
    if not _is_int(value) or value < minimum:
        raise JudgeConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _int(value: object, name: str) -> int:
    if not _is_int(value):
        raise JudgeConfigError(f"{name} must be an integer")
    return value


def _number_at_least(value: object, minimum: float, name: str) -> float:
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or not math.isfinite(value) or value < minimum:
        raise JudgeConfigError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise JudgeConfigError(f"{name} must be true or false")
    return value


def _choice(value: object, allowed: tuple[str, ...] | frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise JudgeConfigError(f"{name} must be one of {sorted(allowed)}")
    return value


# --- sections -------------------------------------------------------------------------------------


def _instructions(raw: object) -> InstructionsInput:
    name = f"inputs.{BLOCK_INSTRUCTIONS}"
    section = _mapping(raw, INSTRUCTIONS_KEYS, name)
    return InstructionsInput(
        include=_flag(section["include"], f"{name}.include"),
        renderer=_choice(section["renderer"], (RENDERER_CHRONOLOGICAL,), f"{name}.renderer"),
        show_ids=_flag(section["show_ids"], f"{name}.show_ids"),
        show_targets=_flag(section["show_targets"], f"{name}.show_targets"),
        show_issue_steps=_flag(section["show_issue_steps"], f"{name}.show_issue_steps"),
    )


def _positive_number(value: object, name: str) -> float:
    number = _number_at_least(value, 0, name)
    if number <= 0:
        raise JudgeConfigError(f"{name} must be a finite number > 0")
    return number


def _video_episode(raw: object, root: Path, name: str) -> VideoEpisode:
    section = _mapping(raw, VIDEO_EPISODE_KEYS, name)
    path = section["path"]
    if not isinstance(path, str) or not path.strip():
        raise JudgeConfigError(f"{name}.path must be a non-empty path")
    return VideoEpisode(
        path=Path(path) if Path(path).is_absolute() else root / path,
        fps=_positive_number(section["fps"], f"{name}.fps"),
        frame_offset=_int_at_least(section["frame_offset"], 0, f"{name}.frame_offset"),
        frames_per_step=_int_at_least(section["frames_per_step"], 1, f"{name}.frames_per_step"),
        step_seconds=_positive_number(section["step_seconds"], f"{name}.step_seconds"),
    )


def _videos(raw: object, root: Path, include: bool, name: str) -> dict[str, VideoEpisode]:
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise JudgeConfigError(f"{name} must be a mapping of episode id to video")
    if include and not raw:
        raise JudgeConfigError(f"{name} must name at least one episode when the video block is included")
    for episode_id in raw:
        if not SAFE_ID_PATTERN.fullmatch(episode_id):
            raise JudgeConfigError(f"{name} keys must match {SAFE_ID_PATTERN.pattern}")
    return {episode_id: _video_episode(item, root, f"{name}.{episode_id}") for episode_id, item in raw.items()}


def _label_format(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise JudgeConfigError(f"{name} must be a non-empty single-line string")
    try:
        parsed = list(Formatter().parse(value))
    except ValueError as exc:
        raise JudgeConfigError(f"{name} is not a valid format string: {exc}") from exc
    fields = []
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if field not in LABEL_FIELDS or spec or conversion:
            raise JudgeConfigError(f"{name} may only use the plain fields {LABEL_FIELDS_TEXT}")
        fields.append(field)
    missing = [field for field in LABEL_REQUIRED_FIELDS if field not in fields]
    if missing:
        raise JudgeConfigError(f"{name} must contain the fields {missing}")
    return value


def _image_tokens(raw: object, name: str) -> ImageTokens:
    section = _mapping(raw, IMAGE_TOKEN_KEYS, name)
    return ImageTokens(
        per_image=_int_at_least(section["per_image"], 0, f"{name}.per_image"),
        status=_choice(section["status"], TOKEN_STATUSES, f"{name}.status"),
    )


def _video(raw: object, root: Path) -> VideoInput:
    name = f"inputs.{BLOCK_VIDEO}"
    section = _mapping(raw, VIDEO_KEYS, name)
    sampling = _mapping(section["sampling"], SAMPLING_KEYS, f"{name}.sampling")
    resize = _mapping(section["resize"], RESIZE_KEYS, f"{name}.resize")
    quality = _int_at_least(resize["jpeg_quality"], 1, f"{name}.resize.jpeg_quality")
    if quality > MAX_JPEG_QUALITY:
        raise JudgeConfigError(f"{name}.resize.jpeg_quality must be <= {MAX_JPEG_QUALITY}")
    include = _flag(section["include"], f"{name}.include")
    return VideoInput(
        include=include,
        renderer=_choice(section["renderer"], (RENDERER_SAMPLED_JPEG,), f"{name}.renderer"),
        source=_choice(section["source"], (VIDEO_SOURCE,), f"{name}.source"),
        videos=_videos(section["videos"], root, include, f"{name}.videos"),
        sampling=VideoSampling(
            strategy=_choice(sampling["strategy"], (VIDEO_STRATEGY,), f"{name}.sampling.strategy"),
            before_steps=_int_at_least(sampling["before_steps"], 0, f"{name}.sampling.before_steps"),
            after_steps=_int_at_least(sampling["after_steps"], 0, f"{name}.sampling.after_steps"),
            stride_steps=_int_at_least(sampling["stride_steps"], 1, f"{name}.sampling.stride_steps"),
            max_frames=_int_at_least(sampling["max_frames"], 1, f"{name}.sampling.max_frames"),
        ),
        resize=VideoResize(
            max_width=_int_at_least(resize["max_width"], 1, f"{name}.resize.max_width"),
            max_height=_int_at_least(resize["max_height"], 1, f"{name}.resize.max_height"),
            jpeg_quality=quality,
        ),
        detail=_choice(section["detail"], IMAGE_DETAILS, f"{name}.detail"),
        label_format=_label_format(section["label_format"], f"{name}.label_format"),
        image_tokens=_image_tokens(section["image_tokens"], f"{name}.image_tokens"),
    )


def _timestamps(raw: object) -> TimestampsInput:
    name = f"inputs.{BLOCK_TIMESTAMPS}"
    section = _mapping(raw, TIMESTAMPS_KEYS, name)
    return TimestampsInput(
        include=_flag(section["include"], f"{name}.include"),
        renderer=_choice(section["renderer"], (RENDERER_CHRONOLOGICAL,), f"{name}.renderer"),
        scope=_choice(section["scope"], (TIMESTAMP_SCOPE,), f"{name}.scope"),
        show_sim_seconds=_flag(section["show_sim_seconds"], f"{name}.show_sim_seconds"),
        show_sources=_flag(section["show_sources"], f"{name}.show_sources"),
    )


def _trajectory(raw: object) -> TrajectoryInput:
    name = f"inputs.{BLOCK_TRAJECTORY}"
    section = _mapping(raw, TRAJECTORY_KEYS, name)
    return TrajectoryInput(
        include=_flag(section["include"], f"{name}.include"),
        renderer=_choice(section["renderer"], (RENDERER_VERBATIM,), f"{name}.renderer"),
        show_window_metadata=_flag(section["show_window_metadata"], f"{name}.show_window_metadata"),
    )


def _renderer_only(raw: object, block: str, renderer: str) -> tuple[bool, str]:
    name = f"inputs.{block}"
    section = _mapping(raw, RENDERER_ONLY_KEYS, name)
    return _flag(section["include"], f"{name}.include"), _choice(section["renderer"], (renderer,), f"{name}.renderer")


def _inputs(raw: object, root: Path) -> JudgeInputs:
    section = _mapping(raw, set(INPUT_BLOCKS), "inputs")
    return JudgeInputs(
        instructions=_instructions(section[BLOCK_INSTRUCTIONS]),
        video=_video(section[BLOCK_VIDEO], root),
        timestamps=_timestamps(section[BLOCK_TIMESTAMPS]),
        trajectory=_trajectory(section[BLOCK_TRAJECTORY]),
        translated_intent=TranslatedIntentInput(
            *_renderer_only(section[BLOCK_TRANSLATED_INTENT], BLOCK_TRANSLATED_INTENT, RENDERER_INTENT_EVIDENCE)
        ),
        key_points=KeyPointsInput(*_renderer_only(section[BLOCK_KEY_POINTS], BLOCK_KEY_POINTS, RENDERER_RUBRIC)),
    )


def _price(raw: object, model: str) -> JudgePrice:
    if not isinstance(raw, dict) or set(raw) != {model}:
        raise JudgeConfigError(f"pricing must have exactly one entry, for the judge model {model!r}")
    name = f"pricing.{model}"
    section = _mapping(raw[model], PRICE_KEYS, name)
    return JudgePrice(
        input=_number_at_least(section["input"], 0, f"{name}.input"),
        cached_input=_number_at_least(section["cached_input"], 0, f"{name}.cached_input"),
        output=_number_at_least(section["output"], 0, f"{name}.output"),
        long_context_threshold_tokens=_int_at_least(
            section["long_context_threshold_tokens"], 1, f"{name}.long_context_threshold_tokens"
        ),
        long_input_multiplier=_number_at_least(section["long_input_multiplier"], 1, f"{name}.long_input_multiplier"),
        long_output_multiplier=_number_at_least(section["long_output_multiplier"], 1, f"{name}.long_output_multiplier"),
    )


def _output_root(raw: object, root: Path) -> Path:
    section = _mapping(raw, OUTPUT_KEYS, "output")
    value = section["root"]
    if not isinstance(value, str) or not value.strip():
        raise JudgeConfigError("output.root must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_config(raw: object, *, root: Path = paths.ROOT) -> JudgeConfig:
    """Validate a decoded YAML mapping into a frozen ``JudgeConfig``."""
    section = _mapping(raw, TOP_KEYS, "config")
    if not _is_int(section["schema_version"]) or section["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise JudgeConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if section["model"] != JUDGE_MODEL:
        raise JudgeConfigError(f"model must be exactly {JUDGE_MODEL!r} (D8)")
    limits = _mapping(section["limits"], LIMIT_KEYS, "limits")
    return JudgeConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        model=JUDGE_MODEL,
        reasoning_effort=_choice(section["reasoning_effort"], REASONING_EFFORTS, "reasoning_effort"),
        max_output_tokens=_int_at_least(section["max_output_tokens"], 1, "max_output_tokens"),
        timeout_s=_positive_timeout(section["timeout_s"]),
        max_judged_events=_int_at_least(limits["max_judged_events"], 1, "limits.max_judged_events"),
        inputs=_inputs(section["inputs"], root),
        output_root=_output_root(section["output"], root),
        price=_price(section["pricing"], JUDGE_MODEL),
    )


def _positive_timeout(value: object) -> float:
    timeout_s = _number_at_least(value, 0, "timeout_s")
    if timeout_s <= 0:
        raise JudgeConfigError("timeout_s must be positive")
    return timeout_s


def load_config(path: Path, *, root: Path = paths.ROOT) -> JudgeConfig:
    return parse_config(read_strict_yaml(path, JudgeConfigError), root=root)
