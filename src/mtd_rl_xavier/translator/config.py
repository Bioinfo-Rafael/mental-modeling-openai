"""Strict Translator run-config and scenario YAML loading (exact key sets, no hidden defaults).

Every mapping must have exactly the documented keys; unknown, missing or wrong-typed values
fail. Booleans never count as numbers. There is no default model in code. Relative output roots
resolve from the repository root; scenario paths resolve relative to the scenario file.
The detector endpoint goes through the System One client's own validators (``checked_tcp_url``,
``checked_env_name``), so a preview rejects exactly what an execute would, before anything is
written. ``uds_path`` may contain the one placeholder ``{user}`` (the current user name).
Checks that need the episode itself (last step, aircraft and team ids) run in
``pipeline.validate_scenario``.
"""

from __future__ import annotations

import getpass
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mtd_rl_xavier import paths
from mtd_rl_xavier.lag.dataset import COLUMNS
from mtd_rl_xavier.lag.rendering import LAG17_RENDERER, RESERVED_COMPACT_KEYS
from mtd_rl_xavier.llm.openai_responses import REASONING_EFFORTS
from mtd_rl_xavier.llm.system_one import SystemOneConfigError, checked_env_name, checked_tcp_url

from .schema import (
    SAFE_ID_PATTERN,
    TARGET_KINDS,
    EpisodeRef,
    ExternalChangePoint,
    Instruction,
    RequestPoint,
    Scenario,
    TargetRef,
)

CONFIG_SCHEMA_VERSION = 1
SCENARIO_SCHEMA_VERSION = 1
VISIBILITY_ONLINE = "online"
VISIBILITY_RETROSPECTIVE = "retrospective"  # LATER (L2); rejected in this phase
DETECTOR_BACKENDS = ("system_one",)
TRANSLATION_OPENAI = "openai_responses"
TRANSLATION_VLLM_CHAT = "vllm_chat"  # LATER (L2); rejected in this phase
TRANSLATION_BACKENDS = (TRANSLATION_OPENAI,)
PAID_BACKENDS = frozenset({TRANSLATION_OPENAI})
ENTITY_SCOPES = ("target_and_opponents",)
QUESTION_TYPES = ("noul",)  # the detector reads p(yes) of one noul question
TRANSLATION_RENDERERS = (LAG17_RENDERER,)
# clef-flash serves at most 16384 tokens; the budget must equal min(16384, --max-model-len).
MAX_DETECTOR_INPUT_TOKEN_BUDGET = 16384
NOT_IN_THIS_PHASE = "is not implemented in this phase (P2)"
USER_PLACEHOLDER = "{user}"  # the only placeholder in uds_path; no environment variable is expanded
USER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}")  # used with fullmatch only
UNEXPANDED_MARKERS = ("{", "}", "$")

TOP_KEYS = {"schema_version", "pipeline", "limits", "detector", "translation", "output", "pricing"}
PIPELINE_KEYS = {"tick_interval_steps", "visibility"}
VISIBILITY_KEYS = {"mode", "future_steps"}
LIMIT_KEYS = {"max_detector_calls", "max_translation_calls"}
DETECTOR_KEYS = {
    "backend",
    "model",
    "revision",
    "served_model_name",
    "connection",
    "timeout_s",
    "threshold",
    "input_token_budget",
    "question",
    "window",
}
CONNECTION_KEYS = {"base_url", "uds_path", "api_key_env"}
QUESTION_KEYS = {"id", "type", "instructions", "criteria"}
CRITERIA_KEYS = {"true", "false"}
DETECTOR_WINDOW_KEYS = {"past_steps", "entity_scope", "fields", "float_precision"}
TRANSLATION_KEYS = {
    "backend",
    "connection",
    "model",
    "reasoning_effort",
    "output_language",
    "max_output_tokens",
    "max_intent_chars",
    "timeout_s",
    "window",
}
TRANSLATION_WINDOW_KEYS = {"past_steps", "renderer"}
OUTPUT_KEYS = {"root"}
PRICE_KEYS = {"input", "cached_input", "output"}
SCENARIO_KEYS = {"schema_version", "scenario_id", "episode", "instructions", "requests", "external_change_points"}
EPISODE_KEYS = {"episode_id", "alias", "csv_path", "video_path"}
INSTRUCTION_KEYS = {"instruction_id", "issue_step", "target", "text"}
REQUEST_KEYS = {"request_id", "step", "targets", "source"}
CHANGE_KEYS = {"change_id", "step", "target", "source"}
TARGET_KEYS = {"kind", "id"}


class TranslatorConfigError(ValueError):
    """The run configuration is malformed or asks for something this phase does not implement."""


class ScenarioError(ValueError):
    """The scenario file is malformed or inconsistent."""


@dataclass(frozen=True)
class VisibilityConfig:
    mode: str
    future_steps: int


@dataclass(frozen=True)
class PipelineConfig:
    tick_interval_steps: int
    visibility: VisibilityConfig


@dataclass(frozen=True)
class LimitsConfig:
    max_detector_calls: int
    max_translation_calls: int


@dataclass(frozen=True)
class ConnectionConfig:
    base_url: str | None
    uds_path: Path | None
    api_key_env: str | None  # the variable NAME; the value is read only by the client


@dataclass(frozen=True)
class DetectorQuestionConfig:
    question_id: str
    question_type: str
    instructions: str
    true_criterion: str
    false_criterion: str

    def to_json(self) -> dict[str, Any]:
        """The full question object, as logged in every ``detector_call`` record."""
        criteria = {"true": self.true_criterion, "false": self.false_criterion}
        return {
            "id": self.question_id,
            "type": self.question_type,
            "instructions": self.instructions,
            "criteria": criteria,
        }


@dataclass(frozen=True)
class DetectorWindowConfig:
    past_steps: int
    entity_scope: str
    fields: tuple[str, ...]
    float_precision: int


@dataclass(frozen=True)
class DetectorConfig:
    backend: str
    model: str
    revision: str
    served_model_name: str
    connection: ConnectionConfig
    timeout_s: float
    threshold: float
    input_token_budget: int
    question: DetectorQuestionConfig
    window: DetectorWindowConfig


@dataclass(frozen=True)
class TranslationWindowConfig:
    past_steps: int
    renderer: str


@dataclass(frozen=True)
class TranslationConfig:
    backend: str
    model: str
    reasoning_effort: str
    output_language: str
    max_output_tokens: int
    max_intent_chars: int
    timeout_s: float
    window: TranslationWindowConfig


@dataclass(frozen=True)
class Price:
    input: float  # USD per 1M tokens
    cached_input: float | None  # None: priced at the input rate
    output: float


@dataclass(frozen=True)
class TranslatorConfig:
    schema_version: int
    pipeline: PipelineConfig
    limits: LimitsConfig
    detector: DetectorConfig
    translation: TranslationConfig
    output_root: Path
    pricing: tuple[tuple[str, Price], ...]

    def price(self, model: str) -> Price | None:
        return dict(self.pricing).get(model)

    def paid_layers(self) -> tuple[str, ...]:
        """Layers whose backend is a paid API; non-empty means ``--confirm-paid-api`` is required."""
        layers = (("detector", self.detector.backend), ("translation", self.translation.backend))
        return tuple(name for name, backend in layers if backend in PAID_BACKENDS)


# --- primitive checks -----------------------------------------------------------------------------


def _mapping(raw: object, keys: set[str], name: str, error: type[ValueError]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise error(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise error(f"{name} keys must be strings (quote 'true'/'false' keys)")
    if set(raw) != keys:
        missing, unknown = sorted(keys - set(raw)), sorted(set(raw) - keys)
        raise error(f"{name} must have exactly {sorted(keys)} (missing {missing}, unknown {unknown})")
    return raw


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value <= 0:
        raise error(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value < 0:
        raise error(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str, error: type[ValueError]) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise error(f"{name} must be a finite number")
    return float(value)


def _text(value: object, name: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a non-empty string")
    return value


def _safe_id(value: object, name: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not SAFE_ID_PATTERN.fullmatch(value):
        raise error(f"{name} must match {SAFE_ID_PATTERN.pattern}")
    return value


def _phase_choice(value: object, name: str, allowed: tuple[str, ...], later: tuple[str, ...]) -> str:
    if isinstance(value, str) and value in later:
        raise TranslatorConfigError(f"{name} {value!r} {NOT_IN_THIS_PHASE}; use one of {list(allowed)}")
    if not isinstance(value, str) or value not in allowed:
        raise TranslatorConfigError(f"{name} must be one of {list(allowed)}")
    return value


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


def _read_yaml(path: Path, error: type[ValueError]) -> object:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"cannot read {path}: {type(exc).__name__}") from exc
    try:
        return yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass: no arbitrary objects
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None)
        raise error(f"{path} is not valid YAML" + (f": {problem}" if problem else "")) from exc


# --- run config -----------------------------------------------------------------------------------


def _pipeline(raw: object) -> PipelineConfig:
    section = _mapping(raw, PIPELINE_KEYS, "pipeline", TranslatorConfigError)
    visibility = _mapping(section["visibility"], VISIBILITY_KEYS, "pipeline.visibility", TranslatorConfigError)
    mode = _phase_choice(
        visibility["mode"], "pipeline.visibility.mode", (VISIBILITY_ONLINE,), (VISIBILITY_RETROSPECTIVE,)
    )
    future_steps = _nonnegative_int(
        visibility["future_steps"], "pipeline.visibility.future_steps", TranslatorConfigError
    )
    if future_steps != 0:
        raise TranslatorConfigError("pipeline.visibility.future_steps must be 0 in online mode")
    tick = _positive_int(section["tick_interval_steps"], "pipeline.tick_interval_steps", TranslatorConfigError)
    return PipelineConfig(tick_interval_steps=tick, visibility=VisibilityConfig(mode, future_steps))


def _limits(raw: object) -> LimitsConfig:
    section = _mapping(raw, LIMIT_KEYS, "limits", TranslatorConfigError)
    return LimitsConfig(
        **{key: _positive_int(section[key], f"limits.{key}", TranslatorConfigError) for key in LIMIT_KEYS}
    )


def _current_user() -> str:
    try:
        user = getpass.getuser()
    except (OSError, KeyError):  # no login name in the environment or the password database
        raise TranslatorConfigError(f"cannot determine the current user name for {USER_PLACEHOLDER}") from None
    if not USER_NAME_PATTERN.fullmatch(user):
        raise TranslatorConfigError(f"the current user name cannot be used for {USER_PLACEHOLDER}")
    return user


def _uds_path(value: object, name: str) -> Path:
    """Absolute socket path; ``{user}`` becomes the current user name, nothing else is expanded."""
    if not isinstance(value, str) or not value:
        raise TranslatorConfigError(f"{name}.uds_path must be an absolute path")
    if any(marker in value.replace(USER_PLACEHOLDER, "") for marker in UNEXPANDED_MARKERS):
        raise TranslatorConfigError(
            f"{name}.uds_path supports only the placeholder {USER_PLACEHOLDER}; environment variables are not expanded"
        )
    path = Path(value.replace(USER_PLACEHOLDER, _current_user()) if USER_PLACEHOLDER in value else value)
    if not path.is_absolute():
        raise TranslatorConfigError(f"{name}.uds_path must be an absolute path")
    return path


def _connection(raw: object, name: str) -> ConnectionConfig:
    """The System One client's own checks, run at load time: preview rejects what execute rejects."""
    section = _mapping(raw, CONNECTION_KEYS, name, TranslatorConfigError)
    base_url, uds_path, key_env = section["base_url"], section["uds_path"], section["api_key_env"]
    if (base_url is None) == (uds_path is None):
        raise TranslatorConfigError(f"{name}: set exactly one of base_url (loopback TCP) or uds_path")
    try:
        key_env = None if key_env is None else checked_env_name(key_env)
        root_url = None if base_url is None else checked_tcp_url(base_url)
    except SystemOneConfigError as exc:  # fixed messages: never the URL, the name or a value
        raise TranslatorConfigError(f"{name}: {exc}") from None
    if uds_path is not None:
        return ConnectionConfig(base_url=None, uds_path=_uds_path(uds_path, name), api_key_env=key_env)
    if key_env is None:
        raise TranslatorConfigError(f"{name}.api_key_env is required for a TCP endpoint")
    return ConnectionConfig(base_url=root_url, uds_path=None, api_key_env=key_env)


def _question(raw: object) -> DetectorQuestionConfig:
    section = _mapping(raw, QUESTION_KEYS, "detector.question", TranslatorConfigError)
    criteria = _mapping(section["criteria"], CRITERIA_KEYS, "detector.question.criteria", TranslatorConfigError)
    question_type = section["type"]
    if question_type not in QUESTION_TYPES:
        raise TranslatorConfigError(f"detector.question.type must be one of {list(QUESTION_TYPES)}")
    return DetectorQuestionConfig(
        question_id=_safe_id(section["id"], "detector.question.id", TranslatorConfigError),
        question_type=question_type,
        instructions=_text(section["instructions"], "detector.question.instructions", TranslatorConfigError),
        true_criterion=_text(criteria["true"], "detector.question.criteria.true", TranslatorConfigError),
        false_criterion=_text(criteria["false"], "detector.question.criteria.false", TranslatorConfigError),
    )


def _detector_window(raw: object) -> DetectorWindowConfig:
    section = _mapping(raw, DETECTOR_WINDOW_KEYS, "detector.window", TranslatorConfigError)
    fields = section["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
        raise TranslatorConfigError("detector.window.fields must be a non-empty list of column names")
    if len(set(fields)) != len(fields):
        raise TranslatorConfigError("detector.window.fields must be unique")
    bad = sorted(set(fields) - set(COLUMNS) | set(fields) & set(RESERVED_COMPACT_KEYS))
    if bad:
        raise TranslatorConfigError(f"detector.window.fields has unknown or reserved names: {bad}")
    scope = section["entity_scope"]
    if scope not in ENTITY_SCOPES:
        raise TranslatorConfigError(f"detector.window.entity_scope must be one of {list(ENTITY_SCOPES)}")
    return DetectorWindowConfig(
        past_steps=_positive_int(section["past_steps"], "detector.window.past_steps", TranslatorConfigError),
        entity_scope=scope,
        fields=tuple(fields),
        float_precision=_nonnegative_int(
            section["float_precision"], "detector.window.float_precision", TranslatorConfigError
        ),
    )


def _detector(raw: object) -> DetectorConfig:
    section = _mapping(raw, DETECTOR_KEYS, "detector", TranslatorConfigError)
    backend = _phase_choice(section["backend"], "detector.backend", DETECTOR_BACKENDS, ())
    threshold = _number(section["threshold"], "detector.threshold", TranslatorConfigError)
    if not 0 <= threshold <= 1:
        raise TranslatorConfigError("detector.threshold must lie in [0, 1]")
    budget = _positive_int(section["input_token_budget"], "detector.input_token_budget", TranslatorConfigError)
    if budget > MAX_DETECTOR_INPUT_TOKEN_BUDGET:
        raise TranslatorConfigError(
            f"detector.input_token_budget must not exceed {MAX_DETECTOR_INPUT_TOKEN_BUDGET} (the serving cap)"
        )
    timeout_s = _number(section["timeout_s"], "detector.timeout_s", TranslatorConfigError)
    if timeout_s <= 0:
        raise TranslatorConfigError("detector.timeout_s must be positive")
    return DetectorConfig(
        backend=backend,
        model=_text(section["model"], "detector.model", TranslatorConfigError),
        revision=_text(section["revision"], "detector.revision", TranslatorConfigError),
        served_model_name=_text(section["served_model_name"], "detector.served_model_name", TranslatorConfigError),
        connection=_connection(section["connection"], "detector.connection"),
        timeout_s=timeout_s,
        threshold=threshold,
        input_token_budget=budget,
        question=_question(section["question"]),
        window=_detector_window(section["window"]),
    )


def _translation(raw: object) -> TranslationConfig:
    section = _mapping(raw, TRANSLATION_KEYS, "translation", TranslatorConfigError)
    backend = _phase_choice(section["backend"], "translation.backend", TRANSLATION_BACKENDS, (TRANSLATION_VLLM_CHAT,))
    if section["connection"] is not None:
        raise TranslatorConfigError("translation.connection must be null for openai_responses")
    effort = section["reasoning_effort"]
    if not isinstance(effort, str) or effort not in REASONING_EFFORTS:
        raise TranslatorConfigError(f"translation.reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    window = _mapping(section["window"], TRANSLATION_WINDOW_KEYS, "translation.window", TranslatorConfigError)
    renderer = window["renderer"]
    if renderer not in TRANSLATION_RENDERERS:
        raise TranslatorConfigError(f"translation.window.renderer must be one of {list(TRANSLATION_RENDERERS)}")
    timeout_s = _number(section["timeout_s"], "translation.timeout_s", TranslatorConfigError)
    if timeout_s <= 0:
        raise TranslatorConfigError("translation.timeout_s must be positive")
    return TranslationConfig(
        backend=backend,
        model=_text(section["model"], "translation.model", TranslatorConfigError),
        reasoning_effort=effort,
        output_language=_text(section["output_language"], "translation.output_language", TranslatorConfigError),
        max_output_tokens=_positive_int(
            section["max_output_tokens"], "translation.max_output_tokens", TranslatorConfigError
        ),
        max_intent_chars=_positive_int(
            section["max_intent_chars"], "translation.max_intent_chars", TranslatorConfigError
        ),
        timeout_s=timeout_s,
        window=TranslationWindowConfig(
            past_steps=_positive_int(window["past_steps"], "translation.window.past_steps", TranslatorConfigError),
            renderer=renderer,
        ),
    )


def _price(model: str, raw: object) -> Price:
    section = _mapping(raw, PRICE_KEYS, f"pricing.{model}", TranslatorConfigError)
    cached = section["cached_input"]
    prices = {
        "input": _number(section["input"], f"pricing.{model}.input", TranslatorConfigError),
        "output": _number(section["output"], f"pricing.{model}.output", TranslatorConfigError),
        "cached_input": None
        if cached is None
        else _number(cached, f"pricing.{model}.cached_input", TranslatorConfigError),
    }
    if any(value is not None and value < 0 for value in prices.values()):
        raise TranslatorConfigError(f"pricing.{model} values must be non-negative")
    return Price(**prices)


def _pricing(raw: object) -> tuple[tuple[str, Price], ...]:
    if not isinstance(raw, dict) or not raw:
        raise TranslatorConfigError("pricing must be a non-empty mapping of model id -> prices")
    return tuple(
        (_text(model, "pricing model id", TranslatorConfigError), _price(model, prices))
        for model, prices in raw.items()
    )


def _output_root(raw: object, root: Path) -> Path:
    section = _mapping(raw, OUTPUT_KEYS, "output", TranslatorConfigError)
    value = Path(_text(section["root"], "output.root", TranslatorConfigError))
    return value if value.is_absolute() else root / value


def parse_config(raw: object, *, root: Path = paths.ROOT) -> TranslatorConfig:
    """Validate a decoded YAML mapping into a frozen ``TranslatorConfig``."""
    section = _mapping(raw, TOP_KEYS, "config", TranslatorConfigError)
    if section["schema_version"] != CONFIG_SCHEMA_VERSION or isinstance(section["schema_version"], bool):
        raise TranslatorConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    config = TranslatorConfig(
        schema_version=CONFIG_SCHEMA_VERSION,
        pipeline=_pipeline(section["pipeline"]),
        limits=_limits(section["limits"]),
        detector=_detector(section["detector"]),
        translation=_translation(section["translation"]),
        output_root=_output_root(section["output"], root),
        pricing=_pricing(section["pricing"]),
    )
    if config.translation.backend in PAID_BACKENDS and config.price(config.translation.model) is None:
        raise TranslatorConfigError(f"pricing has no entry for translation.model {config.translation.model!r}")
    return config


def load_config(path: Path, *, root: Path = paths.ROOT) -> TranslatorConfig:
    return parse_config(_read_yaml(path, TranslatorConfigError), root=root)


# --- scenario -------------------------------------------------------------------------------------


def _target(raw: object, name: str) -> TargetRef:
    section = _mapping(raw, TARGET_KEYS, name, ScenarioError)
    if section["kind"] not in TARGET_KINDS:
        raise ScenarioError(f"{name}.kind must be one of {list(TARGET_KINDS)}")
    return TargetRef(kind=section["kind"], target_id=_safe_id(section["id"], f"{name}.id", ScenarioError))


def _list(raw: object, name: str) -> list[Any]:
    if not isinstance(raw, list):
        raise ScenarioError(f"{name} must be a list (use [] for none)")
    return raw


def _episode(raw: object, base: Path) -> EpisodeRef:
    section = _mapping(raw, EPISODE_KEYS, "episode", ScenarioError)
    episode_id = _text(section["episode_id"], "episode.episode_id", ScenarioError)
    alias = _safe_id(section["alias"], "episode.alias", ScenarioError)
    if alias == episode_id:
        raise ScenarioError("episode.alias must differ from episode_id (the id never reaches a model)")
    video = section["video_path"]
    return EpisodeRef(
        episode_id=episode_id,
        alias=alias,
        csv_path=(base / _text(section["csv_path"], "episode.csv_path", ScenarioError)).resolve(),
        video_path=None if video is None else (base / _text(video, "episode.video_path", ScenarioError)).resolve(),
    )


def _instruction(raw: object, index: int) -> Instruction:
    name = f"instructions[{index}]"
    section = _mapping(raw, INSTRUCTION_KEYS, name, ScenarioError)
    return Instruction(
        instruction_id=_safe_id(section["instruction_id"], f"{name}.instruction_id", ScenarioError),
        issue_step=_nonnegative_int(section["issue_step"], f"{name}.issue_step", ScenarioError),
        target=_target(section["target"], f"{name}.target"),
        text=_text(section["text"], f"{name}.text", ScenarioError),
    )


def _request(raw: object, index: int) -> RequestPoint:
    name = f"requests[{index}]"
    section = _mapping(raw, REQUEST_KEYS, name, ScenarioError)
    items = _list(section["targets"], f"{name}.targets")
    targets = tuple(_target(item, f"{name}.targets[{i}]") for i, item in enumerate(items))
    if not targets or len(set(targets)) != len(targets):
        raise ScenarioError(f"{name}.targets must be a non-empty list of distinct targets")
    return RequestPoint(
        request_id=_safe_id(section["request_id"], f"{name}.request_id", ScenarioError),
        step=_nonnegative_int(section["step"], f"{name}.step", ScenarioError),
        targets=targets,
        source=_safe_id(section["source"], f"{name}.source", ScenarioError),
    )


def _change(raw: object, index: int) -> ExternalChangePoint:
    name = f"external_change_points[{index}]"
    section = _mapping(raw, CHANGE_KEYS, name, ScenarioError)
    return ExternalChangePoint(
        change_id=_safe_id(section["change_id"], f"{name}.change_id", ScenarioError),
        step=_nonnegative_int(section["step"], f"{name}.step", ScenarioError),
        target=_target(section["target"], f"{name}.target"),
        source=_safe_id(section["source"], f"{name}.source", ScenarioError),
    )


def check_scenario_rules(scenario: Scenario) -> None:
    """Rules that need no episode: ids, step-0 start, one instruction per target per step, references.

    ``parse_scenario`` and ``pipeline.validate_scenario`` both run it, so a ``Scenario`` built in
    code obeys the same rules as a loaded one (and ``plan_schedule`` matches the run).
    """
    if not scenario.instructions:
        raise ScenarioError("instructions must not be empty")
    ids = [
        *(item.instruction_id for item in scenario.instructions),
        *(item.request_id for item in scenario.requests),
        *(item.change_id for item in scenario.external_change_points),
    ]
    if len(set(ids)) != len(ids):
        raise ScenarioError("instruction, request and change ids must be unique across the scenario")
    if not any(item.issue_step == 0 for item in scenario.instructions):
        raise ScenarioError("at least one instruction must be issued at step 0 (the start point)")
    keys = [(item.target, item.issue_step) for item in scenario.instructions]
    if len(set(keys)) != len(keys):
        raise ScenarioError("at most one instruction per exact target per step")
    targets = {item.target for item in scenario.instructions}
    first_step = {t: min(i.issue_step for i in scenario.instructions if i.target == t) for t in targets}
    references = [*((r.request_id, t, r.step) for r in scenario.requests for t in r.targets)]
    references += [(c.change_id, c.target, c.step) for c in scenario.external_change_points]
    for reference_id, target, step in references:
        if target not in first_step or first_step[target] > step:
            raise ScenarioError(
                f"{reference_id} references {target.label}, which has no instruction at or before {step}"
            )


def parse_scenario(raw: object, *, base: Path) -> Scenario:
    section = _mapping(raw, SCENARIO_KEYS, "scenario", ScenarioError)
    if section["schema_version"] != SCENARIO_SCHEMA_VERSION or isinstance(section["schema_version"], bool):
        raise ScenarioError(f"scenario schema_version must be {SCENARIO_SCHEMA_VERSION}")
    instructions = _list(section["instructions"], "instructions")
    if not instructions:
        raise ScenarioError("instructions must not be empty")
    scenario = Scenario(
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_id=_safe_id(section["scenario_id"], "scenario_id", ScenarioError),
        episode=_episode(section["episode"], base),
        instructions=tuple(_instruction(item, i) for i, item in enumerate(instructions)),
        requests=tuple(_request(item, i) for i, item in enumerate(_list(section["requests"], "requests"))),
        external_change_points=tuple(
            _change(item, i)
            for i, item in enumerate(_list(section["external_change_points"], "external_change_points"))
        ),
    )
    check_scenario_rules(scenario)
    return scenario


def load_scenario(path: Path) -> Scenario:
    """Load a scenario; ``csv_path`` / ``video_path`` resolve relative to the scenario file."""
    path = Path(path)
    return parse_scenario(_read_yaml(path, ScenarioError), base=path.resolve().parent)
