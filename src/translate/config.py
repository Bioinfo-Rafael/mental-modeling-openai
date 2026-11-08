"""Strict run config (schema_version 2): what to translate when, from which inputs, with which prompt.

One file per run configuration (``configs/translate/*.yaml``); every key is required and explicit,
unknown keys fail, and there is no default model in code. Sections:

- ``scenario``: the scenario file (repository-relative; the CLI's ``--scenario`` overrides it);
- ``triggers``: ``online`` (forced ``start`` at the episode start / ``request`` + the clef-flash detector every
  ``tick_interval_steps``) or ``preset`` (``points``: steps chosen in advance, each with its kind);
- ``inputs``: the selected inputs; ``trajectory`` (the window every event records for the judge,
  inserted into the prompt when selected) and ``policy`` (rollout, spec, window) configure them;
- ``prompt``: the prompt file and its ``vars``; its protocol decides the call (one Responses
  request, or a code session configured by ``session``);
- ``translation`` (model and request options), ``answer`` (limits, ``on_invalid``: ``stop`` | ``skip``),
  ``limits`` (call caps), ``scenario_limits``, ``detector``, ``output``, ``pricing``.

The prompt's placeholders are checked here against the values the selected inputs provide, and
every selected input must be used by the prompt, so a config cannot silently drop an input.
"""

from __future__ import annotations

import getpass
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translate import paths
from translate.answers import AnswerLimits
from translate.events import DETECTOR_LABELS, TARGET_KINDS, TRIGGER_PRECEDENCE, TargetRef
from translate.lag.dataset import COLUMNS
from translate.lag.rendering import LAG17_RENDERER, RESERVED_COMPACT_KEYS
from translate.llm.openai_responses import REASONING_EFFORTS
from translate.llm.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from translate.llm.system_one import SystemOneConfigError, checked_env_name, checked_tcp_url
from translate.prompts import PROTOCOL_CODE_SESSION, PromptTemplate, load_prompt
from translate.scenario import ScenarioLimitsConfig
from translate.strict_yaml import (
    is_int,
    mapping,
    nonnegative_int,
    number,
    positive_int,
    read_yaml,
    safe_id,
    text,
)

CONFIG_SCHEMA_VERSION = 2
VISIBILITY_ONLINE = "online"
TRIGGER_SOURCE_ONLINE = "online"
TRIGGER_SOURCE_PRESET = "preset"
TRIGGER_SOURCES = (TRIGGER_SOURCE_ONLINE, TRIGGER_SOURCE_PRESET)
DETECTOR_BACKENDS = ("system_one",)
TRANSLATION_OPENAI = "openai_responses"
TRANSLATION_BACKENDS = (TRANSLATION_OPENAI,)
RETRY_DEFAULT = "default"  # the shared narrow retry of translate.llm.retry
RETRY_NONE = "none"
NO_RETRY_POLICY = RetryPolicy(max_retries=0)  # every request is sent once; a 5xx is reported, not resent
ON_INVALID_STOP = "stop"
ON_INVALID_SKIP = "skip"
ENTITY_SCOPES = ("target_and_opponents",)
QUESTION_TYPES = ("choice",)  # one choice question over DETECTOR_LABELS (none / change)
TRAJECTORY_RENDERERS = (LAG17_RENDERER,)
# clef-flash serves at most 16384 tokens; the budget must equal min(16384, --max-model-len).
MAX_DETECTOR_INPUT_TOKEN_BUDGET = 16384
USER_PLACEHOLDER = "{user}"  # the only placeholder in uds_path; no environment variable is expanded
USER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}")  # used with fullmatch only
UNEXPANDED_MARKERS = ("{", "}", "$")

INPUT_TRAJECTORY = "trajectory"
INPUT_POLICY = "policy"
# What each selectable input provides to the prompt (src/translate/inputs.py fills these).
INPUT_VALUES = {
    "instructions": {"instructions", "primary_instruction"},
    "alarms": {"alarms"},
    "question": {"question"},
    "previous_translation": {"previous_translation"},
    INPUT_TRAJECTORY: {"trajectory", "trajectory_start", "trajectory_end", "trajectory_points", "trajectory_anchor"},
    INPUT_POLICY: {
        "policy_alias",
        "policy_task",
        "policy_architecture",
        "policy_inputs",
        "policy_outputs",
        "policy_parameters",
        "policy_activations",
        "window_start",
        "window_end",
        "window_length",
    },
}
BLOCK_INPUTS = ("instructions", "alarms", "question", "previous_translation")  # formatted by the prompt's blocks
BASE_VALUES = {
    "episode_alias",
    "step",
    "time_s",
    "t_s",
    "step_seconds",
    "target_label",
    "target_aircraft",
    "max_intent_chars",
}
SESSION_VALUES = {"max_runs", "timeout_s", "stdout_cap_chars", "stderr_tail_chars"}

TOP_KEYS = {
    "schema_version",
    "scenario",
    "triggers",
    "visibility",
    "limits",
    "scenario_limits",
    "detector",
    "inputs",
    "trajectory",
    "policy",
    "prompt",
    "translation",
    "answer",
    "session",
    "output",
    "pricing",
}
TRIGGER_KEYS = {"source", "tick_interval_steps", "points"}
POINT_KEYS = {"step", "kind", "targets"}
VISIBILITY_KEYS = {"mode", "future_steps"}
LIMIT_KEYS = {"max_detector_calls", "max_translation_calls"}
SCENARIO_LIMIT_KEYS = {"max_instruction_chars", "max_alarm_chars", "max_question_chars"}
DETECTOR_KEYS = {
    "backend",
    "model",
    "revision",
    "served_model_name",
    "connection",
    "timeout_s",
    "threshold",
    "input_token_budget",
    "messages_lookback_steps",
    "question",
    "window",
}
CONNECTION_KEYS = {"base_url", "uds_path", "api_key_env"}
QUESTION_KEYS = {"id", "type", "instructions", "criteria"}
CRITERIA_KEYS = set(DETECTOR_LABELS)
DETECTOR_WINDOW_KEYS = {"past_steps", "entity_scope", "fields", "float_precision"}
TRAJECTORY_KEYS = {"renderer", "past_steps"}
POLICY_KEYS = {"rollout_dir", "seed", "spec", "checkpoint", "past_steps"}
PROMPT_KEYS = {"file", "vars"}
TRANSLATION_KEYS = {"backend", "model", "reasoning_effort", "max_output_tokens", "timeout_s", "retry"}
ANSWER_KEYS = {"max_intent_chars", "max_evidence_items", "max_evidence_chars", "on_invalid"}
SESSION_KEYS = {"max_runs", "max_total_tokens", "max_protocol_errors", "answer_at_fraction", "sandbox"}
SANDBOX_KEYS = {
    "python",
    "timeout_s",
    "memory_bytes",
    "file_size_bytes",
    "stdout_cap_chars",
    "stderr_tail_chars",
    "tmp_size_mb",
    "threads",
    "work_max_bytes",
}
OUTPUT_KEYS = {"root"}
PRICE_KEYS = {"input", "cached_input", "output"}


class ConfigError(ValueError):
    """The run configuration is malformed or inconsistent."""


@dataclass(frozen=True)
class PresetPoint:
    step: int
    kind: str  # start | change | request
    targets: tuple[TargetRef, ...] | None  # None: every translation target (instructed or asked about)


@dataclass(frozen=True)
class TriggerConfig:
    source: str  # online | preset
    tick_interval_steps: int | None  # online only
    points: tuple[PresetPoint, ...]  # preset only


@dataclass(frozen=True)
class VisibilityConfig:
    mode: str
    future_steps: int


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
    criteria: tuple[tuple[str, str], ...]  # (label, description) in DETECTOR_LABELS order

    def to_json(self) -> dict[str, Any]:
        """The full question object, as logged in every ``detector_call`` record."""
        return {
            "id": self.question_id,
            "type": self.question_type,
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
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
    messages_lookback_steps: int  # alarms and questions in [t - N + 1, t] go into the state
    question: DetectorQuestionConfig
    window: DetectorWindowConfig


@dataclass(frozen=True)
class TrajectoryConfig:
    renderer: str
    past_steps: int


@dataclass(frozen=True)
class PolicyConfig:
    rollout_dir: Path
    seed: int
    spec: Path
    checkpoint: Path | None  # None: the actor named in the rollout manifest
    past_steps: int


@dataclass(frozen=True)
class TranslationConfig:
    backend: str
    model: str
    reasoning_effort: str
    max_output_tokens: int
    timeout_s: float
    retry: RetryPolicy  # ``retry: none`` is the zero-retry policy (NO_RETRY_POLICY)


@dataclass(frozen=True)
class AnswerConfig:
    limits: AnswerLimits
    on_invalid: str  # stop | skip


@dataclass(frozen=True)
class SandboxConfig:
    python: Path
    timeout_s: int
    memory_bytes: int
    file_size_bytes: int
    stdout_cap_chars: int
    stderr_tail_chars: int
    tmp_size_mb: int
    threads: int
    work_max_bytes: int


@dataclass(frozen=True)
class SessionConfig:
    max_runs: int
    max_total_tokens: int
    max_protocol_errors: int
    answer_at_fraction: float  # once this share of max_total_tokens is used, the answer is requested
    sandbox: SandboxConfig


@dataclass(frozen=True)
class Price:
    input: float  # USD per 1M tokens
    cached_input: float | None  # None: priced at the input rate
    output: float


@dataclass(frozen=True)
class RunConfig:
    path: Path
    sha256: str  # of the config file's bytes
    scenario_path: Path | None
    triggers: TriggerConfig
    visibility: VisibilityConfig
    limits: LimitsConfig
    scenario_limits: ScenarioLimitsConfig
    detector: DetectorConfig | None
    inputs: tuple[str, ...]
    trajectory: TrajectoryConfig
    policy: PolicyConfig | None
    prompt: PromptTemplate
    translation: TranslationConfig
    answer: AnswerConfig
    session: SessionConfig | None
    output_root: Path
    pricing: tuple[tuple[str, Price], ...]

    def price(self, model: str) -> Price | None:
        return dict(self.pricing).get(model)

    @property
    def online(self) -> bool:
        return self.triggers.source == TRIGGER_SOURCE_ONLINE

    @property
    def code_session(self) -> bool:
        return self.prompt.protocol == PROTOCOL_CODE_SESSION


# --- sections -------------------------------------------------------------------------------------


def _choice(value: object, name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(f"{name} must be one of {list(allowed)}")
    return value


def _path(value: object, name: str, root: Path) -> Path:
    path = Path(text(value, name, ConfigError))
    return path if path.is_absolute() else root / path


def _target(raw: object, name: str) -> TargetRef:
    section = mapping(raw, {"kind", "id"}, name, ConfigError)
    kind = _choice(section["kind"], f"{name}.kind", TARGET_KINDS)
    return TargetRef(kind=kind, target_id=safe_id(section["id"], f"{name}.id", ConfigError))


def _point(raw: object, index: int) -> PresetPoint:
    name = f"triggers.points[{index}]"
    section = mapping(raw, POINT_KEYS, name, ConfigError)
    targets = section["targets"]
    if targets is not None and (not isinstance(targets, list) or not targets):
        raise ConfigError(f"{name}.targets must be null (every target) or a non-empty list")
    return PresetPoint(
        step=nonnegative_int(section["step"], f"{name}.step", ConfigError),
        kind=_choice(section["kind"], f"{name}.kind", TRIGGER_PRECEDENCE),
        targets=None
        if targets is None
        else tuple(_target(item, f"{name}.targets[{i}]") for i, item in enumerate(targets)),
    )


def _triggers(raw: object) -> TriggerConfig:
    section = mapping(raw, TRIGGER_KEYS, "triggers", ConfigError)
    source = _choice(section["source"], "triggers.source", TRIGGER_SOURCES)
    if source == TRIGGER_SOURCE_ONLINE:
        if section["points"] is not None:
            raise ConfigError("triggers.points must be null for online triggers")
        tick = positive_int(section["tick_interval_steps"], "triggers.tick_interval_steps", ConfigError)
        return TriggerConfig(source, tick, ())
    if section["tick_interval_steps"] is not None:
        raise ConfigError("triggers.tick_interval_steps must be null for preset triggers")
    raw_points = section["points"]
    if not isinstance(raw_points, list) or not raw_points:
        raise ConfigError("triggers.points must be a non-empty list for preset triggers")
    points = tuple(_point(item, index) for index, item in enumerate(raw_points))
    keys = [(point.step, point.kind, point.targets) for point in points]
    if len(set(keys)) != len(keys):
        raise ConfigError("triggers.points must not repeat a point")
    return TriggerConfig(source, None, points)


def _visibility(raw: object) -> VisibilityConfig:
    section = mapping(raw, VISIBILITY_KEYS, "visibility", ConfigError)
    mode = _choice(section["mode"], "visibility.mode", (VISIBILITY_ONLINE,))
    future_steps = nonnegative_int(section["future_steps"], "visibility.future_steps", ConfigError)
    if future_steps != 0:
        raise ConfigError("visibility.future_steps must be 0 in online mode")
    return VisibilityConfig(mode, future_steps)


def _limits(raw: object) -> LimitsConfig:
    section = mapping(raw, LIMIT_KEYS, "limits", ConfigError)
    return LimitsConfig(**{key: positive_int(section[key], f"limits.{key}", ConfigError) for key in LIMIT_KEYS})


def _scenario_limits(raw: object) -> ScenarioLimitsConfig:
    section = mapping(raw, SCENARIO_LIMIT_KEYS, "scenario_limits", ConfigError)
    return ScenarioLimitsConfig(
        **{key: positive_int(section[key], f"scenario_limits.{key}", ConfigError) for key in SCENARIO_LIMIT_KEYS}
    )


def _current_user() -> str:
    try:
        user = getpass.getuser()
    except (OSError, KeyError):  # no login name in the environment or the password database
        raise ConfigError(f"cannot determine the current user name for {USER_PLACEHOLDER}") from None
    if not USER_NAME_PATTERN.fullmatch(user):
        raise ConfigError(f"the current user name cannot be used for {USER_PLACEHOLDER}")
    return user


def _uds_path(value: object, name: str) -> Path:
    """Absolute socket path; ``{user}`` becomes the current user name, nothing else is expanded."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name}.uds_path must be an absolute path")
    if any(marker in value.replace(USER_PLACEHOLDER, "") for marker in UNEXPANDED_MARKERS):
        raise ConfigError(
            f"{name}.uds_path supports only the placeholder {USER_PLACEHOLDER}; environment variables are not expanded"
        )
    path = Path(value.replace(USER_PLACEHOLDER, _current_user()) if USER_PLACEHOLDER in value else value)
    if not path.is_absolute():
        raise ConfigError(f"{name}.uds_path must be an absolute path")
    return path


def _connection(raw: object, name: str) -> ConnectionConfig:
    """The System One client's own checks, run at load time: preview rejects what execute rejects."""
    section = mapping(raw, CONNECTION_KEYS, name, ConfigError)
    base_url, uds_path, key_env = section["base_url"], section["uds_path"], section["api_key_env"]
    if (base_url is None) == (uds_path is None):
        raise ConfigError(f"{name}: set exactly one of base_url (loopback TCP) or uds_path")
    try:
        key_env = None if key_env is None else checked_env_name(key_env)
        root_url = None if base_url is None else checked_tcp_url(base_url)
    except SystemOneConfigError as exc:  # fixed messages: never the URL, the name or a value
        raise ConfigError(f"{name}: {exc}") from None
    if uds_path is not None:
        return ConnectionConfig(base_url=None, uds_path=_uds_path(uds_path, name), api_key_env=key_env)
    if key_env is None:
        raise ConfigError(f"{name}.api_key_env is required for a TCP endpoint")
    return ConnectionConfig(base_url=root_url, uds_path=None, api_key_env=key_env)


def _question(raw: object) -> DetectorQuestionConfig:
    section = mapping(raw, QUESTION_KEYS, "detector.question", ConfigError)
    criteria = mapping(section["criteria"], CRITERIA_KEYS, "detector.question.criteria", ConfigError)
    question_type = section["type"]
    if question_type not in QUESTION_TYPES:
        raise ConfigError(f"detector.question.type must be one of {list(QUESTION_TYPES)}")
    return DetectorQuestionConfig(
        question_id=safe_id(section["id"], "detector.question.id", ConfigError),
        question_type=question_type,
        instructions=text(section["instructions"], "detector.question.instructions", ConfigError),
        criteria=tuple(
            (label, text(criteria[label], f"detector.question.criteria.{label}", ConfigError))
            for label in DETECTOR_LABELS
        ),
    )


def _detector_window(raw: object) -> DetectorWindowConfig:
    section = mapping(raw, DETECTOR_WINDOW_KEYS, "detector.window", ConfigError)
    fields = section["fields"]
    if not isinstance(fields, list) or not fields or not all(isinstance(field, str) for field in fields):
        raise ConfigError("detector.window.fields must be a non-empty list of column names")
    if len(set(fields)) != len(fields):
        raise ConfigError("detector.window.fields must be unique")
    bad = sorted(set(fields) - set(COLUMNS) | set(fields) & set(RESERVED_COMPACT_KEYS))
    if bad:
        raise ConfigError(f"detector.window.fields has unknown or reserved names: {bad}")
    scope = section["entity_scope"]
    if scope not in ENTITY_SCOPES:
        raise ConfigError(f"detector.window.entity_scope must be one of {list(ENTITY_SCOPES)}")
    return DetectorWindowConfig(
        past_steps=positive_int(section["past_steps"], "detector.window.past_steps", ConfigError),
        entity_scope=scope,
        fields=tuple(fields),
        float_precision=nonnegative_int(section["float_precision"], "detector.window.float_precision", ConfigError),
    )


def _detector(raw: object) -> DetectorConfig:
    section = mapping(raw, DETECTOR_KEYS, "detector", ConfigError)
    backend = _choice(section["backend"], "detector.backend", DETECTOR_BACKENDS)
    threshold = number(section["threshold"], "detector.threshold", ConfigError)
    if not 0 <= threshold <= 1:
        raise ConfigError("detector.threshold must lie in [0, 1]")
    budget = positive_int(section["input_token_budget"], "detector.input_token_budget", ConfigError)
    if budget > MAX_DETECTOR_INPUT_TOKEN_BUDGET:
        raise ConfigError(
            f"detector.input_token_budget must not exceed {MAX_DETECTOR_INPUT_TOKEN_BUDGET} (the serving cap)"
        )
    timeout_s = number(section["timeout_s"], "detector.timeout_s", ConfigError)
    if timeout_s <= 0:
        raise ConfigError("detector.timeout_s must be positive")
    return DetectorConfig(
        backend=backend,
        model=text(section["model"], "detector.model", ConfigError),
        revision=text(section["revision"], "detector.revision", ConfigError),
        served_model_name=text(section["served_model_name"], "detector.served_model_name", ConfigError),
        connection=_connection(section["connection"], "detector.connection"),
        timeout_s=timeout_s,
        threshold=threshold,
        input_token_budget=budget,
        messages_lookback_steps=positive_int(
            section["messages_lookback_steps"], "detector.messages_lookback_steps", ConfigError
        ),
        question=_question(section["question"]),
        window=_detector_window(section["window"]),
    )


def _price(model: str, raw: object) -> Price:
    section = mapping(raw, PRICE_KEYS, f"pricing.{model}", ConfigError)
    cached = section["cached_input"]
    prices = {
        "input": number(section["input"], f"pricing.{model}.input", ConfigError),
        "output": number(section["output"], f"pricing.{model}.output", ConfigError),
        "cached_input": None if cached is None else number(cached, f"pricing.{model}.cached_input", ConfigError),
    }
    if any(value is not None and value < 0 for value in prices.values()):
        raise ConfigError(f"pricing.{model} values must be non-negative")
    return Price(**prices)


def _pricing(raw: object) -> tuple[tuple[str, Price], ...]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("pricing must be a non-empty mapping of model id -> prices")
    return tuple((text(model, "pricing model id", ConfigError), _price(model, prices)) for model, prices in raw.items())


def _inputs(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise ConfigError(f"inputs must be a non-empty list of {sorted(INPUT_VALUES)}")
    unknown = sorted(set(raw) - set(INPUT_VALUES))
    if unknown or len(set(raw)) != len(raw):
        raise ConfigError(f"inputs must be distinct names from {sorted(INPUT_VALUES)} (unknown: {unknown})")
    return tuple(raw)


def _trajectory(raw: object) -> TrajectoryConfig:
    section = mapping(raw, TRAJECTORY_KEYS, "trajectory", ConfigError)
    return TrajectoryConfig(
        renderer=_choice(section["renderer"], "trajectory.renderer", TRAJECTORY_RENDERERS),
        past_steps=positive_int(section["past_steps"], "trajectory.past_steps", ConfigError),
    )


def _policy(raw: object, root: Path) -> PolicyConfig | None:
    if raw is None:
        return None
    section = mapping(raw, POLICY_KEYS, "policy", ConfigError)
    checkpoint = section["checkpoint"]
    return PolicyConfig(
        rollout_dir=_path(section["rollout_dir"], "policy.rollout_dir", root),
        seed=nonnegative_int(section["seed"], "policy.seed", ConfigError),
        spec=_path(section["spec"], "policy.spec", root),
        checkpoint=None if checkpoint is None else _path(checkpoint, "policy.checkpoint", root),
        past_steps=positive_int(section["past_steps"], "policy.past_steps", ConfigError),
    )


def _prompt(raw: object, root: Path) -> PromptTemplate:
    section = mapping(raw, PROMPT_KEYS, "prompt", ConfigError)
    prompt_vars = section["vars"]
    if not isinstance(prompt_vars, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in prompt_vars.items()
    ):
        raise ConfigError("prompt.vars must map names to strings ({} for none)")
    return load_prompt(_path(section["file"], "prompt.file", root), prompt_vars)


def _translation(raw: object) -> TranslationConfig:
    section = mapping(raw, TRANSLATION_KEYS, "translation", ConfigError)
    timeout_s = number(section["timeout_s"], "translation.timeout_s", ConfigError)
    if timeout_s <= 0:
        raise ConfigError("translation.timeout_s must be positive")
    retry = _choice(section["retry"], "translation.retry", (RETRY_DEFAULT, RETRY_NONE))
    return TranslationConfig(
        backend=_choice(section["backend"], "translation.backend", TRANSLATION_BACKENDS),
        model=text(section["model"], "translation.model", ConfigError),
        reasoning_effort=_choice(
            section["reasoning_effort"], "translation.reasoning_effort", tuple(sorted(REASONING_EFFORTS))
        ),
        max_output_tokens=positive_int(section["max_output_tokens"], "translation.max_output_tokens", ConfigError),
        timeout_s=timeout_s,
        retry=DEFAULT_RETRY_POLICY if retry == RETRY_DEFAULT else NO_RETRY_POLICY,
    )


def _optional_positive(value: object, name: str) -> int | None:
    return None if value is None else positive_int(value, name, ConfigError)


def _answer(raw: object) -> AnswerConfig:
    section = mapping(raw, ANSWER_KEYS, "answer", ConfigError)
    limits = AnswerLimits(
        max_intent_chars=positive_int(section["max_intent_chars"], "answer.max_intent_chars", ConfigError),
        max_evidence_items=_optional_positive(section["max_evidence_items"], "answer.max_evidence_items"),
        max_evidence_chars=_optional_positive(section["max_evidence_chars"], "answer.max_evidence_chars"),
    )
    return AnswerConfig(limits, _choice(section["on_invalid"], "answer.on_invalid", (ON_INVALID_STOP, ON_INVALID_SKIP)))


def _session(raw: object, root: Path) -> SessionConfig | None:
    if raw is None:
        return None
    section = mapping(raw, SESSION_KEYS, "session", ConfigError)
    sandbox = mapping(section["sandbox"], SANDBOX_KEYS, "session.sandbox", ConfigError)
    fraction = number(section["answer_at_fraction"], "session.answer_at_fraction", ConfigError)
    if not 0 < fraction <= 1:
        raise ConfigError("session.answer_at_fraction must lie in (0, 1]")
    ints = {key: positive_int(sandbox[key], f"session.sandbox.{key}", ConfigError) for key in SANDBOX_KEYS - {"python"}}
    return SessionConfig(
        max_runs=positive_int(section["max_runs"], "session.max_runs", ConfigError),
        max_total_tokens=positive_int(section["max_total_tokens"], "session.max_total_tokens", ConfigError),
        max_protocol_errors=positive_int(section["max_protocol_errors"], "session.max_protocol_errors", ConfigError),
        answer_at_fraction=fraction,
        sandbox=SandboxConfig(python=_path(sandbox["python"], "session.sandbox.python", root), **ints),
    )


def _output_root(raw: object, root: Path) -> Path:
    section = mapping(raw, OUTPUT_KEYS, "output", ConfigError)
    return _path(section["root"], "output.root", root)


def _check_prompt(config: RunConfig) -> None:
    """The prompt uses exactly the selected inputs, and needs no value nobody provides."""
    prompt = config.prompt
    available = BASE_VALUES | set(prompt.vars) | {value for name in config.inputs for value in INPUT_VALUES[name]}
    if config.code_session:
        available |= SESSION_VALUES
    missing = sorted(prompt.placeholders() - available)
    if missing:
        raise ConfigError(f"prompt {prompt.path.name} needs {missing}, which no selected input or prompt.vars provides")
    unused = [name for name in config.inputs if not INPUT_VALUES[name] & prompt.placeholders()]
    if unused:
        raise ConfigError(f"inputs {unused} are selected but prompt {prompt.path.name} does not use them")
    used = prompt.placeholders()
    unformatted = [
        name for name in config.inputs if name in BLOCK_INPUTS and name in used and name not in prompt.blocks
    ]
    if unformatted:
        raise ConfigError(f"prompt {prompt.path.name} has no blocks entry for {unformatted}")
    if "primary_instruction" in used and "primary_instruction" not in prompt.blocks:  # a start may have none
        raise ConfigError(f"prompt {prompt.path.name} uses primary_instruction but has no primary_instruction block")


def _check_consistency(config: RunConfig) -> None:
    if config.online != (config.detector is not None):
        raise ConfigError("detector is required for online triggers and must be null for preset triggers")
    if config.online and config.detector.messages_lookback_steps < config.triggers.tick_interval_steps:
        raise ConfigError(
            "detector.messages_lookback_steps must be >= triggers.tick_interval_steps, "
            "so every alarm and question reaches a detector call"
        )
    if (INPUT_POLICY in config.inputs) != (config.policy is not None):
        raise ConfigError("policy is required when the policy input is selected, and must be null otherwise")
    if config.code_session != (config.session is not None) or config.code_session != (INPUT_POLICY in config.inputs):
        raise ConfigError("a code-session prompt needs the policy input and a session section; other prompts neither")
    if config.price(config.translation.model) is None:
        raise ConfigError(f"pricing has no entry for translation.model {config.translation.model!r}")
    _check_prompt(config)


def parse_config(raw: object, *, path: Path, sha256: str, root: Path = paths.ROOT) -> RunConfig:
    """Validate a decoded YAML mapping into a frozen ``RunConfig``."""
    section = mapping(raw, TOP_KEYS, "config", ConfigError)
    if section["schema_version"] != CONFIG_SCHEMA_VERSION or not is_int(section["schema_version"]):
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    scenario = section["scenario"]
    config = RunConfig(
        path=Path(path),
        sha256=sha256,
        scenario_path=None if scenario is None else _path(scenario, "scenario", root),
        triggers=_triggers(section["triggers"]),
        visibility=_visibility(section["visibility"]),
        limits=_limits(section["limits"]),
        scenario_limits=_scenario_limits(section["scenario_limits"]),
        detector=None if section["detector"] is None else _detector(section["detector"]),
        inputs=_inputs(section["inputs"]),
        trajectory=_trajectory(section["trajectory"]),
        policy=_policy(section["policy"], root),
        prompt=_prompt(section["prompt"], root),
        translation=_translation(section["translation"]),
        answer=_answer(section["answer"]),
        session=_session(section["session"], root),
        output_root=_output_root(section["output"], root),
        pricing=_pricing(section["pricing"]),
    )
    _check_consistency(config)
    return config


def load_config(path: Path, *, root: Path = paths.ROOT) -> RunConfig:
    data = Path(path).read_bytes()
    raw = read_yaml(path, ConfigError)
    return parse_config(raw, path=Path(path), sha256=hashlib.sha256(data).hexdigest(), root=root)
