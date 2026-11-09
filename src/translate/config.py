"""Run config and scenario: pydantic models with defaults; unknown keys and duplicate YAML keys fail.

A run config names the scenario, the change source (``preset`` steps or the ``clef_flash``
detector), the selected inputs with their options, the inline prompt, the model and the limits.
Errors name the key path. Paths in a run config are repository-relative; the episode path in a
scenario is relative to the scenario file.
"""

from __future__ import annotations

import getpass
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from translate.inputs import INPUTS
from translate.journal import repo_path
from translate.prompts import BASE_VALUES, placeholders

SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
TARGET_ID = r"^(aircraft|team):[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
SERVING_CAP = 16384  # clef-flash serves at most 16384 tokens; the budget must equal the server's max-model-len


class ConfigError(ValueError):
    """A config or scenario file is malformed; the message names the key path."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe loader that rejects a repeated mapping key (PyYAML would keep the last one silently)."""


def _mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    keys = [loader.construct_object(key) for key, _ in node.value]
    repeated = sorted({str(key) for key in keys if keys.count(key) > 1})
    if repeated:
        raise yaml.constructor.ConstructorError(None, None, f"duplicate key {repeated}", node.start_mark)
    return loader.construct_mapping(node)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_yaml(path: Path) -> Any:
    """One YAML document; a duplicate key fails, naming the file."""
    try:
        return yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)  # a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {getattr(exc, 'problem', exc)}") from None


def validated(model: type[BaseModel], raw: object, where: str) -> Any:
    """``model.model_validate(raw)``; a failure becomes a ``ConfigError`` naming every bad key path."""
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '(top)'}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{where}: {problems}") from None


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- scenario -------------------------------------------------------------------------------------


class Message(Strict):
    id: Annotated[str, Field(pattern=SAFE_ID)]
    step: int = Field(ge=0)
    kind: Literal["instruction", "question", "alarm"]
    to: tuple[Annotated[str, Field(pattern=TARGET_ID)], ...] = Field(min_length=1)
    text: str = Field(min_length=1)


class Scenario(Strict):
    id: Annotated[str, Field(pattern=SAFE_ID)]  # never shown to a model
    episode: Path  # the episode CSV, relative to the scenario file
    targets: tuple[Annotated[str, Field(pattern=TARGET_ID)], ...] = Field(min_length=1)
    messages: tuple[Message, ...] = ()

    @model_validator(mode="after")
    def _rules(self) -> Scenario:
        if len(set(self.targets)) != len(self.targets) or len({m.id for m in self.messages}) != len(self.messages):
            raise ValueError("targets and message ids must be unique")
        unknown = sorted({to for m in self.messages for to in m.to} - set(self.targets))
        if unknown:
            raise ValueError(f"messages name {unknown}, which targets does not list")
        asked = [(to, m.step) for m in self.messages if m.kind == "question" for to in m.to]
        if len(set(asked)) != len(asked):
            raise ValueError("at most one question per target per step")
        return self


# --- run config -----------------------------------------------------------------------------------


class PresetChange(Strict):
    source: Literal["preset"]
    steps: tuple[Annotated[int, Field(ge=0)], ...] = ()  # change steps for every live target


class Question(Strict):
    instructions: str = Field(min_length=1)
    none: str = Field(min_length=1)
    change: str = Field(min_length=1)


class ClefFlashChange(Strict):
    source: Literal["clef_flash"]
    every_steps: int = Field(5, gt=0)
    threshold: float = Field(0.65, ge=0, le=1)  # provisional until a labelled check
    model: str = Field(min_length=1)  # the Hugging Face id the server serves; recorded in run.json
    revision: str = Field(min_length=1)
    socket: str | None = "/tmp/{user}-vllm/clef-flash.sock"  # default; or base_url + api_key_env (D39)
    base_url: str | None = None
    api_key_env: Annotated[str, Field(pattern=r"^[A-Z_][A-Z0-9_]{0,63}$")] | None = None  # a NAME, never a key
    timeout_s: float = Field(30, gt=0)
    input_token_budget: int = Field(SERVING_CAP, gt=0, le=SERVING_CAP)
    question: Question
    inputs: dict[str, Any] = Field(min_length=1)
    state: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _tcp_drops_the_socket(cls, raw: Any) -> Any:
        return {**raw, "socket": raw.get("socket")} if isinstance(raw, dict) and raw.get("base_url") else raw

    @model_validator(mode="after")
    def _endpoint(self) -> ClefFlashChange:
        if (self.socket is None) == (self.base_url is None):
            raise ValueError("set exactly one of socket (default) or base_url")
        if self.base_url is not None and self.api_key_env is None:
            raise ValueError("a TCP base_url needs api_key_env")
        if self.socket is not None and ("$" in self.socket or "{" in self.socket.replace("{user}", "")):
            raise ValueError("socket supports only the placeholder {user}; no environment variable is expanded")
        if self.socket is not None and not self.socket.startswith("/"):
            raise ValueError("socket must be an absolute path")
        return self

    @property
    def socket_path(self) -> Path | None:
        """The socket with ``{user}`` replaced by the current user name."""
        return None if self.socket is None else Path(self.socket.replace("{user}", getpass.getuser()))


class PromptConfig(Strict):
    name: str | None = None  # default: the config file stem; recorded with the prompt's sha256
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class ModelConfig(Strict):
    backend: str = "openai"
    name: str = Field(min_length=1)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_output_tokens: int = Field(4000, gt=0)  # reasoning + visible output
    timeout_s: float = Field(1800, gt=0)


class AnswerConfig(Strict):
    max_intent_chars: int = Field(600, gt=0)
    on_invalid: Literal["stop", "skip"] = "stop"  # D41: an invalid answer stops the run, or is skipped


class LimitsConfig(Strict):
    max_translations: int = Field(250, gt=0)
    max_detector_calls: int = Field(500, gt=0)


def _options(inputs: dict[str, Any], where: str) -> dict[str, Any]:
    unknown = sorted(set(inputs) - set(INPUTS))
    if unknown:
        raise ValueError(f"{where}: unknown inputs {unknown} (known: {sorted(INPUTS)})")
    return {name: INPUTS[name].options.model_validate(options or {}) for name, options in inputs.items()}


def _check_prompt(templates: dict[str, str], inputs: dict[str, Any]) -> None:
    """Every placeholder is a base value or a selected input, and every selected input is used."""
    used = set().union(*(placeholders(text, where) for where, text in templates.items()))
    unknown, unused = sorted(used - BASE_VALUES - set(inputs)), sorted(set(inputs) - used)
    if unknown or unused:
        raise ValueError(f"{' and '.join(templates)}: unknown placeholders {unknown}, unused inputs {unused}")


class RunConfig(Strict):
    scenario: Path
    change: Annotated[PresetChange | ClefFlashChange, Field(discriminator="source")]
    inputs: dict[str, Any] = Field(min_length=1)
    prompt: PromptConfig
    model: ModelConfig
    answer: AnswerConfig = AnswerConfig()
    limits: LimitsConfig = LimitsConfig()
    output: Path = Path("outputs/translate")

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, inputs: dict[str, Any]) -> dict[str, Any]:
        return _options(inputs, "inputs")

    @model_validator(mode="after")
    def _templates(self) -> RunConfig:
        _check_prompt({"prompt.system": self.prompt.system, "prompt.user": self.prompt.user}, self.inputs)
        if isinstance(self.change, ClefFlashChange):
            _check_prompt({"change.state": self.change.state}, _options(self.change.inputs, "change.inputs"))
        return self


@dataclass(frozen=True)
class Loaded:
    """A validated run config and scenario with what identifies them."""

    config: RunConfig
    scenario: Scenario
    path: Path
    sha256: str
    scenario_path: Path
    prompt_name: str

    def detector_inputs(self) -> dict[str, Any]:
        return _options(self.config.change.inputs, "change.inputs")


def load_scenario(path: Path) -> Scenario:
    scenario = validated(Scenario, load_yaml(path), str(path))
    episode = scenario.episode if scenario.episode.is_absolute() else (path.parent / scenario.episode).resolve()
    return scenario.model_copy(update={"episode": episode})


def load_config(path: Path, scenario: Path | None = None) -> Loaded:
    config = validated(RunConfig, load_yaml(path), str(path))
    scenario_path = repo_path(scenario if scenario is not None else config.scenario)
    sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return Loaded(
        config, load_scenario(scenario_path), Path(path), sha256, scenario_path, config.prompt.name or Path(path).stem
    )
