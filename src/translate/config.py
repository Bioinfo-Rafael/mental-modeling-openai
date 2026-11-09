"""Run config and scenario: pydantic models with defaults; unknown keys and duplicate YAML keys fail.

A run config names the scenario, the change source (any registered in ``CHANGE_SOURCES``, with its
own options model), the selected inputs with their options, the inline prompt, the model and the limits.
Errors name the key path. Paths in a run config are repository-relative; the episode path in a
scenario is relative to the scenario file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

import translate.detector  # noqa: F401  (registers the clef_flash change source)
from translate.inputs import validate_options
from translate.journal import repo_path
from translate.prompts import check_inputs
from translate.triggers import CHANGE_SOURCES

SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
TARGET_ID = r"^(aircraft|team):[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


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


# The options of every registered change source, picked by ``change.source``: registering a source
# (``triggers.register_change``) adds it here; no model in this file names a source.
ChangeConfig = Annotated[Union[tuple(s.options for s in CHANGE_SOURCES.values())], Field(discriminator="source")]  # noqa: UP007


class RunConfig(Strict):
    scenario: Path
    change: ChangeConfig
    inputs: dict[str, Any] = Field(min_length=1)
    prompt: PromptConfig
    model: ModelConfig
    answer: AnswerConfig = AnswerConfig()
    limits: LimitsConfig = LimitsConfig()
    output: Path = Path("outputs/translate")

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, inputs: dict[str, Any]) -> dict[str, Any]:
        return validate_options(inputs, "inputs")

    @model_validator(mode="after")
    def _templates(self) -> RunConfig:
        check_inputs({"prompt.system": self.prompt.system, "prompt.user": self.prompt.user}, self.inputs)
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
