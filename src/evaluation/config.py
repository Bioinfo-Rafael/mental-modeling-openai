"""The judge config (``configs/evaluation/judge.yaml``): pydantic models with defaults; unknown keys fail.

The prompt texts live here (``prompt.system``, ``prompt.user``, ``prompt.rating_rule``,
``prompt.future``); their placeholders are a fixed set, checked at load, so a typo or a template that
drops judge material fails before anything is rendered. Criteria carry their point kinds; an event
is rated on every criterion of any of its kinds, each once.
"""

from __future__ import annotations

import string
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from translate.strict_yaml import parse_yaml

KINDS = ("start", "change", "request")
JUDGE_MODEL = "gpt-6.1-sol"  # D8: the judge is always gpt-6.1-sol
DEFAULT_FIELDS = (
    "alive",
    "state_valid",
    "lon_deg",
    "lat_deg",
    "alt_m",
    "yaw_deg",
    "tas_mps",
    "vn_mps",
    "ve_mps",
    "vu_mps",
    "altitude_action_id",
    "heading_action_id",
    "speed_action_id",
)
SYSTEM_FIELDS = frozenset({"rating_rule", "rationale_language"})
USER_REQUIRED = frozenset(
    {"instructions", "alarms", "question", "previous_translation", "trajectory", "intent", "evidence"}
    | {"timeline", "future", "criteria", "scale"}
)
USER_FIELDS = USER_REQUIRED | {"target", "target_aircraft", "step", "time_s", "kinds"}
USER_FIELDS |= {"cannot_judge", "not_applicable", "rating_rule"}
FUTURE_FIELDS = frozenset({"last_step", "end_step", "cut", "stride_steps", "fields", "rows", "later"})


class ConfigError(ValueError):
    """The judge config is malformed; the message names the key."""


def placeholders(template: str, where: str) -> set[str]:
    """The ``{name}`` fields of a ``str.format`` template; each must be a bare name (no format spec)."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{where}: malformed template ({exc})") from None
    names = set()
    for _, field, spec, conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise ValueError(f"{where}: {{{field}}} must be a bare name")
        names.add(field)
    return names


def check_fields(template: str, where: str, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> None:
    found = placeholders(template, where)
    if found - allowed:
        raise ValueError(f"{where} uses unknown placeholders {sorted(found - allowed)}")
    if required - found:
        raise ValueError(f"{where} must use {sorted(required - found)}")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSettings(Strict):
    name: Literal["gpt-6.1-sol"] = JUDGE_MODEL
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_output_tokens: int = Field(25000, gt=0)
    timeout_s: float = Field(1800, gt=0)


class Material(Strict):
    past_steps: int = Field(60, gt=0)
    future_steps: int = Field(120, gt=0)
    future_stride: int = Field(5, gt=0)
    fields: tuple[str, ...] = DEFAULT_FIELDS


class Scale(Strict):
    values: tuple[str, ...] = ("1", "2", "3", "4", "5")
    cannot_judge: str = Field(min_length=1)
    not_applicable: str = Field(min_length=1)

    @model_validator(mode="after")
    def _integers(self) -> Scale:
        if len(set(self.values)) != len(self.values) or not all(value.isdigit() for value in self.values):
            raise ValueError("scale.values must be distinct integers written as strings")
        return self


class Criterion(Strict):
    id: str = Field(pattern=r"^[a-z_]+\.[a-z0-9_]+$")
    kinds: tuple[Literal["start", "change", "request"], ...] = Field(min_length=1)
    text: str = Field(min_length=1)
    levels: dict[str, str]
    conditional: bool = False
    not_applicable: str | None = None
    cannot_judge: str | None = None
    may_use_future: bool = False

    @model_validator(mode="after")
    def _conditional(self) -> Criterion:
        if self.conditional != (self.not_applicable is not None):
            raise ValueError(f"{self.id}: conditional: true needs a not_applicable text, and only then")
        return self

    @property
    def aspect(self) -> str:
        return self.id.split(".")[0]


class Prompt(Strict):
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)
    rating_rule: str = ""
    future: str = ""

    @model_validator(mode="after")
    def _fields(self) -> Prompt:
        check_fields(self.system, "prompt.system", SYSTEM_FIELDS)
        check_fields(self.rating_rule, "prompt.rating_rule", frozenset({"rationale_language"}))
        check_fields(self.user, "prompt.user", USER_FIELDS, USER_REQUIRED)
        check_fields(self.future, "prompt.future", FUTURE_FIELDS, frozenset({"rows"}))
        return self


class Timing(Strict):
    request_deadline_s: float = Field(3.0, gt=0)  # D55


class Limits(Strict):
    max_calls: int = Field(100, gt=0)


class JudgeConfig(Strict):
    model: ModelSettings = ModelSettings()
    material: Material = Material()
    rubric_version: str = Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")
    rationale_language: str = "Japanese"
    scale: Scale
    timing: Timing = Timing()
    limits: Limits = Limits()
    output: Path = Path("outputs/evaluation")
    prompt: Prompt
    criteria: tuple[Criterion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _criteria(self) -> JudgeConfig:
        ids = [criterion.id for criterion in self.criteria]
        if len(set(ids)) != len(ids):
            raise ValueError("criteria ids must be unique")
        for criterion in self.criteria:
            if tuple(sorted(criterion.levels, key=int)) != tuple(sorted(self.scale.values, key=int)):
                raise ValueError(f"{criterion.id}: levels must be exactly the scale values {list(self.scale.values)}")
        return self


def validated(model: type[BaseModel], raw: object, where: str) -> BaseModel:
    """``model.model_validate(raw)``; a failure becomes a ``ConfigError`` naming every bad key path."""
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '(top)'}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"{where}: {problems}") from None


def load_config(path: Path) -> JudgeConfig:
    text = Path(path).read_text(encoding="utf-8")
    return validated(JudgeConfig, parse_yaml(text, path, ConfigError), str(path))


def criteria_for(config: JudgeConfig, kinds: tuple[str, ...]) -> tuple[Criterion, ...]:
    """Every criterion of any of the kinds, each once, in config order."""
    return tuple(criterion for criterion in config.criteria if set(criterion.kinds) & set(kinds))
