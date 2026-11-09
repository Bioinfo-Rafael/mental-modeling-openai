"""The judge config (``configs/evaluation/judge.yaml``): pydantic models with defaults; unknown keys fail.

The prompt texts live here (``prompt.system``, ``prompt.user``, ``prompt.rating_rule``,
``prompt.future``); their placeholders are a fixed set, checked at load, so a typo or a template that
drops judge material fails before anything is rendered. Criteria carry their point kinds; an event
is rated on every criterion of any of its kinds, each once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from translate.config import load_yaml, validated
from translate.prompts import check_fields

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
    rating_rule: str = Field(min_length=1)  # filled into {rating_rule}
    future: str = Field(min_length=1)  # the FUTURE block of start and change events

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


def load_config(path: Path) -> JudgeConfig:
    return validated(JudgeConfig, load_yaml(path), str(path))


def criteria_for(config: JudgeConfig, kinds: tuple[str, ...]) -> tuple[Criterion, ...]:
    """Every criterion of any of the kinds, each once, in config order."""
    return tuple(criterion for criterion in config.criteria if set(criterion.kinds) & set(kinds))
