"""Compose the judge's four text templates and response schema, without an API client."""

from __future__ import annotations

from typing import Any

from evaluation.config import Criterion, JudgeConfig
from evaluation.material import TranslationRun, future_values, material_values
from translate.backends import Prompt, input_text
from translate.episode import Episode


def verdict_schema(criteria: tuple[Criterion, ...], config: JudgeConfig) -> dict[str, Any]:
    """Exactly one item per criterion; the rationale comes before the judgement."""
    judgements = [*config.scale.values, "cannot_judge", "not_applicable"]
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "criterion_id": {"type": "string", "enum": [criterion.id for criterion in criteria]},
            "rationale": {"type": "string"},
            "judgement": {"type": "string", "enum": judgements},
        },
        "required": ["criterion_id", "rationale", "judgement"],
    }
    items = {"type": "array", "minItems": len(criteria), "maxItems": len(criteria), "items": item}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"criteria": items},
        "required": ["criteria"],
    }


def build_prompt(
    event: dict[str, Any],
    run: TranslationRun,
    frame: Episode,
    config: JudgeConfig,
    criteria: tuple[Criterion, ...],
) -> Prompt:
    """Fill FUTURE, rating rule, system and user, in that order; inserted material is never re-expanded."""
    future_material = future_values(event, run, frame, config)
    future = config.prompt.future.format_map(future_material) if future_material is not None else ""
    values = material_values(event, run, frame, config, criteria, has_future=bool(future))
    rating_rule = config.prompt.rating_rule.format_map({"rationale_language": config.rationale_language})
    system = config.prompt.system.format_map(
        {"rating_rule": rating_rule, "rationale_language": config.rationale_language}
    )
    user = config.prompt.user.format_map({**values, "future": future, "rating_rule": rating_rule})
    schema = verdict_schema(criteria, config)
    return Prompt(system, (input_text(user),), f"verdict_{config.rubric_version}", schema)
