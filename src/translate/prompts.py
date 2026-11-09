"""Prompt templates: ``str.format`` text with bare ``{name}`` placeholders, checked at load, filled once.

A template may use the base values and the selected inputs; a config fails to load on an unknown
placeholder or on a selected input its prompt does not use. Filling is one pass: a value containing
``{...}`` is never expanded again. Literal braces are doubled.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

BASE_VALUES = frozenset({"step", "time_s", "target", "target_aircraft", "max_intent_chars"})


def placeholders(template: str, where: str) -> set[str]:
    """The ``{name}`` fields of a template; each must be a bare name (a format spec could cut material)."""
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"{where}: malformed template ({exc})") from None
    names = set()
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if not name.isidentifier() or spec or conversion:
            raise ValueError(f"{where}: {{{name}}} must be a bare name")
        names.add(name)
    return names


def check_fields(template: str, where: str, allowed: frozenset[str], required: frozenset[str] = frozenset()) -> None:
    found = placeholders(template, where)
    if found - allowed:
        raise ValueError(f"{where} uses unknown placeholders {sorted(found - allowed)}")
    if required - found:
        raise ValueError(f"{where} must use {sorted(required - found)}")


def check_inputs(templates: dict[str, str], inputs: Mapping[str, Any]) -> None:
    """Every placeholder is a base value or a selected input, and every selected input is used."""
    used = set().union(*(placeholders(text, where) for where, text in templates.items()))
    unknown, unused = sorted(used - BASE_VALUES - set(inputs)), sorted(set(inputs) - used)
    if unknown or unused:
        raise ValueError(f"{' and '.join(templates)}: unknown placeholders {unknown}, unused inputs {unused}")


def fill(template: str, values: Mapping[str, Any]) -> str:
    return template.format_map(values)


def intent_schema() -> dict[str, Any]:
    """The translator's answer: one object with a non-empty intent and at least one evidence item."""
    evidence = {"type": "array", "minItems": 1, "items": {"type": "string"}}
    properties = {"intent": {"type": "string"}, "evidence": evidence}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["intent", "evidence"],
    }
