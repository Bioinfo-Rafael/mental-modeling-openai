"""Strict YAML loading and primitive value checks shared by the run config and the scenario.

Every mapping must have exactly the documented keys; booleans never count as numbers; duplicate
YAML keys are rejected (PyYAML would keep the last one silently).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from translate.events import SAFE_ID_PATTERN


def mapping(raw: object, keys: set[str], name: str, error: type[ValueError]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise error(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise error(f"{name} keys must be strings (quote 'true'/'false' keys)")
    if set(raw) != keys:
        missing, unknown = sorted(keys - set(raw)), sorted(set(raw) - keys)
        raise error(f"{name} must have exactly {sorted(keys)} (missing {missing}, unknown {unknown})")
    return raw


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def positive_int(value: object, name: str, error: type[ValueError]) -> int:
    if not is_int(value) or value <= 0:
        raise error(f"{name} must be a positive integer")
    return value


def nonnegative_int(value: object, name: str, error: type[ValueError]) -> int:
    if not is_int(value) or value < 0:
        raise error(f"{name} must be a non-negative integer")
    return value


def number(value: object, name: str, error: type[ValueError]) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise error(f"{name} must be a finite number")
    return float(value)


def text(value: object, name: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a non-empty string")
    return value


def safe_id(value: object, name: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not SAFE_ID_PATTERN.fullmatch(value):
        raise error(f"{name} must match {SAFE_ID_PATTERN.pattern}")
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


def read_yaml(path: Path, error: type[ValueError]) -> object:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise error(f"cannot read {path}: {type(exc).__name__}") from exc
    try:
        return yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass: no arbitrary objects
    except yaml.YAMLError as exc:
        problem = getattr(exc, "problem", None)
        raise error(f"{path} is not valid YAML" + (f": {problem}" if problem else "")) from exc
