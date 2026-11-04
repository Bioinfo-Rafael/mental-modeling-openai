"""Optional ``evaluation`` section of a spec: where base states are sampled for claim evaluation.

Harness-only (like ``provenance`` and ``notes``): it never reaches the prompt. Shape::

    "evaluation": {
      "notes": ["why these ranges"],
      "inputs": {"<name>": {"range": [lo, hi]} | {"values": [v0, v1, ...]}, ...},
      "angles": [{"sin": "<name>", "cos": "<name>", "range": [lo_rad, hi_rad]}],
      "presence": [{"inputs": ["<name>", ...], "p_present": 0.5}]
    }

Every input is covered exactly once: by ``inputs`` or as one half of an ``angles`` pair. A pair
is sampled as one angle inside ``range`` (strictly within (-pi/2, pi/2) and containing 0, so sin and
cos are both monotone in |angle| on each side of level) and written as (sin, cos). A ``presence`` group is all-zero with
probability ``1 - p_present`` (e.g. "no live missile"), else sampled from its members' entries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SECTION_KEYS = {"notes", "inputs", "angles", "presence"}
RANGE_ENDS = 2
MIN_DISCRETE_VALUES = 2
HALF_PI = math.pi / 2


@dataclass(frozen=True)
class InputRange:
    """Continuous ``[low, high]`` (``values`` empty) or a discrete set (``values``, ascending)."""

    index: int
    low: float
    high: float
    values: tuple[float, ...] = ()

    @property
    def discrete(self) -> bool:
        return bool(self.values)

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class AnglePair:
    sin_index: int
    cos_index: int
    low: float  # radians
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class PresenceGroup:
    indices: tuple[int, ...]
    p_present: float


@dataclass(frozen=True)
class EvalRanges:
    inputs: tuple[InputRange, ...]  # one per non-angle input, ordered by index
    angles: tuple[AnglePair, ...]
    presence: tuple[PresenceGroup, ...]
    notes: tuple[str, ...]

    def input_range(self, index: int) -> InputRange | None:
        return next((item for item in self.inputs if item.index == index), None)

    def angle_of(self, index: int) -> AnglePair | None:
        return next((pair for pair in self.angles if index in (pair.sin_index, pair.cos_index)), None)

    def presence_of(self, index: int) -> PresenceGroup | None:
        return next((group for group in self.presence if index in group.indices), None)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _bounds(raw: object, where: str, problems: list[str]) -> tuple[float, float] | None:
    if not isinstance(raw, list) or len(raw) != RANGE_ENDS or not all(map(_is_number, raw)) or raw[0] >= raw[1]:
        problems.append(f"{where} must be [low, high] with low < high")
        return None
    return float(raw[0]), float(raw[1])


def _input_entry(name: str, raw: object, index: int, problems: list[str]) -> InputRange | None:
    where = f"evaluation.inputs.{name}"
    if not isinstance(raw, dict) or set(raw) not in ({"range"}, {"values"}):
        problems.append(f"{where} needs exactly one of range or values")
        return None
    if "range" in raw:
        bounds = _bounds(raw["range"], f"{where}.range", problems)
        return None if bounds is None else InputRange(index, *bounds)
    values = raw["values"]
    if not isinstance(values, list) or not all(map(_is_number, values)) or len(set(values)) < MIN_DISCRETE_VALUES:
        problems.append(f"{where}.values must list at least {MIN_DISCRETE_VALUES} distinct numbers")
        return None
    ordered = tuple(sorted(float(v) for v in set(values)))
    return InputRange(index, ordered[0], ordered[-1], ordered)


def _parse_inputs(raw: object, names: dict[str, int], problems: list[str]) -> list[InputRange]:
    if not isinstance(raw, dict):
        problems.append("evaluation.inputs must be an object {name: entry}")
        return []
    entries = []
    for name, item in raw.items():
        if name not in names:
            problems.append(f"evaluation.inputs names unknown input {name}")
            continue
        entry = _input_entry(name, item, names[name], problems)
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_angle(raw: object, names: dict[str, int], problems: list[str]) -> AnglePair | None:
    if not isinstance(raw, dict) or set(raw) != {"sin", "cos", "range"}:
        problems.append("evaluation.angles entries need exactly sin, cos and range")
        return None
    sin_name, cos_name = raw["sin"], raw["cos"]
    if sin_name not in names or cos_name not in names or sin_name == cos_name:
        problems.append(f"evaluation.angles pair ({sin_name}, {cos_name}) must name two different inputs")
        return None
    bounds = _bounds(raw["range"], f"evaluation.angles {sin_name}.range", problems)
    if bounds is None:
        return None
    if bounds[0] <= -HALF_PI or bounds[1] >= HALF_PI or not bounds[0] < 0 < bounds[1]:
        problems.append(f"evaluation.angles {sin_name}.range must lie strictly inside (-pi/2, pi/2) and contain 0")
        return None
    return AnglePair(names[sin_name], names[cos_name], *bounds)


def _parse_presence(raw: object, names: dict[str, int], problems: list[str]) -> PresenceGroup | None:
    if not isinstance(raw, dict) or set(raw) != {"inputs", "p_present"}:
        problems.append("evaluation.presence entries need exactly inputs and p_present")
        return None
    members, p_present = raw["inputs"], raw["p_present"]
    if not isinstance(members, list) or not members or not all(name in names for name in members):
        problems.append("evaluation.presence.inputs must be a non-empty list of input names")
        return None
    if not _is_number(p_present) or not 0 < p_present <= 1:
        problems.append("evaluation.presence.p_present must be in (0, 1]")
        return None
    return PresenceGroup(tuple(names[name] for name in members), float(p_present))


def _coverage_problems(
    inputs: list[InputRange], angles: list[AnglePair], presence: list[PresenceGroup], names: dict[str, int]
) -> list[str]:
    covered = [item.index for item in inputs] + [i for pair in angles for i in (pair.sin_index, pair.cos_index)]
    by_index = {index: name for name, index in names.items()}
    problems = [
        f"evaluation covers input {by_index[i]} more than once" for i in sorted(set(covered)) if covered.count(i) > 1
    ]
    problems += [f"evaluation does not cover input {name}" for name, i in names.items() if i not in covered]
    angle_members = {i for pair in angles for i in (pair.sin_index, pair.cos_index)}
    grouped = [i for group in presence for i in group.indices]
    problems += [f"presence member {by_index[i]} must not be an angle input" for i in grouped if i in angle_members]
    problems += [
        f"input {by_index[i]} is in more than one presence group" for i in set(grouped) if grouped.count(i) > 1
    ]
    return problems


def parse_eval_ranges(raw: object, input_names: tuple[str, ...]) -> tuple[EvalRanges | None, list[str]]:
    """``(ranges, problems)``; ``(None, [])`` when the section is absent."""
    if raw is None:
        return None, []
    if not isinstance(raw, dict) or set(raw) - SECTION_KEYS:
        return None, [f"evaluation must be an object with keys from {sorted(SECTION_KEYS)}"]
    names = {name: index for index, name in enumerate(input_names)}
    problems: list[str] = []
    inputs = _parse_inputs(raw.get("inputs", {}), names, problems)
    angle_items = raw.get("angles", [])
    presence_items = raw.get("presence", [])
    if not isinstance(angle_items, list) or not isinstance(presence_items, list):
        return None, ["evaluation.angles and evaluation.presence must be lists"]
    angles = [pair for pair in (_parse_angle(item, names, problems) for item in angle_items) if pair]
    presence = [group for group in (_parse_presence(item, names, problems) for item in presence_items) if group]
    if problems:
        return None, problems
    problems = _coverage_problems(inputs, angles, presence, names)
    if problems:
        return None, problems
    notes_raw = raw.get("notes", [])
    if not isinstance(notes_raw, list):
        return None, ["evaluation.notes must be a list of strings"]
    notes = tuple(str(note) for note in notes_raw)
    ordered = tuple(sorted(inputs, key=lambda item: item.index))
    return EvalRanges(ordered, tuple(angles), tuple(presence), notes), []
