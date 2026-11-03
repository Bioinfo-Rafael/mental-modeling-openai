"""Base-state sampling and one-input perturbations for claim evaluation (pure numpy, seeded).

Perturbation conventions (what "increasing input i" means, per input kind):

* continuous ``range``: clamped central difference ``x -/+ h`` with ``h = step_fraction * width``,
  clamped to ``[low, high]`` (one-sided at an edge);
* discrete ``values``: the lowest value versus the highest value (e.g. side -1 -> +1);
* the ``sin`` half of an angle pair: the underlying angle ``theta -/+ h`` (clamped to the pair's
  range), both sin and cos rewritten - "roll further right/up" near level;
* the ``cos`` half: ``|theta|`` moved towards 0 (cos grows) versus away from 0, keeping the sign of
  ``theta`` - "towards level";
* ``angle_mode="raw"`` (sensitivity check) instead moves only that one coordinate, as a continuous
  input over ``[sin(low), sin(high)]`` or ``[cos(max |angle|), 1]``, leaving the unit circle;
* a member of a presence group is perturbed only in states where the group is present; absent
  states (all-zero entries) are excluded rather than moved off the "absent" encoding.

``Perturbation.scale`` converts ``E[v](high) - E[v](low)`` into "change over the input's whole
range": ``width / (x_high - x_low)`` for continuous inputs and angles, ``1`` for discrete ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightreading.eval_ranges import AnglePair, EvalRanges, InputRange

STEP_FRACTION = 0.05
ANGLE_MODE = "angle"
ANGLE_MODES = ("angle", "raw")


@dataclass(frozen=True)
class BaseStates:
    obs: np.ndarray  # (N, inputs)
    angles: np.ndarray  # (N, pairs): the underlying angle of each pair, radians
    present: np.ndarray  # (N, groups): bool


@dataclass(frozen=True)
class Perturbation:
    low: np.ndarray  # (N, inputs): the "decreased" observation
    high: np.ndarray  # (N, inputs): the "increased" observation
    scale: np.ndarray  # (N,): multiplies E(high) - E(low) into a whole-range change
    valid: np.ndarray  # (N,): bool; False where the input is not defined (absent group)


def _sample_input(item: InputRange, n: int, rng: np.random.Generator) -> np.ndarray:
    if item.discrete:
        return rng.choice(np.asarray(item.values), size=n)
    return rng.uniform(item.low, item.high, size=n)


def sample_base_states(ranges: EvalRanges, n_inputs: int, n: int, rng: np.random.Generator) -> BaseStates:
    """``n`` independent base observations; absent presence groups are all-zero."""
    obs = np.zeros((n, n_inputs), dtype=np.float64)
    for item in ranges.inputs:
        obs[:, item.index] = _sample_input(item, n, rng)
    angles = (
        np.stack([rng.uniform(pair.low, pair.high, size=n) for pair in ranges.angles], axis=1)
        if ranges.angles
        else np.zeros((n, 0))
    )
    for column, pair in enumerate(ranges.angles):
        obs[:, pair.sin_index] = np.sin(angles[:, column])
        obs[:, pair.cos_index] = np.cos(angles[:, column])
    present = (
        np.stack([rng.random(n) < group.p_present for group in ranges.presence], axis=1)
        if ranges.presence
        else np.zeros((n, 0), dtype=bool)
    )
    for column, group in enumerate(ranges.presence):
        obs[np.ix_(~present[:, column], list(group.indices))] = 0.0
    return BaseStates(obs, angles, present)


def _with_column(obs: np.ndarray, index: int, values: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[:, index] = values
    return out


def _with_angle(obs: np.ndarray, pair: AnglePair, theta: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[:, pair.sin_index] = np.sin(theta)
    out[:, pair.cos_index] = np.cos(theta)
    return out


def _continuous(base: BaseStates, item: InputRange, step_fraction: float) -> Perturbation:
    x = base.obs[:, item.index]
    step = step_fraction * item.width
    x_low, x_high = np.maximum(x - step, item.low), np.minimum(x + step, item.high)
    scale = item.width / (x_high - x_low)
    valid = np.ones(len(x), dtype=bool)
    return Perturbation(
        _with_column(base.obs, item.index, x_low), _with_column(base.obs, item.index, x_high), scale, valid
    )


def _discrete(base: BaseStates, item: InputRange) -> Perturbation:
    n = base.obs.shape[0]
    low = _with_column(base.obs, item.index, np.full(n, item.values[0]))
    high = _with_column(base.obs, item.index, np.full(n, item.values[-1]))
    return Perturbation(low, high, np.ones(n), np.ones(n, dtype=bool))


def _angle(base: BaseStates, ranges: EvalRanges, index: int, step_fraction: float) -> Perturbation:
    pair = ranges.angle_of(index)
    if pair is None:
        raise ValueError(f"input {index} is not part of an angle pair")
    theta = base.angles[:, ranges.angles.index(pair)]
    step = step_fraction * pair.width
    valid = np.ones(len(theta), dtype=bool)
    if index == pair.sin_index:
        t_low, t_high = np.maximum(theta - step, pair.low), np.minimum(theta + step, pair.high)
        scale = pair.width / (t_high - t_low)
        return Perturbation(_with_angle(base.obs, pair, t_low), _with_angle(base.obs, pair, t_high), scale, valid)
    sign = np.where(theta < 0, -1.0, 1.0)
    side_max = np.where(theta < 0, -pair.low, pair.high)  # |angle| bound on theta's own side of 0
    magnitude = np.abs(theta)
    towards, away = np.maximum(magnitude - step, 0.0), np.minimum(magnitude + step, side_max)
    scale = side_max / (away - towards)
    # "increase cos" = towards level, so the high observation uses the smaller |theta|
    return Perturbation(
        _with_angle(base.obs, pair, sign * away), _with_angle(base.obs, pair, sign * towards), scale, valid
    )


def raw_angle_range(pair: AnglePair, index: int) -> InputRange:
    """The range one half of an angle pair spans when treated as a plain coordinate."""
    if index == pair.sin_index:
        return InputRange(index, float(np.sin(pair.low)), float(np.sin(pair.high)))
    return InputRange(index, float(np.cos(max(abs(pair.low), abs(pair.high)))), 1.0)


def perturb(
    base: BaseStates, ranges: EvalRanges, index: int, step_fraction: float = STEP_FRACTION, angle_mode: str = ANGLE_MODE
) -> Perturbation:
    """Low/high observations that decrease/increase input ``index``, all other inputs fixed."""
    if angle_mode not in ANGLE_MODES:
        raise ValueError(f"angle_mode must be one of {ANGLE_MODES}")
    pair = ranges.angle_of(index)
    if pair is not None and angle_mode == "raw":
        return _continuous(base, raw_angle_range(pair, index), step_fraction)
    if pair is not None:
        return _angle(base, ranges, index, step_fraction)
    item = ranges.input_range(index)
    if item is None:
        raise ValueError(f"input {index} has no evaluation range")
    result = _discrete(base, item) if item.discrete else _continuous(base, item, step_fraction)
    group = ranges.presence_of(index)
    if group is None:
        return result
    present = base.present[:, ranges.presence.index(group)]
    return Perturbation(result.low, result.high, result.scale, result.valid & present)
