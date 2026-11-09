"""Base-state sampling for the policy parity check (pure numpy, seeded).

``sample_base_states`` draws independent observations from a spec's evaluation ranges: continuous
inputs uniformly, discrete inputs from their values, angle pairs through their underlying angle,
and absent presence groups as all-zero entries (``scripts/policy_forward_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from translate.policy.eval_ranges import EvalRanges, InputRange


@dataclass(frozen=True)
class BaseStates:
    obs: np.ndarray  # (N, inputs)
    angles: np.ndarray  # (N, pairs): the underlying angle of each pair, radians
    present: np.ndarray  # (N, groups): bool


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
