"""Measured input -> head effects of a reference policy under three hidden-state conditions.

Per (input, head, base state) the effect is ``(E[v](high) - E[v](low)) * scale / span``: the change
of the head's expected physical value ``E[v] = sum_k p_k v_k`` over the input's whole evaluation
range (linearised), as a fraction of the head's value span ``|max v - min v|``. ``low``/``high`` and
``scale`` come from ``sampling.perturb``. Secondary measure (the deployed action is the argmax): the
change of the argmax bin's value between ``low`` and ``high``, as a fraction of the span (no
rescaling; usually 0 because a small step rarely crosses an argmax boundary).

Conditions (``CONDITIONS``):

* ``h0``: one step from ``h = 0`` (the first decision of an episode);
* ``warm``: ``h`` after ``warmup_steps`` steps repeating the unperturbed base observation, then one
  perturbed step (immediate response of a settled policy);
* ``sustained``: ``warmup_steps + 1`` steps of the perturbed observation from ``h = 0`` (the
  response once the change has persisted).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weightreading.eval_ranges import EvalRanges
from weightreading.reference import ReferencePolicy, expected_value
from weightreading.sampling import ANGLE_MODE, STEP_FRACTION, BaseStates, perturb
from weightreading.spec import OutputSpec

CONDITIONS = ("h0", "warm", "sustained")
WARMUP_STEPS = 30


@dataclass(frozen=True)
class EffectTable:
    condition: str
    effects: np.ndarray  # (inputs, heads, N): normalised per-state effect; NaN where not defined
    argmax_effects: np.ndarray  # (inputs, heads, N): chosen-value change high - low / span; NaN where not defined


def head_span(output: OutputSpec) -> float:
    return abs(max(output.values) - min(output.values))


def _head_outputs(policy: ReferencePolicy, logits: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(E[v] per head, value of the argmax bin per head), both (N, heads)."""
    outputs = policy.spec.outputs
    pairs = list(zip(logits, outputs, strict=True))
    values = np.stack([expected_value(head_logits, out.values) for head_logits, out in pairs], axis=1)
    chosen = np.stack([np.asarray(out.values)[head_logits.argmax(axis=-1)] for head_logits, out in pairs], axis=1)
    return values, chosen


def warm_hidden(policy: ReferencePolicy, obs: np.ndarray, steps: int) -> np.ndarray:
    """Hidden state after ``steps`` steps that all see ``obs`` (N, inputs), starting from 0."""
    if steps < 1:
        return policy.initial_hidden(obs.shape[0])
    return policy.rollout(np.repeat(obs[None], steps, axis=0), policy.initial_hidden(obs.shape[0])).hidden


def evaluate_condition(
    policy: ReferencePolicy, obs: np.ndarray, condition: str, hidden: np.ndarray, warmup_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Head outputs for ``obs`` under ``condition``; ``hidden`` is the warm state (used by ``warm``)."""
    n = obs.shape[0]
    if condition == "h0":
        return _head_outputs(policy, policy.step(obs, policy.initial_hidden(n)).logits)
    if condition == "warm":
        return _head_outputs(policy, policy.step(obs, hidden).logits)
    if condition == "sustained":
        sequence = np.repeat(obs[None], warmup_steps + 1, axis=0)
        return _head_outputs(policy, policy.rollout(sequence, policy.initial_hidden(n)).logits)
    raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def measure_effects(
    policy: ReferencePolicy,
    ranges: EvalRanges,
    base: BaseStates,
    condition: str,
    warmup_steps: int = WARMUP_STEPS,
    step_fraction: float = STEP_FRACTION,
    angle_mode: str = ANGLE_MODE,
) -> EffectTable:
    """Per-state normalised effects of every input on every head under one condition."""
    n_inputs, n_heads, n = policy.input_size, len(policy.spec.outputs), base.obs.shape[0]
    spans = np.array([head_span(out) for out in policy.spec.outputs])
    hidden = warm_hidden(policy, base.obs, warmup_steps) if condition == "warm" else policy.initial_hidden(n)
    effects = np.full((n_inputs, n_heads, n), np.nan)
    chosen_effects = np.full((n_inputs, n_heads, n), np.nan)
    for index in range(n_inputs):
        move = perturb(base, ranges, index, step_fraction, angle_mode)
        both = np.concatenate([move.low, move.high])
        values, chosen = evaluate_condition(policy, both, condition, np.concatenate([hidden, hidden]), warmup_steps)
        delta = (values[n:] - values[:n]) * move.scale[:, None] / spans[None, :]
        effects[index] = np.where(move.valid[:, None], delta, np.nan).T
        chosen_delta = (chosen[n:] - chosen[:n]) / spans[None, :]
        chosen_effects[index] = np.where(move.valid[:, None], chosen_delta, np.nan).T
    return EffectTable(condition, effects, chosen_effects)
