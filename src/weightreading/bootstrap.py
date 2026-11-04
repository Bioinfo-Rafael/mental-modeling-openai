"""State-level bootstrap: resample base states, recompute every truth label and the claim scores.

Claims share base states, heads and one network, so a per-claim binomial interval would be too
narrow; resampling whole states keeps that dependence. Reported: percentile intervals of the
overall and signed-claim accuracy, and per claim how often its truth label equals the point label
(label stability; low = the claim sits near a threshold).
"""

from __future__ import annotations

import numpy as np

from weightreading.claims import Claim
from weightreading.effects import EffectTable
from weightreading.scoring import SIGNED, truth_labels

BOOTSTRAP_RESAMPLES = 200
CI_PERCENTILES = (2.5, 97.5)


def _interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    low, high = np.percentile(values, CI_PERCENTILES)
    return [float(low), float(high)]


def bootstrap_scores(
    table: EffectTable,
    claims: tuple[Claim, ...],
    agree: float,
    min_effect: float,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    n_states = table.effects.shape[2]
    rng = np.random.default_rng(seed)
    point = truth_labels(table.effects, agree, min_effect)
    pairs = [(c.input_index, c.head_index) for c in claims]
    signed = [c.direction in SIGNED for c in claims]
    overall, signed_acc = [], []
    stable = np.zeros(len(claims))
    pair_stable = np.zeros(point.shape)
    for _ in range(resamples):
        labels = truth_labels(table.effects[:, :, rng.integers(0, n_states, n_states)], agree, min_effect)
        correct = np.array([labels[pair] == c.direction for pair, c in zip(pairs, claims, strict=True)], dtype=bool)
        stable += np.array([labels[pair] == point[pair] for pair in pairs], dtype=float)
        pair_stable += labels == point
        if claims:
            overall.append(float(correct.mean()))
        if any(signed):
            signed_acc.append(float(correct[np.array(signed)].mean()))
    return {
        "resamples": resamples,
        "overall_accuracy_ci95": _interval(overall),
        "signed_accuracy_ci95": _interval(signed_acc),
        "claim_truth_stability": [float(value) for value in stable / resamples],
        "mean_pair_label_stability": float((pair_stable / resamples).mean()),
    }
