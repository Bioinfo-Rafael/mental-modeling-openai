"""Ground-truth directions from measured effects, and claim scores against them.

Primary truth for one (input, head) pair, from its per-state normalised ``E[v]`` effects ``e``:

* ``'+'`` if at least ``agree`` of the valid states have ``e > 0`` AND ``mean(e) >= min_effect``;
* ``'-'`` symmetrically; otherwise ``'none'`` (no consistent direction, or too small to matter).

Secondary truth (argmax chosen value, the deployed action): ``'+'`` if at least ``ARGMAX_MIN_FLIP`` of
the states flip upwards and at least ``agree`` of the flipping states flip upwards; ``'-'``
symmetrically; else ``'none'``.

A claim is correct iff its direction equals the truth. Outcomes: ``correct``, ``wrong_sign``
(signed claim, opposite signed truth), ``overclaim`` (signed claim, truth ``none``), ``missed``
(claim ``none``, signed truth). Unclaimed pairs are abstentions; ``recall`` counts them. The Brier
score reads ``confidence`` as the probability that the claimed label is correct.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from weightreading.claims import Claim
from weightreading.effects import EffectTable

AGREE_FRACTION = 0.7
MIN_EFFECT = 0.02  # fraction of the head's value span, over the input's whole evaluation range
ARGMAX_MIN_FLIP = 0.02  # fraction of states whose argmax value must move for a signed argmax truth
CONFIDENCE_EDGES = (0.6, 0.8)  # buckets [0, .6), [.6, .8), [.8, 1]
TOP_K = 3
SENSITIVITY_AGREE = (0.6, 0.7, 0.8, 0.9)
SENSITIVITY_MIN_EFFECT = (0.005, 0.01, 0.02, 0.05)
SIGNED = ("+", "-")
OUTCOMES = ("correct", "wrong_sign", "overclaim", "missed")


@dataclass(frozen=True)
class PairTruth:
    input_index: int
    head_index: int
    n_valid: int
    frac_pos: float
    frac_neg: float
    mean_effect: float
    median_effect: float
    argmax_up: float  # fraction of valid states whose chosen value rises
    argmax_down: float
    argmax_truth: str
    truth: str


@dataclass(frozen=True)
class ClaimScore:
    position: int
    input_index: int
    head_index: int
    claimed: str
    confidence: float
    truth: str
    outcome: str
    frac_agree: float  # fraction of states whose effect sign matches the claim ('none': max(frac_pos, frac_neg))
    mean_effect: float
    argmax_truth: str

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


def sign_fractions(effects: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(valid count, fraction > 0, fraction < 0) over the last axis, NaN = not valid."""
    valid = np.isfinite(effects)
    count = valid.sum(axis=-1)
    safe = np.maximum(count, 1)
    return count, (effects > 0).sum(axis=-1) / safe, (effects < 0).sum(axis=-1) / safe


def nan_mean(effects: np.ndarray) -> np.ndarray:
    valid = np.isfinite(effects)
    return np.where(valid, effects, 0.0).sum(axis=-1) / np.maximum(valid.sum(axis=-1), 1)


def truth_labels(effects: np.ndarray, agree: float, min_effect: float) -> np.ndarray:
    """Primary truth for every pair of an (inputs, heads, N) effect array -> (inputs, heads) of str."""
    _, pos, neg = sign_fractions(effects)
    mean = nan_mean(effects)
    labels = np.full(mean.shape, "none", dtype=object)
    labels[(pos >= agree) & (mean >= min_effect)] = "+"
    labels[(neg >= agree) & (mean <= -min_effect)] = "-"
    return labels


def argmax_labels(chosen_effects: np.ndarray, agree: float) -> np.ndarray:
    _, up, down = sign_fractions(chosen_effects)
    flips = up + down
    labels = np.full(up.shape, "none", dtype=object)
    labels[(up >= ARGMAX_MIN_FLIP) & (up >= agree * flips)] = "+"
    labels[(down >= ARGMAX_MIN_FLIP) & (down >= agree * flips)] = "-"
    return labels


def pair_truths(
    table: EffectTable, agree: float = AGREE_FRACTION, min_effect: float = MIN_EFFECT
) -> dict[tuple[int, int], PairTruth]:
    count, pos, neg = sign_fractions(table.effects)
    mean = nan_mean(table.effects)
    filled = np.where(np.isfinite(table.effects), table.effects, np.nan)
    _, up, down = sign_fractions(table.argmax_effects)
    labels, argmax = truth_labels(table.effects, agree, min_effect), argmax_labels(table.argmax_effects, agree)
    n_inputs, n_heads = mean.shape
    truths = {}
    for i in range(n_inputs):
        for h in range(n_heads):
            median = float(np.nanmedian(filled[i, h])) if count[i, h] else 0.0
            truths[(i, h)] = PairTruth(
                i, h, int(count[i, h]), float(pos[i, h]), float(neg[i, h]), float(mean[i, h]), median,
                float(up[i, h]), float(down[i, h]), str(argmax[i, h]), str(labels[i, h]),
            )  # fmt: skip
    return truths


def outcome(claimed: str, truth: str) -> str:
    if claimed == truth:
        return "correct"
    if claimed in SIGNED and truth in SIGNED:
        return "wrong_sign"
    return "overclaim" if claimed in SIGNED else "missed"


def score_claim(claim: Claim, truth: PairTruth) -> ClaimScore:
    agree = {"+": truth.frac_pos, "-": truth.frac_neg}.get(claim.direction, max(truth.frac_pos, truth.frac_neg))
    return ClaimScore(
        claim.position, claim.input_index, claim.head_index, claim.direction, claim.confidence, truth.truth,
        outcome(claim.direction, truth.truth), agree, truth.mean_effect, truth.argmax_truth,
    )  # fmt: skip


def score_claims(claims: Iterable[Claim], truths: dict[tuple[int, int], PairTruth]) -> tuple[ClaimScore, ...]:
    return tuple(score_claim(claim, truths[(claim.input_index, claim.head_index)]) for claim in claims)


def accuracy(flags: list[bool]) -> dict[str, float | int | None]:
    correct = sum(flags)
    return {"n": len(flags), "correct": correct, "accuracy": correct / len(flags) if flags else None}


def confidence_bucket(confidence: float) -> str:
    low, high = CONFIDENCE_EDGES
    if confidence < low:
        return f"[0, {low})"
    return f"[{low}, {high})" if confidence < high else f"[{high}, 1]"


def bucket_names() -> tuple[str, ...]:
    low, high = CONFIDENCE_EDGES
    return (f"[0, {low})", f"[{low}, {high})", f"[{high}, 1]")


def recall(scores: tuple[ClaimScore, ...], truths: dict[tuple[int, int], PairTruth]) -> dict:
    """Signed truths over ALL pairs, and how many the run claimed with the right sign."""
    signed_pairs = {key for key, truth in truths.items() if truth.truth in SIGNED}
    hit_pairs = {(s.input_index, s.head_index) for s in scores if s.correct and s.claimed in SIGNED}
    claimed_pairs = {(s.input_index, s.head_index) for s in scores}
    return {
        "pairs_total": len(truths),
        "pairs_claimed": len(claimed_pairs),
        "signed_truths": len(signed_pairs),
        "signed_truths_claimed_correctly": len(hit_pairs & signed_pairs),
        "recall": len(hit_pairs & signed_pairs) / len(signed_pairs) if signed_pairs else None,
    }


def summarize(scores: tuple[ClaimScore, ...], truths: dict[tuple[int, int], PairTruth]) -> dict:
    """Accuracy overall / signed / by confidence, outcomes, Brier, recall, argmax secondary, baselines."""
    items = list(scores)
    signed = [s for s in items if s.claimed in SIGNED]
    buckets = {name: [s.correct for s in items if confidence_bucket(s.confidence) == name] for name in bucket_names()}
    return {
        "overall": accuracy([s.correct for s in items]),
        "signed_claims": accuracy([s.correct for s in signed]),
        "none_claims": accuracy([s.correct for s in items if s.claimed == "none"]),
        "by_confidence": {name: accuracy(flags) for name, flags in buckets.items()},
        "outcomes": {name: sum(s.outcome == name for s in items) for name in OUTCOMES},
        "brier": float(np.mean([(s.confidence - s.correct) ** 2 for s in items])) if items else None,
        "recall": recall(scores, truths),
        "argmax_secondary": accuracy([s.claimed == s.argmax_truth for s in items]),
        "baselines": {
            "always_none_accuracy_on_claimed_pairs": accuracy([s.truth == "none" for s in items])["accuracy"],
            "random_sign_accuracy_on_signed_truth": 0.5,
        },
    }


def coverage(
    truths: dict[tuple[int, int], PairTruth], claims: Iterable[Claim], n_heads: int, top_k: int = TOP_K
) -> list[dict]:
    """Per head: the ``top_k`` strongest signed true effects and how the run claimed each (None = not claimed)."""
    claimed = {(c.input_index, c.head_index): c.direction for c in claims}
    rows = []
    for head in range(n_heads):
        signed = [t for (_, h), t in truths.items() if h == head and t.truth in SIGNED]
        top = sorted(signed, key=lambda t: abs(t.mean_effect), reverse=True)[:top_k]
        for rank, truth in enumerate(top, start=1):
            direction = claimed.get((truth.input_index, head))
            rows.append(
                {
                    "head_index": head,
                    "rank": rank,
                    "input_index": truth.input_index,
                    "truth": truth.truth,
                    "mean_effect": truth.mean_effect,
                    "claimed": direction,
                    "hit": direction == truth.truth,
                }
            )
    return rows


def sensitivity(table: EffectTable, claims: tuple[Claim, ...]) -> list[dict]:
    """Accuracy over the (agree, min_effect) threshold grid."""
    rows = []
    for agree in SENSITIVITY_AGREE:
        for min_effect in SENSITIVITY_MIN_EFFECT:
            labels = truth_labels(table.effects, agree, min_effect)
            flags = [labels[c.input_index, c.head_index] == c.direction for c in claims]
            signed = [flag for flag, c in zip(flags, claims, strict=True) if c.direction in SIGNED]
            rows.append(
                {
                    "agree": agree,
                    "min_effect": min_effect,
                    "accuracy": accuracy(flags)["accuracy"],
                    "signed_accuracy": accuracy(signed)["accuracy"],
                }
            )
    return rows


def truth_record(truth: PairTruth) -> dict:
    return asdict(truth)


def score_record(score: ClaimScore) -> dict:
    return {**asdict(score), "correct": score.correct}
