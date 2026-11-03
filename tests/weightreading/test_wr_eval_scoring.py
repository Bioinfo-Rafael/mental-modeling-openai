"""Claim parsing, truth labels, claim outcomes, summaries, coverage, sensitivity and bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from weightreading.bootstrap import bootstrap_scores
from weightreading.claims import Claim, ClaimsError, parse_claims
from weightreading.effects import EffectTable
from weightreading.scoring import (
    argmax_labels,
    confidence_bucket,
    coverage,
    outcome,
    pair_truths,
    score_claims,
    sensitivity,
    summarize,
    truth_labels,
)

from .wr_eval_helpers import claims_markdown, known_spec


def _table(per_pair: dict[tuple[int, int], list[float]], shape: tuple[int, int] = (2, 2)) -> EffectTable:
    n = len(next(iter(per_pair.values())))
    effects = np.zeros((*shape, n))
    for (i, h), values in per_pair.items():
        effects[i, h] = values
    return EffectTable("h0", effects, np.zeros_like(effects))


def test_truth_labels_need_agreement_and_magnitude() -> None:
    table = _table(
        {
            (0, 0): [0.1] * 8 + [-0.01] * 2,  # 80% positive, mean 0.078 -> '+'
            (0, 1): [-0.1] * 6 + [0.1] * 4,  # 60% negative -> 'none'
            (1, 0): [0.001] * 10,  # consistent but tiny -> 'none'
            (1, 1): [-0.05] * 10,  # -> '-'
        }
    )

    labels = truth_labels(table.effects, agree=0.7, min_effect=0.02)

    assert labels.tolist() == [["+", "none"], ["none", "-"]]


def test_truth_labels_ignore_nan_states() -> None:
    effects = np.array([[[0.1, np.nan, np.nan, 0.1]]])
    assert truth_labels(effects, 0.7, 0.02)[0, 0] == "+"
    assert truth_labels(np.full((1, 1, 3), np.nan), 0.7, 0.02)[0, 0] == "none"


def test_argmax_labels_need_enough_flips_in_one_direction() -> None:
    chosen = np.zeros((1, 2, 100))
    chosen[0, 0, :5] = 0.5  # 5% flip up
    chosen[0, 1, :1] = -0.5  # 1% flip down: below ARGMAX_MIN_FLIP
    assert argmax_labels(chosen, 0.7).tolist() == [["+", "none"]]


@pytest.mark.parametrize(
    ("claimed", "truth", "expected"),
    [("+", "+", "correct"), ("+", "-", "wrong_sign"), ("-", "none", "overclaim"), ("none", "+", "missed"),
     ("none", "none", "correct")],
)  # fmt: skip
def test_outcome_categories(claimed: str, truth: str, expected: str) -> None:
    assert outcome(claimed, truth) == expected


def test_confidence_buckets_are_half_open() -> None:
    assert [confidence_bucket(c) for c in (0.0, 0.59, 0.6, 0.8, 1.0)] == [
        "[0, 0.6)",
        "[0, 0.6)",
        "[0.6, 0.8)",
        "[0.8, 1]",
        "[0.8, 1]",
    ]


def _claims() -> tuple[Claim, ...]:
    return (
        Claim(0, 0, 0, "+", 0.9),  # correct
        Claim(1, 0, 1, "-", 0.7),  # overclaim (truth none)
        Claim(2, 1, 1, "none", 0.5),  # missed (truth '-')
    )


def test_summarize_reports_accuracy_recall_and_brier() -> None:
    table = _table({(0, 0): [0.1] * 10, (0, 1): [0.0] * 10, (1, 0): [0.0] * 10, (1, 1): [-0.1] * 10})
    truths = pair_truths(table)

    summary = summarize(score_claims(_claims(), truths), truths)

    assert summary["overall"] == {"n": 3, "correct": 1, "accuracy": pytest.approx(1 / 3)}
    assert summary["signed_claims"]["accuracy"] == 0.5
    assert summary["outcomes"] == {"correct": 1, "wrong_sign": 0, "overclaim": 1, "missed": 1}
    assert summary["brier"] == pytest.approx(((0.9 - 1) ** 2 + 0.7**2 + 0.5**2) / 3)
    assert summary["recall"]["signed_truths"] == 2
    assert summary["recall"]["recall"] == 0.5
    assert summary["by_confidence"]["[0.8, 1]"]["accuracy"] == 1.0


def test_coverage_lists_strongest_signed_truths_and_unclaimed_as_none() -> None:
    table = _table({(0, 0): [0.1] * 4, (1, 0): [0.3] * 4, (0, 1): [0.0] * 4, (1, 1): [0.0] * 4})
    truths = pair_truths(table)

    rows = coverage(truths, [Claim(0, 0, 0, "+", 0.9)], n_heads=2, top_k=3)

    assert [(r["input_index"], r["rank"], r["claimed"], r["hit"]) for r in rows] == [
        (1, 1, None, False),
        (0, 2, "+", True),
    ]


def test_sensitivity_covers_the_whole_grid() -> None:
    table = _table({(0, 0): [0.03] * 10, (0, 1): [0.0] * 10, (1, 0): [0.0] * 10, (1, 1): [0.0] * 10})

    rows = sensitivity(table, (Claim(0, 0, 0, "+", 0.9),))

    assert len(rows) == 16
    by_threshold = {(r["agree"], r["min_effect"]): r["accuracy"] for r in rows}
    assert by_threshold[(0.7, 0.02)] == 1.0
    assert by_threshold[(0.7, 0.05)] == 0.0


def test_bootstrap_is_stable_for_unanimous_effects() -> None:
    table = _table({(0, 0): [0.1] * 20, (0, 1): [0.0] * 20, (1, 0): [0.0] * 20, (1, 1): [-0.1] * 20})

    result = bootstrap_scores(table, _claims(), 0.7, 0.02, seed=0, resamples=20)

    assert result["claim_truth_stability"] == [1.0, 1.0, 1.0]
    assert result["overall_accuracy_ci95"] == [pytest.approx(1 / 3), pytest.approx(1 / 3)]


def test_parse_claims_resolves_names_and_rejects_bad_entries() -> None:
    spec = known_spec()
    text = claims_markdown(
        [
            {"input": "speed", "head": "climb", "direction": "+", "confidence": 0.9},
            {"input": "speed", "head": "climb", "direction": "-", "confidence": 0.2},
            {"input": "ghost", "head": "climb", "direction": "+", "confidence": 0.9},
            {"input": "speed", "head": "dive", "direction": "up", "confidence": 0.9},
            {"input": "speed", "head": "dive", "direction": "-", "confidence": 1.5},
            "not an object",
            {"input": ["speed"], "head": "dive", "direction": "-", "confidence": 0.5},
            {"input": "noise", "head": "dive", "direction": "none", "confidence": 0.5},
        ]
    )

    parsed = parse_claims(text, spec, "shown")

    assert [(c.input_index, c.head_index, c.direction) for c in parsed.claims] == [(1, 1, "none")]
    assert [r.position for r in parsed.rejected] == [0, 1, 2, 3, 4, 5, 6]
    assert parsed.rejected[0].reason == "pair claimed more than once"
    assert parsed.duplicate_pairs == ((0, 0),)


def test_parse_claims_uses_blind_labels_in_blind_mode() -> None:
    text = claims_markdown([{"input": "x0", "head": "head_1", "direction": "-", "confidence": 0.6}])

    parsed = parse_claims(text, known_spec(), "blind")

    assert [(c.input_index, c.head_index) for c in parsed.claims] == [(0, 1)]


@pytest.mark.parametrize("text", ["no block", "```json\n[1, 2]\n```", "```json\n{bad\n```"])
def test_parse_claims_without_valid_block_raises(text: str) -> None:
    with pytest.raises(ClaimsError):
        parse_claims(text, known_spec(), "shown")
