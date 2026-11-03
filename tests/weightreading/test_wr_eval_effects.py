"""Base-state sampling, perturbation conventions and measured effect signs on a known-sign network."""

from __future__ import annotations

import numpy as np
import pytest

from weightreading.effects import CONDITIONS, evaluate_condition, measure_effects, warm_hidden
from weightreading.reference import ReferencePolicy
from weightreading.sampling import perturb, sample_base_states
from weightreading.scoring import pair_truths
from weightreading.spec import parse_spec

from .wr_eval_helpers import eval_spec_dict, known_sign_arrays, known_spec

N_STATES = 64
SPEED, NOISE, ROLL_SIN, ROLL_COS, SIDE, MSL = range(6)
CLIMB, DIVE = 0, 1


def _setup(n: int = N_STATES, seed: int = 0):  # noqa: ANN202 - (spec, policy, base)
    spec = known_spec()
    params = {k: v.astype(np.float64) for k, v in known_sign_arrays().items()}
    base = sample_base_states(spec.evaluation, len(spec.inputs), n, np.random.default_rng(seed))
    return spec, ReferencePolicy(spec, params), base


def test_sampling_respects_ranges_unit_circle_values_and_presence() -> None:
    spec, _, base = _setup(n=500)
    obs = base.obs

    assert ((obs[:, SPEED] >= -1) & (obs[:, SPEED] <= 1)).all()
    np.testing.assert_allclose(obs[:, ROLL_SIN] ** 2 + obs[:, ROLL_COS] ** 2, 1.0)
    assert (np.abs(base.angles[:, 0]) <= 1.0).all()
    assert set(np.unique(obs[:, SIDE])) == {-1.0, 1.0}
    absent = ~base.present[:, 0]
    assert absent.any() and base.present[:, 0].any()
    assert (obs[absent, MSL] == 0).all()
    assert (obs[~absent, MSL] >= 0.1).all()


def test_sampling_is_deterministic_for_a_seed() -> None:
    _, _, first = _setup(seed=3)
    _, _, second = _setup(seed=3)
    np.testing.assert_array_equal(first.obs, second.obs)


def test_continuous_perturbation_is_clamped_at_range_edges() -> None:
    spec, _, base = _setup()
    obs = base.obs.copy()
    obs[0, SPEED] = 1.0  # at the upper edge
    edge = type(base)(obs, base.angles, base.present)

    move = perturb(edge, spec.evaluation, SPEED, step_fraction=0.05)

    assert move.high[0, SPEED] == 1.0
    assert move.low[0, SPEED] == pytest.approx(0.9)
    assert move.scale[0] == pytest.approx(2.0 / 0.1)
    np.testing.assert_array_equal(np.delete(move.high, SPEED, axis=1), np.delete(obs, SPEED, axis=1))


def test_cos_perturbation_moves_towards_level_and_stays_on_unit_circle() -> None:
    spec, _, base = _setup()

    move = perturb(base, spec.evaluation, ROLL_COS)

    assert (move.high[:, ROLL_COS] >= move.low[:, ROLL_COS]).all()
    np.testing.assert_allclose(move.high[:, ROLL_SIN] ** 2 + move.high[:, ROLL_COS] ** 2, 1.0)
    assert (np.sign(move.high[:, ROLL_SIN]) * np.sign(base.obs[:, ROLL_SIN]) >= 0).all()


def test_cos_perturbation_clamps_to_each_side_of_an_asymmetric_range() -> None:
    raw = eval_spec_dict()
    raw["evaluation"]["angles"][0]["range"] = [-0.2, 1.0]
    spec = parse_spec(raw)
    base = sample_base_states(spec.evaluation, len(spec.inputs), 200, np.random.default_rng(0))

    move = perturb(base, spec.evaluation, ROLL_COS)

    low_theta = np.arctan2(move.low[:, ROLL_SIN], move.low[:, ROLL_COS])
    assert (low_theta >= -0.2 - 1e-12).all() and (low_theta <= 1.0 + 1e-12).all()


def test_sin_perturbation_increases_the_angle() -> None:
    spec, _, base = _setup()
    move = perturb(base, spec.evaluation, ROLL_SIN)
    assert (move.high[:, ROLL_SIN] > move.low[:, ROLL_SIN]).all()


def test_raw_angle_mode_moves_only_one_coordinate() -> None:
    spec, _, base = _setup()

    move = perturb(base, spec.evaluation, ROLL_SIN, angle_mode="raw")

    np.testing.assert_array_equal(move.high[:, ROLL_COS], base.obs[:, ROLL_COS])
    assert (move.high[:, ROLL_SIN] >= move.low[:, ROLL_SIN]).all()


def test_discrete_perturbation_compares_lowest_and_highest_value() -> None:
    spec, _, base = _setup()
    move = perturb(base, spec.evaluation, SIDE)
    assert (move.low[:, SIDE] == -1).all() and (move.high[:, SIDE] == 1).all()
    np.testing.assert_array_equal(move.scale, 1.0)


def test_presence_member_is_invalid_where_absent() -> None:
    spec, _, base = _setup()
    move = perturb(base, spec.evaluation, MSL)
    np.testing.assert_array_equal(move.valid, base.present[:, 0])


def test_unknown_angle_mode_raises() -> None:
    spec, _, base = _setup()
    with pytest.raises(ValueError, match="angle_mode"):
        perturb(base, spec.evaluation, SPEED, angle_mode="degrees")


@pytest.mark.parametrize("condition", CONDITIONS)
def test_known_sign_network_gives_expected_truths(condition: str) -> None:
    # Arrange
    spec, policy, base = _setup()

    # Act
    table = measure_effects(policy, spec.evaluation, base, condition, warmup_steps=3)
    truths = pair_truths(table)

    # Assert
    assert truths[(SPEED, CLIMB)].truth == "+"
    assert truths[(SPEED, CLIMB)].frac_pos == 1.0
    assert truths[(SPEED, DIVE)].truth == "-"
    for index in (NOISE, ROLL_SIN, ROLL_COS, SIDE, MSL):
        assert truths[(index, CLIMB)].truth == "none"
        assert truths[(index, CLIMB)].mean_effect == 0.0
    assert truths[(MSL, CLIMB)].n_valid == int(base.present[:, 0].sum())


def test_effect_is_linearised_change_over_whole_range_as_span_fraction() -> None:
    # Arrange: finite difference by hand at one state
    spec, policy, base = _setup(n=1)
    move = perturb(base, spec.evaluation, SPEED)
    values_low, _ = evaluate_condition(policy, move.low, "h0", policy.initial_hidden(1), 1)
    values_high, _ = evaluate_condition(policy, move.high, "h0", policy.initial_hidden(1), 1)
    expected = (values_high[0, CLIMB] - values_low[0, CLIMB]) * move.scale[0] / 1.0  # span of [0, 1] = 1

    # Act
    table = measure_effects(policy, spec.evaluation, base, "h0")

    # Assert
    assert table.effects[SPEED, CLIMB, 0] == pytest.approx(expected)


def test_warm_hidden_without_gru_is_empty() -> None:
    _, policy, base = _setup()
    assert warm_hidden(policy, base.obs, 5).shape == (N_STATES, 0)


def test_unknown_condition_raises() -> None:
    _, policy, base = _setup()
    with pytest.raises(ValueError, match="unknown condition"):
        evaluate_condition(policy, base.obs, "cold", policy.initial_hidden(N_STATES), 1)
