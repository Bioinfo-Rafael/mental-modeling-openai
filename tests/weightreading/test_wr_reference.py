"""Reference forward pass: hand-computed MLP/GRU values, dataflow, carried state, expected value."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from weightreading.architecture import Block
from weightreading.reference import (
    ReferencePolicy,
    build_policy,
    expected_value,
    gru_step,
    layer_norm,
    mlp_forward,
    sigmoid,
)
from weightreading.spec import load_spec, parse_spec
from weightreading.torch_reader import read_checkpoint

from .wr_helpers import SMALL_SPEC, small_arrays

EPS = 1e-5
CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "weightreading"


def test_mlp_forward_one_layer_matches_hand_computation() -> None:
    # Arrange: x = [1, -1]; rows [1,0], [0,1], [1,1] -> relu [1, 0, 0] -> LN with g = 1, beta = 0
    block = Block("b", "mlp", "fc.", "observation", 2, (3,), eps=EPS)
    params = {
        "fc.0.weight": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        "fc.0.bias": np.zeros(3),
        "fc.2.weight": np.ones(3),
        "fc.2.bias": np.zeros(3),
    }
    # mean 1/3, biased variance 2/9
    scale = 1.0 / math.sqrt(2.0 / 9.0 + EPS)
    expected = np.array([[2.0 / 3.0 * scale, -1.0 / 3.0 * scale, -1.0 / 3.0 * scale]])

    # Act
    out = mlp_forward(np.array([[1.0, -1.0]]), params, block)

    # Assert
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_layer_norm_applies_gain_and_bias_after_normalising() -> None:
    v = np.array([[1.0, 3.0]])
    out = layer_norm(v, np.array([2.0, 2.0]), np.array([0.5, -0.5]), 0.0)
    np.testing.assert_allclose(out, [[-1.5, 1.5]])


def test_gru_step_matches_pytorch_equations_with_b_hn_inside_reset() -> None:
    # Arrange: hidden size 1, input size 1; gate rows [r; z; n]
    block = Block("g", "gru", "gru.", "base", 1, (1,))
    w_ih, w_hh = [0.5, -0.3, 0.8], [0.2, 0.4, -0.7]
    b_ih, b_hh = [0.1, 0.05, -0.2], [-0.1, 0.3, 0.9]
    params = {
        "gru.weight_ih_l0": np.array(w_ih).reshape(3, 1),
        "gru.weight_hh_l0": np.array(w_hh).reshape(3, 1),
        "gru.bias_ih_l0": np.array(b_ih),
        "gru.bias_hh_l0": np.array(b_hh),
    }
    u, h = 1.5, -0.4
    r = 1.0 / (1.0 + math.exp(-(w_ih[0] * u + b_ih[0] + w_hh[0] * h + b_hh[0])))
    z = 1.0 / (1.0 + math.exp(-(w_ih[1] * u + b_ih[1] + w_hh[1] * h + b_hh[1])))
    n = math.tanh(w_ih[2] * u + b_ih[2] + r * (w_hh[2] * h + b_hh[2]))
    expected = (1.0 - z) * n + z * h

    # Act
    out = gru_step(np.array([[u]]), np.array([[h]]), params, block)

    # Assert
    assert out[0, 0] == pytest.approx(expected, rel=1e-12)


def test_sigmoid_is_finite_for_large_arguments() -> None:
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def _small_policy() -> ReferencePolicy:
    spec = parse_spec(SMALL_SPEC)
    params = {key: value.astype(np.float64) for key, value in small_arrays().items()}
    return ReferencePolicy(spec, params)


def test_step_carries_pre_layernorm_state_and_feeds_normalised_state_to_heads() -> None:
    # Arrange
    policy = _small_policy()
    spec, p = policy.spec, policy.params
    obs = np.array([[0.2, -0.1, 0.7], [0.0, 0.5, 0.3]])
    hidden = np.array([[0.1, -0.2, 0.3, 0.0], [0.0, 0.0, 0.0, 0.0]])
    base = mlp_forward(obs, p, spec.blocks[0])
    h_new = gru_step(base, hidden, p, spec.blocks[1])
    normed = layer_norm(h_new, p["rnn.norm.weight"], p["rnn.norm.bias"], EPS)
    expected_logits = [
        normed @ p[f"act.action_outs.{k}.logits_net.weight"].T + p[f"act.action_outs.{k}.logits_net.bias"]
        for k in range(2)
    ]

    # Act
    out = policy.step(obs, hidden)

    # Assert
    np.testing.assert_allclose(out.hidden, h_new, rtol=1e-12)
    assert not np.allclose(out.hidden, normed)
    for got, want in zip(out.logits, expected_logits, strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-12)


def test_rollout_equals_repeated_steps() -> None:
    policy = _small_policy()
    rng = np.random.default_rng(1)
    sequence = rng.standard_normal((5, 3, 3))
    hidden = policy.initial_hidden(3)
    for obs in sequence:
        last = policy.step(obs, hidden)
        hidden = last.hidden

    out = policy.rollout(sequence, policy.initial_hidden(3))

    np.testing.assert_allclose(out.hidden, last.hidden)
    np.testing.assert_allclose(out.logits[0], last.logits[0])


def test_step_with_wrong_observation_shape_raises() -> None:
    policy = _small_policy()
    with pytest.raises(ValueError, match="obs must have shape"):
        policy.step(np.zeros((2, 4)), policy.initial_hidden(2))


def test_rollout_with_empty_sequence_raises() -> None:
    policy = _small_policy()
    with pytest.raises(ValueError, match="obs_sequence"):
        policy.rollout(np.zeros((0, 2, 3)), policy.initial_hidden(2))


def test_expected_value_uniform_logits_is_mean_of_bin_values() -> None:
    out = expected_value(np.zeros((1, 3)), (100.0, 0.0, -40.0))
    assert out[0] == pytest.approx(20.0)


def test_expected_value_binary_head_is_probability_of_yes() -> None:
    logits = np.array([[0.3, 1.1]])
    out = expected_value(logits, (0.0, 1.0))
    assert out[0] == pytest.approx(1.0 / (1.0 + math.exp(-0.8)))


def test_baseline_checkpoint_forward_has_expected_head_shapes(lag_model_dir: Path) -> None:
    checkpoint = read_checkpoint(lag_model_dir / "baseline_model.pt")
    policy = build_policy(load_spec(CONFIGS / "lag_baseline_12.spec.json"), checkpoint.tensors)

    out = policy.step(np.zeros((4, 12)), policy.initial_hidden(4))

    assert [logits.shape for logits in out.logits] == [(4, 41), (4, 41), (4, 41), (4, 30)]
    assert out.hidden.shape == (4, 128)
    assert all(np.isfinite(logits).all() for logits in out.logits)
