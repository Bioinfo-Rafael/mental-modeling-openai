"""Numpy float64 reference forward pass of a spec'd policy, for evaluation only (never in the LLM run).

Semantics mirror the mtd-rl-lag source exactly:

* ``mlp``: ``nn.Sequential(Linear, ReLU, LayerNorm)`` per layer (``algorithms/utils/mlp.py:15-17``,
  ``envs/JSBSim/model/baseline_actor.py:27-30``); activation_id 1 = ReLU.
* ``gru``: one ``nn.GRU`` step, PyTorch gate order ``[r; z; n]`` with ``b_hn`` inside ``r * (...)``;
  ``GRULayer.forward`` returns ``(norm(x), hxs)`` (``algorithms/utils/gru.py:76-123``,
  ``baseline_actor.py:57-63``), so the carried state is the pre-LayerNorm ``h'`` and the following
  ``layernorm`` block only feeds the next block.
* ``layernorm``: ``nn.LayerNorm`` with the default ``eps = 1e-5`` (taken from the spec block), biased
  variance over the last axis.
* ``categorical_heads``: one ``Linear`` per head, logits only (``distributions.py:105-119``); the
  optional ACT MLP is a separate ``mlp`` block (``act.py:55-58``, ``:223-224``).

The GRU mask (``gru.py:76``) is 1 inside an episode and is not modelled; ``h = 0`` starts an episode.
No input clipping happens inside the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from weightreading.architecture import GRU_GATES, MLP_NORM_OFFSET, MLP_STRIDE, Block
from weightreading.spec import Spec
from weightreading.tensors import Tensor

OBSERVATION = "observation"


def load_params(tensors: tuple[Tensor, ...]) -> dict[str, np.ndarray]:
    """``{key: float64 copy}`` of every checkpoint tensor."""
    return {tensor.name: tensor.to_numpy().astype(np.float64) for tensor in tensors}


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def layer_norm(v: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    """``nn.LayerNorm`` over the last axis (biased variance, ``eps`` inside the square root)."""
    mean = v.mean(axis=-1, keepdims=True)
    var = ((v - mean) ** 2).mean(axis=-1, keepdims=True)
    return gain * (v - mean) / np.sqrt(var + eps) + bias


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def mlp_forward(x: np.ndarray, params: Mapping[str, np.ndarray], block: Block) -> np.ndarray:
    for layer in range(len(block.sizes)):
        lin = f"{block.prefix}{MLP_STRIDE * layer}"
        norm = f"{block.prefix}{MLP_STRIDE * layer + MLP_NORM_OFFSET}"
        hidden = np.maximum(linear(x, params[f"{lin}.weight"], params[f"{lin}.bias"]), 0.0)
        x = layer_norm(hidden, params[f"{norm}.weight"], params[f"{norm}.bias"], block.eps)
    return x


def gru_step(u: np.ndarray, h: np.ndarray, params: Mapping[str, np.ndarray], block: Block) -> np.ndarray:
    """One ``nn.GRU`` step; returns the new (pre-LayerNorm) hidden state ``h'``."""
    p = block.prefix
    gi = linear(u, params[f"{p}weight_ih_l0"], params[f"{p}bias_ih_l0"])
    gh = linear(h, params[f"{p}weight_hh_l0"], params[f"{p}bias_hh_l0"])
    i_r, i_z, i_n = np.split(gi, GRU_GATES, axis=-1)
    h_r, h_z, h_n = np.split(gh, GRU_GATES, axis=-1)
    reset = sigmoid(i_r + h_r)
    update = sigmoid(i_z + h_z)
    candidate = np.tanh(i_n + reset * h_n)  # h_n already holds W_hn h + b_hn
    return (1.0 - update) * candidate + update * h


def heads_forward(v: np.ndarray, params: Mapping[str, np.ndarray], block: Block) -> tuple[np.ndarray, ...]:
    keys = (f"{block.prefix}{head}.{block.param}" for head in range(len(block.sizes)))
    return tuple(linear(v, params[f"{key}.weight"], params[f"{key}.bias"]) for key in keys)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def expected_value(logits: np.ndarray, values: tuple[float, ...]) -> np.ndarray:
    """``E[v] = sum_k p_k v_k`` per row (for a ``[0, 1]`` shoot head this is ``p(yes)``)."""
    return softmax(logits) @ np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class StepOutput:
    logits: tuple[np.ndarray, ...]  # one (N, bins) array per head
    hidden: np.ndarray  # carried state after this step, (N, hidden); (N, 0) without a GRU


@dataclass(frozen=True)
class ReferencePolicy:
    """A spec plus float64 parameters. ``step`` is batched: ``obs`` is (N, inputs)."""

    spec: Spec
    params: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        grus = [block for block in self.spec.blocks if block.block_type == "gru"]
        if len(grus) > 1:
            raise ValueError("the reference forward supports at most one GRU block")

    @property
    def hidden_size(self) -> int:
        grus = [block.sizes[0] for block in self.spec.blocks if block.block_type == "gru"]
        return grus[0] if grus else 0

    @property
    def input_size(self) -> int:
        return len(self.spec.inputs)

    def initial_hidden(self, batch: int) -> np.ndarray:
        return np.zeros((batch, self.hidden_size), dtype=np.float64)

    def step(self, obs: np.ndarray, hidden: np.ndarray) -> StepOutput:
        obs = np.asarray(obs, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != self.input_size:
            raise ValueError(f"obs must have shape (N, {self.input_size}), got {obs.shape}")
        if hidden.shape != (obs.shape[0], self.hidden_size):
            raise ValueError(f"hidden must have shape ({obs.shape[0]}, {self.hidden_size}), got {hidden.shape}")
        values: dict[str, np.ndarray] = {OBSERVATION: obs}
        new_hidden = hidden
        for block in self.spec.blocks:
            source = values[block.input]
            if block.block_type == "mlp":
                values[block.block_id] = mlp_forward(source, self.params, block)
            elif block.block_type == "gru":
                new_hidden = gru_step(source, hidden, self.params, block)
                values[block.block_id] = new_hidden
            elif block.block_type == "layernorm":
                p = block.prefix
                values[block.block_id] = layer_norm(
                    source, self.params[f"{p}weight"], self.params[f"{p}bias"], block.eps
                )
            else:
                return StepOutput(heads_forward(source, self.params, block), new_hidden)
        raise ValueError("spec has no categorical_heads block")

    def rollout(self, obs_sequence: np.ndarray, hidden: np.ndarray) -> StepOutput:
        """Run ``obs_sequence`` (T, N, inputs) from ``hidden``; returns the LAST step's logits and state."""
        if obs_sequence.ndim != 3 or obs_sequence.shape[0] < 1:
            raise ValueError("obs_sequence must have shape (T >= 1, N, inputs)")
        output = StepOutput((), hidden)
        for obs in obs_sequence:
            output = self.step(obs, output.hidden)
        return output


def build_policy(spec: Spec, tensors: tuple[Tensor, ...]) -> ReferencePolicy:
    return ReferencePolicy(spec, load_params(tensors))
