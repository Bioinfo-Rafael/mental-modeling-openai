"""Per-step activations of a spec'd recurrent policy (numpy float64) for the intent-B mode.

``trace_episode`` replays a recorded episode free-running from ``h = 0`` (the recorded
observations are the only input) and keeps every intermediate value the equations in the prompt
name. ``validate_trace`` compares the replay with the recorded torch outputs, and
``window_arrays`` cuts one aircraft's window out of the trace. Key names follow the spec block
ids, so the prompt's architecture section and the arrays use the same vocabulary:

``x``, ``h_prev``, ``<mlp>.a<k>``, ``<gru>.r|z|n|h_new``, ``<layernorm>.y``,
``<heads>.logits_<h>``, ``<heads>.probs_<h>``, ``action`` (and ``step`` in a window).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from translate.policy.architecture import GRU_GATES, MLP_NORM_OFFSET, MLP_STRIDE, Block
from translate.policy.reference import OBSERVATION, ReferencePolicy, layer_norm, linear, sigmoid, softmax

KEY_X = "x"
KEY_H_PREV = "h_prev"
KEY_ACTION = "action"
KEY_STEP = "step"
# float32 vs float64 differences seen on the recorded episodes are ~1.7e-6; anything far above
# that means the trace does not describe the recorded policy and must not be shown to the model.
LOGITS_TOLERANCE = 1e-4
HIDDEN_TOLERANCE = 1e-4
INT_KEYS = (KEY_ACTION, KEY_STEP)


class TraceError(ValueError):
    """The replayed activations do not reproduce the recording."""


@dataclass(frozen=True)
class ActivationDoc:
    key: str
    shape_tail: tuple[int, ...]  # per-step shape (the window adds a leading L axis)
    description: str


def _heads_block(policy: ReferencePolicy) -> Block:
    return next(block for block in policy.spec.blocks if block.block_type == "categorical_heads")


def _block_docs(block: Block, policy: ReferencePolicy) -> list[ActivationDoc]:
    source = "x" if block.input == OBSERVATION else f"the output of `{block.input}`"
    bid = block.block_id
    if block.block_type == "mlp":
        return [
            ActivationDoc(
                f"{bid}.a{layer + 1}", (size,), f"MLP `{bid}` layer {layer + 1} output a{layer + 1} (after LN)"
            )
            for layer, size in enumerate(block.sizes)
        ]
    if block.block_type == "gru":
        size = (block.sizes[0],)
        return [
            ActivationDoc(f"{bid}.r", size, f"GRU `{bid}` reset gate r (input u = {source}, state h = h_prev)"),
            ActivationDoc(f"{bid}.z", size, f"GRU `{bid}` update gate z"),
            ActivationDoc(f"{bid}.n", size, f"GRU `{bid}` candidate n"),
            ActivationDoc(f"{bid}.h_new", size, f"GRU `{bid}` new state h' (carried: h_prev of the next step)"),
        ]
    if block.block_type == "layernorm":
        return [ActivationDoc(f"{bid}.y", (block.sizes[0],), f"LayerNorm `{bid}` output y")]
    docs = []
    for head, bins in enumerate(block.sizes):
        name = policy.spec.outputs[head].name
        docs.append(ActivationDoc(f"{bid}.logits_{head}", (bins,), f"head {head} ({name}) logits"))
        docs.append(ActivationDoc(f"{bid}.probs_{head}", (bins,), f"head {head} ({name}) softmax of the logits"))
    return docs


def activation_docs(policy: ReferencePolicy) -> tuple[ActivationDoc, ...]:
    """One entry per trace key, in trace order (``step`` first, as it appears in a window)."""
    heads = len(_heads_block(policy).sizes)
    docs = [
        ActivationDoc(KEY_STEP, (), "absolute decision index of each row (int64)"),
        ActivationDoc(KEY_X, (policy.input_size,), "observation x the network read (see Inputs)"),
        ActivationDoc(KEY_H_PREV, (policy.hidden_size,), "carried recurrent state h before this step (0 at step 0)"),
    ]
    for block in policy.spec.blocks:
        docs += _block_docs(block, policy)
    docs.append(ActivationDoc(KEY_ACTION, (heads,), "chosen bin per head = argmax of its logits (int64)"))
    return tuple(docs)


def _mlp_trace(x: np.ndarray, policy: ReferencePolicy, block: Block, out: dict[str, np.ndarray]) -> np.ndarray:
    params = policy.params
    for layer in range(len(block.sizes)):
        lin = f"{block.prefix}{MLP_STRIDE * layer}"
        norm = f"{block.prefix}{MLP_NORM_OFFSET + MLP_STRIDE * layer}"
        hidden = np.maximum(linear(x, params[f"{lin}.weight"], params[f"{lin}.bias"]), 0.0)
        x = layer_norm(hidden, params[f"{norm}.weight"], params[f"{norm}.bias"], block.eps)
        out[f"{block.block_id}.a{layer + 1}"] = x
    return x


def _gru_trace(
    u: np.ndarray, h: np.ndarray, policy: ReferencePolicy, block: Block, out: dict[str, np.ndarray]
) -> np.ndarray:
    p, params = block.prefix, policy.params
    gi = linear(u, params[f"{p}weight_ih_l0"], params[f"{p}bias_ih_l0"])
    gh = linear(h, params[f"{p}weight_hh_l0"], params[f"{p}bias_hh_l0"])
    i_r, i_z, i_n = np.split(gi, GRU_GATES, axis=-1)
    h_r, h_z, h_n = np.split(gh, GRU_GATES, axis=-1)
    reset = sigmoid(i_r + h_r)
    update = sigmoid(i_z + h_z)
    candidate = np.tanh(i_n + reset * h_n)
    new_h = (1.0 - update) * candidate + update * h
    bid = block.block_id
    out.update({f"{bid}.r": reset, f"{bid}.z": update, f"{bid}.n": candidate, f"{bid}.h_new": new_h})
    return new_h


def _heads_trace(v: np.ndarray, policy: ReferencePolicy, block: Block, out: dict[str, np.ndarray]) -> None:
    actions = []
    for head in range(len(block.sizes)):
        key = f"{block.prefix}{head}.{block.param}"
        logits = linear(v, policy.params[f"{key}.weight"], policy.params[f"{key}.bias"])
        out[f"{block.block_id}.logits_{head}"] = logits
        out[f"{block.block_id}.probs_{head}"] = softmax(logits)
        actions.append(logits.argmax(axis=-1))
    out[KEY_ACTION] = np.stack(actions, axis=-1)


def trace_step(
    policy: ReferencePolicy, obs: np.ndarray, hidden: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One batched decision: (activations by key, new carried state). ``obs`` (N, inputs)."""
    out: dict[str, np.ndarray] = {KEY_X: obs, KEY_H_PREV: hidden}
    values: dict[str, np.ndarray] = {OBSERVATION: obs}
    new_hidden = hidden
    for block in policy.spec.blocks:
        source = values[block.input]
        if block.block_type == "mlp":
            values[block.block_id] = _mlp_trace(source, policy, block, out)
        elif block.block_type == "gru":
            new_hidden = _gru_trace(source, hidden, policy, block, out)
            values[block.block_id] = new_hidden
        elif block.block_type == "layernorm":
            p = block.prefix
            y = layer_norm(source, policy.params[f"{p}weight"], policy.params[f"{p}bias"], block.eps)
            out[f"{block.block_id}.y"] = y
            values[block.block_id] = y
        else:
            _heads_trace(source, policy, block, out)
    return out, new_hidden


def trace_episode(policy: ReferencePolicy, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Free-running replay of ``obs`` (T, N, inputs) from ``h = 0``; every key gets shape (T, N, ...)."""
    if obs.ndim != 3 or obs.shape[0] < 1:
        raise TraceError("obs must have shape (T >= 1, N, inputs)")
    hidden = policy.initial_hidden(obs.shape[1])
    steps: list[dict[str, np.ndarray]] = []
    for row in np.asarray(obs, dtype=np.float64):
        values, hidden = trace_step(policy, row, hidden)
        steps.append(values)
    return {key: np.stack([step[key] for step in steps]) for key in steps[0]}


def concatenated_logits(trace: dict[str, np.ndarray], policy: ReferencePolicy) -> np.ndarray:
    block = _heads_block(policy)
    return np.concatenate([trace[f"{block.block_id}.logits_{h}"] for h in range(len(block.sizes))], axis=-1)


def _gru_key(policy: ReferencePolicy) -> str:
    block = next(block for block in policy.spec.blocks if block.block_type == "gru")
    return f"{block.block_id}.h_new"


def validate_trace(
    trace: dict[str, np.ndarray],
    policy: ReferencePolicy,
    logits: np.ndarray,
    gru_out: np.ndarray,
    actions: np.ndarray,
    rnn_in: np.ndarray | None = None,
) -> dict[str, float | int | bool]:
    """Replay vs recording: max |logit|, |h'| and (if given) |h_prev - rnn_in| differences, argmax
    agreement, and that every replayed value is finite."""
    logits_diff = float(np.max(np.abs(concatenated_logits(trace, policy) - logits)))
    hidden_diff = float(np.max(np.abs(trace[_gru_key(policy)] - gru_out)))
    if rnn_in is not None:
        state_in = np.asarray(rnn_in).reshape(trace[KEY_H_PREV].shape)
        hidden_diff = max(hidden_diff, float(np.max(np.abs(trace[KEY_H_PREV] - state_in))))
    finite = all(bool(np.all(np.isfinite(values))) for values in trace.values())
    mismatches = int(np.sum(trace[KEY_ACTION] != actions))
    return {
        "decisions": int(logits.shape[0]),
        "rows": int(logits.shape[1]),
        "max_abs_logits_diff": logits_diff,
        "max_abs_hidden_diff": hidden_diff,
        "action_agreement": 1.0 - mismatches / actions.size,
        "action_mismatches": mismatches,
        "logits_tolerance": LOGITS_TOLERANCE,
        "hidden_tolerance": HIDDEN_TOLERANCE,
        "checked_state_in": rnn_in is not None,
        "finite": finite,
        "ok": finite and logits_diff <= LOGITS_TOLERANCE and hidden_diff <= HIDDEN_TOLERANCE and mismatches == 0,
    }


def window_bounds(step: int, window_steps: int) -> tuple[int, int]:
    """Inclusive ``[max(0, step - window_steps + 1), step]``."""
    if step < 0 or window_steps < 1:
        raise TraceError("step must be >= 0 and window_steps >= 1")
    return max(0, step - window_steps + 1), step


def window_arrays(trace: dict[str, np.ndarray], row: int, start: int, end: int) -> dict[str, np.ndarray]:
    """Rows ``start..end`` (inclusive) of one aircraft: float32 values, int64 ``action`` and ``step``."""
    total = next(iter(trace.values())).shape[0]
    if not 0 <= start <= end < total:
        raise TraceError(f"window [{start}, {end}] is outside the episode (0..{total - 1})")
    arrays = {KEY_STEP: np.arange(start, end + 1, dtype=np.int64)}
    for key, values in trace.items():
        dtype = np.int64 if key in INT_KEYS else np.float32
        arrays[key] = np.ascontiguousarray(values[start : end + 1, row], dtype=dtype)
    return arrays
