#!/usr/bin/env python3
"""Torch parity for weightreading's numpy reference forward pass (``weightreading.reference``).

Run with torch available and ``src`` on ``PYTHONPATH`` (see ``scripts/weightreading_forward_parity.sbatch``);
torch is NOT a dependency of this repository. Checks, on random base states and on multi-step sequences:

1. ``torch.load(weights_only=True)`` tensors equal the restricted reader's tensors bit for bit;
2. a plain-torch mirror (``nn.Sequential(Linear, ReLU, LayerNorm)``, ``nn.GRU``, ``nn.LayerNorm``,
   ``Linear`` heads; loaded with ``strict=True`` under the checkpoint's own keys) matches the numpy
   reference in float64 (semantics, tolerance ``TOL_FLOAT64``) and float32 (``TOL_FLOAT32``);
3. with ``--lag-root``, the REAL mtd-rl-lag modules match too: ``BaselineActor`` for networks without an
   ACT MLP, and ``algorithms/utils/gru.py`` ``GRULayer`` (including its mask path) for every network.

Real-module argmax actions must agree on every decision (a float32 near-tie would show up here).
Writes a JSON report; exit 0 = every check passed, 1 = a mismatch, 2 = bad arguments.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from weightreading.reference import ReferencePolicy, build_policy, layer_norm
from weightreading.sampling import sample_base_states
from weightreading.spec import Spec, check_against_tensors, load_spec
from weightreading.torch_reader import read_checkpoint

TOL_FLOAT64 = 1e-9
TOL_FLOAT32 = 1e-4
SAMPLES = 256
SEQUENCE_STEPS = 50
SEQUENCE_BATCH = 16
SEED = 0


def _set_module(root: nn.Module, dotted: str, module: nn.Module) -> None:
    parts = dotted.strip(".").split(".")
    node = root
    for part in parts[:-1]:
        if not hasattr(node, part):
            node.add_module(part, nn.Module())
        node = getattr(node, part)
    node.add_module(parts[-1], module)


def _mlp(block_input: int, sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = block_input
    for size in sizes:
        layers += [nn.Linear(previous, size), nn.ReLU(), nn.LayerNorm(size)]
        previous = size
    return nn.Sequential(*layers)


def build_mirror(spec: Spec) -> nn.Module:
    root = nn.Module()
    for block in spec.blocks:
        if block.block_type == "mlp":
            _set_module(root, block.prefix, _mlp(block.input_size, block.sizes))
        elif block.block_type == "gru":
            _set_module(root, block.prefix, nn.GRU(block.input_size, block.sizes[0], num_layers=1))
        elif block.block_type == "layernorm":
            _set_module(root, block.prefix, nn.LayerNorm(block.sizes[0], eps=block.eps))
        else:
            heads = nn.ModuleList()
            for bins in block.sizes:
                head = nn.Module()
                head.add_module(block.param, nn.Linear(block.input_size, bins))
                heads.append(head)
            _set_module(root, block.prefix, heads)
    return root


def _module(root: nn.Module, prefix: str) -> nn.Module:
    return root.get_submodule(prefix.strip("."))


def mirror_step(root: nn.Module, spec: Spec, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[list, torch.Tensor]:
    values = {"observation": obs}
    new_hidden = hidden
    for block in spec.blocks:
        source = values[block.input]
        module = _module(root, block.prefix)
        if block.block_type == "gru":
            out, h_n = module(source.unsqueeze(0), hidden.unsqueeze(0))
            values[block.block_id], new_hidden = out[0], h_n[0]
        elif block.block_type == "categorical_heads":
            return [getattr(head, block.param)(source) for head in module], new_hidden
        else:
            values[block.block_id] = module(source)
    raise ValueError("spec has no categorical_heads block")


def _max_diff(torch_values: list, numpy_values: tuple) -> float:
    return max(
        float(np.abs(t.detach().cpu().numpy().astype(np.float64) - n).max())
        for t, n in zip(torch_values, numpy_values, strict=True)
    )


def compare(root: nn.Module, spec: Spec, policy: ReferencePolicy, obs_seq: np.ndarray, dtype: torch.dtype) -> dict:
    """Max |torch - numpy| over logits and hidden state along a (T, N, inputs) sequence."""
    model = root.to(dtype)
    hidden_t = torch.zeros(obs_seq.shape[1], policy.hidden_size, dtype=dtype)
    hidden_n = policy.initial_hidden(obs_seq.shape[1])
    logits_diff = hidden_diff = 0.0
    argmax_agree = []
    with torch.no_grad():
        for obs in obs_seq:
            logits_t, hidden_t = mirror_step(model, spec, torch.as_tensor(obs, dtype=dtype), hidden_t)
            out = policy.step(obs, hidden_n)
            hidden_n = out.hidden
            logits_diff = max(logits_diff, _max_diff(logits_t, out.logits))
            hidden_diff = max(hidden_diff, _max_diff([hidden_t], (hidden_n,)))
            argmax_agree += [
                float((t.argmax(-1).numpy() == n.argmax(-1)).mean()) for t, n in zip(logits_t, out.logits, strict=True)
            ]
    return {
        "logits_max_abs_diff": logits_diff,
        "hidden_max_abs_diff": hidden_diff,
        "argmax_agreement_min": min(argmax_agree),
    }


def _load_file_module(path: Path, name: str) -> object:
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def compare_real_lag(lag_root: Path, spec: Spec, state: dict, policy: ReferencePolicy, obs_seq: np.ndarray) -> dict:
    """mtd-rl-lag's own GRULayer (mask path, T=1) and, without an ACT MLP, its BaselineActor."""
    gru_module = _load_file_module(lag_root / "algorithms/utils/gru.py", "lag_gru")
    gru_block = next(b for b in spec.blocks if b.block_type == "gru")
    layer = gru_module.GRULayer(gru_block.input_size, gru_block.sizes[0], 1)
    layer.load_state_dict({k.removeprefix("rnn."): v for k, v in state.items() if k.startswith("rnn.")}, strict=True)
    mirror = build_mirror(spec)
    mirror.load_state_dict(state, strict=True)
    report: dict = {}
    start = np.random.default_rng(SEED).uniform(-1.0, 1.0, (obs_seq.shape[1], policy.hidden_size))
    with torch.no_grad():
        base_out = _module(mirror, spec.blocks[0].prefix)(torch.as_tensor(obs_seq[0], dtype=torch.float32))
        hxs = torch.as_tensor(start, dtype=torch.float32).unsqueeze(1)  # [N, L=1, hidden]; nonzero h
        normed, new_hxs = layer(base_out, hxs, torch.ones(base_out.shape[0], 1))
    out = policy.step(obs_seq[0], start)
    norm_block = next(b for b in spec.blocks if b.block_type == "layernorm")
    p = policy.params
    ref_normed = layer_norm(out.hidden, p[f"{norm_block.prefix}weight"], p[f"{norm_block.prefix}bias"], norm_block.eps)
    report["gru_layer_hidden_max_abs_diff"] = float(np.abs(new_hxs[:, 0].numpy() - out.hidden).max())
    report["gru_layer_output_max_abs_diff"] = float(np.abs(normed.numpy() - ref_normed).max())
    has_act_mlp = any(b.block_type == "mlp" and b.input != "observation" for b in spec.blocks)
    if has_act_mlp:
        return report
    actor_module = _load_file_module(lag_root / "envs/JSBSim/model/baseline_actor.py", "lag_baseline_actor")
    actor = actor_module.BaselineActor(input_dim=len(spec.inputs))
    actor.load_state_dict(state, strict=True)
    hidden = np.zeros((obs_seq.shape[1], 1, policy.hidden_size), dtype=np.float32)
    hidden_n = policy.initial_hidden(obs_seq.shape[1])
    agree, diff = [], 0.0
    for obs in obs_seq:
        actions, hidden_t = actor(obs.astype(np.float32), hidden)
        hidden = hidden_t.numpy()
        ref = policy.step(obs, hidden_n)
        hidden_n = ref.hidden
        diff = max(diff, float(np.abs(hidden[:, 0] - hidden_n).max()))
        agree.append(float((actions.numpy() == np.stack([lg.argmax(-1) for lg in ref.logits], axis=1)).mean()))
    report |= {"baseline_actor_hidden_max_abs_diff": diff, "baseline_actor_action_agreement_min": min(agree)}
    return report


def run(checkpoint: Path, spec_path: Path, lag_root: Path | None) -> dict:
    spec = load_spec(spec_path)
    checkpoint_data = read_checkpoint(checkpoint)
    check_against_tensors(spec, checkpoint_data.tensors, checkpoint_data.sha256)
    policy = build_policy(spec, checkpoint_data.tensors)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reader = {t.name: t.to_numpy() for t in checkpoint_data.tensors}
    tensors_equal = list(state) == list(reader) and all(np.array_equal(state[k].numpy(), reader[k]) for k in reader)
    rng = np.random.default_rng(SEED)
    if spec.evaluation is not None:
        states = sample_base_states(spec.evaluation, len(spec.inputs), SAMPLES, rng).obs
        sequence = sample_base_states(spec.evaluation, len(spec.inputs), SEQUENCE_STEPS * SEQUENCE_BATCH, rng).obs
    else:
        states = rng.standard_normal((SAMPLES, len(spec.inputs)))
        sequence = rng.standard_normal((SEQUENCE_STEPS * SEQUENCE_BATCH, len(spec.inputs)))
    sequence = sequence.reshape(SEQUENCE_STEPS, SEQUENCE_BATCH, len(spec.inputs))
    report: dict = {
        "checkpoint": str(checkpoint),
        "spec": str(spec_path),
        "torch": torch.__version__,
        "tensors_equal": tensors_equal,
    }
    for label, dtype, tol in (("float64", torch.float64, TOL_FLOAT64), ("float32", torch.float32, TOL_FLOAT32)):
        mirror = build_mirror(spec)
        mirror.load_state_dict(state, strict=True)
        single = compare(mirror, spec, policy, states[None], dtype)
        multi = compare(mirror, spec, policy, sequence, dtype)
        diffs = [single["logits_max_abs_diff"], single["hidden_max_abs_diff"]]
        diffs += [multi["logits_max_abs_diff"], multi["hidden_max_abs_diff"]]
        ok = max(diffs) < tol
        report[f"mirror_{label}"] = {"tolerance": tol, "single_step": single, "sequence": multi, "ok": ok}
    if lag_root is not None:
        real = compare_real_lag(lag_root, spec, state, policy, sequence)
        diffs = [v for k, v in real.items() if k.endswith("max_abs_diff")]
        agreement = [v for k, v in real.items() if k.endswith("agreement_min")]
        ok = max(diffs) < TOL_FLOAT32 and all(value == 1.0 for value in agreement)
        report["real_lag_modules"] = {**real, "tolerance": TOL_FLOAT32, "ok": ok}
    checks = [tensors_equal] + [report[k]["ok"] for k in report if isinstance(report[k], dict) and "ok" in report[k]]
    report["ok"] = all(checks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lag-root", type=Path, help="mtd-rl-lag checkout: also compare its real modules")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run(args.checkpoint, args.spec, args.lag_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "ok": report["ok"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
