"""A tiny MLP policy with KNOWN effect signs, plus a fake run directory, for evaluation tests.

Base layer (3 units): relu(speed + 2), relu(2 - speed), relu(1), then LayerNorm (g = 1, beta = 0).
For speed in [-1, 1] all three are positive and LN0 - LN1 = 2*speed / std is strictly increasing in
speed. Head ``climb`` (values [0, 1]) has logit difference LN0 - LN1 -> ``'+'``; head ``dive`` has
LN1 - LN0 -> ``'-'``. Every other input has zero weight -> effect exactly 0 -> ``'none'``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from weightreading.spec import parse_spec
from weightreading.torch_reader import read_checkpoint

from .wr_helpers import checkpoint_bytes

INPUT_NAMES = ("speed", "noise", "roll_sin", "roll_cos", "side", "msl_dist")
EVAL_SPEC = {
    "schema_version": 1,
    "network_alias": "net-feedbeef",
    "hidden_from_prompt": ["provenance", "notes", "evaluation"],
    "description": "Synthetic evaluation test network.",
    "inputs": [{"index": i, "name": name, "definition": name} for i, name in enumerate(INPUT_NAMES)],
    "input_clip": None,
    "architecture": [
        {"id": "base", "type": "mlp", "prefix": "base.mlp.fc.", "input": "observation", "input_size": 6, "hidden_sizes": [3]},
        {"id": "heads", "type": "categorical_heads", "prefix": "act.action_outs.", "input": "base", "input_size": 3},
    ],
    "outputs": [
        {"head": 0, "name": "climb", "bins": 2, "bin_value": {"values": [0, 1]}},
        {"head": 1, "name": "dive", "bins": 2, "bin_value": {"values": [0, 1]}},
    ],
    "evaluation": {
        "inputs": {
            "speed": {"range": [-1.0, 1.0]},
            "noise": {"range": [0.0, 2.0]},
            "side": {"values": [-1, 1]},
            "msl_dist": {"range": [0.1, 1.0]},
        },
        "angles": [{"sin": "roll_sin", "cos": "roll_cos", "range": [-1.0, 1.0]}],
        "presence": [{"inputs": ["msl_dist"], "p_present": 0.5}],
    },
}  # fmt: skip


def eval_spec_dict() -> dict:
    return copy.deepcopy(EVAL_SPEC)


def known_sign_arrays() -> dict[str, np.ndarray]:
    w1 = np.zeros((3, 6), dtype=np.float32)
    w1[0, 0], w1[1, 0] = 1.0, -1.0
    contrast = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]], dtype=np.float32)
    return {
        "base.mlp.fc.0.weight": w1,
        "base.mlp.fc.0.bias": np.array([2.0, 2.0, 1.0], dtype=np.float32),
        "base.mlp.fc.2.weight": np.ones(3, dtype=np.float32),
        "base.mlp.fc.2.bias": np.zeros(3, dtype=np.float32),
        "act.action_outs.0.logits_net.weight": contrast,
        "act.action_outs.0.logits_net.bias": np.zeros(2, dtype=np.float32),
        "act.action_outs.1.logits_net.weight": -contrast,
        "act.action_outs.1.logits_net.bias": np.zeros(2, dtype=np.float32),
    }


def claims_markdown(claims: list[dict]) -> str:
    return "## Summary\n\nfine\n\n```json\n" + json.dumps({"claims": claims}) + "\n```\n"


def write_eval_files(root: Path, claims: list[dict], task_mode: str = "shown") -> tuple[Path, Path, Path]:
    """(run dir, checkpoint, spec) for the known-sign network; the run dir holds run.json + final_answer.md."""
    checkpoint = root / "known.pt"
    checkpoint.write_bytes(checkpoint_bytes(known_sign_arrays()))
    spec = root / "known.spec.json"
    spec.write_text(json.dumps(EVAL_SPEC), encoding="utf-8")
    run_dir = root / "runs" / f"20260101T000000Z_net-feedbeef_{task_mode}"
    run_dir.mkdir(parents=True)
    record = {
        "network_alias": "net-feedbeef",
        "task_mode": task_mode,
        "model": "fake-model",
        "status": "complete",
        "inputs": {"checkpoint_sha256": read_checkpoint(checkpoint).sha256, "spec_sha256": "0" * 64},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    (run_dir / "final_answer.md").write_text(claims_markdown(claims), encoding="utf-8")
    return run_dir, checkpoint, spec


def known_spec():  # noqa: ANN201 - returns weightreading.spec.Spec
    return parse_spec(eval_spec_dict())
