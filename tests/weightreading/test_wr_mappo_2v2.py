"""2v2 MAPPO actor support: explicit-value heads, the ACT-MLP variant, and the repository spec."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from weightreading.architecture import equations, expected_params
from weightreading.prompt import ExecutorLimits, first_message
from weightreading.spec import SpecError, check_against_tensors, load_spec, parse_spec
from weightreading.tensors import Tensor
from weightreading.torch_reader import read_checkpoint

from .wr_helpers import MHI_ACTORS, SMALL_SPEC

SPEC_DIR = Path(__file__).resolve().parents[2] / "configs" / "weightreading"
MAPPO_SPEC = SPEC_DIR / "lag_mappo_2v2_33.spec.json"
LIMITS = ExecutorLimits(max_runs=7, timeout_s=60, stdout_cap_chars=1000, stderr_tail_chars=200)
MAPPO_PARAMS = 155_917
MAPPO_TENSORS = 30
HIDDEN = 4
LABELLED_OUTPUTS = [
    {"head": 0, "name": "altitude_command", "bins": 3, "bin_value": {"values": [100, 0, -100]},
     "labels": ["up", "hold", "down"], "unit": "m"},
    {"head": 1, "name": "shoot", "bins": 2, "bin_value": {"values": [0, 1]}, "labels": ["no", "yes"]},
]  # fmt: skip
ACT_MLP_BLOCK = {
    "id": "act_mlp", "type": "mlp", "prefix": "act.mlp.fc.", "input": "gru_norm", "input_size": HIDDEN,
    "hidden_sizes": [HIDDEN, HIDDEN],
}  # fmt: skip


def _act_mlp_spec() -> dict:
    """SMALL_SPEC with labelled heads and an ACT MLP between the GRU LayerNorm and the heads."""
    spec = json.loads(json.dumps(SMALL_SPEC))
    heads = {**spec["architecture"][-1], "input": "act_mlp"}
    fresh = json.loads(json.dumps({"act_mlp": ACT_MLP_BLOCK, "outputs": LABELLED_OUTPUTS}))  # tests may mutate
    return {**spec, "architecture": [*spec["architecture"][:-1], fresh["act_mlp"], heads], "outputs": fresh["outputs"]}


def _tensors_for(spec_dict: dict) -> tuple[Tensor, ...]:
    rng = np.random.default_rng(0)
    shapes = {k: v for block in parse_spec(spec_dict).blocks for k, v in expected_params(block).items()}
    return tuple(
        Tensor(key, "float32", shape, rng.standard_normal(shape).astype("<f4").tobytes())
        for key, shape in shapes.items()
    )


# --- explicit per-bin values and labels ----------------------------------------------------------


def test_parse_spec_with_explicit_values_keeps_values_and_labels() -> None:
    spec = parse_spec(_act_mlp_spec())

    altitude = spec.outputs[0]
    assert altitude.values == (100.0, 0.0, -100.0)
    assert altitude.labels == ("up", "hold", "down")
    assert altitude.explicit and altitude.unit == "m"
    assert (altitude.low, altitude.high) == (100.0, -100.0)


def test_parse_spec_with_linspace_keeps_exact_end_points(small_spec_dict: dict) -> None:
    throttle = parse_spec(small_spec_dict).outputs[1]

    assert throttle.values == (0.4, 0.9)
    assert not throttle.explicit and throttle.labels == ()


@pytest.mark.parametrize(
    ("bin_value", "labels", "message"),
    [
        ({"values": [1, 0]}, ["up", "hold", "down"], "one number per bin"),
        ({"values": [1, 0, 0]}, ["up", "hold", "down"], "strictly monotonic"),
        ({"values": [0, 1, -1]}, ["up", "hold", "down"], "strictly monotonic"),
        ({"values": [1, 0, True]}, ["up", "hold", "down"], "one number per bin"),
        ({"values": [1, 0, float("nan")]}, ["up", "hold", "down"], "one number per bin"),
        ({"values": [1, 0, -1], "linspace": [0, 1]}, ["up", "hold", "down"], "exactly one of"),
        ({}, ["up", "hold", "down"], "exactly one of"),
        ({"values": [1, 0, -1]}, ["up", "down"], "3 distinct entries"),
        ({"values": [1, 0, -1]}, ["up", "up", "down"], "3 distinct entries"),
        ({"values": [1, 0, -1]}, ["up", "", "down"], "non-empty strings"),
        ({"values": [1, 0, -1]}, "up", "non-empty strings"),
    ],
)
def test_parse_spec_with_invalid_explicit_head_raises(bin_value: dict, labels: object, message: str) -> None:
    spec = _act_mlp_spec()
    spec["outputs"][0] = {**spec["outputs"][0], "bin_value": bin_value, "labels": labels}

    with pytest.raises(SpecError, match=message):
        parse_spec(spec)


def test_first_message_shown_lists_bin_values_labels_and_unit() -> None:
    spec_dict = _act_mlp_spec()

    message = first_message(parse_spec(spec_dict), _tensors_for(spec_dict), "shown", LIMITS)

    assert "head 0 `altitude_command`: 3 bins; bin 0 = 100 (up), bin 1 = 0 (hold), bin 2 = -100 (down) [m]" in message
    assert "head 1 `shoot`: 2 bins; bin 0 = 0 (no), bin 1 = 1 (yes)" in message
    assert "not the bin index" in message


def test_first_message_blind_hides_bin_labels_and_unit() -> None:
    spec_dict = _act_mlp_spec()

    message = first_message(parse_spec(spec_dict), _tensors_for(spec_dict), "blind", LIMITS)

    assert "head 0 `head_0`: 3 bins; bin 0 = 100, bin 1 = 0, bin 2 = -100\n" in message
    for hidden in ["(up)", "(hold)", "(yes)", "[m]", "altitude_command", "shoot"]:
        assert hidden not in message


def test_first_message_with_only_linspace_heads_omits_value_note(small_spec_dict: dict) -> None:
    spec = parse_spec(small_spec_dict)
    message = first_message(spec, _tensors_for(small_spec_dict), "shown", LIMITS)

    assert "bin k (0..2) means value -1 + k * 2 / 2" in message
    assert "not the bin index" not in message


# --- ACT-MLP variant: ownership and equations ---------------------------------------------------


def test_check_against_tensors_with_act_mlp_variant_claims_act_mlp_keys() -> None:
    spec_dict = _act_mlp_spec()
    tensors = _tensors_for(spec_dict)

    check_against_tensors(parse_spec(spec_dict), tensors)

    assert {"act.mlp.fc.0.weight", "act.mlp.fc.2.weight", "act.mlp.fc.3.bias", "act.mlp.fc.5.bias"} <= {
        t.name for t in tensors
    }


def test_check_against_tensors_without_act_mlp_block_reports_unclaimed_keys() -> None:
    spec_dict = _act_mlp_spec()
    tensors = _tensors_for(spec_dict)
    without = {**spec_dict, "architecture": json.loads(json.dumps(SMALL_SPEC["architecture"]))}

    with pytest.raises(SpecError, match="not claimed by any block: act.mlp.fc.0.weight"):
        check_against_tensors(parse_spec(without), tensors)


def test_equations_for_act_mlp_name_input_and_follow_linear_relu_layernorm() -> None:
    blocks = {block.block_id: block for block in parse_spec(_act_mlp_spec()).blocks}

    act_mlp = equations(blocks["act_mlp"])
    heads = equations(blocks["heads"])
    norm = equations(blocks["gru_norm"])

    assert "input u: the output of block `gru_norm`" in act_mlp
    assert "a1 = LN1(relu(W['act.mlp.fc.0.weight'] @ u + W['act.mlp.fc.0.bias']))" in act_mlp
    assert "LN2 gain W['act.mlp.fc.5.weight'], bias W['act.mlp.fc.5.bias']" in act_mlp
    assert "input v: the output of block `act_mlp`" in heads
    assert "input v: the output of block `gru`" in norm


def test_equations_for_gru_state_gate_order_and_pre_norm_carry() -> None:
    gru = next(block for block in parse_spec(_act_mlp_spec()).blocks if block.block_id == "gru")

    text = equations(gru)

    assert f"rows 0-{HIDDEN - 1} = r, {HIDDEN}-{2 * HIDDEN - 1} = z" in text
    assert "n  = tanh(W_in u + b_in + r * (W_hn h + b_hn))" in text
    assert "The carried recurrent state is h' itself (before any LayerNorm)" in text


def test_first_message_data_flow_runs_through_act_mlp() -> None:
    spec_dict = _act_mlp_spec()

    message = first_message(parse_spec(spec_dict), _tensors_for(spec_dict), "shown", LIMITS)

    assert "data flow: observation -> base -> gru -> gru_norm -> act_mlp -> heads" in message


# --- the repository spec and the delivered actors -------------------------------------------------


def test_mappo_spec_has_33_inputs_and_four_labelled_heads() -> None:
    spec = load_spec(MAPPO_SPEC)

    assert len(spec.inputs) == 33
    assert [input_.name for input_ in spec.inputs[9:15]][0] == "partner_delta_speed"
    assert spec.inputs[27].name == "missile_delta_speed"
    assert [(o.bins, o.labels) for o in spec.outputs] == [
        (3, ("up", "hold", "down")),
        (5, ("left30", "left15", "hold", "right15", "right30")),
        (3, ("up", "hold", "down")),
        (2, ("no", "yes")),
    ]
    assert spec.input_clip is None


def test_mappo_spec_prompt_hides_provenance_and_notes() -> None:
    spec = load_spec(MAPPO_SPEC)
    spec_dict = json.loads(MAPPO_SPEC.read_text(encoding="utf-8"))

    message = first_message(spec, _tensors_for(spec_dict), "shown", LIMITS)

    for secret in ["mtd-rl", "v1.3", "MHI", "multiplecombat_task", "7cdf5ebc"]:
        assert secret not in message


@pytest.mark.parametrize("actor", MHI_ACTORS)
def test_mappo_spec_validates_against_delivered_actor(mhi_model_dir: Path, actor: str) -> None:
    checkpoint = read_checkpoint(mhi_model_dir / actor)

    check_against_tensors(load_spec(MAPPO_SPEC), checkpoint.tensors, checkpoint.sha256)

    assert len(checkpoint.tensors) == MAPPO_TENSORS
    assert sum(t.numel for t in checkpoint.tensors) == MAPPO_PARAMS
