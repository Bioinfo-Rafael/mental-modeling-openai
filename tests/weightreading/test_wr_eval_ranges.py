"""Spec ``evaluation`` section: parsing, coverage and structural constraints; the shipped specs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from weightreading.prompt import ExecutorLimits, first_message
from weightreading.spec import SpecError, load_spec, parse_spec
from weightreading.tensors import Tensor

from .wr_helpers import SMALL_SPEC, small_arrays

# A synthetic spec with every kind of `evaluation` entry (moved here from the removed W eval helpers).
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


CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "weightreading"
SHIPPED = ("lag_mappo_2v2_33",)  # the 1v1 W specs were removed


def test_parse_valid_section_resolves_indices_and_kinds() -> None:
    spec = parse_spec(eval_spec_dict())

    ranges = spec.evaluation
    assert ranges is not None
    assert [item.index for item in ranges.inputs] == [0, 1, 4, 5]
    assert ranges.input_range(4).values == (-1.0, 1.0)
    assert ranges.angle_of(3).sin_index == 2
    assert ranges.presence_of(5).p_present == 0.5
    assert ranges.presence_of(0) is None


def test_spec_without_section_has_no_ranges() -> None:
    assert parse_spec(SMALL_SPEC).evaluation is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e["inputs"].pop("noise"), "does not cover input noise"),
        (lambda e: e["inputs"].update({"roll_sin": {"range": [0, 1]}}), "more than once"),
        (lambda e: e["inputs"].update({"ghost": {"range": [0, 1]}}), "unknown input ghost"),
        (lambda e: e["inputs"].update({"speed": {"range": [1, -1]}}), "low < high"),
        (
            lambda e: e["inputs"].update({"speed": {"range": [0, 1], "values": [0, 1]}}),
            "exactly one of range or values",
        ),
        (lambda e: e["inputs"].update({"side": {"values": [1, 1]}}), "at least 2 distinct"),
        (lambda e: e["angles"][0].update({"range": [-2.0, 1.0]}), "strictly inside"),
        (lambda e: e["angles"][0].update({"range": [0.1, 1.0]}), "contain 0"),
        (lambda e: e.update({"notes": "one string"}), "notes must be a list"),
        (lambda e: e["angles"][0].update({"cos": "roll_sin"}), "two different inputs"),
        (lambda e: e["presence"][0].update({"p_present": 0}), "p_present"),
        (lambda e: e["presence"][0].update({"inputs": ["roll_cos"]}), "must not be an angle input"),
        (lambda e: e.update({"extra": 1}), "evaluation must be an object"),
    ],
)
def test_parse_invalid_section_raises_spec_error(mutate, message: str) -> None:  # noqa: ANN001
    raw = eval_spec_dict()
    mutate(raw["evaluation"])

    with pytest.raises(SpecError, match=message):
        parse_spec(raw)


def test_section_never_reaches_the_prompt(small_spec_dict: dict) -> None:
    # Arrange: the small spec plus an evaluation section with a recognisable number
    with_section = {
        **small_spec_dict,
        "evaluation": {
            "inputs": {"airspeed": {"range": [0.123456, 0.9]}},
            "angles": [{"sin": "roll_sin", "cos": "pitch_sin", "range": [-0.5, 0.5]}],
        },
    }
    tensors = tuple(Tensor(k, "float32", v.shape, v.tobytes()) for k, v in small_arrays().items())
    limits = ExecutorLimits(10, 30, 1000, 500)

    # Act
    plain = first_message(parse_spec(small_spec_dict), tensors, "shown", limits)
    extended = first_message(parse_spec(with_section), tensors, "shown", limits)

    # Assert
    assert plain == extended
    assert "0.123456" not in extended


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_specs_cover_every_input(name: str) -> None:
    spec = load_spec(CONFIGS / f"{name}.spec.json")

    ranges = spec.evaluation
    assert ranges is not None
    covered = {item.index for item in ranges.inputs} | {i for p in ranges.angles for i in (p.sin_index, p.cos_index)}
    assert covered == set(range(len(spec.inputs)))
