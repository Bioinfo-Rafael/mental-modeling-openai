"""Spec validation and cross-check against checkpoint tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from translate.policy.spec import SpecError, check_against_tensors, load_spec, parse_spec
from translate.policy.tensors import Tensor

from .wr_helpers import small_arrays

SPEC_DIR = Path(__file__).resolve().parents[2] / "configs" / "translate" / "policies"


def _tensors(arrays: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
    return tuple(Tensor(k, "float32", v.shape, v.astype("<f4").tobytes()) for k, v in arrays.items())


def test_check_against_tensors_with_matching_weights_passes(small_spec_dict: dict) -> None:
    spec = parse_spec(small_spec_dict)

    check_against_tensors(spec, _tensors(small_arrays()))


def test_check_against_tensors_with_wrong_shape_reports_key(small_spec_dict: dict) -> None:
    arrays = {**small_arrays(), "rnn.norm.weight": np.zeros(5, np.float32)}

    with pytest.raises(SpecError, match=r"rnn.norm.weight: checkpoint shape \(5,\)"):
        check_against_tensors(parse_spec(small_spec_dict), _tensors(arrays))


def test_check_against_tensors_with_unclaimed_key_fails(small_spec_dict: dict) -> None:
    arrays = {**small_arrays(), "act.mlp.fc.0.weight": np.zeros((4, 4), np.float32)}

    with pytest.raises(SpecError, match="not claimed by any block: act.mlp.fc.0.weight"):
        check_against_tensors(parse_spec(small_spec_dict), _tensors(arrays))


def test_check_against_tensors_with_missing_key_fails(small_spec_dict: dict) -> None:
    arrays = {k: v for k, v in small_arrays().items() if k != "rnn.gru.bias_hh_l0"}

    with pytest.raises(SpecError, match="missing from checkpoint: rnn.gru.bias_hh_l0"):
        check_against_tensors(parse_spec(small_spec_dict), _tensors(arrays))


def test_check_against_tensors_with_double_claim_fails(small_spec_dict: dict) -> None:
    extra = {"id": "again", "type": "layernorm", "prefix": "rnn.norm.", "input": "gru", "size": 4}
    small_spec_dict["architecture"].insert(3, extra)

    with pytest.raises(SpecError, match="claimed by both"):
        check_against_tensors(parse_spec(small_spec_dict), _tensors(small_arrays()))


def test_check_against_tensors_with_other_checkpoint_hash_fails(small_spec_dict: dict) -> None:
    small_spec_dict["provenance"]["checkpoint_sha256"] = "0" * 64

    with pytest.raises(SpecError, match="sha256 differs"):
        check_against_tensors(parse_spec(small_spec_dict), _tensors(small_arrays()), checkpoint_sha256="1" * 64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.update(schema_version=2), "schema_version"),
        (lambda s: s.update(network_alias="baseline"), "network_alias"),
        (lambda s: s.update(surprise=1), "unknown top-level"),
        (lambda s: s["inputs"][1].update(index=5), "indices must be 0..n-1"),
        (lambda s: s["outputs"][0].update(bins=0), "bins >= 1"),
        (lambda s: s["architecture"][1].update(input_size=7), "expects 7 inputs"),
        (lambda s: s["architecture"][1].update(input="nowhere"), "unknown input"),
        (lambda s: s["architecture"][0].update(type="conv"), "unknown block type"),
        (lambda s: s["architecture"].pop(), "last architecture block"),
        (lambda s: s["architecture"][1].update(num_layers=2), "single-layer"),
        (lambda s: s.update(input_clip=[1.0]), "input_clip"),
        (lambda s: s["inputs"][0].pop("name"), "missing field: name"),
        (lambda s: s["inputs"][0].update(index=True), "wrong type"),
    ],
)
def test_parse_spec_with_invalid_field_raises(small_spec_dict: dict, mutate, message: str) -> None:
    mutate(small_spec_dict)

    with pytest.raises(SpecError, match=message):
        parse_spec(small_spec_dict)


def test_load_spec_with_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.spec.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(SpecError, match="not valid JSON"):
        load_spec(path)
