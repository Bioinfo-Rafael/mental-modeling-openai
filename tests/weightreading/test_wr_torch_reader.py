"""Restricted checkpoint reader: happy paths and mandatory malicious-payload rejections."""

from __future__ import annotations

import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
from wr_helpers import (
    PREFIX,
    Entry,
    build_zip,
    checkpoint_bytes,
    contiguous,
    op_global,
    op_int,
    op_str,
    op_tensor,
    op_tuple,
    state_dict_pickle,
)

from weightreading import torch_reader
from weightreading.torch_reader import CheckpointError, parse_checkpoint_bytes, read_checkpoint

DATA_PKL = f"{PREFIX}/data.pkl"


def _float_entry(name: str = "w", numel: int = 6, offset: int = 0, shape=(2, 3), stride=None) -> Entry:
    return Entry(name, "0", numel, offset, shape, contiguous(shape) if stride is None else stride)


def _single(entry: Entry, storage: bytes | None = None, extra: dict | None = None) -> bytes:
    payload = state_dict_pickle([(op_str(entry.name), op_tensor(entry))])
    data = storage if storage is not None else np.arange(entry.storage_numel, dtype="<f4").tobytes()
    return build_zip({DATA_PKL: payload, f"{PREFIX}/data/0": data, **(extra or {})})


def _payload_zip(payload: bytes) -> bytes:
    return build_zip({DATA_PKL: payload})


# --- happy paths --------------------------------------------------------------------------------


def test_parse_checkpoint_with_two_tensors_returns_values_in_order() -> None:
    arrays = {"a.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "a.bias": np.array([1.5, -2.0], np.float32)}

    tensors = parse_checkpoint_bytes(checkpoint_bytes(arrays))

    assert [t.name for t in tensors] == ["a.weight", "a.bias"]
    np.testing.assert_array_equal(tensors[0].to_numpy(), arrays["a.weight"])
    np.testing.assert_array_equal(tensors[1].to_numpy(), arrays["a.bias"])


def test_parse_checkpoint_with_shared_storage_honours_offsets() -> None:
    first = Entry("first", "0", 10, 0, (2, 2), (2, 1))
    second = Entry("second", "0", 10, 4, (6,), (1,))
    payload = state_dict_pickle([(op_str("first"), op_tensor(first)), (op_str("second"), op_tensor(second))])
    raw = build_zip({DATA_PKL: payload, f"{PREFIX}/data/0": np.arange(10, dtype="<f4").tobytes()})

    tensors = parse_checkpoint_bytes(raw)

    np.testing.assert_array_equal(tensors[0].to_numpy(), [[0, 1], [2, 3]])
    np.testing.assert_array_equal(tensors[1].to_numpy(), [4, 5, 6, 7, 8, 9])


def test_parse_checkpoint_with_big_endian_storage_returns_little_endian_bytes() -> None:
    values = np.array([1.0, -2.5, 3.25], dtype=">f4")
    raw = _single(_float_entry(numel=3, shape=(3,)), values.tobytes(), {f"{PREFIX}/byteorder": b"big"})

    tensors = parse_checkpoint_bytes(raw)

    np.testing.assert_array_equal(tensors[0].to_numpy(), [1.0, -2.5, 3.25])


def test_parse_checkpoint_with_metadata_build_and_parameter_wrapper_is_accepted() -> None:
    entry = _float_entry()
    wrapped = op_global("torch._utils", "_rebuild_parameter") + op_tuple([op_tensor(entry), pickle.NEWTRUE])
    metadata = pickle.EMPTY_DICT + op_str("_metadata") + pickle.EMPTY_DICT + pickle.SETITEM + pickle.BUILD
    payload = state_dict_pickle([(op_str("w"), wrapped + pickle.REDUCE)], tail=metadata)
    raw = build_zip({DATA_PKL: payload, f"{PREFIX}/data/0": np.arange(6, dtype="<f4").tobytes()})

    tensors = parse_checkpoint_bytes(raw)

    assert tensors[0].shape == (2, 3)


def test_read_checkpoint_on_lag_baseline_matches_known_counts(lag_model_dir: Path) -> None:
    checkpoint = read_checkpoint(lag_model_dir / "baseline_model.pt")

    assert len(checkpoint.tensors) == 22
    assert sum(t.numel for t in checkpoint.tensors) == 137_753
    assert checkpoint.sha256.startswith("54e9c996d981")


def test_read_checkpoint_on_lag_dodge_matches_known_counts(lag_model_dir: Path) -> None:
    checkpoint = read_checkpoint(lag_model_dir / "dodge_missile_model.pt")

    assert len(checkpoint.tensors) == 30
    assert sum(t.numel for t in checkpoint.tensors) == 172_441


# --- malicious or malformed pickles ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name"),
    [("os", "system"), ("posix", "system"), ("builtins", "eval"), ("builtins", "getattr"), ("subprocess", "Popen")],
)
def test_parse_checkpoint_with_dangerous_global_raises_before_call(tmp_path: Path, module: str, name: str) -> None:
    marker = tmp_path / "pwned"
    payload = pickle.PROTO + b"\x02" + op_global(module, name)
    payload += op_tuple([op_str(f"touch {marker}")]) + pickle.REDUCE + pickle.STOP

    with pytest.raises(CheckpointError, match="global not allowed"):
        parse_checkpoint_bytes(_payload_zip(payload))
    assert not marker.exists()


def test_parse_checkpoint_with_inst_opcode_global_raises(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    payload = pickle.MARK + op_str(f"touch {marker}") + pickle.INST + b"os\nsystem\n" + pickle.STOP

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(payload))
    assert not marker.exists()


def test_parse_checkpoint_with_stack_global_raises(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    payload = pickle.PROTO + b"\x04" + op_str("os") + op_str("system") + pickle.STACK_GLOBAL
    payload += op_tuple([op_str(f"touch {marker}")]) + pickle.REDUCE + pickle.STOP

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(payload))
    assert not marker.exists()


def test_parse_checkpoint_with_dangerous_global_inside_state_dict_raises(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    evil = op_global("os", "system") + op_tuple([op_str(f"touch {marker}")]) + pickle.REDUCE
    payload = state_dict_pickle([(op_str("w"), evil)])

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(payload))
    assert not marker.exists()


@pytest.mark.parametrize(
    "global_name", [("torch._utils", "_rebuild_qtensor"), ("torch", "load"), ("collections", "defaultdict")]
)
def test_parse_checkpoint_with_unlisted_torch_global_raises(global_name: tuple[str, str]) -> None:
    payload = pickle.PROTO + b"\x02" + op_global(*global_name) + pickle.STOP

    with pytest.raises(CheckpointError, match="global not allowed"):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_calling_storage_type_directly_raises() -> None:
    payload = state_dict_pickle(
        [(op_str("w"), op_global("torch", "FloatStorage") + pickle.EMPTY_TUPLE + pickle.REDUCE)]
    )

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_with_build_on_tensor_raises() -> None:
    tensor = op_tensor(_float_entry()) + pickle.EMPTY_DICT + op_str("x") + op_int(1) + pickle.SETITEM + pickle.BUILD
    payload = state_dict_pickle([(op_str("w"), tensor)])
    raw = build_zip({DATA_PKL: payload, f"{PREFIX}/data/0": np.zeros(6, "<f4").tobytes()})

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_malformed_persistent_id_raises() -> None:
    pid = op_tuple([op_str("module"), op_str("x")]) + pickle.BINPERSID
    payload = state_dict_pickle([(op_str("w"), pid)])

    with pytest.raises(CheckpointError, match="persistent id"):
        parse_checkpoint_bytes(_payload_zip(payload))


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (_float_entry(offset=-1), "negative"),
        (_float_entry(offset=5), "past the end"),
        (_float_entry(shape=(-2, 3), stride=(3, 1)), "negative"),
        (_float_entry(shape=(2, 3), stride=(1, 2)), "non-contiguous"),
        (_float_entry(numel=6, shape=(2, 3), stride=(3,)), "rank mismatch"),
        (_float_entry(numel=6, shape=(4, 3)), "past the end"),
    ],
)
def test_parse_checkpoint_with_bad_view_raises(entry: Entry, message: str) -> None:
    raw = _single(entry, np.zeros(6, "<f4").tobytes())

    with pytest.raises(CheckpointError, match=message):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_storage_size_mismatch_raises() -> None:
    raw = _single(_float_entry(), np.zeros(5, "<f4").tobytes())

    with pytest.raises(CheckpointError, match="size does not match"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_missing_storage_member_raises() -> None:
    payload = state_dict_pickle([(op_str("w"), op_tensor(_float_entry()))])

    with pytest.raises(CheckpointError, match="missing storage"):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_with_double_dtype_fails_closed() -> None:
    entry = Entry("w", "0", 2, 0, (2,), (1,), storage_type="DoubleStorage")

    with pytest.raises(CheckpointError, match="unsupported dtype float64"):
        parse_checkpoint_bytes(_single(entry, np.zeros(2, "<f8").tobytes()))


@pytest.mark.parametrize(
    "items",
    [
        [(op_int(3), op_tensor(_float_entry()))],
        [(op_str("w"), op_str("not a tensor"))],
        [(op_str("../bad key"), op_tensor(_float_entry()))],
    ],
)
def test_parse_checkpoint_with_bad_state_dict_entry_raises(items: list) -> None:
    raw = build_zip({DATA_PKL: state_dict_pickle(items), f"{PREFIX}/data/0": np.zeros(6, "<f4").tobytes()})

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_top_level_list_raises() -> None:
    payload = pickle.PROTO + b"\x02" + pickle.EMPTY_LIST + pickle.STOP

    with pytest.raises(CheckpointError, match="not a plain state_dict"):
        parse_checkpoint_bytes(_payload_zip(payload))


# --- zip layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["../evil", "/abs/data.pkl", "archive/../data.pkl", "archive\\data.pkl", "archive/extra.py", "archive/"]
)
def test_parse_checkpoint_with_unexpected_member_name_raises(name: str) -> None:
    raw = build_zip({DATA_PKL: pickle.PROTO + b"\x02" + pickle.EMPTY_DICT + pickle.STOP, name: b"x"})

    with pytest.raises(CheckpointError, match="unexpected zip member"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_two_prefixes_raises() -> None:
    raw = build_zip({DATA_PKL: b"x", "other/data.pkl": b"y"})

    with pytest.raises(CheckpointError, match="exactly one top-level"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_deflated_member_raises() -> None:
    raw = build_zip({DATA_PKL: b"\0" * 1000}, compression=zipfile.ZIP_DEFLATED)

    with pytest.raises(CheckpointError, match="compressed"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_over_size_cap_raises() -> None:
    raw = checkpoint_bytes({"w": np.zeros(1000, np.float32)})

    with pytest.raises(CheckpointError, match="size cap"):
        parse_checkpoint_bytes(raw, max_total_bytes=1000)


def test_parse_checkpoint_without_data_pkl_raises() -> None:
    with pytest.raises(CheckpointError, match="no data.pkl"):
        parse_checkpoint_bytes(build_zip({f"{PREFIX}/version": b"3\n"}))


@pytest.mark.parametrize("raw", [b"", b"not a zip at all", pickle.dumps({"w": [1.0, 2.0]})])
def test_parse_checkpoint_with_non_zip_input_raises(raw: bytes) -> None:
    with pytest.raises(CheckpointError, match="not a zip"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_truncated_pickle_raises() -> None:
    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(pickle.PROTO + b"\x02" + op_global("collections", "OrderedDict")))


def test_read_checkpoint_file_over_cap_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch_reader, "MAX_RAW_OVERHEAD", 0)
    path = tmp_path / "big.pt"
    path.write_bytes(b"\0" * 2048)

    with pytest.raises(CheckpointError, match="size cap"):
        read_checkpoint(path, max_total_bytes=1024)


@pytest.mark.parametrize(
    "target", [op_global("torch", "DoubleStorage"), op_global("torch._utils", "_rebuild_tensor_v2")]
)
def test_parse_checkpoint_with_build_on_allowed_global_raises_and_does_not_persist(target: bytes) -> None:
    mutate = target + pickle.EMPTY_DICT + op_str("dtype") + op_str("float32") + pickle.SETITEM + pickle.BUILD
    evil = state_dict_pickle([(op_str("w"), mutate)])

    with pytest.raises(CheckpointError):
        parse_checkpoint_bytes(_payload_zip(evil))
    entry = Entry("w", "0", 2, 0, (2,), (1,), storage_type="DoubleStorage")
    with pytest.raises(CheckpointError, match="unsupported dtype float64"):
        parse_checkpoint_bytes(_single(entry, np.zeros(2, "<f8").tobytes()))


def test_parse_checkpoint_with_non_empty_ordered_dict_call_raises() -> None:
    call = op_global("collections", "OrderedDict") + op_tuple([op_str("x")]) + pickle.REDUCE
    payload = pickle.PROTO + b"\x02" + call + pickle.STOP

    with pytest.raises(CheckpointError, match="constructed empty"):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_with_second_reference_overclaiming_shared_storage_raises() -> None:
    honest = Entry("a", "0", 6, 0, (6,), (1,))
    liar = Entry("b", "0", 100, 50, (10,), (1,))
    payload = state_dict_pickle([(op_str("a"), op_tensor(honest)), (op_str("b"), op_tensor(liar))])
    raw = build_zip({DATA_PKL: payload, f"{PREFIX}/data/0": np.zeros(6, "<f4").tobytes()})

    with pytest.raises(CheckpointError, match="size does not match"):
        parse_checkpoint_bytes(raw)


@pytest.mark.parametrize("name", [f"{PREFIX}/data.pkl\n", f"{PREFIX}/data/0\n"])
def test_parse_checkpoint_with_trailing_newline_member_name_raises(name: str) -> None:
    raw = build_zip({DATA_PKL: b"x", name: b"y"})

    with pytest.raises(CheckpointError, match="unexpected zip member"):
        parse_checkpoint_bytes(raw)


def test_parse_checkpoint_with_trailing_newline_key_raises() -> None:
    raw = build_zip(
        {
            DATA_PKL: state_dict_pickle([(op_str("w\n"), op_tensor(_float_entry()))]),
            f"{PREFIX}/data/0": np.zeros(6, "<f4").tobytes(),
        }
    )

    with pytest.raises(CheckpointError, match="unsupported state_dict key"):
        parse_checkpoint_bytes(raw)


# --- findings from the Codex security review ------------------------------------------------------


@pytest.mark.parametrize("ext", [pickle.EXT1 + b"\x01", pickle.EXT2 + b"\x01\x00", pickle.EXT4 + b"\x01\x00\x00\x00"])
def test_parse_checkpoint_with_extension_registry_opcode_raises(ext: bytes) -> None:
    payload = pickle.PROTO + b"\x02" + ext + pickle.EMPTY_TUPLE + pickle.REDUCE + pickle.STOP

    with pytest.raises(CheckpointError, match="opcode not allowed: EXT"):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_with_huge_memo_index_raises_before_allocation() -> None:
    payload = pickle.PROTO + b"\x02" + pickle.NONE + pickle.LONG_BINPUT + struct.pack("<I", 100_000_000) + pickle.STOP

    with pytest.raises(CheckpointError, match="memo index"):
        parse_checkpoint_bytes(_payload_zip(payload))


def test_parse_checkpoint_with_encrypted_member_raises() -> None:
    raw = bytearray(build_zip({DATA_PKL: b"x"}))
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):  # local + central header
        position = raw.index(signature)
        raw[position + flag_offset] |= 0x1

    with pytest.raises(CheckpointError, match="encrypted"):
        parse_checkpoint_bytes(bytes(raw))


def test_parse_checkpoint_with_too_many_central_directory_entries_raises(monkeypatch) -> None:
    monkeypatch.setattr(torch_reader, "MAX_MEMBERS", 1)

    with pytest.raises(CheckpointError, match="central directory"):
        parse_checkpoint_bytes(checkpoint_bytes({"w": np.zeros(2, np.float32)}))


def test_parse_checkpoint_with_large_prefix_over_raw_cap_raises(monkeypatch) -> None:
    monkeypatch.setattr(torch_reader, "MAX_RAW_OVERHEAD", 1024)
    raw = b"\0" * 4096 + checkpoint_bytes({"w": np.zeros(2, np.float32)})

    with pytest.raises(CheckpointError, match="size cap"):
        parse_checkpoint_bytes(raw, max_total_bytes=64)
