"""Restricted reader for PyTorch zip checkpoints (``torch.save`` of a plain ``state_dict``).

Checkpoints are untrusted input: ``torch.load`` without ``weights_only`` is arbitrary code
execution, and torch is not a dependency here anyway. This reader never imports torch and never
lets the pickle reach a real global:

* ``find_class`` maps a small allow-list of names to local inert stand-ins and raises
  ``UnpicklingError`` for anything else (``os.system``, ``builtins.eval``, ...) before any call.
* ``persistent_load`` accepts only ``('storage', <storage type>, key, location, numel)``.
* The zip must be uncompressed (no decompression bombs), have exactly one top-level prefix and
  only known member names; sizes are capped; storage byte lengths and tensor views are checked.

Only float32 tensors are decoded; any other dtype fails closed with a named error.
"""

from __future__ import annotations

import array
import hashlib
import io
import math
import pickle
import pickletools
import re
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from translate.policy.tensors import Tensor

MIB = 1024 * 1024
MAX_TOTAL_BYTES = 512 * MIB
MAX_PICKLE_BYTES = 16 * MIB
MAX_MEMBERS = 10_000
MAX_RAW_OVERHEAD = 16 * MIB  # zip headers, alignment padding and any prefix beyond member bytes
MAX_CENTRAL_DIRECTORY = 4 * MIB
MAX_OPCODES = 1_000_000
MAX_MEMO_INDEX = 1_000_000
EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_SIZE = 22
EOCD_SEARCH = EOCD_SIZE + 0xFFFF  # fixed record + maximum comment length
ZIP64_MARKER_16 = 0xFFFF
ZIP64_MARKER_32 = 0xFFFFFFFF
ENCRYPTED_FLAG = 0x1
FLOAT32_ITEMSIZE = 4
PERSISTENT_ID_FIELDS = 5

STORAGE_DTYPES = {
    "FloatStorage": "float32",
    "DoubleStorage": "float64",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "BoolStorage": "bool",
}
SUPPORTED_DTYPES = {"float32": FLOAT32_ITEMSIZE}

_PREFIX = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
_MEMBER_RE = re.compile(
    rf"(?P<prefix>{_PREFIX})/(?P<rest>data\.pkl|version|byteorder|\.format_version|\.storage_alignment"
    r"|\.data/serialization_id|\.data/version|data/(?P<key>[A-Za-z0-9_]+))"
)
# Opcodes torch.save (protocol 2) emits, plus harmless protocol-4 framing. Everything else -
# notably EXT1/EXT2/EXT4, which the C unpickler resolves from a process-wide cache WITHOUT calling
# find_class - is rejected before unpickling starts.
_ALLOWED_OPCODES = frozenset(
    {
        "PROTO", "FRAME", "STOP", "MARK", "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "BINPERSID",
        "BINPUT", "LONG_BINPUT", "MEMOIZE", "BINGET", "LONG_BINGET",
        "EMPTY_TUPLE", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "EMPTY_DICT", "SETITEM", "SETITEMS",
        "EMPTY_LIST", "APPEND", "APPENDS",
        "BINUNICODE", "SHORT_BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT",
        "NONE", "NEWTRUE", "NEWFALSE",
    }
)  # fmt: skip
_MEMO_OPCODES = frozenset({"BINPUT", "LONG_BINPUT", "BINGET", "LONG_BINGET"})
_KEY_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.]*")  # used with fullmatch (no trailing newline)


class CheckpointError(ValueError):
    """The file is not a checkpoint this reader accepts. Nothing from it was executed."""


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    sha256: str
    tensors: tuple[Tensor, ...]


# --- inert stand-ins for the allow-listed globals ---------------------------------------------
# Every object a pickle can reach defines ``__setstate__`` (raising) and has no ``__dict__``, so a
# BUILD opcode can neither mutate it nor the shared allow-list between loads. The only exception
# is the per-load state dict, whose BUILD state (torch's ``_metadata``) is accepted and ignored.


def _refuse_build(_self: object, _state: object) -> None:
    raise pickle.UnpicklingError("BUILD is not allowed on this object")


class _StateDict(dict):
    """Stand-in for ``collections.OrderedDict`` (dicts keep insertion order)."""

    __slots__ = ()

    def __setstate__(self, state: object) -> None:
        if not isinstance(state, dict):
            raise pickle.UnpicklingError("unexpected BUILD state on state dict")


@dataclass(frozen=True, slots=True)
class _StorageRef:
    dtype: str
    key: str
    numel: int

    __setstate__ = _refuse_build


@dataclass(frozen=True, slots=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    __setstate__ = _refuse_build


class _StorageType:
    __slots__ = ("dtype",)
    __setstate__ = _refuse_build

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def __call__(self, *args: object) -> None:
        raise pickle.UnpicklingError("storage types may only appear in persistent ids")


class _Factory:
    """Callable wrapper so module-level functions and classes are never handed to the pickle."""

    __slots__ = ("_function",)
    __setstate__ = _refuse_build

    def __init__(self, function: Callable[..., object]) -> None:
        self._function = function

    def __call__(self, *args: object) -> object:
        return self._function(*args)


def _ordered_dict(*args: object) -> _StateDict:
    if args:
        raise pickle.UnpicklingError("OrderedDict must be constructed empty")
    return _StateDict()


def _rebuild_tensor_v2(storage: object, offset: object, size: object, stride: object, *_rest: object) -> _TensorRef:
    if not isinstance(storage, _StorageRef):
        raise pickle.UnpicklingError("tensor storage must come from a persistent id")
    return _TensorRef(storage, _int(offset), _int_tuple(size), _int_tuple(stride))


def _rebuild_parameter(data: object, *_rest: object) -> _TensorRef:
    if not isinstance(data, _TensorRef):
        raise pickle.UnpicklingError("parameter data must be a tensor")
    return data


def _int(value: object) -> int:
    if type(value) is not int:
        raise pickle.UnpicklingError("expected an integer")
    return value


def _int_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise pickle.UnpicklingError("expected a tuple of integers")
    return tuple(_int(item) for item in value)


def _allowed_globals() -> dict[tuple[str, str], object]:
    """Fresh stand-ins per load, so nothing a pickle touches outlives that load."""
    return {
        ("collections", "OrderedDict"): _Factory(_ordered_dict),
        ("torch._utils", "_rebuild_tensor_v2"): _Factory(_rebuild_tensor_v2),
        ("torch._utils", "_rebuild_parameter"): _Factory(_rebuild_parameter),
        **{("torch", name): _StorageType(dtype) for name, dtype in STORAGE_DTYPES.items()},
    }


def _prescan(payload: bytes) -> None:
    """Reject unknown opcodes, extension-registry lookups and memo indices that would force huge allocations."""
    try:
        for count, (opcode, arg, _pos) in enumerate(pickletools.genops(payload)):
            if count >= MAX_OPCODES:
                raise CheckpointError("pickle has too many opcodes")
            if opcode.name not in _ALLOWED_OPCODES:
                raise CheckpointError(f"pickle opcode not allowed: {opcode.name}")
            if opcode.name in _MEMO_OPCODES and not 0 <= arg < MAX_MEMO_INDEX:
                raise CheckpointError("pickle memo index out of range")
    except (ValueError, struct.error) as exc:
        raise CheckpointError(f"malformed pickle: {exc}") from exc


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, payload: bytes) -> None:
        super().__init__(io.BytesIO(payload))
        self._allowed = _allowed_globals()

    def find_class(self, module: str, name: str) -> object:
        allowed = self._allowed.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(f"global not allowed: {module}.{name}")
        return allowed

    def persistent_load(self, pid: object) -> _StorageRef:
        if not isinstance(pid, tuple) or len(pid) != PERSISTENT_ID_FIELDS or pid[0] != "storage":
            raise pickle.UnpicklingError("unsupported persistent id")
        _, storage_type, key, _location, numel = pid
        if not isinstance(storage_type, _StorageType) or not isinstance(key, str):
            raise pickle.UnpicklingError("unsupported storage reference")
        numel = _int(numel)
        if numel < 0:
            raise pickle.UnpicklingError("negative storage size")
        return _StorageRef(storage_type.dtype, key, numel)


# --- zip layer ---------------------------------------------------------------------------------


def _checked_members(archive: zipfile.ZipFile, max_total_bytes: int) -> tuple[str, dict[str, zipfile.ZipInfo]]:
    infos = archive.infolist()
    if not infos or len(infos) > MAX_MEMBERS:
        raise CheckpointError("unexpected number of zip members")
    prefixes = set()
    by_name: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        match = _MEMBER_RE.fullmatch(info.filename)
        if match is None:
            raise CheckpointError(f"unexpected zip member name: {info.filename!r}")
        if info.flag_bits & ENCRYPTED_FLAG:
            raise CheckpointError(f"encrypted zip member not accepted: {info.filename!r}")
        if info.compress_type != zipfile.ZIP_STORED or info.compress_size != info.file_size:
            raise CheckpointError(f"compressed zip member not accepted: {info.filename!r}")
        if info.filename in by_name:
            raise CheckpointError(f"duplicate zip member: {info.filename!r}")
        prefixes.add(match.group("prefix"))
        by_name[info.filename] = info
    if len(prefixes) != 1:
        raise CheckpointError("checkpoint must have exactly one top-level directory")
    if sum(info.file_size for info in infos) > max_total_bytes:
        raise CheckpointError("checkpoint exceeds the size cap")
    prefix = prefixes.pop()
    if f"{prefix}/data.pkl" not in by_name:
        raise CheckpointError("checkpoint has no data.pkl")
    return prefix, by_name


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    if info.file_size > limit:
        raise CheckpointError(f"zip member too large: {info.filename!r}")
    with archive.open(info) as handle:
        data = handle.read(limit + 1)
    if len(data) != info.file_size:
        raise CheckpointError(f"zip member size mismatch: {info.filename!r}")
    return data


def _byteorder(archive: zipfile.ZipFile, prefix: str, members: dict[str, zipfile.ZipInfo]) -> str:
    info = members.get(f"{prefix}/byteorder")
    if info is None:
        return "little"
    value = _read_member(archive, info, limit=16).decode("ascii", errors="replace").strip()
    if value not in ("little", "big"):
        raise CheckpointError("unsupported byteorder record")
    return value


# --- tensor materialisation --------------------------------------------------------------------


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    step = 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


def _check_view(name: str, ref: _TensorRef) -> int:
    if ref.offset < 0 or any(size < 0 for size in ref.shape):
        raise CheckpointError(f"negative offset or size in tensor {name!r}")
    if len(ref.shape) != len(ref.stride):
        raise CheckpointError(f"shape/stride rank mismatch in tensor {name!r}")
    expected = _contiguous_strides(ref.shape)
    if any(size > 1 and got != want for size, got, want in zip(ref.shape, ref.stride, expected, strict=True)):
        raise CheckpointError(f"non-contiguous tensor not supported: {name!r}")
    numel = math.prod(ref.shape)
    if ref.offset + numel > ref.storage.numel:
        raise CheckpointError(f"tensor {name!r} reads past the end of its storage")
    return numel


def _to_little_endian(raw: bytes, byteorder: str) -> bytes:
    if byteorder == "little":
        return raw
    values = array.array("f", raw)
    values.byteswap()
    return values.tobytes()


def _materialise(
    state: dict, archive: zipfile.ZipFile, prefix: str, members: dict[str, zipfile.ZipInfo], max_total_bytes: int
) -> tuple[Tensor, ...]:
    byteorder = _byteorder(archive, prefix, members)
    storages: dict[str, bytes] = {}
    tensors = []
    for name, ref in state.items():
        if not isinstance(name, str) or not _KEY_RE.fullmatch(name):
            raise CheckpointError(f"unsupported state_dict key: {name!r}")
        if not isinstance(ref, _TensorRef):
            raise CheckpointError(f"state_dict value is not a tensor: {name!r}")
        itemsize = SUPPORTED_DTYPES.get(ref.storage.dtype)
        if itemsize is None:
            raise CheckpointError(f"unsupported dtype {ref.storage.dtype} in tensor {name!r} (float32 only)")
        numel = _check_view(name, ref)
        key = ref.storage.key
        info = members.get(f"{prefix}/data/{key}")
        if info is None:
            raise CheckpointError(f"missing storage {key!r} for tensor {name!r}")
        # Checked for every reference: a later one may claim a different size for the same storage.
        if info.file_size != ref.storage.numel * itemsize:
            raise CheckpointError(f"storage {key!r} size does not match its element count")
        if key not in storages:
            storages[key] = _to_little_endian(_read_member(archive, info, max_total_bytes), byteorder)
        start = ref.offset * itemsize
        data = storages[key][start : start + numel * itemsize]
        tensors.append(Tensor(name=name, dtype=ref.storage.dtype, shape=ref.shape, data=data))
    if not tensors:
        raise CheckpointError("state_dict is empty")
    return tuple(tensors)


def _check_end_record(raw: bytes) -> None:
    """Bound the central directory before ``zipfile`` parses it (member count and size, no ZIP64)."""
    position = raw.rfind(EOCD_SIGNATURE, max(0, len(raw) - EOCD_SEARCH))
    if position < 0 or position + EOCD_SIZE > len(raw):
        raise CheckpointError("not a zip-format torch checkpoint (legacy or non-torch files are rejected)")
    entries, directory_size = struct.unpack_from("<HI", raw, position + 10)
    if entries == ZIP64_MARKER_16 or directory_size == ZIP64_MARKER_32:
        raise CheckpointError("ZIP64 checkpoints are not accepted")
    if entries > MAX_MEMBERS or directory_size > MAX_CENTRAL_DIRECTORY:
        raise CheckpointError("zip central directory exceeds the cap")


def max_raw_bytes(max_total_bytes: int) -> int:
    return max_total_bytes + MAX_RAW_OVERHEAD


def parse_checkpoint_bytes(raw: bytes, *, max_total_bytes: int = MAX_TOTAL_BYTES) -> tuple[Tensor, ...]:
    """Parse checkpoint bytes into tensors in state_dict order. Raises ``CheckpointError``."""
    if len(raw) > max_raw_bytes(max_total_bytes):
        raise CheckpointError("checkpoint file exceeds the size cap")
    if not zipfile.is_zipfile(io.BytesIO(raw)):
        raise CheckpointError("not a zip-format torch checkpoint (legacy or non-torch files are rejected)")
    _check_end_record(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            prefix, members = _checked_members(archive, max_total_bytes)
            payload = _read_member(archive, members[f"{prefix}/data.pkl"], MAX_PICKLE_BYTES)
            _prescan(payload)
            state = _RestrictedUnpickler(payload).load()
            if not isinstance(state, _StateDict):
                raise CheckpointError("checkpoint is not a plain state_dict")
            return _materialise(state, archive, prefix, members, max_total_bytes)
    except CheckpointError:
        raise
    except (
        pickle.UnpicklingError,
        zipfile.BadZipFile,
        EOFError,
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RecursionError,
        MemoryError,
    ) as exc:
        raise CheckpointError(f"rejected checkpoint: {exc}") from exc


def read_checkpoint(path: Path, *, max_total_bytes: int = MAX_TOTAL_BYTES) -> Checkpoint:
    """Read a checkpoint file. The whole file is hashed and parsed from memory."""
    limit = max_raw_bytes(max_total_bytes)
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)  # bounded even if the file grows after any stat()
    tensors = parse_checkpoint_bytes(raw, max_total_bytes=max_total_bytes)
    return Checkpoint(path=path, sha256=hashlib.sha256(raw).hexdigest(), tensors=tensors)
