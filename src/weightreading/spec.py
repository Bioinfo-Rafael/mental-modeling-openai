"""Network spec (``configs/weightreading/*.spec.json``): load, validate, cross-check against tensors.

Only ``description``, ``inputs``, ``input_clip``, ``architecture`` and ``outputs`` ever reach the
prompt. ``provenance`` and ``notes`` stay on the harness side (they name the source checkpoint).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from weightreading import SCHEMA_VERSION
from weightreading.architecture import BLOCK_TYPES, Block, expected_params
from weightreading.tensors import Tensor

ALIAS_RE = re.compile(r"^net-[0-9a-f]{8}$")
TOP_LEVEL_KEYS = {
    "schema_version",
    "network_alias",
    "hidden_from_prompt",
    "provenance",
    "description",
    "inputs",
    "input_clip",
    "architecture",
    "outputs",
    "notes",
}
CLIP_BOUNDS = 2
LINSPACE_ENDS = 2


class SpecError(ValueError):
    """The spec is malformed or does not match the checkpoint. ``problems`` lists every issue."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


@dataclass(frozen=True)
class InputSpec:
    index: int
    name: str
    definition: str
    unit: str


@dataclass(frozen=True)
class OutputSpec:
    head: int
    name: str
    bins: int
    low: float
    high: float


@dataclass(frozen=True)
class Spec:
    alias: str
    description: str
    inputs: tuple[InputSpec, ...]
    input_clip: tuple[float, float] | None
    blocks: tuple[Block, ...]
    outputs: tuple[OutputSpec, ...]
    provenance: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]


def _require(raw: dict, key: str, kind: type | tuple[type, ...]) -> object:
    if key not in raw:
        raise SpecError([f"missing field: {key}"])
    value = raw[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise SpecError([f"field {key} has the wrong type"])
    return value


def _parse_inputs(raw: list) -> tuple[InputSpec, ...]:
    inputs = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SpecError([f"inputs[{position}] must be an object"])
        index = _require(item, "index", int)
        if index != position:
            raise SpecError([f"inputs[{position}] has index {index}; indices must be 0..n-1 in order"])
        inputs.append(
            InputSpec(
                index=position,
                name=str(_require(item, "name", str)),
                definition=str(_require(item, "definition", str)),
                unit=str(item.get("unit", "")),
            )
        )
    if not inputs:
        raise SpecError(["inputs must not be empty"])
    return tuple(inputs)


def _parse_outputs(raw: list) -> tuple[OutputSpec, ...]:
    outputs = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or _require(item, "head", int) != position:
            raise SpecError([f"outputs[{position}] must be an object with head == {position}"])
        bins = _require(item, "bins", int)
        linspace = _require(_require(item, "bin_value", dict), "linspace", list)
        if bins < 1 or len(linspace) != LINSPACE_ENDS or not all(isinstance(v, int | float) for v in linspace):
            raise SpecError([f"outputs[{position}] needs bins >= 1 and bin_value.linspace [low, high]"])
        name = str(_require(item, "name", str))
        outputs.append(OutputSpec(position, name, bins, float(linspace[0]), float(linspace[1])))
    if not outputs:
        raise SpecError(["outputs must not be empty"])
    return tuple(outputs)


def _parse_block(item: dict, heads: tuple[OutputSpec, ...]) -> Block:
    block_type = _require(item, "type", str)
    if block_type not in BLOCK_TYPES:
        raise SpecError([f"unknown block type: {block_type}"])
    common = {
        "block_id": str(_require(item, "id", str)),
        "block_type": block_type,
        "prefix": str(_require(item, "prefix", str)),
        "input": str(_require(item, "input", str)),
        "eps": float(item.get("eps", 1e-5)),
    }
    if block_type == "mlp":
        sizes = tuple(int(v) for v in _require(item, "hidden_sizes", list))
        return Block(**common, input_size=int(_require(item, "input_size", int)), sizes=sizes)
    if block_type == "gru":
        if item.get("num_layers", 1) != 1:
            raise SpecError(["only single-layer GRUs are supported"])
        hidden = int(_require(item, "hidden_size", int))
        return Block(**common, input_size=int(_require(item, "input_size", int)), sizes=(hidden,))
    if block_type == "layernorm":
        size = int(_require(item, "size", int))
        return Block(**common, input_size=size, sizes=(size,))
    return Block(
        **common,
        input_size=int(_require(item, "input_size", int)),
        sizes=tuple(head.bins for head in heads),
        param=str(item.get("param", "logits_net")),
    )


def _check_dataflow(blocks: tuple[Block, ...], n_inputs: int) -> list[str]:
    problems = []
    sizes = {"observation": n_inputs}
    for block in blocks:
        if block.block_id in sizes:
            problems.append(f"duplicate block id: {block.block_id}")
        if block.input not in sizes:
            problems.append(f"block {block.block_id} reads unknown input {block.input}")
        elif sizes[block.input] != block.input_size:
            problems.append(f"block {block.block_id} expects {block.input_size} inputs, gets {sizes[block.input]}")
        sizes[block.block_id] = block.output_size
    if not blocks or blocks[-1].block_type != "categorical_heads":
        problems.append("the last architecture block must be categorical_heads")
    return problems


def parse_spec(raw: object) -> Spec:
    """Validate a decoded spec object. Raises ``SpecError``."""
    if not isinstance(raw, dict):
        raise SpecError(["spec must be a JSON object"])
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise SpecError([f"unknown top-level fields: {unknown}"])
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise SpecError([f"schema_version must be {SCHEMA_VERSION}"])
    alias = str(_require(raw, "network_alias", str))
    if not ALIAS_RE.match(alias):
        raise SpecError(["network_alias must look like net-<8 hex digits>"])
    inputs = _parse_inputs(_require(raw, "inputs", list))
    outputs = _parse_outputs(_require(raw, "outputs", list))
    blocks = tuple(_parse_block(item, outputs) for item in _require(raw, "architecture", list))
    problems = _check_dataflow(blocks, len(inputs))
    if problems:
        raise SpecError(problems)
    clip = raw.get("input_clip")
    if clip is not None and (not isinstance(clip, list) or len(clip) != CLIP_BOUNDS):
        raise SpecError(["input_clip must be null or [low, high]"])
    provenance = raw.get("provenance", {})
    return Spec(
        alias=alias,
        description=str(_require(raw, "description", str)),
        inputs=inputs,
        input_clip=None if clip is None else (float(clip[0]), float(clip[1])),
        blocks=blocks,
        outputs=outputs,
        provenance=tuple((str(k), str(v)) for k, v in provenance.items()),
        notes=tuple(str(note) for note in raw.get("notes", [])),
    )


def load_spec(path: Path) -> Spec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError([f"spec is not valid JSON: {exc}"]) from exc
    return parse_spec(raw)


def check_against_tensors(spec: Spec, tensors: tuple[Tensor, ...], checkpoint_sha256: str | None = None) -> None:
    """Every tensor claimed by exactly one block with the expected shape. Raises ``SpecError``."""
    problems = []
    claimed: dict[str, str] = {}
    expected: dict[str, tuple[int, ...]] = {}
    for block in spec.blocks:
        for key, shape in expected_params(block).items():
            if key in claimed:
                problems.append(f"{key} claimed by both {claimed[key]} and {block.block_id}")
            claimed[key] = block.block_id
            expected[key] = shape
    actual = {tensor.name: tensor for tensor in tensors}
    problems += [f"tensor not claimed by any block: {key}" for key in actual if key not in expected]
    problems += [f"expected tensor missing from checkpoint: {key}" for key in expected if key not in actual]
    for key, shape in expected.items():
        tensor = actual.get(key)
        if tensor is not None and tensor.shape != shape:
            problems.append(f"{key}: checkpoint shape {tensor.shape} != spec shape {shape}")
    recorded = dict(spec.provenance).get("checkpoint_sha256")
    if checkpoint_sha256 is not None and recorded is not None and recorded != checkpoint_sha256:
        problems.append("checkpoint sha256 differs from provenance.checkpoint_sha256")
    if problems:
        raise SpecError(problems)
