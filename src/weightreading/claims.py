"""Read the ``claims`` list out of a run's ``final_answer.md`` and resolve names to spec indices.

Names are resolved with the labels the model saw (``prompt.input_label`` / ``head_label``: real
names in ``shown`` mode, ``x<i>`` / ``head_<h>`` in ``blind`` mode). Malformed entries, and every
copy of an (input, head) pair claimed more than once, are kept in ``rejected`` with a reason, never
dropped silently.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from weightreading.prompt import head_label, input_label
from weightreading.protocol import JSON_BLOCK_RE
from weightreading.spec import Spec

DIRECTIONS = ("+", "-", "none")
CLAIM_KEYS = {"input", "head", "direction", "confidence"}


class ClaimsError(ValueError):
    """The final answer has no parseable claims block."""


@dataclass(frozen=True)
class Claim:
    position: int  # index in the model's claims list
    input_index: int
    head_index: int
    direction: str
    confidence: float


@dataclass(frozen=True)
class Rejected:
    position: int
    raw: object
    reason: str


@dataclass(frozen=True)
class ParsedClaims:
    claims: tuple[Claim, ...]
    rejected: tuple[Rejected, ...]
    duplicate_pairs: tuple[tuple[int, int], ...]  # (input, head) claimed more than once


def claims_payload(text: str) -> list:
    """The raw ``claims`` list of the single ```json block. Raises ``ClaimsError``."""
    blocks = JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        raise ClaimsError(f"expected exactly one ```json block, found {len(blocks)}")
    try:
        payload = json.loads(blocks[0])
    except json.JSONDecodeError as exc:
        raise ClaimsError("the ```json block is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ClaimsError('the ```json block must be an object with a "claims" list')
    return payload["claims"]


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value) if 0.0 <= value <= 1.0 else None


def _resolve(raw: object, position: int, inputs: dict[str, int], heads: dict[str, int]) -> Claim | Rejected:
    if not isinstance(raw, dict) or set(raw) != CLAIM_KEYS:
        return Rejected(position, raw, f"entry must have exactly the keys {sorted(CLAIM_KEYS)}")
    if not all(isinstance(raw[key], str) for key in ("input", "head", "direction")):
        return Rejected(position, raw, "input, head and direction must be strings")
    if raw["input"] not in inputs:
        return Rejected(position, raw, "unknown input name")
    if raw["head"] not in heads:
        return Rejected(position, raw, "unknown head name")
    if raw["direction"] not in DIRECTIONS:
        return Rejected(position, raw, f"direction must be one of {DIRECTIONS}")
    confidence = _confidence(raw["confidence"])
    if confidence is None:
        return Rejected(position, raw, "confidence must be a number in [0, 1]")
    return Claim(position, inputs[raw["input"]], heads[raw["head"]], raw["direction"], confidence)


def parse_claims(text: str, spec: Spec, task_mode: str) -> ParsedClaims:
    inputs = {input_label(spec, i, task_mode): i for i in range(len(spec.inputs))}
    heads = {head_label(spec, h, task_mode): h for h in range(len(spec.outputs))}
    payload = claims_payload(text)
    resolved = [_resolve(raw, position, inputs, heads) for position, raw in enumerate(payload)]
    valid = [item for item in resolved if isinstance(item, Claim)]
    pairs = [(claim.input_index, claim.head_index) for claim in valid]
    duplicates = tuple(sorted({pair for pair in pairs if pairs.count(pair) > 1}))
    # A pair claimed twice (possibly with opposite signs) is ambiguous: every copy is rejected.
    claims = tuple(c for c in valid if (c.input_index, c.head_index) not in duplicates)
    twice = [Rejected(c.position, payload[c.position], "pair claimed more than once") for c in valid if c not in claims]
    rejected = tuple(sorted([*(i for i in resolved if isinstance(i, Rejected)), *twice], key=lambda r: r.position))
    return ParsedClaims(claims, rejected, duplicates)
