"""Translator data types: scenario, targets, triggers, translations and event-record helpers.

Everything here is a frozen dataclass or a pure function. The event-log contract
(``events.jsonl``) is documented in ``translator/README.md``; the helpers below build the parts
that several record types share, so every record spells them the same way.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtd_rl_xavier.lag.rendering import SECONDS_DECIMALS, STEP_SECONDS, RenderedTrajectory

EVENT_SCHEMA_VERSION = 2
RECORD_DETECTOR_CALL = "detector_call"
RECORD_TRANSLATION_EVENT = "translation_event"
RECORD_TARGET_CLOSED = "target_closed"
RECORD_TRIGGER_SUPPRESSED = "trigger_suppressed"
RECORD_TYPES = (RECORD_DETECTOR_CALL, RECORD_TRANSLATION_EVENT, RECORD_TARGET_CLOSED, RECORD_TRIGGER_SUPPRESSED)

TARGET_AIRCRAFT = "aircraft"
TARGET_TEAM = "team"
TARGET_KINDS = (TARGET_AIRCRAFT, TARGET_TEAM)

# Trigger kinds in primary-trigger precedence order (first = highest). ``initial`` is the forced
# bootstrap at a target's first instruction; ``start`` (開始点) and ``change`` (変化点) are detector
# labels, exclusive by construction; ``request`` (要求点) is forced at a scenario request.
TRIGGER_INITIAL = "initial"
TRIGGER_START = "start"
TRIGGER_CHANGE = "change"
TRIGGER_REQUEST = "request"
TRIGGER_PRECEDENCE = (TRIGGER_INITIAL, TRIGGER_START, TRIGGER_CHANGE, TRIGGER_REQUEST)
INITIAL_SOURCE = "first_instruction"
DETECTOR_SOURCE = "detector"
DOWN_REASON = "target_down"
# The detector's one choice question: its option names, in request order. ``none`` triggers nothing.
DETECTOR_LABEL_NONE = "none"
DETECTOR_LABELS = (DETECTOR_LABEL_NONE, TRIGGER_START, TRIGGER_CHANGE)

STEP_DIGITS = 6
# Identifiers that end up in event ids, file names and prompts: no spaces, slashes or colons.
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # used with fullmatch only


class ModelCallError(Exception):
    """A layer's model call failed. Raised by the layer adapters; the pipeline stops the run.

    ``error_type`` is the original exception class name and the message is ``str()`` of the
    original (the LLM parts keep both free of keys and bodies). ``uncertain`` means the request
    may have reached the server; such a call is never resent. ``raw`` is a received response body.
    """

    def __init__(self, message: str, *, error_type: str, uncertain: bool, raw: Any = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.uncertain = uncertain
        self.raw = raw


UNEXPECTED_ERROR_MESSAGE = "unexpected exception; details withheld"


def redacted_error(exc: BaseException) -> dict[str, str]:
    """``{type, message}`` for records. Only messages built by this project or the LLM parts are kept.

    ``ModelCallError`` (sanitised by the LLM parts), ``ValueError`` (config, scenario, data, render and
    parse errors raised here) and ``OSError`` (file paths) keep ``str(exc)``; anything else could echo
    request text or a credential, so only its class name is recorded.
    """
    if isinstance(exc, (ModelCallError, ValueError, OSError)):
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": type(exc).__name__, "message": UNEXPECTED_ERROR_MESSAGE}


@dataclass(frozen=True, order=True)
class TargetRef:
    """One exact translation unit (D17). Ordering is the canonical (kind, target_id) order."""

    kind: str
    target_id: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.target_id}

    @property
    def label(self) -> str:
        return f"{self.kind} {self.target_id}"


@dataclass(frozen=True)
class EpisodeRef:
    episode_id: str
    alias: str  # neutral alias; the only episode name a model ever sees
    csv_path: Path
    video_path: Path | None


@dataclass(frozen=True)
class Instruction:
    instruction_id: str
    issue_step: int
    target: TargetRef
    text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "issue_step": self.issue_step,
            "target": self.target.to_json(),
            "text": self.text,
        }


@dataclass(frozen=True)
class Alarm:
    """An external warning; it reaches every target that shares an aircraft with ``targets``."""

    alarm_id: str
    step: int
    targets: tuple[TargetRef, ...]
    text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "alarm_id": self.alarm_id,
            "step": self.step,
            "targets": [target.to_json() for target in self.targets],
            "text": self.text,
        }


@dataclass(frozen=True)
class RequestPoint:
    request_id: str
    step: int
    targets: tuple[TargetRef, ...]
    source: str
    question: str  # what the human asked; shown to the translation layer of each target


@dataclass(frozen=True)
class Scenario:
    schema_version: int
    scenario_id: str
    episode: EpisodeRef
    instructions: tuple[Instruction, ...]
    alarms: tuple[Alarm, ...]
    requests: tuple[RequestPoint, ...]


@dataclass(frozen=True)
class TriggerReason:
    kind: str  # one of TRIGGER_PRECEDENCE
    source: str  # "first_instruction", "detector", or the request source label
    reference_id: str  # instruction id (initial), request id, or the detector call id
    probability: float | None  # detector p(label); None for forced triggers

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "reference_id": self.reference_id,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class IntentTranslation:
    intent: str
    evidence: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"intent": self.intent, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class PreviousTranslation:
    """The target's own latest translation; it feeds the next detector and translation calls."""

    event_id: str
    step: int
    translation: IntentTranslation

    def to_json(self) -> dict[str, Any]:
        return {"event_id": self.event_id, "step": self.step, **self.translation.to_json()}


@dataclass(frozen=True)
class TargetState:
    target: TargetRef
    instruction: Instruction  # the target's current instruction (D21: a later one replaces it)
    previous: PreviousTranslation | None
    closed: bool


def sort_reasons(reasons: tuple[TriggerReason, ...]) -> tuple[TriggerReason, ...]:
    """Precedence order, then reference id: the same reasons always serialise the same way."""
    return tuple(sorted(reasons, key=lambda reason: (TRIGGER_PRECEDENCE.index(reason.kind), reason.reference_id)))


def primary_trigger(reasons: tuple[TriggerReason, ...]) -> str:
    """initial > start > change > request."""
    if not reasons:
        raise ValueError("a translation needs at least one trigger reason")
    return sort_reasons(reasons)[0].kind


def step_time_s(step: int) -> float:
    """Simulation seconds of a step, rounded exactly like the 17-format ``time_s``."""
    return round(step * STEP_SECONDS, SECONDS_DECIMALS)


def _step_key(scenario_id: str, target: TargetRef, step: int) -> str:
    return f"{scenario_id}:{target.kind}:{target.target_id}:{step:0{STEP_DIGITS}d}"


def event_id(scenario_id: str, target: TargetRef, step: int) -> str:
    """``<scenario_id>:<kind>:<target_id>:<step:06d>``; unique (one translation per target per step)."""
    return _step_key(scenario_id, target, step)


def detector_call_id(scenario_id: str, target: TargetRef, step: int) -> str:
    """``<scenario_id>:detector:<kind>:<target_id>:<step:06d>``; one detector call per target per tick."""
    return f"{scenario_id}:{DETECTOR_SOURCE}:{target.kind}:{target.target_id}:{step:0{STEP_DIGITS}d}"


def trajectory_json(rendered: RenderedTrajectory) -> dict[str, Any]:
    """The exact rendered text the model saw plus the renderer metadata (clipped steps)."""
    return {
        "renderer": rendered.renderer,
        "renderer_version": rendered.renderer_version,
        "start_step": rendered.start_step,
        "end_step": rendered.end_step,
        "anchor_step": rendered.anchor_step,
        "object_ids": list(rendered.object_ids),
        "fields": list(rendered.fields),
        "text": rendered.text,
        "sha256": rendered.sha256,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    mapping = dict(pairs)
    if len(mapping) != len(pairs):
        raise ValueError("JSON object repeats a key")
    return mapping


def _reject_constant(constant: str) -> None:
    raise ValueError(f"non-standard JSON constant {constant}")


def strict_json_loads(text: str) -> Any:
    """``json.loads`` for logs and model output: duplicate keys, NaN and Infinity raise ``ValueError``.

    A repeated key would let a reader show one value while a parser keeps another (last wins), so it
    is rejected. Deep nesting (``RecursionError``) is reported as ``ValueError`` too.
    """
    try:
        return json.loads(text, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except RecursionError:
        raise ValueError("JSON nesting is too deep") from None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
