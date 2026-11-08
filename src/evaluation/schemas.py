"""Judge data contracts: the Translator event log, key points, the output schema and its parser.

- ``load_translator_run`` reads a Translator run directory (``run.json`` + ``events.jsonl``) and
  checks the parts the judge relies on: ``schema_version`` 1, a gap-free ``sequence`` and the
  required ``translation_event`` fields (the trajectory text must match its SHA-256 and be JSON
  lines). Lines are split on ``"\\n"`` only (U+2028, U+2029 and U+0085 are legal inside a JSON
  string), and a repeated JSON key is an error. Every id the judge prints (event, target,
  aircraft, instruction, trigger source, trajectory objects) must be a safe id, so no field but
  the free texts can carry a line break or a heading into the request.
- Key points (D36: partial judging): every entry names an existing translation event exactly once;
  key-point ids are unique per event; lists are non-empty and texts non-blank.
- The output schema is built per event: an enum of that event's key-point ids (0/1/2 each, D29)
  plus an overall 1-5 score, each with a rationale. ``minLength`` is not used because strict
  structured outputs do not support it; the parser rejects blank rationales instead.
- The parser never repairs, coerces or retries.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translate.events import (
    EVENT_SCHEMA_VERSION,
    RECORD_TRANSLATION_EVENT,
    RECORD_TYPES,
    SAFE_ID_PATTERN,
    STEP_DIGITS,
    TARGET_KINDS,
    TRIGGER_PRECEDENCE,
    sha256_text,
    strict_json_loads,
)

from .configuration import read_strict_yaml

RUN_JSON = "run.json"
EVENTS_FILE = "events.jsonl"
KEY_POINTS_SCHEMA_VERSION = 1
KEY_POINTS_KEYS = {"schema_version", "events"}
EVENT_ENTRY_KEYS = {"event_id", "key_points"}
KEY_POINT_KEYS = {"key_point_id", "text"}
KEY_POINT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # used with fullmatch only
OUTPUT_SCHEMA_NAME = "judge_verdict"
KEY_POINT_SCORES = (0, 1, 2)  # D29: 2 = core purpose and main action, 1 = partial, 0 = missing / different
OVERALL_SCORES = (1, 2, 3, 4, 5)
VERDICT_KEYS = frozenset({"key_points", "overall"})
OVERALL_ONLY_KEYS = frozenset({"overall"})
CRITERION_KEYS = frozenset({"key_point_id", "rationale", "score"})
OVERALL_KEYS = frozenset({"rationale", "score"})
MAX_LISTED_IDS = 5  # unknown ids quoted in an error message
JSONL_SEPARATOR = "\n"  # the only record separator; str.splitlines() would also split on U+2028 etc.
# ``<scenario_id>:<kind>:<target_id>:<step:06d>`` (translator.schema.event_id)
EVENT_ID_PATTERN = re.compile(
    rf"{SAFE_ID_PATTERN.pattern}:(?:{'|'.join(TARGET_KINDS)}):{SAFE_ID_PATTERN.pattern}:[0-9]{{{STEP_DIGITS},}}"
)
UNICODE_LINE_BREAKS = ("\x85", "\u2028", "\u2029")
# A carriage return is legal JSON whitespace between tokens, but a reader may start a new line there.
TRAJECTORY_FORBIDDEN = ("\r", *UNICODE_LINE_BREAKS)


class EventLogError(ValueError):
    """The Translator run directory or its event log violates the event contract."""


class KeyPointsError(ValueError):
    """The key-points file is malformed or does not match the event log."""


class JudgeSchemaError(ValueError):
    """The judge answered, but not with one valid verdict. Never repaired or retried."""


# --- Translator run ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class TranslatorRun:
    run_dir: Path
    run_json: dict[str, Any]
    run_json_sha256: str
    events_sha256: str
    records: tuple[dict[str, Any], ...]  # every record, in sequence order
    translation_events: tuple[dict[str, Any], ...]

    def event_ids(self) -> tuple[str, ...]:
        return tuple(event["event_id"] for event in self.translation_events)


@dataclass(frozen=True)
class RunEpisode:
    """The episode a Translator run covered, as its ``run.json`` records it."""

    episode_id: str
    last_complete_step: int


def run_episode(run: TranslatorRun) -> RunEpisode:
    """``inputs.episode`` of ``run.json``. Needed only to find an episode's video (judge ``video`` block)."""
    inputs = run.run_json.get("inputs")
    episode = inputs.get("episode") if isinstance(inputs, dict) else None
    if not isinstance(episode, dict):
        raise EventLogError(f"{RUN_JSON} has no inputs.episode (the video block needs the episode id)")
    episode_id, last_step = episode.get("episode_id"), episode.get("last_complete_step")
    if not isinstance(episode_id, str) or not SAFE_ID_PATTERN.fullmatch(episode_id):
        raise EventLogError(f"{RUN_JSON} inputs.episode.episode_id must be a safe id")
    if not isinstance(last_step, int) or isinstance(last_step, bool) or last_step < 0:
        raise EventLogError(f"{RUN_JSON} inputs.episode.last_complete_step must be a non-negative integer")
    return RunEpisode(episode_id=episode_id, last_complete_step=last_step)


ANSWER_PROBLEM = "answer_problem"  # set by the translation pipeline on a skipped invalid answer


def jsonl_lines(text: str) -> list[str]:
    """The lines of a JSONL text: split on ``"\\n"`` only, without the empty tail after the last one."""
    lines = text.split(JSONL_SEPARATOR)
    return lines[:-1] if lines and not lines[-1] else lines


def _field(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise EventLogError(f"{where}: missing field {key!r}")
    return mapping[key]


def _text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = _field(mapping, key, where)
    if not isinstance(value, str) or not value.strip():
        raise EventLogError(f"{where}.{key} must be a non-empty string")
    return value


def _string(mapping: dict[str, Any], key: str, where: str) -> str:
    value = _field(mapping, key, where)
    if not isinstance(value, str):
        raise EventLogError(f"{where}.{key} must be a string")
    return value


def _count(mapping: dict[str, Any], key: str, where: str) -> int:
    value = _field(mapping, key, where)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise EventLogError(f"{where}.{key} must be a non-negative integer")
    return value


def _object(mapping: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    value = _field(mapping, key, where)
    if not isinstance(value, dict):
        raise EventLogError(f"{where}.{key} must be an object")
    return value


def _items(mapping: dict[str, Any], key: str, where: str, item_type: type) -> list[Any]:
    value = _field(mapping, key, where)
    if not isinstance(value, list) or not value or not all(isinstance(item, item_type) for item in value):
        raise EventLogError(f"{where}.{key} must be a non-empty list of {item_type.__name__}")
    return value


def _safe_id(mapping: dict[str, Any], key: str, where: str, pattern: re.Pattern[str] = SAFE_ID_PATTERN) -> str:
    value = _field(mapping, key, where)
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise EventLogError(f"{where}.{key} must match {pattern.pattern}")
    return value


def _safe_ids(mapping: dict[str, Any], key: str, where: str) -> list[str]:
    values = _items(mapping, key, where, str)
    if not all(SAFE_ID_PATTERN.fullmatch(value) for value in values):
        raise EventLogError(f"{where}.{key} must list ids matching {SAFE_ID_PATTERN.pattern}")
    return values


def _check_target(mapping: dict[str, Any], key: str, where: str) -> None:
    target = _object(mapping, key, where)
    if _field(target, "kind", f"{where}.{key}") not in TARGET_KINDS:
        raise EventLogError(f"{where}.{key}.kind must be one of {list(TARGET_KINDS)}")
    _safe_id(target, "id", f"{where}.{key}")


def _is_json_object(line: str) -> bool:
    try:
        return isinstance(strict_json_loads(line), dict)
    except ValueError:
        return False


def _check_trajectory_text(trajectory: dict[str, Any], where: str) -> None:
    """The exact window text: its SHA-256 matches, and it is JSON lines with no other line break."""
    text = _text(trajectory, "text", where)
    if sha256_text(text) != trajectory["sha256"]:
        raise EventLogError(f"{where}.text does not match its sha256")
    if any(mark in text for mark in TRAJECTORY_FORBIDDEN) or not all(map(_is_json_object, text.split(JSONL_SEPARATOR))):
        raise EventLogError(f"{where}.text must be JSON lines, one object per line")


def _check_reasons(event: dict[str, Any], where: str) -> None:
    if _field(event, "primary_trigger", where) not in TRIGGER_PRECEDENCE:
        raise EventLogError(f"{where}.primary_trigger must be one of {list(TRIGGER_PRECEDENCE)}")
    for index, reason in enumerate(_items(event, "trigger_reasons", where, dict)):
        name = f"{where}.trigger_reasons[{index}]"
        if _field(reason, "kind", name) not in TRIGGER_PRECEDENCE:
            raise EventLogError(f"{name}.kind must be one of {list(TRIGGER_PRECEDENCE)}")
        _safe_id(reason, "source", name)
        _string(reason, "reference_id", name)
        probability = _field(reason, "probability", name)
        if probability is not None and (not isinstance(probability, (int, float)) or isinstance(probability, bool)):
            raise EventLogError(f"{name}.probability must be a number or null")


def _check_input(event: dict[str, Any], where: str) -> None:
    payload = _object(event, "input", where)
    name = f"{where}.input"
    instruction_ids = []
    for index, instruction in enumerate(_items(payload, "instructions_in_effect", name, dict)):
        item = f"{name}.instructions_in_effect[{index}]"
        instruction_ids.append(_safe_id(instruction, "instruction_id", item))
        _count(instruction, "issue_step", item)
        _check_target(instruction, "target", item)
        _text(instruction, "text", item)
    if _text(payload, "primary_instruction_id", name) not in instruction_ids:
        raise EventLogError(f"{name}.primary_instruction_id is not among instructions_in_effect")
    _safe_ids(payload, "target_aircraft", name)
    trajectory = _object(payload, "trajectory", name)
    for key in ("renderer", "sha256"):
        _text(trajectory, key, f"{name}.trajectory")
    _safe_id(trajectory, "renderer_version", f"{name}.trajectory")
    for key in ("start_step", "end_step", "anchor_step"):
        _count(trajectory, key, f"{name}.trajectory")
    _safe_ids(trajectory, "object_ids", f"{name}.trajectory")
    _check_trajectory_text(trajectory, f"{name}.trajectory")


def _check_translation_event(event: dict[str, Any], where: str) -> None:
    _safe_id(event, "event_id", where, EVENT_ID_PATTERN)
    _count(event, "step", where)
    time_s = _field(event, "time_s", where)
    if not isinstance(time_s, (int, float)) or isinstance(time_s, bool) or not math.isfinite(time_s):
        raise EventLogError(f"{where}.time_s must be a finite number")
    _check_target(event, "target", where)
    _check_reasons(event, where)
    _check_input(event, where)
    _check_output(event, where)


def _check_output(event: dict[str, Any], where: str) -> None:
    """A non-blank answer, or a skipped invalid one: blank output allowed only with ``answer_problem``.

    The translation pipeline writes a rejected answer (``answer.on_invalid: skip``) as a blank output
    plus the reason. It is judged as given, like the assess judge does (rubric level 1: empty intent).
    """
    output = _object(event, "output", where)
    if ANSWER_PROBLEM not in event:
        _text(output, "intent", f"{where}.output")
        _items(output, "evidence", f"{where}.output", str)
        return
    _text(event, ANSWER_PROBLEM, where)
    _string(output, "intent", f"{where}.output")
    evidence = _field(output, "evidence", f"{where}.output")
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise EventLogError(f"{where}.output.evidence must be a list of str")


def _parse_records(text: str) -> tuple[dict[str, Any], ...]:
    records = []
    for index, line in enumerate(jsonl_lines(text)):
        where = f"{EVENTS_FILE} line {index + 1}"
        try:
            record = strict_json_loads(line)
        except ValueError as exc:
            raise EventLogError(f"{where} is not valid JSON (or repeats an object key)") from exc
        if not isinstance(record, dict):
            raise EventLogError(f"{where} must be a JSON object")
        version = record.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool) or version != EVENT_SCHEMA_VERSION:
            raise EventLogError(f"{where}: schema_version must be {EVENT_SCHEMA_VERSION}")
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence != index + 1:
            raise EventLogError(f"{where}: sequence must be {index + 1} (1..n without gaps)")
        if record.get("record_type") not in RECORD_TYPES:
            raise EventLogError(f"{where}: record_type must be one of {list(RECORD_TYPES)}")
        if record["record_type"] == RECORD_TRANSLATION_EVENT:
            _check_translation_event(record, where)
        records.append(record)
    return tuple(records)


def load_translator_run(run_dir: Path) -> TranslatorRun:
    """Read and check a Translator run directory. Nothing is written."""
    run_dir = Path(run_dir)
    run_path, events_path = run_dir / RUN_JSON, run_dir / EVENTS_FILE
    if not run_path.is_file():
        raise EventLogError(f"{run_dir} has no {RUN_JSON}")
    if not events_path.is_file():
        raise EventLogError(f"{run_dir} has no {EVENTS_FILE} (a preview run has no events to judge)")
    run_bytes, events_bytes = run_path.read_bytes(), events_path.read_bytes()
    try:
        run_json = strict_json_loads(run_bytes.decode("utf-8"))
        events_text = events_bytes.decode("utf-8")
    except ValueError as exc:
        raise EventLogError(f"{RUN_JSON} or {EVENTS_FILE} is not valid UTF-8 JSON (or repeats an object key)") from exc
    if not isinstance(run_json, dict):
        raise EventLogError(f"{RUN_JSON} must be a JSON object")
    records = _parse_records(events_text)
    events = tuple(record for record in records if record["record_type"] == RECORD_TRANSLATION_EVENT)
    ids = [event["event_id"] for event in events]
    if len(set(ids)) != len(ids):
        raise EventLogError(f"{EVENTS_FILE} repeats a translation event_id")
    return TranslatorRun(
        run_dir=run_dir,
        run_json=run_json,
        run_json_sha256=hashlib.sha256(run_bytes).hexdigest(),
        events_sha256=hashlib.sha256(events_bytes).hexdigest(),
        records=records,
        translation_events=events,
    )


# --- key points -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyPoint:
    key_point_id: str
    text: str


@dataclass(frozen=True)
class EventKeyPoints:
    event_id: str
    key_points: tuple[KeyPoint, ...]


def _key_mapping(raw: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise KeyPointsError(f"{name} must be a mapping with exactly {sorted(keys)}")
    return raw


def _key_point(raw: object, name: str) -> KeyPoint:
    section = _key_mapping(raw, KEY_POINT_KEYS, name)
    key_point_id, text = section["key_point_id"], section["text"]
    if not isinstance(key_point_id, str) or not KEY_POINT_ID_PATTERN.fullmatch(key_point_id):
        raise KeyPointsError(f"{name}.key_point_id must match {KEY_POINT_ID_PATTERN.pattern}")
    if not isinstance(text, str) or not text.strip():
        raise KeyPointsError(f"{name}.text must be a non-blank string")
    return KeyPoint(key_point_id=key_point_id, text=text)


def _event_entry(raw: object, index: int) -> EventKeyPoints:
    name = f"events[{index}]"
    section = _key_mapping(raw, EVENT_ENTRY_KEYS, name)
    event_id, items = section["event_id"], section["key_points"]
    if not isinstance(event_id, str) or not event_id.strip():
        raise KeyPointsError(f"{name}.event_id must be a non-empty string")
    if not isinstance(items, list) or not items:
        raise KeyPointsError(f"{name}.key_points must be a non-empty list ({event_id})")
    key_points = tuple(_key_point(item, f"{name}.key_points[{i}]") for i, item in enumerate(items))
    ids = [key_point.key_point_id for key_point in key_points]
    if len(set(ids)) != len(ids):
        raise KeyPointsError(f"{name} repeats a key_point_id ({event_id})")
    return EventKeyPoints(event_id=event_id, key_points=key_points)


def parse_key_points(raw: object) -> tuple[EventKeyPoints, ...]:
    """Entries in file order. ``events: []`` is valid (nothing is judged)."""
    section = _key_mapping(raw, KEY_POINTS_KEYS, "key points file")
    version = section["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != KEY_POINTS_SCHEMA_VERSION:
        raise KeyPointsError(f"key points schema_version must be {KEY_POINTS_SCHEMA_VERSION}")
    if not isinstance(section["events"], list):
        raise KeyPointsError("events must be a list (use [] for none)")
    entries = tuple(_event_entry(item, index) for index, item in enumerate(section["events"]))
    event_ids = [entry.event_id for entry in entries]
    if len(set(event_ids)) != len(event_ids):
        duplicates = sorted({event_id for event_id in event_ids if event_ids.count(event_id) > 1})
        raise KeyPointsError(f"duplicate event_id in key points: {duplicates[:MAX_LISTED_IDS]}")
    return entries


def load_key_points(path: Path) -> tuple[EventKeyPoints, ...]:
    return parse_key_points(read_strict_yaml(path, KeyPointsError))


def match_key_points(event_ids: tuple[str, ...], entries: tuple[EventKeyPoints, ...]) -> dict[str, EventKeyPoints]:
    """event id -> key points. Every entry must name an existing translation event (D36)."""
    known = set(event_ids)
    unknown = [entry.event_id for entry in entries if entry.event_id not in known]
    if unknown:
        raise KeyPointsError(
            f"{len(unknown)} key-point entries name no translation event in the run: {unknown[:MAX_LISTED_IDS]}"
        )
    return {entry.event_id: entry for entry in entries}


# --- output schema and parser -----------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionResult:
    key_point_id: str
    rationale: str
    score: int  # 0 / 1 / 2


@dataclass(frozen=True)
class JudgeVerdict:
    key_points: tuple[CriterionResult, ...]  # in the requested key-point order; () when key points are excluded
    overall_rationale: str
    overall_score: int  # 1..5

    def to_json(self) -> dict[str, Any]:
        criteria = [
            {"key_point_id": item.key_point_id, "rationale": item.rationale, "score": item.score}
            for item in self.key_points
        ]
        return {"key_points": criteria, "overall": {"rationale": self.overall_rationale, "score": self.overall_score}}


def _overall_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rationale": {"type": "string"},
            "score": {"type": "integer", "enum": list(OVERALL_SCORES)},
        },
        "required": ["rationale", "score"],
    }


def build_judge_output_schema(key_points: tuple[KeyPoint, ...]) -> dict[str, Any]:
    """Strict schema for one event. No key points (block excluded) -> the overall score only."""
    if not key_points:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"overall": _overall_schema()},
            "required": ["overall"],
        }
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "key_point_id": {"type": "string", "enum": [key_point.key_point_id for key_point in key_points]},
            "rationale": {"type": "string"},
            "score": {"type": "integer", "enum": list(KEY_POINT_SCORES)},
        },
        "required": ["key_point_id", "rationale", "score"],
    }
    count = len(key_points)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "key_points": {"type": "array", "minItems": count, "maxItems": count, "items": item},
            "overall": _overall_schema(),
        },
        "required": ["key_points", "overall"],
    }


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _rationale(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or not _is_utf8(value):
        raise JudgeSchemaError(f"{name} must be a non-blank UTF-8 string")
    return value


def _score(value: object, allowed: tuple[int, ...], name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in allowed:
        raise JudgeSchemaError(f"{name} must be an integer in {list(allowed)}")
    return value


def _exact(value: object, keys: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise JudgeSchemaError(f"{name} must be an object with exactly {sorted(keys)}")
    return value


def _criteria(raw: object, key_points: tuple[KeyPoint, ...]) -> tuple[CriterionResult, ...]:
    if not isinstance(raw, list):
        raise JudgeSchemaError("key_points must be a list")
    by_id: dict[str, CriterionResult] = {}
    for index, item in enumerate(raw):
        name = f"key_points[{index}]"
        section = _exact(item, CRITERION_KEYS, name)
        key_point_id = section["key_point_id"]
        if not isinstance(key_point_id, str) or key_point_id not in {kp.key_point_id for kp in key_points}:
            raise JudgeSchemaError(f"{name}.key_point_id is not a requested key point")
        if key_point_id in by_id:
            raise JudgeSchemaError(f"{name}.key_point_id {key_point_id!r} appears more than once")
        result = CriterionResult(
            key_point_id=key_point_id,
            rationale=_rationale(section["rationale"], f"{name}.rationale"),
            score=_score(section["score"], KEY_POINT_SCORES, f"{name}.score"),
        )
        by_id = {**by_id, key_point_id: result}
    missing = [kp.key_point_id for kp in key_points if kp.key_point_id not in by_id]
    if missing:
        raise JudgeSchemaError(f"key points missing from the verdict: {missing}")
    return tuple(by_id[kp.key_point_id] for kp in key_points)


def parse_judge_output(output_text: str, key_points: tuple[KeyPoint, ...]) -> JudgeVerdict:
    """Exactly one JSON object (no repeated key) with exactly the requested ids; scores 0/1/2 and 1-5;
    non-blank rationales."""
    try:
        verdict = strict_json_loads(output_text)
    except (TypeError, ValueError) as exc:
        raise JudgeSchemaError("judge output is not exactly one JSON object without repeated keys") from exc
    section = _exact(verdict, VERDICT_KEYS if key_points else OVERALL_ONLY_KEYS, "verdict")
    overall = _exact(section["overall"], OVERALL_KEYS, "overall")
    return JudgeVerdict(
        key_points=_criteria(section["key_points"], key_points) if key_points else (),
        overall_rationale=_rationale(overall["rationale"], "overall.rationale"),
        overall_score=_score(overall["score"], OVERALL_SCORES, "overall.score"),
    )
