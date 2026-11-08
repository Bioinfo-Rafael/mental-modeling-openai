"""Translator evaluation, 3 time points x 4 aspects: vocabulary, config, Translator run, annotation, criteria.

- The Translator event contract v2 constants are an own copy; ``test_assess_e2e`` checks them
  against ``translator.schema``.
- The judge config (``configs/evaluation/assess*.yaml``), its prompt file, the run files and the
  annotation load strictly: exact keys (``null`` = undecided), JSONL split on ``"\\n"`` only, and a
  repeated JSON or YAML key is an error.
- An event is rated on the union of the criteria of all its point kinds (``event_criteria``).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translate import paths
from translate.llm.openai_responses import REASONING_EFFORTS
from translate.events import canonical_json as canonical_json  # re-exported for the assess modules
from translate.events import sha256_text as sha256_text  # re-exported for the assess modules
from translate.events import strict_json_loads

from .configuration import JUDGE_MODEL, read_strict_yaml
from .prompt_files import ASSESS_PROMPT_SPEC, JudgePrompt, JudgePromptError, load_judge_prompt

# --- Translator event contract v2 (own copy; see the module docstring) ----------------------------

EVENT_SCHEMA_VERSION = 2
TRIGGER_INITIAL, TRIGGER_START, TRIGGER_CHANGE, TRIGGER_REQUEST = "initial", "start", "change", "request"
TRIGGER_PRECEDENCE = (TRIGGER_INITIAL, TRIGGER_START, TRIGGER_CHANGE, TRIGGER_REQUEST)
RECORD_DETECTOR_CALL, RECORD_TRANSLATION_EVENT = "detector_call", "translation_event"
RECORD_TARGET_CLOSED, RECORD_TRIGGER_SUPPRESSED = "target_closed", "trigger_suppressed"
RECORD_TYPES = (RECORD_DETECTOR_CALL, RECORD_TRANSLATION_EVENT, RECORD_TARGET_CLOSED, RECORD_TRIGGER_SUPPRESSED)
TARGET_KINDS = ("aircraft", "team")
LAYER_OF_RECORD = {RECORD_TRANSLATION_EVENT: "translation", RECORD_DETECTOR_CALL: "detector"}
STEP_DIGITS = 6
SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # used with fullmatch only

# --- evaluation vocabulary ------------------------------------------------------------------------

POINT_START, POINT_CHANGE, POINT_REQUEST = "start", "change", "request"
POINT_KINDS = (POINT_START, POINT_CHANGE, POINT_REQUEST)
# 開始点 = the forced "initial" bootstrap or a detector start; 変化点 = detector change; 要求点 = request.
POINT_KIND_OF_TRIGGER = {
    TRIGGER_INITIAL: POINT_START,
    TRIGGER_START: POINT_START,
    TRIGGER_CHANGE: POINT_CHANGE,
    TRIGGER_REQUEST: POINT_REQUEST,
}
ASPECT_ELEMENTS, ASPECT_CORRECTNESS, ASPECT_TIMING = "elements", "correctness", "timing"
ASPECTS = (ASPECT_ELEMENTS, ASPECT_CORRECTNESS, ASPECT_TIMING, "contribution")
ASPECT_OVERALL = "overall"
REFERENCE_FUTURE = "future_behaviour"
REFERENCES = ("observable_records", REFERENCE_FUTURE)
EVALUATOR_MACHINE, EVALUATOR_HUMAN, EVALUATOR_JUDGE = "machine", "human", "judge"
STATUS_EVALUATED, STATUS_NOT_EVALUATED = "evaluated", "not_evaluated"
JUDGEMENT_CANNOT_JUDGE, JUDGEMENT_NOT_APPLICABLE = "cannot_judge", "not_applicable"
REASON_AGGREGATION_UNSET = "aggregation_unset"  # D54
REASON_FORCED_TRIGGER = "forced_trigger"  # the start timing of a forced "initial" translation
REASON_INPUT_INVALID = "generation_input_invalid"  # a leak or error on any call of the target
RESULT_SCHEMA_VERSION = 1
RUN_FILES = ("run.json", "events.jsonl", "requests.jsonl", "responses.jsonl")
RESULT_FILES = {
    EVALUATOR_MACHINE: "machine_results.jsonl",
    EVALUATOR_HUMAN: "human_results.jsonl",
    EVALUATOR_JUDGE: "judge_results.jsonl",
}
DEFAULT_CONFIG = paths.ROOT / "configs" / "evaluation" / "assess.yaml"


class AssessError(ValueError):
    """An input of the evaluation is malformed; the message names the field."""


class ConfigError(AssessError):
    """``assess.yaml`` is malformed."""


class RunFormatError(AssessError):
    """The Translator run directory violates the v2 event contract."""


class AnnotationError(AssessError):
    """The reference annotation is malformed or does not match the run."""


class RatingsError(AssessError):
    """The human-ratings file is malformed or does not match the run or the rubric."""


# --- files and value checks -----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path, error: type[AssessError]) -> list[dict[str, Any]]:
    """Every line one JSON object, split on ``"\\n"`` only (U+2028 and friends are legal in strings)."""
    try:
        lines = Path(path).read_bytes().decode("utf-8").split("\n")
    except (OSError, UnicodeDecodeError) as exc:
        raise error(f"cannot read {Path(path).name}: {type(exc).__name__}") from exc
    records = []
    for index, line in enumerate(lines if lines[-1] else lines[:-1]):
        try:
            record = strict_json_loads(line)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            raise error(f"{Path(path).name} line {index + 1} must be one JSON object (no repeated key)")
        records.append(record)
    return records


Check = Callable[[Any], bool]


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_count(value: object) -> bool:
    return is_int(value) and value >= 0


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_non_negative(value: object) -> bool:
    return is_number(value) and value >= 0


def is_str(value: object) -> bool:
    return isinstance(value, str)


def is_bool(value: object) -> bool:
    return isinstance(value, bool)


def is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_dict(value: object) -> bool:
    return isinstance(value, dict)


def is_safe_id(value: object) -> bool:
    return isinstance(value, str) and SAFE_ID_PATTERN.fullmatch(value) is not None


def is_call_id(value: object) -> bool:
    """Event and detector-call ids: safe ids joined by ``:``."""
    return isinstance(value, str) and all(SAFE_ID_PATTERN.fullmatch(part) for part in value.split(":"))


def is_target(value: object) -> bool:
    return is_dict(value) and set(value) == {"kind", "id"} and value["kind"] in TARGET_KINDS and is_safe_id(value["id"])


def one_of(*allowed: object) -> Check:
    return lambda value: isinstance(value, str) and value in allowed


def list_of(check: Check, *, non_empty: bool = False) -> Check:
    return lambda value: isinstance(value, list) and (bool(value) or not non_empty) and all(map(check, value))


def nullable(check: Check) -> Check:
    return lambda value: value is None or check(value)


def shaped(shape: dict[str, Check]) -> Check:
    """A mapping holding at least these keys, each passing its check."""
    return lambda value: is_dict(value) and all(key in value and check(value[key]) for key, check in shape.items())


def require(mapping: object, shape: dict[str, Check], where: str, error: type[AssessError]) -> None:
    for key, check in shape.items():
        if not is_dict(mapping) or key not in mapping or not check(mapping[key]):
            raise error(f"{where}.{key} is missing or malformed")


def exact_keys(raw: object, keys: set[str], name: str, error: type[AssessError] = ConfigError) -> dict[str, Any]:
    if not is_dict(raw) or not all(isinstance(key, str) for key in raw):
        raise error(f"{name} must be a mapping with string keys")
    if set(raw) != keys:
        missing, unknown = sorted(keys - set(raw)), sorted(set(raw) - keys)
        raise error(f"{name} must have exactly {sorted(keys)} (missing {missing}, unknown {unknown})")
    return raw


def target_key(target: dict[str, Any]) -> tuple[str, str]:
    return (target["kind"], target["id"])


# --- config ---------------------------------------------------------------------------------------

CONFIG_SCHEMA_VERSION = 3  # v2: per-criterion level anchors (D65); v3: the prompt file
CONFIG_KEYS = {"schema_version", "rubric_version", "timing", "future_evidence", "leakage", "judge"}
CONFIG_KEYS |= {"open_items", "scale", "criteria", "output", "prompt"}
POSITIVE = (lambda value: is_number(value) and value > 0, "a number > 0")
NON_NEGATIVE = (is_non_negative, "a number >= 0")
INT_GE_0, INT_GE_1 = (is_count, "an integer >= 0"), (lambda value: is_int(value) and value >= 1, "an integer >= 1")
# Spec tables: key -> (check, wording). Every timing value may be null (undecided).
TIMING_SPEC = {
    "request_deadline_s": POSITIVE,
    "min_question_interval_s": POSITIVE,
    "start_min_lead_steps": INT_GE_0,
    "change_max_delay_steps": INT_GE_0,
    "max_events_per_segment": INT_GE_1,
    "max_utterances_per_target": INT_GE_1,
}
FUTURE_SPEC = {
    "fields": (lambda v: list_of(is_text, non_empty=True)(v) and len(set(v)) == len(v), "a list of unique columns"),
    "stride_steps": INT_GE_1,
    "max_steps": INT_GE_1,
    "float_precision": INT_GE_0,
}
PRICE_SPEC = {"input": NON_NEGATIVE, "cached_input": NON_NEGATIVE, "output": NON_NEGATIVE}
JUDGE_SPEC = {
    "model": (one_of(JUDGE_MODEL), f"exactly {JUDGE_MODEL!r} (D8)"),
    "reasoning_effort": (one_of(*REASONING_EFFORTS), f"one of {sorted(REASONING_EFFORTS)}"),
    "max_output_tokens": INT_GE_1,
    "timeout_s": POSITIVE,
    "max_calls": INT_GE_1,
    "rationale_language": (is_text, "a non-blank string"),
    "pricing": (is_dict, "a mapping"),
}
OPEN_ITEM_KEYS = {"tactic_vocabulary", "annotated_units", "expected_raters", "overall_aggregation"}
OPEN_ITEM_KEYS |= {"judge_substitution_agreement"}
CRITERIA_SECTIONS = (ASPECT_ELEMENTS, *POINT_KINDS)
CRITERION_OPTIONAL = {"conditional", "reference", "cannot_judge", "not_applicable"}
CRITERION_KEYS = {"id", "text", "levels", *CRITERION_OPTIONAL}
CRITERION_ID_PATTERN = re.compile(rf"(?:{'|'.join(ASPECTS)})\.[a-z0-9_]{{1,64}}")  # fullmatch only


@dataclass(frozen=True)
class TimingConfig:
    """Machine timing thresholds; ``None`` = undecided (raw measurement, ``<key>_unset``)."""

    request_deadline_s: float | None
    min_question_interval_s: float | None
    start_min_lead_steps: int | None
    change_max_delay_steps: int | None
    max_events_per_segment: int | None
    max_utterances_per_target: int | None


@dataclass(frozen=True)
class FutureEvidenceConfig:
    fields: tuple[str, ...]
    stride_steps: int
    max_steps: int
    float_precision: int


@dataclass(frozen=True)
class JudgeSettings:
    model: str
    reasoning_effort: str
    max_output_tokens: int
    timeout_s: float
    max_calls: int
    rationale_language: str
    pricing: dict[str, float]  # USD per 1M tokens (input, cached_input, output); preview estimate only


@dataclass(frozen=True)
class Criterion:
    criterion_id: str
    aspect: str  # the id prefix
    text: str
    reference: str | None  # correctness only: observable_records | future_behaviour
    conditional: bool  # "not_applicable" is accepted (the scene lacks the element, or no PREVIOUS)
    levels: dict[str, str]  # every scale value -> its anchor (D65), in scale order
    cannot_judge: str | None  # criterion-specific boundary; None => only the scale's general wording
    not_applicable: str | None  # set exactly when conditional


@dataclass(frozen=True)
class ScaleConfig:
    values: tuple[str, ...]  # ordered, e.g. "1".."5"
    cannot_judge: str
    not_applicable: str


@dataclass(frozen=True)
class AssessConfig:
    path: Path
    sha256: str  # of the config file bytes
    rubric_version: str
    timing: TimingConfig
    future_evidence: FutureEvidenceConfig
    allow_action_at_t: bool
    judge: JudgeSettings
    open_items: dict[str, None]
    scale: ScaleConfig
    criteria: dict[str, tuple[Criterion, ...]]  # elements / start / change / request -> criteria
    output_root: Path
    prompt: JudgePrompt  # every word of the judge request


def _section(raw: object, spec: dict[str, tuple[Check, str]], name: str, *, allow_null: bool = False) -> dict[str, Any]:
    section = exact_keys(raw, set(spec), name)
    for key, (check, wording) in spec.items():
        if not ((allow_null and section[key] is None) or check(section[key])):
            raise ConfigError(f"{name}.{key} must be {'null or ' if allow_null else ''}{wording}")
    return section


def _scale(raw: object) -> ScaleConfig:
    section = exact_keys(raw, {"values", "cannot_judge", "not_applicable"}, "scale")
    values = section["values"]
    if not list_of(lambda v: isinstance(v, str) and v.isdigit())(values) or len(values) < 2:
        raise ConfigError('scale.values must list at least two integers as strings, e.g. ["1", "2"]')
    if [int(value) for value in values] != sorted({int(value) for value in values}):
        raise ConfigError("scale.values must be strictly increasing")
    if not is_text(section["cannot_judge"]) or not is_text(section["not_applicable"]):
        raise ConfigError("scale.cannot_judge and scale.not_applicable must be non-blank")
    return ScaleConfig(tuple(values), section["cannot_judge"], section["not_applicable"])


def _levels(raw: object, values: tuple[str, ...], name: str) -> dict[str, str]:
    """D65: exactly one non-blank anchor per scale value."""
    levels = exact_keys(raw, set(values), f"{name}.levels")
    if not all(map(is_text, levels.values())):
        raise ConfigError(f"{name}.levels anchors must be non-blank")
    return {value: levels[value] for value in values}


def _criterion(raw: object, section: str, name: str, values: tuple[str, ...]) -> Criterion:
    if not is_dict(raw) or not {"id", "text", "levels"} <= set(raw) <= CRITERION_KEYS:
        raise ConfigError(f"{name} must have id, text and levels, and optionally {sorted(CRITERION_OPTIONAL)}")
    criterion_id, conditional, reference = raw["id"], raw.get("conditional", False), raw.get("reference")
    if not isinstance(criterion_id, str) or not CRITERION_ID_PATTERN.fullmatch(criterion_id):
        raise ConfigError(f"{name}.id must be <aspect>.<name> with aspect one of {list(ASPECTS)}")
    aspect = criterion_id.split(".", 1)[0]
    if (section == ASPECT_ELEMENTS) != (aspect == ASPECT_ELEMENTS):
        raise ConfigError(f"{name}: elements.* criteria belong in criteria.elements, and only there")
    if not is_bool(conditional):
        raise ConfigError(f"{name}.conditional must be a boolean")
    bad_reference = reference is not None and reference not in REFERENCES
    if (aspect == ASPECT_CORRECTNESS) != ("reference" in raw) or bad_reference:
        raise ConfigError(f"{name}.reference ({list(REFERENCES)}) is required for correctness criteria, and only there")
    if not is_text(raw["text"]):
        raise ConfigError(f"{name}.text must be non-blank")
    cannot_judge, not_applicable = raw.get("cannot_judge"), raw.get("not_applicable")
    if "cannot_judge" in raw and not is_text(cannot_judge):
        raise ConfigError(f"{name}.cannot_judge must be non-blank when given")
    if conditional != ("not_applicable" in raw) or ("not_applicable" in raw and not is_text(not_applicable)):
        raise ConfigError(f"{name}.not_applicable (non-blank) is required for conditional criteria, and only there")
    levels = _levels(raw["levels"], values, name)
    return Criterion(criterion_id, aspect, raw["text"], reference, conditional, levels, cannot_judge, not_applicable)


def _criteria(raw: object, values: tuple[str, ...]) -> dict[str, tuple[Criterion, ...]]:
    section = exact_keys(raw, set(CRITERIA_SECTIONS), "criteria")
    parsed: dict[str, tuple[Criterion, ...]] = {}
    defined: dict[str, Criterion] = {}  # an id listed in several sections must be defined the same way
    for name in CRITERIA_SECTIONS:
        if not isinstance(section[name], list) or not section[name]:
            raise ConfigError(f"criteria.{name} must be a non-empty list")
        items = tuple(_criterion(item, name, f"criteria.{name}[{i}]", values) for i, item in enumerate(section[name]))
        if len({item.criterion_id for item in items}) != len(items):
            raise ConfigError(f"criteria.{name} repeats a criterion id")
        parsed[name] = items
    for criterion in (criterion for items in parsed.values() for criterion in items):
        if defined.setdefault(criterion.criterion_id, criterion) != criterion:
            raise ConfigError(f"{criterion.criterion_id} is defined differently in two sections")
    return parsed


def _prompt(raw: object, root: Path) -> JudgePrompt:
    if not is_text(raw):
        raise ConfigError("prompt must be the path of a prompt file (configs/evaluation/prompts/)")
    path = Path(raw) if Path(raw).is_absolute() else root / raw
    try:
        return load_judge_prompt(path, ASSESS_PROMPT_SPEC)
    except JudgePromptError as exc:
        raise ConfigError(f"prompt: {exc}") from exc


def parse_config(raw: object, *, path: Path, sha256: str, root: Path = paths.ROOT) -> AssessConfig:
    section = exact_keys(raw, CONFIG_KEYS, "config")
    if not is_int(section["schema_version"]) or section["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
    if not is_safe_id(section["rubric_version"]):
        raise ConfigError(f"rubric_version must match {SAFE_ID_PATTERN.pattern}")
    leakage = _section(section["leakage"], {"allow_action_at_t": (is_bool, "true or false")}, "leakage")
    output = _section(section["output"], {"root": (is_text, "a non-empty path")}, "output")
    open_items = exact_keys(section["open_items"], OPEN_ITEM_KEYS, "open_items")
    for key, value in open_items.items():
        if value is not None:
            raise ConfigError(f"open_items.{key} must stay null: no check uses a value yet (D58); decide it first")
    future = _section(section["future_evidence"], FUTURE_SPEC, "future_evidence")
    judge = _section(section["judge"], JUDGE_SPEC, "judge")
    pricing = _section(judge["pricing"], PRICE_SPEC, "judge.pricing")
    output_root = Path(output["root"])
    scale = _scale(section["scale"])
    return AssessConfig(
        path=Path(path),
        sha256=sha256,
        rubric_version=section["rubric_version"],
        timing=TimingConfig(**_section(section["timing"], TIMING_SPEC, "timing", allow_null=True)),
        future_evidence=FutureEvidenceConfig(**{**future, "fields": tuple(future["fields"])}),
        allow_action_at_t=leakage["allow_action_at_t"],
        judge=JudgeSettings(**{**judge, "timeout_s": float(judge["timeout_s"]), "pricing": dict(pricing)}),
        open_items=dict.fromkeys(sorted(open_items)),
        scale=scale,
        criteria=_criteria(section["criteria"], scale.values),
        output_root=output_root if output_root.is_absolute() else root / output_root,
        prompt=_prompt(section["prompt"], root),
    )


def load_config(path: Path = DEFAULT_CONFIG) -> AssessConfig:
    return parse_config(read_strict_yaml(path, ConfigError), path=Path(path), sha256=sha256_file(path))


# --- Translator run v2 ----------------------------------------------------------------------------

INSTRUCTION = shaped({"instruction_id": is_safe_id, "issue_step": is_count, "target": is_target, "text": is_str})
ALARM = shaped({"alarm_id": is_safe_id, "step": is_count, "text": is_str})
QUESTION = shaped({"request_id": is_safe_id, "step": is_count, "text": is_str})
MESSAGE = shaped({"kind": one_of("alarm", "question"), "id": is_safe_id, "step": is_count, "text": is_str})
PREVIOUS = shaped({"event_id": is_str, "step": is_count})
TRAJECTORY = shaped(
    {"text": is_str, "sha256": is_str, "renderer": is_str, "start_step": is_count, "end_step": is_count}
)
REASON = shaped(
    {"kind": one_of(*TRIGGER_PRECEDENCE), "source": is_str, "reference_id": is_str, "probability": nullable(is_number)}
)
COMMON_SHAPE = {
    "schema_version": lambda v: is_int(v) and v == EVENT_SCHEMA_VERSION,
    "record_type": one_of(*RECORD_TYPES),
    "scenario_id": is_safe_id,
    "episode_id": is_safe_id,
    "step": is_count,
    "target": is_target,
}
EVENT_INPUT_SHAPE = {
    "target_aircraft": list_of(is_safe_id, non_empty=True),
    "instructions_in_effect": list_of(INSTRUCTION),
    "alarms": list_of(ALARM),
    "question": nullable(QUESTION),
    "previous_translation": nullable(PREVIOUS),
    "trajectory": TRAJECTORY,
}
EVENT_SHAPE = {
    "event_id": is_call_id,
    "time_s": is_number,
    "trigger_reasons": list_of(REASON, non_empty=True),
    "request_sha256": is_str,
    "processing_s": is_non_negative,
    "input": shaped(EVENT_INPUT_SHAPE),
    # Blank intents and empty evidence load: detecting them is the machine element check's job.
    "output": shaped({"intent": is_str, "evidence": list_of(is_str)}),
}
DETECTOR_INPUT_SHAPE = {
    "trajectory": TRAJECTORY,
    "instructions_in_effect": list_of(INSTRUCTION),
    "messages": list_of(MESSAGE),
    "previous_translation": PREVIOUS,
}
RECORD_SHAPES = {
    RECORD_TRANSLATION_EVENT: EVENT_SHAPE,
    RECORD_DETECTOR_CALL: {"call_id": is_call_id, "request_sha256": is_str, "input": shaped(DETECTOR_INPUT_SHAPE)},
    RECORD_TARGET_CLOSED: {"down_step": is_count},
    RECORD_TRIGGER_SUPPRESSED: {},
}
REQUEST_SHAPE = {
    "call_id": is_call_id,
    "layer": one_of(*LAYER_OF_RECORD.values()),
    "step": is_count,
    "target": is_target,
    "payload": is_dict,
}


@dataclass(frozen=True)
class TranslatorRun:
    run_dir: Path
    run_json: dict[str, Any]
    file_sha256: dict[str, str]  # file name -> SHA-256 of its bytes
    records: tuple[dict[str, Any], ...]  # events.jsonl, in sequence order
    events: tuple[dict[str, Any], ...]  # the translation_event records
    requests: tuple[dict[str, Any], ...]  # requests.jsonl, in file order
    latency_s: dict[str, float | None]  # call_id -> latency_s of its response line (None if absent)
    scenario_id: str
    episode_id: str
    last_step: int  # last complete step of the episode
    future_steps: int | None  # declared visible future: 0 online, F retrospective, None undeclared
    retrospective: bool

    @property
    def status(self) -> object:
        return self.run_json.get("status")


def dig(mapping: object, *keys: str) -> object:
    for key in keys:
        mapping = mapping.get(key) if is_dict(mapping) else None
    return mapping


def declared_future_steps(run_json: dict[str, Any]) -> tuple[int | None, bool]:
    """``(future_steps, retrospective)`` from ``config.pipeline.visibility``; ``(None, False)`` if undeclared.

    Online must declare ``future_steps: 0``; any other declared mode is a retrospective run (D46).
    """
    mode, future_steps = (dig(run_json, "config", "pipeline", "visibility", key) for key in ("mode", "future_steps"))
    if not isinstance(mode, str) or not is_count(future_steps):
        return None, False
    if mode == "online":
        return (0, False) if future_steps == 0 else (None, False)
    return future_steps, True


def _records(run_dir: Path) -> tuple[dict[str, Any], ...]:
    records = read_jsonl(run_dir / "events.jsonl", RunFormatError)
    for index, record in enumerate(records):
        where = f"events.jsonl line {index + 1}"
        require(record, COMMON_SHAPE, where, RunFormatError)
        if record.get("sequence") != index + 1 or not is_int(record.get("sequence")):
            raise RunFormatError(f"{where}: sequence must be {index + 1} (1..n without gaps)")
        require(record, RECORD_SHAPES[record["record_type"]], where, RunFormatError)
        target = record["target"]
        expected = f"{record['scenario_id']}:{target['kind']}:{target['id']}:{record['step']:0{STEP_DIGITS}d}"
        if record["record_type"] == RECORD_TRANSLATION_EVENT and record["event_id"] != expected:
            raise RunFormatError(f"{where}.event_id must be {expected!r}")
    return tuple(records)


def _checked_lines(path: Path, shape: dict[str, Check]) -> list[dict[str, Any]]:
    lines = read_jsonl(path, RunFormatError)
    for index, line in enumerate(lines):
        require(line, shape, f"{path.name} line {index + 1}", RunFormatError)
    return lines


def load_run(run_dir: Path) -> TranslatorRun:
    """Read and check a Translator run directory (event contract v2). Nothing is written."""
    run_dir = Path(run_dir)
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RunFormatError(f"{run_dir} has no {', '.join(missing)} (a preview run has nothing to evaluate)")
    try:
        run_json = strict_json_loads((run_dir / "run.json").read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RunFormatError("run.json is not valid UTF-8 JSON (or repeats a key)") from exc
    records = _records(run_dir)
    requests = _checked_lines(run_dir / "requests.jsonl", REQUEST_SHAPE)
    responses = _checked_lines(run_dir / "responses.jsonl", {"call_id": is_call_id})
    scenario_ids, episode_ids = {r["scenario_id"] for r in records}, {r["episode_id"] for r in records}
    if not is_dict(run_json) or len(scenario_ids) != 1 or len(episode_ids) != 1:
        raise RunFormatError("run.json must be an object, and events.jsonl must hold exactly one scenario and episode")
    last_step = dig(run_json, "inputs", "episode", "last_complete_step")
    if not is_count(last_step):
        raise RunFormatError("run.json inputs.episode.last_complete_step must be a non-negative integer")
    future_steps, retrospective = declared_future_steps(run_json)
    latency = {response["call_id"]: response.get("latency_s") for response in responses}
    return TranslatorRun(
        run_dir=run_dir,
        run_json=run_json,
        file_sha256={name: sha256_file(run_dir / name) for name in RUN_FILES},
        records=records,
        events=tuple(r for r in records if r["record_type"] == RECORD_TRANSLATION_EVENT),
        requests=tuple(requests),
        latency_s={call_id: float(value) if is_non_negative(value) else None for call_id, value in latency.items()},
        scenario_id=scenario_ids.pop(),
        episode_id=episode_ids.pop(),
        last_step=last_step,
        future_steps=future_steps,
        retrospective=retrospective,
    )


def target_end_step(run: TranslatorRun, key: tuple[str, str]) -> int:
    """Exclusive end of a target's evaluable episode: its down step, or ``last_step + 1``."""
    closed = (r for r in run.records if r["record_type"] == RECORD_TARGET_CLOSED and target_key(r["target"]) == key)
    return min((r["down_step"] for r in closed), default=run.last_step + 1)


# --- reference annotation (optional; start/change points, may use hindsight) ----------------------

ANNOTATION_SCHEMA_VERSION = 1
POINT_SHAPE = {"point_id": is_safe_id, "kind": one_of(POINT_START, POINT_CHANGE), "target": is_target, "step": is_count}


@dataclass(frozen=True)
class ReferencePoint:
    point_id: str
    kind: str  # start | change
    target: tuple[str, str]
    step: int
    action_onset_step: int | None  # start points: the annotated onset of the action; null = unknown


@dataclass(frozen=True)
class Annotation:
    path: Path
    sha256: str
    scenario_id: str
    annotator: str
    points: tuple[ReferencePoint, ...]


def _point(raw: object, name: str) -> ReferencePoint:
    section = exact_keys(raw, {*POINT_SHAPE, "action_onset_step"}, name, AnnotationError)
    require(section, POINT_SHAPE, name, AnnotationError)
    onset = section["action_onset_step"]
    if onset is not None and (not is_int(onset) or onset < section["step"]):
        raise AnnotationError(f"{name}.action_onset_step must be null or an integer >= step")
    return ReferencePoint(section["point_id"], section["kind"], target_key(section["target"]), section["step"], onset)


def load_annotation(path: Path, run: TranslatorRun) -> Annotation:
    """Strict annotation, checked against the run: scenario id, known targets, steps in the episode."""
    keys = {"schema_version", "scenario_id", "annotator", "points"}
    section = exact_keys(read_strict_yaml(path, AnnotationError), keys, "annotation", AnnotationError)
    if not is_int(section["schema_version"]) or section["schema_version"] != ANNOTATION_SCHEMA_VERSION:
        raise AnnotationError(f"annotation schema_version must be {ANNOTATION_SCHEMA_VERSION}")
    if section["scenario_id"] != run.scenario_id or not is_safe_id(section["annotator"]):
        raise AnnotationError("scenario_id must match the run, and annotator must be a pseudonymous safe id")
    if not isinstance(section["points"], list):
        raise AnnotationError("points must be a list (use [] for none)")
    points = tuple(_point(item, f"points[{index}]") for index, item in enumerate(section["points"]))
    if len({p.point_id for p in points}) != len(points) or len({(p.target, p.step) for p in points}) != len(points):
        raise AnnotationError("point_id values must be unique, with at most one point per target and step")
    targets = {target_key(record["target"]) for record in run.records}
    for point in points:
        if point.target not in targets or point.step > run.last_step:
            raise AnnotationError(f"point {point.point_id}: unknown target, or a step after {run.last_step}")
    return Annotation(Path(path), sha256_file(path), section["scenario_id"], section["annotator"], points)


# --- points of an event and the criteria it is rated on -------------------------------------------


def point_kinds(event: dict[str, Any]) -> tuple[str, ...]:
    """Point kinds of every trigger reason of the event, in ``POINT_KINDS`` order."""
    kinds = {POINT_KIND_OF_TRIGGER[reason["kind"]] for reason in event["trigger_reasons"]}
    return tuple(kind for kind in POINT_KINDS if kind in kinds)


@dataclass(frozen=True)
class EventCriteria:
    asked: tuple[Criterion, ...]  # rated by humans and the judge, each once
    excluded: tuple[tuple[Criterion, str], ...]  # (criterion, not_evaluated reason) for every evaluator
    kinds_of: dict[str, tuple[str, ...]]  # criterion id -> the event's point kinds it is counted under


def event_criteria(event: dict[str, Any], config: AssessConfig) -> EventCriteria:
    """Union of ``criteria.elements`` and the criteria of each point kind; a shared id is asked once.

    The forced ``initial`` translation is rated with the start rules, but its start timing is
    ``not_evaluated/forced_trigger``: every ``timing.*`` criterion that only the start kind brings.
    """
    kinds = point_kinds(event)
    by_id = {criterion.criterion_id: criterion for criterion in config.criteria[ASPECT_ELEMENTS]}
    kinds_of = dict.fromkeys(by_id, kinds)
    for kind in kinds:
        for criterion in config.criteria[kind]:
            by_id.setdefault(criterion.criterion_id, criterion)
            kinds_of[criterion.criterion_id] = (*kinds_of.get(criterion.criterion_id, ()), kind)
    forced = any(reason["kind"] == TRIGGER_INITIAL for reason in event["trigger_reasons"])
    excluded_ids = {
        criterion_id
        for criterion_id, criterion in by_id.items()
        if forced and criterion.aspect == ASPECT_TIMING and kinds_of[criterion_id] == (POINT_START,)
    }
    return EventCriteria(
        asked=tuple(c for c in by_id.values() if c.criterion_id not in excluded_ids),
        excluded=tuple((c, REASON_FORCED_TRIGGER) for c in by_id.values() if c.criterion_id in excluded_ids),
        kinds_of=kinds_of,
    )
