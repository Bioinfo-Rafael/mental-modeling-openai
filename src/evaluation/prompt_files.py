"""Judge prompt files: every word a judge sends lives in a YAML file; the code only fills it.

A prompt file (``configs/evaluation/prompts/*.yaml``) holds:

- ``schema_version`` (1) and ``judge``: which judge reads it (``assess`` or ``key_point``);
- ``version``: recorded as ``prompt_version`` with every judgement;
- ``texts``: the judge's top-level templates (system prompt, user message or its blocks);
- ``blocks``: how each list or optional piece of material becomes text (item template, joiner,
  text when empty, ...).

Templates use ``str.format`` syntax with bare fields only (``{name}``: no format spec, conversion,
index or attribute, which could cut the material short; literal braces are doubled). Each judge declares
the texts and blocks it reads and the fields each template may use (``PromptSpec`` below); the
loader checks exact keys and every field when the judge config loads, so a typo fails before
anything is rendered or sent. Free texts (instructions, alarms, questions, intents, evidence, key
points) reach a template already quoted as one JSON string literal by the judge code.
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .strict_yaml import parse_strict_yaml

PROMPT_SCHEMA_VERSION = 1
FILE_KEYS = {"schema_version", "judge", "version", "texts", "blocks"}
JUDGE_ASSESS, JUDGE_KEY_POINT = "assess", "key_point"


class JudgePromptError(ValueError):
    """A judge prompt file is malformed, or a template names a field nobody provides."""


def placeholders(template: str) -> set[str]:
    """Names of the ``{name}`` fields of a template; every field must be a bare name."""
    try:
        fields = [(name, spec, conversion) for _, name, spec, conversion in string.Formatter().parse(template)]
    except ValueError as exc:
        raise JudgePromptError(f"malformed template: {exc}") from None
    names = set()
    for name, spec, conversion in fields:
        if name is None:
            continue
        if not name:
            raise JudgePromptError("a template has an empty {} field; name every value")
        if not name.isidentifier() or spec or conversion is not None:
            raise JudgePromptError(
                f"field {{{name}}} must be a bare name (no format spec, conversion, index or attribute)"
            )
        names.add(name)
    return names


def fill(template: str, values: Mapping[str, object]) -> str:
    missing = placeholders(template) - set(values)
    if missing:
        raise JudgePromptError(f"no value for {sorted(missing)}")
    return template.format_map(values)


@dataclass(frozen=True)
class TextSpec:
    fields: frozenset[str]  # what the template may use
    required: frozenset[str] = frozenset()  # what it must use (the material it may not drop)


@dataclass(frozen=True)
class PromptSpec:
    judge: str
    texts: Mapping[str, TextSpec]
    blocks: Mapping[str, Mapping[str, frozenset[str]]]  # block -> key -> fields its template may use
    open_blocks: frozenset[str] = field(default_factory=frozenset)  # any keys, plain text (no field)


def _fields(*names: str) -> frozenset[str]:
    return frozenset(names)


NO_FIELDS = frozenset()
LIST_BLOCK = ("joiner", "empty")

# --- assess judge (3 points x 4 aspects) ----------------------------------------------------------

ASSESS_USER_FIELDS = _fields(
    "target",
    "target_aircraft",
    "step",
    "time_s",
    "point_kinds",
    "trigger_reasons",
    "instructions",
    "alarms",
    "question",
    "previous_translation",
    "window_start",
    "window_end",
    "renderer",
    "trajectory",
    "intent",
    "evidence",
    "timeline",
    "future",
    "criteria",
    "not_rated",
    "scale_values",
    "scale_cannot_judge",
    "scale_not_applicable",
    "rating_rule",
)
ASSESS_MATERIAL = _fields(
    "instructions",
    "alarms",
    "question",
    "previous_translation",
    "trajectory",
    "intent",
    "evidence",
    "timeline",
    "future",
    "criteria",
    "not_rated",
)
ASSESS_PROMPT_SPEC = PromptSpec(
    judge=JUDGE_ASSESS,
    texts={
        "system": TextSpec(_fields("rating_rule", "rationale_language"), _fields("rating_rule")),
        "rating_rule": TextSpec(_fields("rationale_language")),
        "user": TextSpec(ASSESS_USER_FIELDS, ASSESS_MATERIAL),
    },
    blocks={
        "trigger_reason": {"item": _fields("kind", "point_kind", "source"), "joiner": NO_FIELDS},
        "instructions": {"item": _fields("issue_step", "target", "text"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
        "alarms": {"item": _fields("step", "text"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
        "question": {"present": _fields("step", "text"), "empty": NO_FIELDS},
        "previous_translation": {
            "present": _fields("step", "intent", "evidence"),
            "evidence_item": _fields("item"),
            "evidence_joiner": NO_FIELDS,
            "empty": NO_FIELDS,
        },
        "evidence": {"item": _fields("item"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
        "timeline": {
            "item": _fields("step", "time_s", "kinds", "point_kinds", "processing_s", "mark"),
            "joiner": NO_FIELDS,
            "this_event": NO_FIELDS,
        },
        "future": {
            "present": _fields("last_step", "end_step", "cut", "stride_steps", "fields", "rows", "later"),
            "row_joiner": NO_FIELDS,
            "no_rows": NO_FIELDS,
            "no_later": NO_FIELDS,
            "empty": NO_FIELDS,
        },
        "criterion": {
            "item": _fields("criterion_id", "notes", "text", "anchors", "boundaries"),
            "joiner": NO_FIELDS,
            "note_joiner": NO_FIELDS,
            "conditional": NO_FIELDS,
            "reference": _fields("reference"),
            "may_use_future": NO_FIELDS,
            "must_not_use_future": NO_FIELDS,
            "anchor": _fields("value", "anchor"),
            "cannot_judge": _fields("text"),
            "not_applicable": _fields("text"),
        },
        "not_rated": {
            "present": _fields("items"),
            "item": _fields("criterion_id", "reason"),
            "joiner": NO_FIELDS,
            "empty": NO_FIELDS,
        },
        "scale_value": {"item": _fields("value"), "joiner": NO_FIELDS},
    },
)

# --- key-point judge (D29) ------------------------------------------------------------------------

KEY_POINT_PROMPT_SPEC = PromptSpec(
    judge=JUDGE_KEY_POINT,
    texts={
        "system": TextSpec(NO_FIELDS),
        "header": TextSpec(_fields("target", "target_aircraft")),
        "instructions": TextSpec(_fields("instructions"), _fields("instructions")),
        "video": TextSpec(NO_FIELDS),
        "timestamps": TextSpec(
            _fields("episode_start_step", "episode_start_time", "earlier", "step", "time", "triggers"),
            _fields("earlier", "step", "triggers"),
        ),
        "trajectory": TextSpec(_fields("window", "legend", "trajectory"), _fields("legend", "trajectory")),
        "translated_intent": TextSpec(_fields("intent", "evidence"), _fields("intent", "evidence")),
        "key_points": TextSpec(_fields("key_points"), _fields("key_points")),
    },
    blocks={
        "instruction": {
            "item": _fields("primary", "details", "text"),
            "primary_marker": NO_FIELDS,
            "details": _fields("parts"),
            "id": _fields("instruction_id"),
            "target": _fields("target"),
            "issue_step": _fields("issue_step"),
            "part_joiner": NO_FIELDS,
            **dict.fromkeys(LIST_BLOCK, NO_FIELDS),
        },
        "time": {"present": _fields("time_s"), "empty": NO_FIELDS},
        "earlier": {"item": _fields("step", "time", "primary_trigger"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
        "trigger": {"item": _fields("reason"), "joiner": NO_FIELDS},
        "reason": {
            "plain": _fields("kind"),
            "detailed": _fields("kind", "details"),
            "source": _fields("source"),
            "probability": _fields("probability"),
            "detail_joiner": NO_FIELDS,
        },
        "window": {
            "present": _fields("start_step", "end_step", "anchor_step", "object_ids", "renderer", "renderer_version"),
            "empty": NO_FIELDS,
        },
        "evidence": {"item": _fields("item"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
        "key_point": {"item": _fields("index", "key_point_id", "text"), **dict.fromkeys(LIST_BLOCK, NO_FIELDS)},
    },
    open_blocks=_fields("legends"),  # trajectory renderer -> what its text means
)


# --- the file -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgePrompt:
    path: Path
    sha256: str  # of the prompt file's bytes
    judge: str
    version: str
    texts: Mapping[str, str]
    blocks: Mapping[str, Mapping[str, str]]

    def text(self, name: str, **values: object) -> str:
        return fill(self.texts[name], values)

    def word(self, block: str, key: str) -> str:
        """A block's plain text (a joiner, a marker, the text when empty)."""
        return self.blocks[block][key]

    def part(self, block: str, key: str, **values: object) -> str:
        return fill(self.blocks[block][key], values)

    def items(self, block: str, items: list[dict[str, object]], key: str = "item", joiner: str = "joiner") -> str:
        """Each item through ``key``, joined by ``joiner``; the block's ``empty`` when there is none."""
        words = self.blocks[block]
        if not items:
            return words["empty"]
        return words[joiner].join(fill(words[key], item) for item in items)

    def optional(self, block: str, item: dict[str, object] | None) -> str:
        """``present`` filled with the item, or ``empty`` when there is none."""
        return self.blocks[block]["empty"] if item is None else fill(self.blocks[block]["present"], item)


def _strings(raw: object, keys: set[str] | None, name: str) -> dict[str, str]:
    if not isinstance(raw, dict) or (keys is not None and set(raw) != keys):
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise JudgePromptError(f"{name} must have exactly {sorted(keys or ())} (found {found})")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in raw.items()):
        raise JudgePromptError(f"{name} must map names to strings")
    return dict(raw)


def _check(template: str, allowed: frozenset[str], required: frozenset[str], name: str) -> None:
    try:
        used = placeholders(template)
    except JudgePromptError as exc:
        raise JudgePromptError(f"{name}: {exc}") from None
    unknown, missing = used - allowed, required - used
    if unknown:
        raise JudgePromptError(f"{name} uses unknown fields {sorted(unknown)} (allowed: {sorted(allowed)})")
    if missing:
        raise JudgePromptError(f"{name} must use {sorted(missing)}: the judge would not see that material")


def _texts(raw: object, spec: PromptSpec, where: str) -> dict[str, str]:
    texts = _strings(raw, set(spec.texts), f"{where}: texts")
    for name, text_spec in spec.texts.items():
        _check(texts[name], text_spec.fields, text_spec.required, f"{where}: texts.{name}")
    return texts


def _blocks(raw: object, spec: PromptSpec, where: str) -> dict[str, dict[str, str]]:
    names = set(spec.blocks) | set(spec.open_blocks)
    if not isinstance(raw, dict) or set(raw) != names:
        raise JudgePromptError(f"{where}: blocks must have exactly {sorted(names)}")
    blocks = {}
    for name in sorted(names):
        keys = spec.blocks.get(name)
        block = _strings(raw[name], set(keys) if keys is not None else None, f"{where}: blocks.{name}")
        for key, template in block.items():
            allowed = keys[key] if keys is not None else NO_FIELDS
            _check(template, allowed, NO_FIELDS, f"{where}: blocks.{name}.{key}")
        blocks[name] = block
    return blocks


def load_judge_prompt(path: Path, spec: PromptSpec) -> JudgePrompt:
    """Read a prompt file strictly (UTF-8, no repeated key, exact keys, known fields only)."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise JudgePromptError(f"cannot read prompt file {path}: {type(exc).__name__}") from exc
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JudgePromptError(f"{path} is not valid UTF-8 YAML") from exc
    raw = parse_strict_yaml(text, path, JudgePromptError)
    if not isinstance(raw, dict) or set(raw) != FILE_KEYS:
        raise JudgePromptError(f"{path} must have exactly {sorted(FILE_KEYS)}")
    if raw["schema_version"] != PROMPT_SCHEMA_VERSION or isinstance(raw["schema_version"], bool):
        raise JudgePromptError(f"{path}: schema_version must be {PROMPT_SCHEMA_VERSION}")
    if raw["judge"] != spec.judge:
        raise JudgePromptError(f"{path} is a prompt for judge {raw['judge']!r}, not {spec.judge!r}")
    if not isinstance(raw["version"], str) or not raw["version"].strip():
        raise JudgePromptError(f"{path}: version must be a non-blank string")
    return JudgePrompt(
        path=Path(path),
        sha256=hashlib.sha256(data).hexdigest(),
        judge=spec.judge,
        version=raw["version"],
        texts=_texts(raw["texts"], spec, str(path)),
        blocks=_blocks(raw["blocks"], spec, str(path)),
    )
