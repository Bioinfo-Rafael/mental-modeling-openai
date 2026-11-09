"""Prompt files: the translation prompt text lives in config, the code only fills it.

A prompt file (``configs/translate/prompts/*.yaml``) holds:

- ``version``: recorded as ``prompt_version`` in every event;
- ``protocol``: ``one_shot`` (one strict-schema Responses call);
- ``output_schema``: the named answer schema (``intent_evidence``);
- ``system`` and ``user``: templates with ``{name}``
  placeholders, filled from the values of the selected inputs and the run config's ``prompt.vars``
  (``str.format`` syntax: ``{t_s:g}`` formats a number, literal braces are doubled);
- ``blocks``: how list and optional inputs become text (item template, joiner, text when empty;
  ``primary_instruction`` words the target's own instruction, or its absence);
- ``turns``: ``null`` (the former code session's follow-up messages).

Every placeholder is checked when the run config is loaded, so a typo fails before anything is
written or sent.
"""

from __future__ import annotations

import hashlib
import string
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from translate.strict_yaml import parse_yaml

PROMPT_SCHEMA_VERSION = 1
PROTOCOL_ONE_SHOT = "one_shot"
PROTOCOLS = (PROTOCOL_ONE_SHOT,)
OUTPUT_SCHEMAS = ("intent_evidence",)
PROMPT_KEYS = {"schema_version", "version", "protocol", "output_schema", "system", "user", "blocks", "turns"}
# The fields each block template may use, per block name; the block's keys are fixed too.
BLOCK_KEYS = {
    "instructions": {"item", "primary_marker", "joiner", "empty"},
    "primary_instruction": {"present", "empty"},
    "alarms": {"item", "joiner", "empty"},
    "question": {"present", "empty"},
    "previous_translation": {"present", "evidence_item", "evidence_joiner", "empty"},
}
BLOCK_FIELDS = {
    "instructions": {"item": {"issue_step", "target", "primary", "text"}},
    "primary_instruction": {"present": {"text"}},
    "alarms": {"item": {"step", "targets", "text"}},
    "question": {"present": {"step", "text"}},
    "previous_translation": {"present": {"step", "intent", "evidence"}, "evidence_item": {"item"}},
}


class PromptError(ValueError):
    """A prompt file is malformed, or a template names a value nobody provides."""


def placeholders(template: str) -> set[str]:
    """Base names of the ``{name}`` fields of a ``str.format`` template."""
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]
    except ValueError as exc:
        raise PromptError(f"malformed template: {exc}") from None
    if any(not field for field in fields):
        raise PromptError("a template has an empty {} field; name every value")
    return {field.split(".")[0].split("[")[0] for field in fields}


def fill(template: str, values: Mapping[str, object]) -> str:
    missing = placeholders(template) - set(values)
    if missing:
        raise PromptError(f"no value for {sorted(missing)}")
    return template.format_map(values)


def _mapping(raw: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        found = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise PromptError(f"{name} must have exactly {sorted(keys)} (found {found})")
    if not all(isinstance(value, str) for value in raw.values()):
        raise PromptError(f"{name} values must be strings")
    return raw


def _blocks(raw: object) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict) or not set(raw) <= set(BLOCK_KEYS):
        raise PromptError(f"blocks may only define {sorted(BLOCK_KEYS)}")
    blocks = {name: _mapping(section, BLOCK_KEYS[name], f"blocks.{name}") for name, section in raw.items()}
    for name, block in blocks.items():
        for key, allowed in BLOCK_FIELDS[name].items():
            unknown = placeholders(block[key]) - allowed
            if unknown:
                raise PromptError(f"blocks.{name}.{key} uses unknown fields {sorted(unknown)}")
    return blocks


@dataclass(frozen=True)
class PromptTemplate:
    path: Path
    sha256: str  # of the prompt file's bytes
    version: str
    protocol: str
    output_schema: str | None
    system: str | None
    user: str
    blocks: Mapping[str, Mapping[str, str]]
    vars: Mapping[str, str]  # from the run config

    def placeholders(self) -> set[str]:
        """Every value name the system and user templates need."""
        return placeholders(self.user) | (placeholders(self.system) if self.system is not None else set())

    def render(self, values: Mapping[str, object]) -> tuple[str | None, str]:
        """(system, user) for one slot; ``vars`` fill what the inputs do not."""
        merged = {**self.vars, **values}
        system = fill(self.system, merged) if self.system is not None else None
        return system, fill(self.user, merged)


def _check_protocol(raw: dict[str, Any]) -> None:
    protocol = raw["protocol"]
    if protocol not in PROTOCOLS:
        raise PromptError(f"protocol must be one of {list(PROTOCOLS)}")
    if protocol == PROTOCOL_ONE_SHOT:
        if raw["output_schema"] not in OUTPUT_SCHEMAS or not isinstance(raw["system"], str) or raw["turns"] is not None:
            raise PromptError(
                f"a one_shot prompt needs output_schema {list(OUTPUT_SCHEMAS)}, a system text, turns: null"
            )


def load_prompt(path: Path, prompt_vars: Mapping[str, str]) -> PromptTemplate:
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptError(f"{path} is not valid UTF-8 YAML") from exc
    raw = parse_yaml(text, path, PromptError)  # duplicate keys are rejected, as in run configs
    if not isinstance(raw, dict) or set(raw) != PROMPT_KEYS:
        raise PromptError(f"{path} must have exactly {sorted(PROMPT_KEYS)}")
    if raw["schema_version"] != PROMPT_SCHEMA_VERSION:
        raise PromptError(f"{path}: schema_version must be {PROMPT_SCHEMA_VERSION}")
    if not isinstance(raw["version"], str) or not raw["version"] or not isinstance(raw["user"], str):
        raise PromptError(f"{path}: version and user must be strings")
    _check_protocol(raw)
    return PromptTemplate(
        path=Path(path),
        sha256=hashlib.sha256(data).hexdigest(),
        version=raw["version"],
        protocol=raw["protocol"],
        output_schema=raw["output_schema"],
        system=raw["system"],
        user=raw["user"],
        blocks=_blocks(raw["blocks"]),
        vars=dict(prompt_vars),
    )


# --- blocks ---------------------------------------------------------------------------------------


def list_block(block: Mapping[str, str], items: list[dict[str, object]]) -> str:
    """Each item through ``item``, joined by ``joiner``; ``empty`` when there is none."""
    if not items:
        return block["empty"]
    return block["joiner"].join(fill(block["item"], item) for item in items)


def optional_block(block: Mapping[str, str], item: dict[str, object] | None) -> str:
    """``present`` filled with the item, or ``empty`` when there is none."""
    return block["empty"] if item is None else fill(block["present"], item)
