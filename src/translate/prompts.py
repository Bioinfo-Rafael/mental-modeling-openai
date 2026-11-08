"""Prompt files: the translation prompt text lives in config, the code only fills it.

A prompt file (``configs/translate/prompts/*.yaml``) holds:

- ``version``: recorded as ``prompt_version`` in every event;
- ``protocol``: ``one_shot`` (one strict-schema Responses call) or ``code_session`` (a multi-turn
  session whose model runs Python in the sandbox); the protocol decides how the call is made;
- ``output_schema``: the named answer schema of a one-shot call (``intent_evidence``);
- ``system`` (one-shot only; ``null`` for a code session) and ``user``: templates with ``{name}``
  placeholders, filled from the values of the selected inputs and the run config's ``prompt.vars``
  (``str.format`` syntax: ``{t_s:g}`` formats a number, literal braces are doubled);
- ``blocks``: how list and optional inputs become text (item template, joiner, text when empty);
- ``turns`` (code session only): the wording of the follow-up messages.

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

import yaml

PROMPT_SCHEMA_VERSION = 1
PROTOCOL_ONE_SHOT = "one_shot"
PROTOCOL_CODE_SESSION = "code_session"
PROTOCOLS = (PROTOCOL_ONE_SHOT, PROTOCOL_CODE_SESSION)
OUTPUT_SCHEMAS = ("intent_evidence",)
PROMPT_KEYS = {"schema_version", "version", "protocol", "output_schema", "system", "user", "blocks", "turns"}
# The fields each block template may use, per block name; the block's keys are fixed too.
BLOCK_KEYS = {
    "instructions": {"item", "primary_marker", "joiner", "empty"},
    "alarms": {"item", "joiner", "empty"},
    "question": {"present", "empty"},
    "previous_translation": {"present", "evidence_item", "evidence_joiner", "empty"},
}
BLOCK_FIELDS = {
    "instructions": {"item": {"issue_step", "target", "primary", "text"}},
    "alarms": {"item": {"step", "targets", "text"}},
    "question": {"present": {"step", "text"}},
    "previous_translation": {"present": {"step", "intent", "evidence"}, "evidence_item": {"item"}},
}
RUN_RESULT_KEYS = {
    "header",
    "status_exit",
    "status_timed_out",
    "merged_note",
    "truncated_note",
    "stderr_header",
    "runs_remaining",
    "no_runs_remaining",
}
PROTOCOL_ERROR_KEYS = {"text", "problem_item", "runs_remaining", "no_runs_remaining"}
TURN_KEYS = {"run_result", "protocol_error", "budget_exhausted"}
CODE_FENCE_OPEN = "```text"
CODE_FENCE_CLOSE = "```"


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


def _turns(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != TURN_KEYS:
        raise PromptError(f"turns must have exactly {sorted(TURN_KEYS)}")
    if not isinstance(raw["budget_exhausted"], str):
        raise PromptError("turns.budget_exhausted must be a string")
    return {
        "run_result": _mapping(raw["run_result"], RUN_RESULT_KEYS, "turns.run_result"),
        "protocol_error": _mapping(raw["protocol_error"], PROTOCOL_ERROR_KEYS, "turns.protocol_error"),
        "budget_exhausted": raw["budget_exhausted"],
    }


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
    turns: Mapping[str, Any] | None
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
        return
    if raw["output_schema"] is not None or raw["system"] is not None or raw["turns"] is None:
        raise PromptError("a code_session prompt needs output_schema: null, system: null and turns")


def load_prompt(path: Path, prompt_vars: Mapping[str, str]) -> PromptTemplate:
    data = Path(path).read_bytes()
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PromptError(f"{path} is not valid UTF-8 YAML") from exc
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
        turns=_turns(raw["turns"]) if raw["turns"] is not None else None,
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


# --- code-session follow-up turns -----------------------------------------------------------------


@dataclass(frozen=True)
class ConfiguredTurns:
    """``sandbox.session.TurnMessages`` with the wording of a prompt file's ``turns``."""

    turns: Mapping[str, Any]

    def run_result(
        self,
        run_index: int,
        max_runs: int,
        stdout: str,
        stderr_tail: str,
        exit_code: int | None,
        timed_out: bool,
        stdout_truncated: bool,
        merged_blocks: int = 1,
    ) -> str:
        words = self.turns["run_result"]
        status = words["status_timed_out"] if timed_out else fill(words["status_exit"], {"exit_code": exit_code})
        header = fill(words["header"], {"run_index": run_index, "max_runs": max_runs, "status": status})
        parts = [header, CODE_FENCE_OPEN, stdout, CODE_FENCE_CLOSE]
        if merged_blocks > 1:
            parts.append(fill(words["merged_note"], {"blocks": merged_blocks}))
        if stdout_truncated:
            parts.append(words["truncated_note"])
        if stderr_tail:
            parts += [words["stderr_header"], CODE_FENCE_OPEN, stderr_tail, CODE_FENCE_CLOSE]
        remaining = max_runs - run_index
        parts.append(
            fill(words["runs_remaining"], {"remaining": remaining}) if remaining else words["no_runs_remaining"]
        )
        return "\n".join(parts)

    def protocol_error(self, problems: tuple[str, ...], runs_remaining: int) -> str:
        words = self.turns["protocol_error"]
        listed = "\n".join(fill(words["problem_item"], {"problem": problem}) for problem in problems)
        remaining = (
            fill(words["runs_remaining"], {"remaining": runs_remaining})
            if runs_remaining > 0
            else words["no_runs_remaining"]
        )
        return fill(words["text"], {"problems": listed, "remaining": remaining})

    def budget_exhausted(self) -> str:
        return self.turns["budget_exhausted"]
