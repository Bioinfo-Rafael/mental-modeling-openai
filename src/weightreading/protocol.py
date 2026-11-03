"""Reply protocol: ```python block(s) = code to run; a well-formed final answer = stop; else error.

Several ```python blocks in one reply are merged, in order, into a single run. A reply without a
code block is a final answer only if it carries the required sections and exactly one parseable
```json block with a ``claims`` list; anything else is a protocol error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

CODE_BLOCK_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```json[ \t]*\n(.*?)```", re.DOTALL)

KIND_CODE = "code"
KIND_FINAL = "final"
KIND_ERROR = "error"

# (heading, description shown to the model); shared with prompt.py so the instructions and the
# validation cannot drift apart.
FINAL_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Observations", "quote the weights and numbers you computed"),
    ("Inputs", "the role of each input"),
    ("Heads", "what drives each head, with sign and evidence"),
    ("Recurrence", "what the recurrent memory is used for"),
    ("Summary", ""),
)
# Asked for in blind mode only, so it is not required by the validation.
HYPOTHESIS_SECTION = (
    "Task hypothesis",
    "what system this controls and what it is trying to do",
)


@dataclass(frozen=True)
class ParsedReply:
    kind: str
    code: str = ""
    block_count: int = 0
    problems: tuple[str, ...] = ()


def merge_code_blocks(blocks: list[str]) -> str:
    """Concatenate blocks in order, each later one introduced by a ``# --- block N ---`` line."""
    if len(blocks) == 1:
        return blocks[0]
    parts = [blocks[0]] + [f"# --- block {number} ---\n{block}" for number, block in enumerate(blocks[1:], start=2)]
    return "\n".join(part if part.endswith("\n") else part + "\n" for part in parts)


def _has_heading(text: str, name: str) -> bool:
    return re.search(rf"^[ \t]*#{{1,6}}[ \t]*{re.escape(name)}\b", text, re.IGNORECASE | re.MULTILINE) is not None


def _claims_problem(text: str) -> str | None:
    """Return what is wrong with the ```json block(s) of ``text``, or None if it is fine."""
    blocks = JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        return f'exactly one ```json block with a "claims" list (found {len(blocks)})'
    try:
        payload = json.loads(blocks[0])
    except json.JSONDecodeError:
        return 'the ```json block is not valid JSON; it must be {"claims": [...]}'
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        return 'the ```json block must be an object with a "claims" list'
    return None


def final_answer_problems(text: str) -> tuple[str, ...]:
    """List what is missing for ``text`` to be a well-formed final answer (empty = well-formed)."""
    missing = [f"section heading '## {name}'" for name, _ in FINAL_SECTIONS if not _has_heading(text, name)]
    claims_problem = _claims_problem(text)
    return tuple(missing + ([claims_problem] if claims_problem else []))


def parse_reply(text: str) -> ParsedReply:
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return ParsedReply(kind=KIND_CODE, code=merge_code_blocks(blocks), block_count=len(blocks))
    problems = final_answer_problems(text)
    if problems:
        return ParsedReply(kind=KIND_ERROR, problems=problems)
    return ParsedReply(kind=KIND_FINAL)
