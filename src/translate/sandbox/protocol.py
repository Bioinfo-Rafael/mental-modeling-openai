"""Reply protocol of a code session: ```python block(s) = code to run; a ```json answer = stop.

Several ```python blocks in one reply are merged, in order, into a single run. A reply without a
code block that carries a ```json block is the final answer attempt and ends the session even if
it is invalid (a one-shot call gets no repair either); ``parse_answer`` then says what is wrong.
A reply with neither is a protocol error: the model is told what is missing and may retry.
The answer has the event contract's keys and the configured ``AnswerLimits``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from translate.answers import DEFAULT_LIMITS, AnswerLimits, payload_problem

CODE_BLOCK_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```json[ \t]*\n(.*?)```", re.DOTALL)

KIND_CODE = "code"
KIND_FINAL = "final"
KIND_ERROR = "error"


@dataclass(frozen=True)
class ParsedReply:
    kind: str
    code: str = ""
    block_count: int = 0
    problems: tuple[str, ...] = ()


class IntentAnswerError(ValueError):
    """A final answer is not one valid intent object."""


@dataclass(frozen=True)
class IntentAnswer:
    intent: str
    evidence: tuple[str, ...]


def merge_code_blocks(blocks: list[str]) -> str:
    """Concatenate blocks in order, each later one introduced by a ``# --- block N ---`` line."""
    if len(blocks) == 1:
        return blocks[0]
    parts = [blocks[0]] + [f"# --- block {number} ---\n{block}" for number, block in enumerate(blocks[1:], start=2)]
    return "\n".join(part if part.endswith("\n") else part + "\n" for part in parts)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise IntentAnswerError("the ```json block repeats a key")
    return dict(pairs)


def answer_problems(text: str, limits: AnswerLimits = DEFAULT_LIMITS) -> tuple[str, ...]:
    """What is wrong with ``text`` as a final answer (empty = well-formed)."""
    blocks = JSON_BLOCK_RE.findall(text)
    if len(blocks) != 1:
        return (f'exactly one ```json block {{"intent": ..., "evidence": [...]}} (found {len(blocks)})',)
    try:
        payload = json.loads(blocks[0], object_pairs_hook=_reject_duplicates)
    except json.JSONDecodeError:
        return ("the ```json block is not valid JSON",)
    except IntentAnswerError as exc:
        return (str(exc),)
    problem = payload_problem(payload, limits)
    return (problem,) if problem else ()


def parse_answer(text: str, limits: AnswerLimits = DEFAULT_LIMITS) -> IntentAnswer:
    """The validated intent object of a final answer; raises ``IntentAnswerError`` otherwise."""
    problems = answer_problems(text, limits)
    if problems:
        raise IntentAnswerError("; ".join(problems))
    payload = json.loads(JSON_BLOCK_RE.findall(text)[0])
    return IntentAnswer(intent=payload["intent"], evidence=tuple(payload["evidence"]))


def parse_reply(text: str) -> ParsedReply:
    """Code to run, the final answer attempt, or a protocol error with what is missing."""
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return ParsedReply(kind=KIND_CODE, code=merge_code_blocks(blocks), block_count=len(blocks))
    if JSON_BLOCK_RE.search(text):
        return ParsedReply(kind=KIND_FINAL)  # validated (and possibly rejected) by parse_answer
    return ParsedReply(kind=KIND_ERROR, problems=answer_problems(text))
