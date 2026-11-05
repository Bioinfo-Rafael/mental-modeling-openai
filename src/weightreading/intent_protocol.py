"""Reply protocol of the intent-B mode: ```python block(s) = code to run; one ```json intent object = stop.

The final answer has the Translator's output schema (method A, ``translator/schema.py``
``INTENT_OUTPUT_KEYS``): exactly the keys ``intent`` (non-blank string, at most ``MAX_INTENT_CHARS``
characters) and ``evidence`` (non-empty list of non-blank strings), no duplicate keys, valid UTF-8.
The optional ``AnswerLimits`` (from the compare config) tighten the intent length and cap the number
and the length of the evidence items; all lengths are counted in characters, not bytes. The limits,
their prompt sentence and the payload validation live in ``mtd_rl_xavier.intent_compare`` (shared
with method A) and are re-exported here.

A reply without a ```python block and without any ```json block is a protocol error (the model is
told what is missing and may retry). A reply that DOES carry a ```json block is the final answer
attempt and ends the session even if it is invalid: method A gets one structured answer per call
and no repair, so B gets no schema retries either. ``parse_answer`` then reports what is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mtd_rl_xavier.intent_compare import (  # the comparison's shared answer contract
    DEFAULT_LIMITS,
    MAX_INTENT_CHARS,
    AnswerLimits,
    limits_sentence,
    payload_problem,
)
from weightreading.protocol import (
    CODE_BLOCK_RE,
    JSON_BLOCK_RE,
    KIND_CODE,
    KIND_ERROR,
    KIND_FINAL,
    ParsedReply,
    merge_code_blocks,
)


class IntentAnswerError(ValueError):
    """A final answer is not one valid intent object."""


@dataclass(frozen=True)
class IntentAnswer:
    intent: str
    evidence: tuple[str, ...]


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


def parse_intent_reply(text: str) -> ParsedReply:
    """Session parser for the intent-B mode (same ``ParsedReply`` kinds as ``protocol.parse_reply``)."""
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return ParsedReply(kind=KIND_CODE, code=merge_code_blocks(blocks), block_count=len(blocks))
    if JSON_BLOCK_RE.search(text):
        return ParsedReply(kind=KIND_FINAL)  # validated (and possibly rejected) by parse_answer
    return ParsedReply(kind=KIND_ERROR, problems=answer_problems(text))


__all__ = [
    "DEFAULT_LIMITS",
    "MAX_INTENT_CHARS",
    "AnswerLimits",
    "IntentAnswer",
    "IntentAnswerError",
    "answer_problems",
    "limits_sentence",
    "parse_answer",
    "parse_intent_reply",
    "payload_problem",
]
