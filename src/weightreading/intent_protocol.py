"""Reply protocol of the intent-B mode: ```python block(s) = code to run; one ```json intent object = stop.

The final answer has the Translator's output schema (method A, ``translator/translation.py``):
exactly the keys ``intent`` (non-blank string, at most ``MAX_INTENT_CHARS`` characters) and
``evidence`` (non-empty list of non-blank strings), no duplicate keys, valid UTF-8.

A reply without a ```python block and without any ```json block is a protocol error (the model is
told what is missing and may retry). A reply that DOES carry a ```json block is the final answer
attempt and ends the session even if it is invalid: method A gets one structured answer per call
and no repair, so B gets no schema retries either. ``parse_answer`` then reports what is wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from weightreading.protocol import (
    CODE_BLOCK_RE,
    JSON_BLOCK_RE,
    KIND_CODE,
    KIND_ERROR,
    KIND_FINAL,
    ParsedReply,
    merge_code_blocks,
)

MAX_INTENT_CHARS = 1000
OUTPUT_KEYS = frozenset({"intent", "evidence"})


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


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _payload_problem(payload: object) -> str | None:
    if not isinstance(payload, dict) or set(payload) != OUTPUT_KEYS:
        return f"the ```json block must be an object with exactly the keys {sorted(OUTPUT_KEYS)}"
    intent, evidence = payload["intent"], payload["evidence"]
    if not isinstance(intent, str) or not intent.strip():
        return '"intent" must be a non-blank string'
    if len(intent) > MAX_INTENT_CHARS:
        return f'"intent" has {len(intent)} characters; the limit is {MAX_INTENT_CHARS}'
    if not isinstance(evidence, list) or not evidence:
        return '"evidence" must be a non-empty list of strings'
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        return 'every "evidence" item must be a non-blank string'
    if not all(_is_utf8(text) for text in (intent, *evidence)):
        return "the answer contains text that is not valid UTF-8 (lone surrogate)"
    return None


def answer_problems(text: str) -> tuple[str, ...]:
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
    problem = _payload_problem(payload)
    return (problem,) if problem else ()


def parse_answer(text: str) -> IntentAnswer:
    """The validated intent object of a final answer; raises ``IntentAnswerError`` otherwise."""
    problems = answer_problems(text)
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
