"""Reply protocol of the intent-B mode: ```python block(s) = code to run; one ```json intent object = stop.

The final answer has the Translator's output schema (method A, ``translator/translation.py``):
exactly the keys ``intent`` (non-blank string, at most ``MAX_INTENT_CHARS`` characters) and
``evidence`` (non-empty list of non-blank strings), no duplicate keys, valid UTF-8. The optional
``AnswerLimits`` (from the compare config) tighten the intent length and cap the number and the
length of the evidence items; all lengths are counted in characters, not bytes.

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


@dataclass(frozen=True)
class AnswerLimits:
    """Length limits of a final answer; ``None`` means unlimited (the default keeps today's rules)."""

    max_intent_chars: int = MAX_INTENT_CHARS
    max_evidence_items: int | None = None
    max_evidence_chars: int | None = None


DEFAULT_LIMITS = AnswerLimits()


def _evidence_limit_text(limits: AnswerLimits) -> str:
    items, chars = limits.max_evidence_items, limits.max_evidence_chars
    if items is not None and chars is not None:
        return f"at most {items} evidence items, each at most {chars} characters"
    if items is not None:
        return f"at most {items} evidence items"
    if chars is not None:
        return f"each evidence item at most {chars} characters"
    return ""


def limits_sentence(limits: AnswerLimits) -> str:
    """The one prompt sentence that states the limits; ``""`` when they equal the defaults.

    Single source of the wording: method B puts it into its final-answer section and method A appends
    it to the Translator's instructions, so the two prompts cannot drift apart.
    """
    if limits == DEFAULT_LIMITS:
        return ""
    parts = [f"intent at most {limits.max_intent_chars} characters", _evidence_limit_text(limits)]
    stated = "; ".join(part for part in parts if part)
    return f"Length limits (counted in characters, not bytes; a longer answer is rejected): {stated}."


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


def _evidence_length_problem(evidence: list, limits: AnswerLimits) -> str | None:
    if limits.max_evidence_items is not None and len(evidence) > limits.max_evidence_items:
        return f'"evidence" has {len(evidence)} items; the limit is {limits.max_evidence_items}'
    if limits.max_evidence_chars is None:
        return None
    too_long = [len(item) for item in evidence if len(item) > limits.max_evidence_chars]
    if too_long:
        return f'an "evidence" item has {max(too_long)} characters; the limit is {limits.max_evidence_chars}'
    return None


def payload_problem(payload: object, limits: AnswerLimits) -> str | None:
    """What is wrong with a parsed final-answer object (None = valid); shared by methods A and B."""
    if not isinstance(payload, dict) or set(payload) != OUTPUT_KEYS:
        return f"the ```json block must be an object with exactly the keys {sorted(OUTPUT_KEYS)}"
    intent, evidence = payload["intent"], payload["evidence"]
    if not isinstance(intent, str) or not intent.strip():
        return '"intent" must be a non-blank string'
    if len(intent) > limits.max_intent_chars:
        return f'"intent" has {len(intent)} characters; the limit is {limits.max_intent_chars}'
    if not isinstance(evidence, list) or not evidence:
        return '"evidence" must be a non-empty list of strings'
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        return 'every "evidence" item must be a non-blank string'
    length_problem = _evidence_length_problem(evidence, limits)
    if length_problem:
        return length_problem
    if not all(_is_utf8(text) for text in (intent, *evidence)):
        return "the answer contains text that is not valid UTF-8 (lone surrogate)"
    return None


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
