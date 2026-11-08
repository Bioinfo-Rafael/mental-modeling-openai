"""The translation answer contract: one ``{"intent", "evidence"}`` object within the configured limits.

Both call kinds validate their parsed answer here, counting characters (not bytes), so a one-shot
answer and a code-session answer obey exactly the same rules. The keys come from the event
contract (``events.INTENT_OUTPUT_KEYS``). How the limits are worded in a prompt is the prompt
file's business, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

from translate.events import INTENT_OUTPUT_KEYS

MAX_INTENT_CHARS = 1000  # default intent cap (characters)
INVALID_ANSWER_PREFIX = "invalid final answer: "  # how a rejected answer's reason starts (events, cues)


@dataclass(frozen=True)
class AnswerLimits:
    """Length limits of an answer; ``None`` means unlimited."""

    max_intent_chars: int = MAX_INTENT_CHARS
    max_evidence_items: int | None = None
    max_evidence_chars: int | None = None


DEFAULT_LIMITS = AnswerLimits()


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
    """What is wrong with a parsed answer object (None = valid)."""
    if not isinstance(payload, dict) or set(payload) != INTENT_OUTPUT_KEYS:
        return f"the ```json block must be an object with exactly the keys {sorted(INTENT_OUTPUT_KEYS)}"
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
