"""One-shot call: one strict-schema Responses request per translation, one JSON answer.

- ``IntentModel`` is the consumer-side Protocol; any model can sit behind it.
- ``OneShotCall`` renders the configured prompt, gives the exact request body to the journal,
  makes the call and validates the answer; ``OpenAIIntentModel`` adapts ``OpenAIResponsesClient``.
  The SDK client is built only in the composition root (``translate/cli.py``) after the paid gate.
- Parsing never repairs, coerces or retries: one JSON object with exactly ``intent`` and
  ``evidence``; then the configured answer limits apply.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from translate.answers import AnswerLimits, payload_problem
from translate.events import INTENT_OUTPUT_KEYS, IntentTranslation, ModelCallError, TargetRef, strict_json_loads
from translate.llm.openai_responses import (
    OpenAIClientError,
    OpenAIRequest,
    OpenAIResponsesClient,
    build_responses_body,
    input_text,
)
from translate.llm.retry import RetryEvent, log_to_stderr
from translate.prompts import PromptTemplate

OUTPUT_SCHEMA_NAME = "intent_translation"
OUTPUT_KEYS = INTENT_OUTPUT_KEYS  # the shared intent output contract (events.py)
INVALID_PREFIX = "invalid final answer: "  # the wording of a rejected answer in events and cues


class TranslationOutputError(ValueError):
    """The model answered, but the answer is not one valid intent object. Never repaired or retried."""


@dataclass(frozen=True)
class TranslationPrompt:
    instructions: str  # system-level instructions
    user_text: str  # the exact user message
    schema_name: str
    schema: dict[str, Any]


@dataclass(frozen=True)
class IntentModelOutput:
    model: str  # the model name the API reported
    response_id: str | None
    output_text: str
    usage: dict[str, int]
    raw: dict[str, Any]


class IntentModel(Protocol):
    def request_payload(self, prompt: TranslationPrompt) -> dict[str, Any]:
        """The exact request body; persisted before the call is sent."""
        ...

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        """One call. Raises ``ModelCallError`` on failure."""
        ...


def output_schema() -> dict[str, Any]:
    """Strict JSON schema of the translation output (``intent_evidence``)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string"},
            "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "required": ["intent", "evidence"],
    }


def build_prompt(template: PromptTemplate, values: Mapping[str, object]) -> TranslationPrompt:
    """The full prompt of one translation from the configured templates."""
    system, user = template.render(values)
    if system is None:
        raise ValueError("a one-shot prompt needs a system text")
    return TranslationPrompt(
        instructions=system, user_text=user, schema_name=OUTPUT_SCHEMA_NAME, schema=output_schema()
    )


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def parse_intent_output(output_text: str, *, max_intent_chars: int) -> IntentTranslation:
    """Strict parse: exactly one JSON object (no repeated key) with a non-blank intent and evidence."""
    try:
        parsed = strict_json_loads(output_text)
    except (TypeError, ValueError) as exc:
        raise TranslationOutputError("translation output is not one JSON object without repeated keys") from exc
    if not isinstance(parsed, dict) or set(parsed) != OUTPUT_KEYS:
        raise TranslationOutputError(f"translation output must have exactly {sorted(OUTPUT_KEYS)}")
    intent, evidence = parsed["intent"], parsed["evidence"]
    if not isinstance(intent, str) or not intent.strip():
        raise TranslationOutputError("intent must be a non-blank string")
    if len(intent) > max_intent_chars:
        raise TranslationOutputError(f"intent has {len(intent)} characters; the limit is {max_intent_chars}")
    if not isinstance(evidence, list) or not evidence:
        raise TranslationOutputError("evidence must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in evidence):
        raise TranslationOutputError("every evidence item must be a non-blank string")
    if not all(_is_utf8(text) for text in (intent, *evidence)):
        raise TranslationOutputError("translation output contains text that is not valid UTF-8 (lone surrogate)")
    return IntentTranslation(intent=intent, evidence=tuple(evidence))


@dataclass
class RetryJournal:
    """Failed attempts of the call in flight: logged to stderr, then drained into its ``responses.jsonl`` line."""

    pending: tuple[dict[str, Any], ...] = ()

    def record(self, event: RetryEvent) -> None:
        log_to_stderr(event)
        self.pending = (*self.pending, event.to_dict())

    def drain(self) -> tuple[dict[str, Any], ...]:
        events, self.pending = self.pending, ()
        return events


@dataclass(frozen=True)
class SlotRef:
    """One translation slot as a call sees it."""

    call_id: str
    step: int
    target: TargetRef
    session_dir: Path | None  # code session only: where the session's inputs and logs go


@dataclass(frozen=True)
class CallResult:
    """What a call produced: the journaled response body, the answer text and the model facts."""

    raw: Any
    answer_text: str | None  # None: the call ended without an answer (code session caps)
    response_model: str | None
    response_id: str | None
    usage: dict[str, int]
    ended_without_answer: str | None = None  # why there is no answer text


class OneShotCall:
    """Translate one slot with one strict-schema request."""

    def __init__(self, model: IntentModel, template: PromptTemplate, limits: AnswerLimits) -> None:
        self._model = model
        self._template = template
        self._limits = limits

    def prepare(self, values: Mapping[str, object], slot: SlotRef) -> tuple[dict[str, Any], TranslationPrompt]:
        """(exact request body to journal, the prompt to send); ``slot`` is not needed here."""
        prompt = build_prompt(self._template, values)
        return self._model.request_payload(prompt), prompt

    def run(self, prompt: TranslationPrompt) -> CallResult:
        output = self._model.generate(prompt)
        return CallResult(output.raw, output.output_text, output.model, output.response_id, output.usage)

    def answer(self, result: CallResult) -> tuple[IntentTranslation | None, str | None]:
        """(translation, None) or (None, why the answer is not valid)."""
        try:
            translation = parse_intent_output(result.answer_text or "", max_intent_chars=self._limits.max_intent_chars)
        except TranslationOutputError as exc:
            return None, f"{INVALID_PREFIX}{exc}"
        problem = payload_problem({"intent": translation.intent, "evidence": list(translation.evidence)}, self._limits)
        return (None, f"{INVALID_PREFIX}{problem}") if problem else (translation, None)


# --- OpenAI Responses adapter ---------------------------------------------------------------------


def build_openai_request(
    prompt: TranslationPrompt, *, model: str, reasoning_effort: str, max_output_tokens: int
) -> OpenAIRequest:
    return OpenAIRequest(
        model=model,
        instructions=prompt.instructions,
        input_items=(input_text(prompt.user_text),),
        schema_name=prompt.schema_name,
        schema=prompt.schema,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )


class OpenAIIntentModel:
    """``IntentModel`` over ``OpenAIResponsesClient``. Model, effort and token cap come from the config."""

    def __init__(
        self, client: OpenAIResponsesClient, *, model: str, reasoning_effort: str, max_output_tokens: int
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens

    def _request(self, prompt: TranslationPrompt) -> OpenAIRequest:
        return build_openai_request(
            prompt,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            max_output_tokens=self._max_output_tokens,
        )

    def request_payload(self, prompt: TranslationPrompt) -> dict[str, Any]:
        return build_responses_body(self._request(prompt))

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        try:
            result = self._client.create(self._request(prompt))
        except OpenAIClientError as exc:
            raw = exc.result.raw if exc.result is not None else exc.raw
            raise ModelCallError(str(exc), error_type=type(exc).__name__, uncertain=exc.uncertain, raw=raw) from exc
        return IntentModelOutput(
            model=result.model,
            response_id=result.response_id,
            output_text=result.output_text,
            usage=asdict(result.usage),
            raw=result.raw,
        )
