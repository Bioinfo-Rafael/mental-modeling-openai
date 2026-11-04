"""Translation layer (D3, D10, D22-D24, D32): trajectory -> natural-language intent of one target.

- ``IntentModel`` is the consumer-side Protocol; any model can sit behind it.
- ``TranslationLayer`` owns the prompt, the strict output schema and the strict parser; models
  only run a ``TranslationPrompt``. The trajectory text goes into the prompt verbatim.
- ``OpenAIIntentModel`` adapts ``OpenAIResponsesClient``; the SDK client is built only in the
  composition root (``translator/cli.py``) after the paid-call gate.
- Parsing never repairs, coerces or retries: one JSON object with exactly ``intent`` and
  ``evidence``; a blank or over-long intent, or empty evidence, is an error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from mtd_rl_xavier.lag.rendering import STEP_SECONDS, RenderedTrajectory
from mtd_rl_xavier.llm.openai_responses import (
    OpenAIClientError,
    OpenAIRequest,
    OpenAIResponsesClient,
    build_responses_body,
    input_text,
)

from .schema import (
    Instruction,
    IntentTranslation,
    ModelCallError,
    PreviousTranslation,
    TargetRef,
    step_time_s,
    strict_json_loads,
)

PROMPT_VERSION = "translator-intent-v1"
OUTPUT_SCHEMA_NAME = "intent_translation"
OUTPUT_KEYS = frozenset({"intent", "evidence"})

SYSTEM_PROMPT_TEMPLATE = """\
You translate the recorded behaviour of aircraft in Light Aircraft Game (LAG), an aircraft combat
simulation using JSBSim, into a natural-language statement of intent.

Environment:
A two-versus-two engagement between F-16 aircraft equipped with missiles. Team A consists of
A0100 and A0200. Team B consists of B0100 and B0200. Control is hierarchical: high-level
altitude, heading and speed commands are turned into control inputs by a low-level controller.
The recorded telemetry is not the complete observation of each aircraft's policy.

Data:
One JSON line per step (0.2 s apart) with all recorded aircraft and missiles. Each state is
[lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg, tas_mps, vn_mps, ve_mps, vu_mps]:
degrees, metres above mean sea level, and m/s (northward, eastward, upward velocity). Heading is
clockwise from north (0 north, 90 east); account for wraparound. Raw longitude and latitude
differences are not distances in metres. Aircraft actions are [altitude_action, heading_action,
speed_action]: altitude 0 up / 1 hold / 2 down; heading 0 left30 / 1 left15 / 2 hold / 3 right15 /
4 right30; speed 0 up / 1 hold / 2 down. They are controller commands, not completed state
changes. null means unavailable, not zero. state_valid = 0 masks the state; alive = 0 marks an
inactive aircraft. Team B's actions are not recorded. launcher_id names a missile's launcher,
not its target. A remove event alone does not establish a hit. missiles_remaining is the recorded
ammunition count.

Task:
The user message names one translation target: a single aircraft or a whole team. State the
target's current intent: what it is trying to achieve and how, judged from its recorded behaviour
up to the current step. The trajectory ends at the current step; nothing later is available.
The human instructions in effect are context: they say what a human asked each aircraft or team
to do. Do not assume the target follows its instruction; report the intent its behaviour shows.
A previous translation, when given, is the earlier statement for the same target: keep what
still holds and revise what newer behaviour contradicts. Treat intentions as hypotheses
supported by evidence. Do not invent missile targets, hits, cooperation or outcomes.

Output: one JSON object.
- "intent": the target's current intent in at most {max_intent_chars} characters.
- "evidence": one or more short observations from the trajectory that support the intent,
  citing object ids, steps and recorded values.
Write both in {output_language}."""


class TranslationOutputError(ValueError):
    """The model answered, but the answer is not one valid intent object. Never repaired or retried."""


@dataclass(frozen=True)
class TranslationModelRequest:
    target: TargetRef
    target_aircraft: tuple[str, ...]
    step: int
    episode_alias: str
    instructions_in_effect: tuple[Instruction, ...]  # every instructed target's current instruction
    primary_instruction_id: str  # the instruction of this exact target
    trajectory: RenderedTrajectory
    previous: PreviousTranslation | None  # target-specific; None for the target's first translation


@dataclass(frozen=True)
class TranslationPrompt:
    instructions: str  # system-level instructions
    user_text: str  # the exact user message, trajectory included verbatim
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
    """Strict JSON schema of the translation output."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string"},
            "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
        "required": ["intent", "evidence"],
    }


def system_prompt(*, output_language: str, max_intent_chars: int) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(output_language=output_language, max_intent_chars=max_intent_chars)


def _instruction_lines(request: TranslationModelRequest) -> list[str]:
    lines = []
    for instruction in request.instructions_in_effect:
        primary = " | THIS TARGET'S INSTRUCTION" if instruction.instruction_id == request.primary_instruction_id else ""
        lines.append(f"- issued at step {instruction.issue_step} for {instruction.target.label}{primary}:")
        lines.append(f"  {instruction.text}")
    return lines


def _previous_lines(previous: PreviousTranslation | None) -> list[str]:
    if previous is None:
        return ["(none: this is the first translation for this target)"]
    evidence = [f"  - {item}" for item in previous.translation.evidence]
    return [f"From step {previous.step}:", f"intent: {previous.translation.intent}", "evidence:", *evidence]


def render_user_text(request: TranslationModelRequest) -> str:
    """The exact user message. Never contains the episode id, scenario id or event ids."""
    trajectory = request.trajectory
    points = trajectory.end_step - trajectory.start_step + 1
    lines = [
        f"Episode: {request.episode_alias}",
        f"Current step: {request.step} (time {step_time_s(request.step)} s)",
        f"Translation target: {request.target.label} (aircraft {', '.join(request.target_aircraft)})",
        "",
        "Instructions in effect:",
        *_instruction_lines(request),
        "",
        "Previous translation for this target:",
        *_previous_lines(request.previous),
        "",
        f"Trajectory: steps {trajectory.start_step} through {trajectory.end_step} inclusive "
        f"({points} time points, {STEP_SECONDS} s apart); relative_step = step - {trajectory.anchor_step}.",
        "BEGIN WINDOW DATA",
        trajectory.text,
        "END WINDOW DATA",
    ]
    return "\n".join(lines)


def build_prompt(request: TranslationModelRequest, *, output_language: str, max_intent_chars: int) -> TranslationPrompt:
    """The full prompt for one translation; models only execute it."""
    return TranslationPrompt(
        instructions=system_prompt(output_language=output_language, max_intent_chars=max_intent_chars),
        user_text=render_user_text(request),
        schema_name=OUTPUT_SCHEMA_NAME,
        schema=output_schema(),
    )


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


def _is_utf8(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class TranslationLayer:
    def __init__(self, model: IntentModel, *, output_language: str, max_intent_chars: int) -> None:
        self._model = model
        self._output_language = output_language
        self._max_intent_chars = max_intent_chars

    @property
    def output_language(self) -> str:
        return self._output_language

    def build_prompt(self, request: TranslationModelRequest) -> TranslationPrompt:
        return build_prompt(request, output_language=self._output_language, max_intent_chars=self._max_intent_chars)

    def request_payload(self, prompt: TranslationPrompt) -> dict[str, Any]:
        return self._model.request_payload(prompt)

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        return self._model.generate(prompt)

    def parse(self, output: IntentModelOutput) -> IntentTranslation:
        return parse_intent_output(output.output_text, max_intent_chars=self._max_intent_chars)


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
