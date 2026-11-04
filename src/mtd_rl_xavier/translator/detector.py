"""Detector layer (D6, D9, D27): at each tick, is the motion still explained by the previous intent?

- ``DetectionModel`` is the consumer-side Protocol; any model can sit behind it.
- ``DetectorLayer`` builds the exact state string (target, the target's own previous intent, the
  compact trajectory window) and applies the threshold: ``changed = probability >= threshold``.
- ``SystemOneDetectionModel`` adapts a ``SystemOneClient`` (one ``noul`` question). The client
  itself is constructed only in the composition root (``translator/cli.py``).
- ``detector_state_bound`` is the conservative token bound used by the preflight budget check:
  the exact state framing plus the worst-case UTF-8 size of a previous intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mtd_rl_xavier.lag.rendering import STEP_SECONDS, RenderedTrajectory
from mtd_rl_xavier.llm.system_one import (
    NoulAnswer,
    NoulQuestion,
    SystemOneClient,
    SystemOneError,
    SystemOneQuestion,
    SystemOneTransportError,
    conservative_input_token_bound,
    question_payload,
    questions_payload,
)

from .config import DetectorQuestionConfig
from .schema import ModelCallError, PreviousTranslation, TargetRef

DETECTOR_STATE_VERSION = "1"
MAX_UTF8_BYTES_PER_CHAR = 4  # worst case for any character of a previous intent


@dataclass(frozen=True)
class DetectionModelReply:
    model: str  # the model name the server reported
    probability: float  # p(yes): the motion is NOT explained by the previous intent
    usage: dict[str, int]
    raw: dict[str, Any]  # the full response body


class DetectionModel(Protocol):
    @property
    def question(self) -> dict[str, Any]:
        """The full question object (id, type, instructions, criteria) sent with every call."""
        ...

    def request_payload(self, state: str) -> dict[str, Any]:
        """The exact request body for ``state``; persisted before the call is sent."""
        ...

    def decide(self, state: str) -> DetectionModelReply:
        """One call. Raises ``ModelCallError`` on failure."""
        ...


@dataclass(frozen=True)
class DetectionContext:
    target: TargetRef
    step: int
    target_aircraft: tuple[str, ...]
    opponent_aircraft: tuple[str, ...]
    previous: PreviousTranslation  # the target's own previous translation
    trajectory: RenderedTrajectory  # compact window; objects = target aircraft, then opponents


@dataclass(frozen=True)
class DetectionResult:
    changed: bool
    probability: float
    threshold: float
    reply: DetectionModelReply


def render_detector_state(context: DetectionContext, *, intent: str | None = None) -> str:
    """The exact state string sent to the detector. ``intent`` overrides the previous intent (budget checks)."""
    trajectory = context.trajectory
    previous_intent = context.previous.translation.intent if intent is None else intent
    lines = (
        f"Target: {context.target.label} (aircraft {', '.join(context.target_aircraft)}). "
        f"Opponents: {', '.join(context.opponent_aircraft)}.",
        f"Previous intent translation for this target (from step {context.previous.step}):",
        previous_intent,
        "",
        f"Trajectory window: steps {trajectory.start_step}-{trajectory.end_step}, {STEP_SECONDS} s per step, "
        f"one JSON line per step, newest last; relative_step = step - {trajectory.anchor_step}. "
        f"Fields: {', '.join(trajectory.fields)}. null = unavailable.",
        trajectory.text,
    )
    return "\n".join(lines)


def detector_state_bound(
    context: DetectionContext, *, questions: tuple[SystemOneQuestion, ...], max_intent_chars: int
) -> int:
    """Conservative token bound for any state of this context whose intent has at most ``max_intent_chars``."""
    framing = render_detector_state(context, intent="")
    intent_reserve = MAX_UTF8_BYTES_PER_CHAR * max_intent_chars
    return conservative_input_token_bound(state=framing, questions=questions) + intent_reserve


class DetectorLayer:
    def __init__(self, model: DetectionModel, *, threshold: float) -> None:
        self._model = model
        self._threshold = threshold

    @property
    def question(self) -> dict[str, Any]:
        return self._model.question

    @property
    def threshold(self) -> float:
        return self._threshold

    def state_text(self, context: DetectionContext) -> str:
        return render_detector_state(context)

    def request_payload(self, state: str) -> dict[str, Any]:
        return self._model.request_payload(state)

    def detect(self, state: str) -> DetectionResult:
        reply = self._model.decide(state)
        return DetectionResult(
            changed=reply.probability >= self._threshold,
            probability=reply.probability,
            threshold=self._threshold,
            reply=reply,
        )


# --- System One adapter ---------------------------------------------------------------------------


def noul_question(config: DetectorQuestionConfig) -> NoulQuestion:
    return NoulQuestion(
        question_id=config.question_id,
        instructions=config.instructions,
        true_criterion=config.true_criterion,
        false_criterion=config.false_criterion,
    )


class SystemOneDetectionModel:
    """``DetectionModel`` over a ``SystemOneClient`` whose fixed question set is one noul question."""

    def __init__(self, client: SystemOneClient, *, question_id: str) -> None:
        questions = client.questions
        if len(questions) != 1 or not isinstance(questions[0], NoulQuestion):
            raise ValueError("the detector client must carry exactly one noul question")
        if questions[0].question_id != question_id:
            raise ValueError("question_id does not match the client's question")
        self._client = client
        self._question = questions[0]

    @property
    def question(self) -> dict[str, Any]:
        return {"id": self._question.question_id, **question_payload(self._question)}

    def request_payload(self, state: str) -> dict[str, Any]:
        """Identical to the body ``SystemOneClient.decide`` sends."""
        return {"model": self._client.model, "state": state, "questions": questions_payload(self._client.questions)}

    def decide(self, state: str) -> DetectionModelReply:
        try:
            result = self._client.decide(state)
        except SystemOneError as exc:
            timed_out = isinstance(exc, SystemOneTransportError) and exc.timed_out
            raise ModelCallError(str(exc), error_type=type(exc).__name__, uncertain=timed_out, raw=exc.raw) from exc
        answer = result.answer(self._question.question_id)
        if not isinstance(answer, NoulAnswer):  # the client parses a noul question into a NoulAnswer
            raise ModelCallError("detector answer is not a noul answer", error_type="ProtocolError", uncertain=False)
        usage = {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens}
        return DetectionModelReply(model=result.model, probability=answer.p_yes, usage=usage, raw=result.raw)
