"""Detector layer (D9, D27, D50, D51): at each tick, which point kind does the target's behaviour show?

- ``DetectionModel`` is the consumer-side Protocol; any model can sit behind it.
- ``DetectorLayer`` builds the exact state string (target, the target's own previous intent, its
  instructions in effect, the recent alarms and questions, the compact trajectory window) and
  applies the decision rule: ``label`` is the model's choice; the tick triggers when
  ``label != none`` and ``p(label) >= threshold``.
- ``SystemOneDetectionModel`` adapts a ``SystemOneClient`` (one ``choice`` question over
  ``none`` / ``change``). The client itself is constructed only in the composition root
  (``translator/cli.py``).
- ``detector_state_bound`` is the conservative token bound used by the preflight budget check:
  the exact state (instruction, alarm and question texts included) plus the worst-case UTF-8 size
  of a previous intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from translate.config import DetectorQuestionConfig
from translate.events import (
    DETECTOR_LABEL_NONE,
    DETECTOR_LABELS,
    Instruction,
    ModelCallError,
    PreviousTranslation,
    TargetRef,
)
from translate.lag.rendering import STEP_SECONDS, RenderedTrajectory
from translate.llm.system_one import (
    ChoiceAnswer,
    ChoiceQuestion,
    SystemOneClient,
    SystemOneError,
    SystemOneQuestion,
    SystemOneTransportError,
    conservative_input_token_bound,
    question_payload,
    questions_payload,
)

DETECTOR_STATE_VERSION = "2"
MAX_UTF8_BYTES_PER_CHAR = 4  # worst case for any character of a previous intent
MESSAGE_ALARM = "alarm"
MESSAGE_QUESTION = "question"


@dataclass(frozen=True)
class DetectorMessage:
    """An alarm or a human question shown in the detector state (issued at or before the tick)."""

    kind: str  # MESSAGE_ALARM | MESSAGE_QUESTION
    message_id: str  # alarm_id or request_id
    step: int
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.message_id, "step": self.step, "text": self.text}


@dataclass(frozen=True)
class DetectionModelReply:
    model: str  # the model name the server reported
    label: str  # the model's choice: one of DETECTOR_LABELS
    probabilities: tuple[tuple[str, float], ...]  # (label, p) in DETECTOR_LABELS order
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
    instructions: tuple[Instruction, ...]  # the target's own and, for an aircraft, its team's
    messages: tuple[DetectorMessage, ...]  # alarms and questions in the lookback, most recent last
    trajectory: RenderedTrajectory  # compact window; objects = target aircraft, then opponents


@dataclass(frozen=True)
class DetectionResult:
    label: str
    probabilities: tuple[tuple[str, float], ...]
    threshold: float
    triggered: bool
    reply: DetectionModelReply

    @property
    def probability(self) -> float:
        """p(label)."""
        return dict(self.probabilities)[self.label]

    def output_json(self) -> dict[str, Any]:
        """``detector_call.output``."""
        return {
            "label": self.label,
            "probabilities": dict(self.probabilities),
            "threshold": self.threshold,
            "triggered": self.triggered,
        }


def _instruction_lines(instructions: tuple[Instruction, ...]) -> list[str]:
    lines = [f"- {item.target.label}, issued at step {item.issue_step}: {item.text}" for item in instructions]
    return lines or ["- none"]  # no instruction need be in effect


def _message_lines(messages: tuple[DetectorMessage, ...]) -> list[str]:
    return [f"- {item.kind} at step {item.step}: {item.text}" for item in messages] or ["- none"]


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
        "Instructions in effect for this target:",
        *_instruction_lines(context.instructions),
        "",
        "Recent messages (alarms and human questions up to this step, most recent last):",
        *_message_lines(context.messages),
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
        probability = dict(reply.probabilities)[reply.label]
        return DetectionResult(
            label=reply.label,
            probabilities=reply.probabilities,
            threshold=self._threshold,
            triggered=reply.label != DETECTOR_LABEL_NONE and probability >= self._threshold,
            reply=reply,
        )


# --- System One adapter ---------------------------------------------------------------------------


def choice_question(config: DetectorQuestionConfig) -> ChoiceQuestion:
    return ChoiceQuestion(question_id=config.question_id, instructions=config.instructions, criteria=config.criteria)


class SystemOneDetectionModel:
    """``DetectionModel`` over a ``SystemOneClient`` whose fixed question set is one choice question."""

    def __init__(self, client: SystemOneClient, *, question_id: str) -> None:
        questions = client.questions
        if len(questions) != 1 or not isinstance(questions[0], ChoiceQuestion):
            raise ValueError("the detector client must carry exactly one choice question")
        if {name for name, _ in questions[0].criteria} != set(DETECTOR_LABELS):
            raise ValueError(f"the detector choice question must have exactly the options {list(DETECTOR_LABELS)}")
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
        if not isinstance(answer, ChoiceAnswer):  # the client parses a choice question into a ChoiceAnswer
            raise ModelCallError("detector answer is not a choice answer", error_type="ProtocolError", uncertain=False)
        probabilities = dict(answer.probabilities)
        usage = {"input_tokens": result.usage.input_tokens, "output_tokens": result.usage.output_tokens}
        return DetectionModelReply(
            model=result.model,
            label=answer.choice,
            probabilities=tuple((label, probabilities[label]) for label in DETECTOR_LABELS),
            usage=usage,
            raw=result.raw,
        )
