"""First message of an intent-B session: translate one aircraft's intent from weights + activations.

Reuses the whitelisted spec sections of ``prompt.py`` (task description, architecture equations,
input semantics, output bins, parameter table) in ``shown`` mode, so the spec's ``evaluation``,
``notes`` and ``provenance`` never reach the model. No world trajectory is rendered: the model sees
only the policy's own observations and activations, inside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass

from weightreading.intent_protocol import MAX_INTENT_CHARS, IntentAnswer
from weightreading.intent_trace import ActivationDoc
from weightreading.prompt import ExecutorLimits, spec_sections
from weightreading.spec import Spec
from weightreading.tensors import Tensor

TASK_MODE = "shown"
PROMPT_VERSION = "intent-b-v1"


@dataclass(frozen=True)
class PreviousTranslation:
    step: int
    answer: IntentAnswer  # the latest valid translation of this aircraft by method B


@dataclass(frozen=True)
class IntentRequest:
    aircraft: str
    step: int
    t_s: float
    window: tuple[int, int]  # inclusive absolute steps
    step_seconds: float
    instruction: str
    previous: PreviousTranslation | None  # None: first translation for this aircraft


RULES = """\
# Rules (strict)
- Use only W, A and the information in this message.
- A holds validated replay values for this aircraft over the stated causal window (the network was
  replayed on the recorded observations from h = 0 and matches the recorded logits and states).
- A['h_prev'] of the first row summarises the policy's own history before the window; it is not
  world data.
- You may compute statistics, temporal differences and any linear algebra directly on W and A.
- You may exactly reconstruct one recorded step of one block from its recorded direct inputs and W
  (linear maps, ReLU, sigmoid, tanh, softmax and LayerNorm included), only to verify or
  algebraically decompose the matching value in A (e.g. per-unit contributions to a logit, input
  vs recurrent parts of a gate). With LayerNorm, a decomposition at the row's fixed mean/variance
  is algebra, not a causal attribution.
- Do not pass perturbed, scaled, interpolated, averaged, zeroed or hand-built vectors through any
  layer. Do not modify W or A, or use other weights, observations or hidden states.
- Do not compute gradients, finite differences, ablations or counterfactuals; do not iterate the
  recurrence or synthesize another trajectory.
- Do not import torch, any other deep-learning or automatic-differentiation library, or search
  for anything. Your code is audited for these rules.
- Base every evidence item on named A/W keys, absolute steps and numbers you computed."""


def _intro(spec: Spec, request: IntentRequest) -> str:
    return (
        f'A neural network policy called "{spec.alias}" flies aircraft {request.aircraft}. State its current '
        f"intent at decision step {request.step} (time {request.t_s:g} s): what it is trying to achieve and how. "
        "You do not see the world. You see only the policy: its trained weights and, for every step of the "
        "window, the observation it read, every intermediate activation and the actions it chose "
        "(validated replay values). "
        "Nothing after the current step is available."
    )


def _context_section(request: IntentRequest) -> str:
    start, end = request.window
    return "\n".join(
        [
            "# Context",
            f"- Aircraft: {request.aircraft} (team A; its teammate runs its own copy of the same network).",
            f"- Current step: {request.step} (time {request.t_s:g} s). Window: steps {start} through {end} "
            f"inclusive ({end - start + 1} decisions, {request.step_seconds:g} s apart).",
            "- Human instruction in effect (context only; the policy does not read it, so do not assume it "
            f"is followed): {request.instruction}",
        ]
    )


def _previous_section(previous: PreviousTranslation | None) -> str:
    lines = ["# Previous translation for this aircraft (yours, from the same method)"]
    if previous is None:
        return "\n".join([*lines, "(none: this is the first translation for this aircraft)"])
    evidence = [f"  - {item}" for item in previous.answer.evidence]
    lines += [f"From step {previous.step}:", f"intent: {previous.answer.intent}", "evidence:", *evidence]
    lines.append("Keep what still holds and revise what the newer steps contradict.")
    return "\n".join(lines)


def _protocol_section(limits: ExecutorLimits, window_length: int) -> str:
    return f"""\
# How you work
- Every reply must be EITHER a ```python code block, which is executed and whose output you receive
  in the next message, OR your final answer (with no ```python block at all). Prefer exactly one
  block per reply: several blocks are merged in order into ONE run and count as one run.
- A reply with neither a code block nor a complete final answer is a protocol error.
- Each code block runs in a fresh Python 3 process with numpy. Preloaded: `np` (numpy), `json`,
  `W` (dict: parameter name -> float32 array, see Parameters) and `A` (dict: activation name ->
  array with a leading axis of {window_length} rows, one per window step, oldest first; see
  Activations). Files you write under /work persist between runs; /work/weights is read-only.
  There is no network and no other data.
- stdout is truncated to {limits.stdout_cap_chars} characters; the last {limits.stderr_tail_chars}
  characters of stderr are shown; each run is killed after {limits.timeout_s} s.
- You may execute at most {limits.max_runs} code blocks. Print compact summaries, not whole arrays."""


def _activations_section(docs: tuple[ActivationDoc, ...], window_length: int) -> str:
    rows = [f"- `A['{doc.key}']` shape {[window_length, *doc.shape_tail]}: {doc.description}" for doc in docs]
    note = (
        "Row i of every array is the same decision. `h_prev` of the first row is the state carried in from "
        "before the window (zero only if the window starts at step 0). Probabilities are exact softmaxes of "
        "the recorded logits; ties within ~1e-4 in the logits are numerical ties."
    )
    return "\n".join(
        [f"# Activations (keys of A, read-only; float32 unless stated, {window_length} rows)", *rows, note]
    )


def _final_answer_section() -> str:
    return f"""\
# Final answer
When you are done, reply WITHOUT any ```python block and with exactly one ```json block:
{{"intent": "<the aircraft's current intent: what it is trying to achieve and how, at most {MAX_INTENT_CHARS} characters>",
 "evidence": ["<short observation from the weights or activations that supports the intent, citing steps,
 activation names and values>", ...]}}
Treat the intent as a hypothesis supported by evidence; do not invent missile targets, hits,
cooperation or outcomes. Write both fields in English. Text outside the ```json block is ignored."""


def intent_message(
    spec: Spec,
    tensors: tuple[Tensor, ...],
    docs: tuple[ActivationDoc, ...],
    request: IntentRequest,
    limits: ExecutorLimits,
) -> str:
    start, end = request.window
    window_length = end - start + 1
    shown = spec_sections(spec, tensors, TASK_MODE)
    sections = [
        _intro(spec, request),
        "# Task of the policy\n" + spec.description,
        _context_section(request),
        _previous_section(request.previous),
        _protocol_section(limits, window_length),
        RULES,
        shown["architecture"],
        shown["inputs"],
        shown["outputs"],
        _activations_section(docs, window_length),
        shown["parameters"],
        _final_answer_section(),
    ]
    return "\n\n".join(sections) + "\n"
