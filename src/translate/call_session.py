"""Code-session call: the model reads the policy input with Python in the sandbox, then answers in JSON.

Per slot: write the read-only inputs bundle (``W`` and this aircraft's window ``A``) under
``sessions/<slot>/inputs``, run the fail-closed sandbox preflight for that directory (a failure
refuses the slot before any request), write ``message.txt``, drive the session loop with the
configured follow-up wording, verify the bundle was not changed, write the advisory audit, and
validate the final ```json answer against the answer limits. The first request body (the first
user message plus the request options) is what the run journals before the session starts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from translate.answers import INVALID_ANSWER_PREFIX, AnswerLimits
from translate.call_one_shot import CallResult, SlotRef
from translate.events import IntentTranslation
from translate.policy.bundle import MANIFEST_FILE, verify_bundle
from translate.policy.material import PolicyMaterial, write_session_inputs
from translate.policy.session_inputs import PRELOAD, SessionInputs
from translate.prompts import ConfiguredTurns, PromptTemplate
from translate.sandbox.audit import write_session_audit
from translate.sandbox.executor import Interpreter, SandboxExecutor, SandboxLimits
from translate.sandbox.preflight import run_preflight
from translate.sandbox.protocol import IntentAnswerError, parse_answer, parse_reply
from translate.sandbox.session import FINAL_ANSWER, RunLog, SessionLimits, SessionState, run_session
from translate.session_backend import ChatBackend, user_item

STOP_FINAL_ANSWER = "final_answer"

ExecutorFactory = Callable[[Path, Path], SandboxExecutor]  # (work dir, weights dir) -> executor
Preflight = Callable[[SandboxExecutor, list[Path]], Any]  # returns an object with ``ok`` and ``to_dict()``


class SandboxRefused(RuntimeError):
    """The sandbox preflight failed for a session directory; no request was sent for it."""


@dataclass(frozen=True)
class SessionSettings:
    limits: SessionLimits
    sandbox: SandboxLimits
    past_steps: int  # window of the policy activations
    hidden: tuple[Path, ...]  # host paths the preflight must find hidden (repository, home, checkpoint, rollout)


def sandbox_executor(interp: Interpreter, limits: SandboxLimits) -> ExecutorFactory:
    def make(work: Path, weights: Path) -> SandboxExecutor:
        return SandboxExecutor(interp, work, weights, limits, PRELOAD)

    return make


@dataclass(frozen=True)
class PreparedSession:
    directory: Path
    message: str
    inputs: SessionInputs
    executor: SandboxExecutor


class SessionCall:
    """Translate one slot with one code session."""

    def __init__(
        self,
        backend: ChatBackend,
        options: Mapping[str, Any],
        template: PromptTemplate,
        settings: SessionSettings,
        material: PolicyMaterial,
        executors: ExecutorFactory,
        limits: AnswerLimits,
        preflight: Preflight = run_preflight,
    ) -> None:
        if template.turns is None:
            raise ValueError("a code session needs the prompt's turns")
        self._backend = backend
        self._options = dict(options)
        self._template = template
        self._turns = ConfiguredTurns(template.turns)
        self._settings = settings
        self._material = material
        self._executors = executors
        self._limits = limits
        self._preflight = preflight

    def first_request(self, message: str) -> dict[str, Any]:
        """The body of the session's first request (the journaled payload)."""
        return {"input": [user_item(message)], **self._options}

    def prepare(self, values: Mapping[str, object], slot: SlotRef) -> tuple[dict[str, Any], PreparedSession]:
        if slot.session_dir is None:
            raise ValueError("a code session needs a session directory")
        directory = slot.session_dir
        _, message = self._template.render(values)
        past_steps = self._settings.past_steps
        inputs = write_session_inputs(
            self._material, slot.target.target_id, slot.step, past_steps, directory / "inputs"
        )
        executor = self._executors(directory / "work", inputs.bundle.weights)
        result = self._preflight(executor, [*self._settings.hidden, inputs.bundle.path])
        record = {"inputs_manifest_sha256": inputs.bundle.manifest_sha256, "sandbox_preflight": result.to_dict()}
        (directory / "preflight.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if not result.ok:
            raise SandboxRefused(f"sandbox preflight failed for {directory.name}")
        (directory / "message.txt").write_text(message, encoding="utf-8")
        return self.first_request(message), PreparedSession(directory, message, inputs, executor)

    def run(self, prepared: PreparedSession) -> CallResult:
        log = RunLog(prepared.directory)
        state = run_session(
            self._backend, prepared.executor, prepared.message, self._settings.limits, log, self._turns, parse_reply
        )
        verify_bundle(prepared.inputs.bundle.path, prepared.inputs.bundle.manifest_sha256)
        write_session_audit(prepared.directory, self._template.version)
        return replace(_result(prepared.directory, state), observed_inputs=observed_inputs(prepared))

    def answer(self, result: CallResult) -> tuple[IntentTranslation | None, str | None]:
        """(translation, None) or (None, why there is no valid answer)."""
        if result.answer_text is None:
            return None, result.ended_without_answer
        try:
            answer = parse_answer(result.answer_text, self._limits)
        except IntentAnswerError as exc:
            return None, f"{INVALID_ANSWER_PREFIX}{exc}"
        return IntentTranslation(answer.intent, answer.evidence), None


def observed_inputs(prepared: PreparedSession) -> dict[str, Any]:
    """The bundle the sandbox read, as written to disk: its manifest's SHA-256 and the window it states."""
    bundle = prepared.inputs.bundle
    manifest = json.loads((bundle.path / MANIFEST_FILE).read_text(encoding="utf-8"))
    return {
        "session_dir": prepared.directory.name,
        "inputs_manifest_sha256": bundle.manifest_sha256,
        "window": manifest["window"],
    }


def _result(directory: Path, state: SessionState) -> CallResult:
    usage = {**asdict(state), "total_tokens": state.total_tokens}
    raw = {"session_dir": directory.name, "stop_reason": state.stop_reason, "turns": state.turn}
    if state.stop_reason != STOP_FINAL_ANSWER:
        reason = f"session ended without a final answer ({state.stop_reason})"
        return CallResult(raw, None, None, None, usage, ended_without_answer=reason)
    text = (directory / FINAL_ANSWER).read_text(encoding="utf-8")
    return CallResult(raw, text, None, None, usage)
