"""The Exp-3-style loop: send -> parse -> run in the sandbox -> report the output -> repeat.

Records (append + fsync per turn, so a crash leaves a readable partial run):
``transcript.jsonl`` (one line per model response), ``tool_calls.jsonl`` (one line per code run,
blog fields ``t``, ``s``, ``run``, ``code``, ``out`` plus ``err``, ``exit``, ``timed_out``,
``truncated``), ``state.json`` (replaced atomically) and ``final_answer.md``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from weightreading.backends import BackendReply, ChatBackend, user_item
from weightreading.prompt import budget_exhausted_message, protocol_error_message, run_result_message
from weightreading.protocol import KIND_ERROR, KIND_FINAL, parse_reply
from weightreading.sandbox import ExecResult

TRANSCRIPT = "transcript.jsonl"
TOOL_CALLS = "tool_calls.jsonl"
STATE = "state.json"
FINAL_ANSWER = "final_answer.md"


class Executor(Protocol):
    def run(self, code: str, run_index: int) -> ExecResult: ...


@dataclass(frozen=True)
class SessionLimits:
    max_runs: int
    max_total_tokens: int
    max_protocol_errors: int


@dataclass(frozen=True)
class SessionState:
    turn: int = 0
    runs: int = 0
    protocol_errors: int = 0
    budget_notice_sent: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    done: bool = False
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    """Durable, append-only records of one session in ``directory``."""

    def __init__(self, directory: Path, now: Callable[[], str] = utc_now) -> None:
        self.directory = directory
        self.now = now

    def _append(self, name: str, record: dict) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def response(self, state: SessionState, reply: BackendReply) -> None:
        self._append(
            TRANSCRIPT,
            {
                "t": self.now(),
                "turn": state.turn,
                "request": reply.request_record,
                "response_id": reply.response_id,
                "status": reply.status,
                "usage": {**asdict(reply.usage), "total_tokens": reply.usage.total_tokens},
                "retry_count": len(reply.retries),
                "retries": list(reply.retries),
                "text": reply.text,
                "output_items": list(reply.output_items),
            },
        )

    def tool_call(self, run_index: int, code: str, result: ExecResult) -> None:
        self._append(
            TOOL_CALLS,
            {
                "t": self.now(),
                "s": result.duration_s,
                "run": run_index,
                "code": code,
                "out": result.stdout,
                "err": result.stderr_tail,
                "exit": result.exit_code,
                "timed_out": result.timed_out,
                "truncated": result.stdout_truncated,
            },
        )

    def state(self, state: SessionState) -> None:
        payload = {**asdict(state), "total_tokens": state.total_tokens, "updated_utc": self.now()}
        temporary = self.directory / f".{STATE}.partial"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.directory / STATE)

    def final_answer(self, text: str) -> None:
        (self.directory / FINAL_ANSWER).write_text(text, encoding="utf-8")


def _with_usage(state: SessionState, reply: BackendReply) -> SessionState:
    return replace(
        state,
        turn=state.turn + 1,
        input_tokens=state.input_tokens + reply.usage.input_tokens,
        output_tokens=state.output_tokens + reply.usage.output_tokens,
        reasoning_tokens=state.reasoning_tokens + reply.usage.reasoning_tokens,
    )


def _stop(state: SessionState, reason: str) -> SessionState:
    return replace(state, done=True, stop_reason=reason)


def _reply_items(reply: BackendReply) -> tuple[dict, ...]:
    if reply.output_items:
        return reply.output_items
    return ({"role": "assistant", "content": reply.text},)


def run_session(
    backend: ChatBackend, executor: Executor, first_message: str, limits: SessionLimits, log: RunLog
) -> SessionState:
    """Drive one session to a final answer or a cap. Backend/executor exceptions are recorded, then re-raised."""
    state = SessionState()
    history: tuple[dict, ...] = (user_item(first_message),)
    log.state(state)
    try:
        while not state.done:
            state, history = _turn(backend, executor, history, state, limits, log)
            log.state(state)
    except Exception as exc:
        log.state(_stop(state, f"error: {type(exc).__name__}"))
        raise
    return state


def _turn(
    backend: ChatBackend,
    executor: Executor,
    history: tuple[dict, ...],
    state: SessionState,
    limits: SessionLimits,
    log: RunLog,
) -> tuple[SessionState, tuple[dict, ...]]:
    reply = backend.respond(history)
    state = _with_usage(state, reply)
    log.response(state, reply)
    history = history + _reply_items(reply)
    if reply.status != "completed":
        return _stop(state, f"response_{reply.status}"), history
    parsed = parse_reply(reply.text)
    if parsed.kind == KIND_FINAL:
        log.final_answer(reply.text)
        return _stop(state, "final_answer"), history
    if state.total_tokens >= limits.max_total_tokens:
        return _stop(state, "max_total_tokens"), history
    if parsed.kind == KIND_ERROR:
        state = replace(state, protocol_errors=state.protocol_errors + 1)
        if state.protocol_errors >= limits.max_protocol_errors:
            return _stop(state, "protocol_errors"), history
        message = protocol_error_message(parsed.problems, limits.max_runs - state.runs)
        return state, history + (user_item(message),)
    state = replace(state, protocol_errors=0)
    if state.runs >= limits.max_runs:
        if state.budget_notice_sent:
            return _stop(state, "max_runs"), history
        return replace(state, budget_notice_sent=True), history + (user_item(budget_exhausted_message()),)
    state = replace(state, runs=state.runs + 1)
    result = executor.run(parsed.code, state.runs)
    log.tool_call(state.runs, parsed.code, result)
    message = run_result_message(
        state.runs,
        limits.max_runs,
        result.stdout,
        result.stderr_tail,
        result.exit_code,
        result.timed_out,
        result.stdout_truncated,
        parsed.block_count,
    )
    return state, history + (user_item(message),)
