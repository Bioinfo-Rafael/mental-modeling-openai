"""Session loop with a fake backend and executor: caps, protocol errors, durable logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from translate import paths
from translate.config import load_config
from translate.prompts import ConfiguredTurns
from translate.sandbox.executor import ExecResult
from translate.sandbox.session import RunLog, SessionLimits, run_session
from translate.session_backend import BackendReply, OpenAIResponsesBackend, Usage

LIMITS = SessionLimits(max_runs=3, max_total_tokens=10_000, max_protocol_errors=2)
TURNS = ConfiguredTurns(load_config(paths.CONFIGS / "translate" / "seed100601-policy.yaml").prompt.turns)
CODE = "```python\nprint(len(W))\n```"
FINAL = 'Analysis complete.\n```json\n{"intent": "It closes on the target.", "evidence": ["A shows closing"]}\n```'
UNABLE = "I'm unable to analyze the network without the requested weight-summary execution results."


@dataclass
class FakeBackend:
    replies: list[str]
    tokens_per_turn: int = 100
    model: str = "fake-model"
    seen: list[tuple[dict, ...]] = field(default_factory=list)

    def respond(self, history: tuple[dict, ...]) -> BackendReply:
        self.seen.append(history)
        if not self.replies:
            raise RuntimeError("backend exploded")
        text = self.replies.pop(0)
        return BackendReply(
            text=text,
            output_items=(
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]},
            ),
            usage=Usage(self.tokens_per_turn, self.tokens_per_turn, 10, 0),
            status="completed",
            response_id=f"resp_{len(self.seen)}",
            request_record={"model": self.model},
        )


@dataclass
class FakeExecutor:
    calls: list[tuple[str, int]] = field(default_factory=list)

    def run(self, code: str, run_index: int) -> ExecResult:
        self.calls.append((code, run_index))
        return ExecResult(f"out {run_index}", False, "", 0, False, 0.01)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_session_code_then_final_logs_everything(tmp_path: Path) -> None:
    backend, executor = FakeBackend([CODE, FINAL]), FakeExecutor()

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert (state.stop_reason, state.runs, state.turn) == ("final_answer", 1, 2)
    assert executor.calls == [("print(len(W))\n", 1)]
    assert [r["run"] for r in _jsonl(tmp_path / "tool_calls.jsonl")] == [1]
    assert len(_jsonl(tmp_path / "transcript.jsonl")) == 2
    assert json.loads((tmp_path / "state.json").read_text())["done"] is True
    assert "It closes on the target." in (tmp_path / "final_answer.md").read_text()
    assert "OUTPUT of run 1 of at most 3" in json.dumps(backend.seen[1][-1])


def test_run_session_over_run_budget_sends_notice_then_stops(tmp_path: Path) -> None:
    backend, executor = FakeBackend([CODE] * 5), FakeExecutor()

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert state.stop_reason == "max_runs"
    assert len(executor.calls) == LIMITS.max_runs
    assert "budget is used up" in json.dumps(backend.seen[-1][-1])


def test_run_session_over_token_budget_stops(tmp_path: Path) -> None:
    backend = FakeBackend([CODE] * 5, tokens_per_turn=6_000)

    state = run_session(backend, FakeExecutor(), "first", LIMITS, RunLog(tmp_path), TURNS)

    assert state.stop_reason == "max_total_tokens"
    assert state.total_tokens >= LIMITS.max_total_tokens


def test_run_session_repeated_protocol_errors_stop(tmp_path: Path) -> None:
    backend, executor = FakeBackend([UNABLE, UNABLE, FINAL]), FakeExecutor()

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert (state.stop_reason, state.protocol_errors, state.turn) == ("protocol_errors", 2, 2)
    assert executor.calls == []
    assert not (tmp_path / "final_answer.md").exists()


def test_run_session_multi_block_reply_runs_merged_code_once(tmp_path: Path) -> None:
    two_blocks = "```python\na = 1\n```\ntext\n```python\nprint(a)\n```"
    backend, executor = FakeBackend([two_blocks, FINAL]), FakeExecutor()
    merged = "a = 1\n\n# --- block 2 ---\nprint(a)\n"

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert (state.stop_reason, state.runs, state.protocol_errors) == ("final_answer", 1, 0)
    assert executor.calls == [(merged, 1)]
    assert [r["code"] for r in _jsonl(tmp_path / "tool_calls.jsonl")] == [merged]
    feedback = json.dumps(backend.seen[1][-1])
    assert "OUTPUT of run 1 of at most 3" in feedback
    assert "2 ```python blocks" in feedback and "merged into this one run" in feedback


def test_run_session_single_block_reply_has_no_merge_note(tmp_path: Path) -> None:
    backend = FakeBackend([CODE, FINAL])

    run_session(backend, FakeExecutor(), "first", LIMITS, RunLog(tmp_path), TURNS)

    assert "merged" not in json.dumps(backend.seen[1][-1])


def test_run_session_reply_without_code_or_final_is_protocol_error(tmp_path: Path) -> None:
    backend, executor = FakeBackend([UNABLE, CODE, FINAL]), FakeExecutor()

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert (state.stop_reason, state.runs, state.turn) == ("final_answer", 1, 3)
    assert state.protocol_errors == 0
    error_message = json.dumps(backend.seen[1][-1])
    assert "neither a ```python code block nor a well-formed final answer" in error_message
    assert "exactly one ```json block" in error_message
    assert "Runs remaining: 3." in error_message


def test_run_session_final_answer_with_malformed_json_ends_the_session_without_a_retry(tmp_path: Path) -> None:
    malformed = FINAL.replace('{"intent": "It closes on the target.", "evidence": ["A shows closing"]}', "{not json}")
    backend = FakeBackend([malformed, FINAL])

    state = run_session(backend, FakeExecutor(), "first", LIMITS, RunLog(tmp_path), TURNS)

    assert (state.stop_reason, state.turn) == ("final_answer", 1)
    assert len(backend.seen) == 1
    assert (tmp_path / "final_answer.md").read_text() == malformed


def test_run_session_evidence_replay_does_not_stop_as_final_answer_after_zero_runs(tmp_path: Path) -> None:
    # Real paid run 20261103T205222Z_net-35cbb0ab_shown: two python blocks, then "unable to analyze".
    two_blocks = "```python\nprint(1)\n```\n```python\nprint(2)\n```"
    backend, executor = FakeBackend([two_blocks, UNABLE, UNABLE]), FakeExecutor()

    state = run_session(backend, executor, "first", LIMITS, RunLog(tmp_path), TURNS)

    assert state.stop_reason == "protocol_errors"
    assert state.runs == 1
    assert len(executor.calls) == 1
    assert not (tmp_path / "final_answer.md").exists()


def test_run_session_backend_crash_leaves_readable_partial_logs(tmp_path: Path) -> None:
    backend = FakeBackend([CODE])

    with pytest.raises(RuntimeError, match="exploded"):
        run_session(backend, FakeExecutor(), "first", LIMITS, RunLog(tmp_path), TURNS)

    assert json.loads((tmp_path / "state.json").read_text())["stop_reason"] == "error: RuntimeError"
    assert len(_jsonl(tmp_path / "transcript.jsonl")) == 1
    assert len(_jsonl(tmp_path / "tool_calls.jsonl")) == 1


def test_run_session_incomplete_response_stops(tmp_path: Path) -> None:
    class Truncating(FakeBackend):
        def respond(self, history: tuple[dict, ...]) -> BackendReply:
            reply = super().respond(history)
            return BackendReply(reply.text, reply.output_items, reply.usage, "incomplete", None, {})

    state = run_session(Truncating([CODE]), FakeExecutor(), "first", LIMITS, RunLog(tmp_path), TURNS)

    assert state.stop_reason == "response_incomplete"


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs: object) -> object:
        self.kwargs = kwargs

        class Response:
            id = "resp_1"
            status = "completed"
            output_text = "done"
            output = [{"type": "reasoning", "encrypted_content": "xyz"}]

            class usage:  # noqa: N801 - mirrors the SDK attribute name
                input_tokens, output_tokens = 11, 7
                output_tokens_details = type("D", (), {"reasoning_tokens": 5})()
                input_tokens_details = type("D", (), {"cached_tokens": 3})()

        return Response()


def test_openai_backend_sends_no_tools_and_stateless_replay() -> None:
    client = type("Client", (), {})()
    client.responses = _FakeResponses()
    backend = OpenAIResponsesBackend(client, "model-x", "high", 1234)

    reply = backend.respond(({"role": "user", "content": "hi"},))

    sent = client.responses.kwargs
    assert "tools" not in sent
    assert sent["store"] is False
    assert sent["include"] == ["reasoning.encrypted_content"]
    assert sent["model"] == "model-x" and sent["max_output_tokens"] == 1234
    assert reply.usage == Usage(11, 7, 5, 3)
    assert reply.output_items == ({"type": "reasoning", "encrypted_content": "xyz"},)
    assert "api_key" not in json.dumps(reply.request_record)
