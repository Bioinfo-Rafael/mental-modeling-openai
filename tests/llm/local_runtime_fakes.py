"""Offline fakes for the local runtime parts (``vllm_chat``, ``vllm_server``).

No vLLM, GPU, network, SLURM or real key is used: the launcher gets a fake process, clock and
signal function; the chat client gets a scripted SDK-shaped client or a fake ``openai`` module.
"""

from __future__ import annotations

import copy
import subprocess
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FAKE_PID = 4242
CHAT_MODEL = "local-chat"


# --- launcher ---------------------------------------------------------------------------------------


class FakeProcess:
    """A ``subprocess.Popen`` stand-in and its process group. The leader exits on the signals in
    ``exits_on`` (code ``-signal``) or by itself after ``exit_after_polls`` polls (code ``exit_code``).
    With ``workers_exit_on`` set, worker processes share its group and outlive it until they get
    one of those signals."""

    def __init__(
        self,
        *,
        exits_on: frozenset[int] = frozenset(),
        exit_after_polls: int | None = None,
        exit_code: int = 1,
        wait_raises: BaseException | None = None,
        workers_exit_on: frozenset[int] | None = None,
    ) -> None:
        self.pid = FAKE_PID
        self.returncode: int | None = None
        self.exits_on = exits_on
        self.workers_alive = workers_exit_on is not None
        self.workers_exit_on = workers_exit_on or frozenset()
        self.signals: list[int] = []
        self.wait_timeouts: list[float | None] = []
        self._polls_left = exit_after_polls
        self._exit_code = exit_code
        self._wait_raises = wait_raises

    def poll(self) -> int | None:
        if self.returncode is None and self._polls_left is not None:
            if self._polls_left <= 0:
                self.returncode = self._exit_code
            self._polls_left -= 1
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self._wait_raises is not None:
            error, self._wait_raises = self._wait_raises, None
            raise error
        if self.poll() is not None:  # like Popen.wait, waiting also reaps a child that exited by itself
            return self.returncode
        raise subprocess.TimeoutExpired("vllm", timeout if timeout is not None else 0.0)

    def send_signal(self, sig: int) -> None:
        self.receive(sig)

    def receive(self, sig: int) -> None:
        self.signals.append(sig)
        if sig in self.exits_on and self.returncode is None:
            self.returncode = -sig

    def receive_group(self, sig: int) -> None:
        if self.returncode is None:
            self.receive(sig)
        if sig in self.workers_exit_on:
            self.workers_alive = False

    @property
    def group_alive(self) -> bool:
        return self.returncode is None or self.workers_alive


@dataclass
class RecordingPopen:
    """Popen factory that records every call and returns ``process``."""

    process: FakeProcess
    calls: list[tuple[list[str], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((list(argv), dict(kwargs)))
        return self.process


@dataclass
class GroupSignals:
    """``os.killpg`` stand-in: delivers to ``process``'s group and records ``(pgid, signal)``.
    Existence probes (signal 0) are counted in ``probes``, not in ``calls``. An empty group, or
    ``missing_group``, raises ``ProcessLookupError`` like the kernel."""

    process: FakeProcess
    calls: list[tuple[int, int]] = field(default_factory=list)
    probes: int = 0
    missing_group: bool = False

    def __call__(self, pgid: int, sig: int) -> None:
        if sig == 0:
            self.probes += 1
        else:
            self.calls.append((pgid, sig))
        if self.missing_group or not self.process.group_alive:
            raise ProcessLookupError(pgid)
        if sig != 0:
            self.process.receive_group(sig)


@dataclass
class FakeClock:
    """Monotonic clock whose ``sleep`` advances time instantly."""

    now: float = 0.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --- chat client --------------------------------------------------------------------------------------


class ScriptedChatClient:
    """Stands in for ``openai.OpenAI``: ``chat.completions.create`` and ``models.list`` pop outcomes."""

    def __init__(self, outcomes: list[object]) -> None:
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
        self.models = types.SimpleNamespace(list=self._list)
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.model_lists = 0

    def _next(self) -> object:
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def _create(self, **kwargs: Any) -> object:
        self.requests.append(copy.deepcopy(kwargs))
        return self._next()

    def _list(self) -> object:
        self.model_lists += 1
        return self._next()


def chat_body(
    content: str | None = '{"ok": true}',
    *,
    finish_reason: str | None = "stop",
    refusal: str | None = None,
    usage: dict[str, Any] | None = None,
    choices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A ``/v1/chat/completions`` response as ``model_dump(mode="json")`` returns it."""
    message = {"role": "assistant", "content": content, "refusal": refusal, "tool_calls": []}
    default_choices = [{"index": 0, "message": message, "finish_reason": finish_reason, "logprobs": None}]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": CHAT_MODEL,
        "choices": choices if choices is not None else default_choices,
        "usage": usage if usage is not None else {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57},
    }


class FakeOmit:
    """Mimics ``openai.Omit`` (a sentinel that removes a default header)."""


class RecordingChatOpenAI:
    """Replaces ``openai.OpenAI`` in a fake ``openai`` module; records constructor arguments."""

    instances: list[RecordingChatOpenAI] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        RecordingChatOpenAI.instances.append(self)


def fake_chat_openai_module() -> types.ModuleType:
    """A stand-in ``openai`` module with ``OpenAI`` and ``Omit`` (the real SDK import costs seconds)."""
    RecordingChatOpenAI.instances = []
    module = types.ModuleType("openai")
    module.OpenAI = RecordingChatOpenAI  # type: ignore[attr-defined]
    module.Omit = FakeOmit  # type: ignore[attr-defined]
    return module


def make_runtime_dir(base: Path) -> Path:
    """A fake separate vLLM env (an executable ``vllm-env/bin/vllm`` stub) and an ``hf`` cache dir."""
    executable = base / "vllm-env" / "bin" / "vllm"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    (base / "hf").mkdir()
    return base
