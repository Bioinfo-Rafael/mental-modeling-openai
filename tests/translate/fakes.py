"""Offline helpers for the translate tests: scenario and config files, a fake backend and a fake detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from translate import cli
from translate.backends import Reply
from translate.backends.openai import OpenAIBackend
from translate.detector import Answer

ROOT = Path(__file__).resolve().parents[2]
EPISODE_CSV = ROOT / "data" / "mhi" / "processed" / "mhi_v1.3_ic100601_win.csv"  # B0200 is down from step 136
ONLINE, PRESET = ROOT / "configs" / "translate" / "online.yaml", ROOT / "configs" / "translate" / "preset.yaml"
INSTRUCTION = {"id": "i1", "step": 0, "kind": "instruction", "to": ["aircraft:A0100"], "text": "[TEST] hold the line"}
QUESTION = {"id": "q1", "step": 10, "kind": "question", "to": ["aircraft:A0100"], "text": "[TEST] why turn?"}
PROMPT = {
    "system": "Answer in at most {max_intent_chars} characters.",
    "user": "{step} {target}\nI: {instructions}\nA: {alarms}\nQ: {question}\nP: {previous_translation}\nT: {trajectory}",
}
INPUTS = {"instructions": {}, "alarms": {}, "question": {}, "previous_translation": {}, "trajectory": {"past_steps": 2}}


def write_yaml(path: Path, raw: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def scenario(tmp_path: Path, messages: list[dict] | None = None, targets: list[str] | None = None) -> Path:
    raw = {
        "id": "test-01",
        "episode": str(EPISODE_CSV),
        "targets": targets or ["aircraft:A0100"],
        "messages": messages or [],
    }
    return write_yaml(tmp_path / "scenario.yaml", raw)


def config(tmp_path: Path, scenario_path: Path, change: dict | None = None, **changes: Any) -> Path:
    """A small preset config (2-step windows) with its output under tmp_path."""
    raw = {
        "scenario": str(scenario_path),
        "change": change or {"source": "preset", "steps": []},
        "inputs": INPUTS,
        "prompt": PROMPT,
        "model": {"name": "test-model"},
        "output": str(tmp_path / "out"),
        **changes,
    }
    return write_yaml(tmp_path / "config.yaml", raw)


def clef(**changes: Any) -> dict[str, Any]:
    """The online config's clef_flash change source with a 2-step detector window."""
    change = yaml.safe_load(ONLINE.read_text(encoding="utf-8"))["change"]
    trajectory = {**change["inputs"]["trajectory"], "past_steps": 2}
    return {**change, "inputs": {**change["inputs"], "trajectory": trajectory}, **changes}


class FakeBackend:
    """One JSON answer per call (``answers`` overrides call by call: a string or an exception)."""

    request_body = staticmethod(OpenAIBackend.request_body)

    def __init__(self, answers: list[object] | None = None, journal: Path | None = None) -> None:
        self.bodies: list[dict] = []
        self.answers, self.journal = list(answers or []), journal
        self.journal_lines_at_send: list[int] = []

    def send(self, body: dict) -> Reply:
        self.bodies.append(body)
        if self.journal is not None:
            self.journal_lines_at_send.append(len(self.journal.read_text().splitlines()))
        answer = self.answers.pop(0) if self.answers else None
        if isinstance(answer, BaseException):
            raise answer
        text = (
            answer
            if isinstance(answer, str)
            else json.dumps({"intent": f"intent {len(self.bodies)}", "evidence": ["e"]})
        )
        return Reply(
            text, "test-model", f"resp_{len(self.bodies)}", {"input_tokens": 1}, {"id": f"resp_{len(self.bodies)}"}
        )


class FakeDetector:
    """Answers ``change`` with ``p`` at the (target, step) pairs in ``changes``, ``none`` elsewhere."""

    def __init__(self, changes: dict[tuple[str, int], float] | None = None) -> None:
        self.changes, self.bodies, self.healthy = changes or {}, [], 0

    def health(self) -> None:
        self.healthy += 1

    def decide(self, body: dict) -> Answer:
        self.bodies.append(body)
        step = int(body["state"].split("Step ")[1].split(".")[0])
        target = body["state"].split("Target: ")[1].split(" (")[0].replace(" ", ":")
        p = self.changes.get((target, step), 0.1)
        choice = "change" if p >= 0.5 else "none"
        answer = {"type": "choice", "choice": choice, "probabilities": {"none": round(1 - p, 3), "change": p}}
        usage = {"input_tokens": 10, "output_tokens": 1}
        return Answer({"model": "clef", "answers": {"point_kind": answer}, "usage": usage})


def run(
    config_path: Path, *, paid: bool = True, backend: FakeBackend | None = None, detector: FakeDetector | None = None
) -> int:
    flags = ["--execute", "--confirm-paid-api"] if paid else []
    return cli.main(
        ["--config", str(config_path), "--run-name", "run", *flags],
        backend_factory=lambda model: backend or FakeBackend(),
        detector_factory=lambda change: detector or FakeDetector(),
    )


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def events(tmp_path: Path, record_type: str = "translation_event") -> list[dict]:
    return [r for r in lines(tmp_path / "out" / "run" / "events.jsonl") if r["record_type"] == record_type]
