"""Offline helpers for the judge tests: a synthetic schema-3 translation run and a scripted SDK client.

The run is written directly (no translator involved) over the committed MHI episode, so the judge
renders real trajectory windows. Nothing here reaches a network or a paid API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from translate.episode import load_episode

ROOT = Path(__file__).resolve().parents[2]
EPISODE_CSV = ROOT / "data" / "mhi" / "processed" / "mhi_v1.3_ic100601_win.csv"
JUDGE_CONFIG = ROOT / "configs" / "evaluation" / "judge.yaml"
SCENARIO_ID, EPISODE_ID = "test-scenario", "mhi_v1.3_ic100601_win"
A0100 = {"kind": "aircraft", "id": "A0100"}
TEAM_A = {"kind": "team", "id": "A"}


def event(step: int, kinds: tuple[str, ...] = ("start",), target: dict | None = None, **input_fields: Any) -> dict:
    """One translation_event of today's schema 3 (the fields the judge reads, plus noise it must ignore)."""
    target = target or A0100
    aircraft = ["A0100"] if target["kind"] == "aircraft" else ["A0100", "A0200"]
    reasons = [
        {"kind": k, "source": "preset", "reference_id": f"preset-step-{step}", "probability": None} for k in kinds
    ]
    payload = {"target_aircraft": aircraft, "instructions_in_effect": [], "alarms": [], "question": None}
    payload = {**payload, "previous_translation": None, "translator_inputs": ["trajectory"], **input_fields}
    return {
        "schema_version": 3,
        "record_type": "translation_event",
        "scenario_id": SCENARIO_ID,
        "episode_id": EPISODE_ID,
        "step": step,
        "time_s": round(step * 0.2, 1),
        "target": target,
        "event_id": f"{SCENARIO_ID}:{target['kind']}:{target['id']}:{step:06d}",
        "trigger_reasons": reasons,
        "input": payload,
        "output": {"intent": f"[TEST] intent at {step}", "evidence": [f"[TEST] evidence at {step}"]},
        "model": {"model": "gpt-6.1-terra"},
        "request_sha256": "0" * 64,
        "processing_s": 1.5,
    }


def request_line(record: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    """The journaled translation request of a record: by default a causal window line at its step."""
    step = record["step"]
    text = text if text is not None else json.dumps({"step": step, "relative_step": 0}, separators=(",", ":"))
    body = {"input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}]}
    return {
        "call_id": record["event_id"],
        "layer": "translation",
        "step": step,
        "target": record["target"],
        "body": body,
    }


def write_run(directory: Path, records: list[dict], requests: list[dict] | None = None) -> Path:
    """A translation run directory: run.json (episode facts), events.jsonl, requests.jsonl."""
    directory.mkdir(parents=True)
    loaded = load_episode(EPISODE_CSV)
    episode = {"csv_path": str(EPISODE_CSV), "csv_sha256": loaded.csv_sha256}
    run_json = {"inputs": {"episode": {**episode, "last_complete_step": loaded.last_step}}}
    (directory / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    lines = [{**record, "sequence": index} for index, record in enumerate(records, start=1)]
    (directory / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    journal = requests if requests is not None else [request_line(r) for r in records if "event_id" in r]
    (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in journal), encoding="utf-8")
    return directory


def judge_config(tmp_path: Path, **changes: Any) -> Path:
    """The shipped judge.yaml with its output under tmp_path (and top-level keys replaced)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(JUDGE_CONFIG.read_text(encoding="utf-8"))
    path = tmp_path / "judge.yaml"
    path.write_text(yaml.safe_dump({**raw, "output": str(tmp_path / "evaluation"), **changes}), encoding="utf-8")
    return path


def asked_criteria(body: dict[str, Any]) -> list[str]:
    schema = body["text"]["format"]["schema"]
    return schema["properties"]["criteria"]["items"]["properties"]["criterion_id"]["enum"]


class FakeSDK:
    """Stands in for the OpenAI SDK client: answers every asked criterion with ``judgement``.

    ``outcomes`` overrides call by call: an exception is raised, a string is the output text.
    """

    def __init__(self, judgement: str = "4", outcomes: list[object] | None = None) -> None:
        self.responses = self
        self.bodies: list[dict[str, Any]] = []
        self.judgement, self.outcomes = judgement, list(outcomes or [])

    def create(self, **body: Any) -> dict[str, Any]:
        self.bodies.append(body)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, BaseException):
            raise outcome
        items = [
            {"criterion_id": c, "rationale": "[FAKE] レベル4", "judgement": self.judgement}
            for c in asked_criteria(body)
        ]
        text = outcome if isinstance(outcome, str) else json.dumps({"criteria": items}, ensure_ascii=False)
        output = [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
        return {"id": f"resp_{len(self.bodies)}", "model": body["model"], "status": "completed", "output": output}


class Factory:
    """Stands in for ``make_openai_client``: counts calls and returns the fake SDK client."""

    def __init__(self, sdk: FakeSDK | None = None) -> None:
        self.sdk, self.calls = sdk or FakeSDK(), 0

    def __call__(self, **_: Any) -> FakeSDK:
        self.calls += 1
        return self.sdk
