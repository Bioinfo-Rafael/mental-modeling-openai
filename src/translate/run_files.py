"""The files of a run directory: JSONL journals written before each send, and the cues projection.

``events.jsonl`` is the authoritative record (the judges read it). ``cues.jsonl`` is derived from
it after the run, one line per translation slot, for the side-by-side viewer: the answer, or
``skipped`` with the reason (target down, or an invalid answer that the run skipped).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from translate.events import RECORD_TRANSLATION_EVENT, RECORD_TRIGGER_SUPPRESSED
from translate.output_safety import derived_path

EVENTS_FILE = "events.jsonl"
REQUESTS_FILE = "requests.jsonl"
RESPONSES_FILE = "responses.jsonl"
CUES_FILE = "cues.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    derived_path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def finite_json(value: Any) -> Any:
    """Standard JSON only: a non-finite float (a lax server's NaN) becomes the string "NaN" / "Infinity"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class JsonlRunSink:
    """Append-only JSONL files of one run. Each line is flushed and fsynced before the call proceeds.

    A lone surrogate (e.g. ``"\\ud800"`` decoded from a response body) cannot be encoded as UTF-8; it
    is written as the JSON escape ``\\ud800`` instead (``backslashreplace``). The line stays valid JSON
    and reads back as the same text. So a fully received response is always journaled and never
    turns ``uncertain``.
    """

    def __init__(self, run_dir: Path) -> None:
        names = (EVENTS_FILE, REQUESTS_FILE, RESPONSES_FILE)
        self._files: dict[str, TextIO] = {
            name: (run_dir / name).open("x", encoding="utf-8", errors="backslashreplace") for name in names
        }

    def _write(self, name: str, record: dict[str, Any]) -> None:
        handle = self._files[name]
        handle.write(json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_event(self, record: dict[str, Any]) -> None:
        self._write(EVENTS_FILE, record)

    def write_request(self, record: dict[str, Any]) -> None:
        self._write(REQUESTS_FILE, {**record, "written_at": utc_now()})

    def write_response(self, record: dict[str, Any]) -> None:
        self._write(RESPONSES_FILE, {**record, "written_at": utc_now()})

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()

    def __enter__(self) -> JsonlRunSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _cue(record: dict[str, Any]) -> dict[str, Any]:
    target = record["target"]
    common = {
        "step": record["step"],
        "time_s": record["time_s"],
        "target": target,
        "aircraft": target["id"],
        "trigger": record["primary_trigger"],
    }
    if record["record_type"] == RECORD_TRIGGER_SUPPRESSED:
        return {**common, "skipped": f"target_down: alive == 0 since step {record['down_step']}"}
    model = record["model"]
    common = {
        **common,
        "model": model["model"],
        "reasoning_effort": model["reasoning_effort"],
        "event_id": record["event_id"],
    }
    if "answer_problem" in record:
        return {**common, "skipped": record["answer_problem"]}
    return {**common, "intent": record["output"]["intent"], "evidence": record["output"]["evidence"]}


def cues_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One cue per translation slot, in event order."""
    kinds = (RECORD_TRANSLATION_EVENT, RECORD_TRIGGER_SUPPRESSED)
    return [_cue(record) for record in events if record["record_type"] in kinds]


def write_cues(run_dir: Path) -> Path:
    """Derive ``cues.jsonl`` from ``events.jsonl`` (rewritten whole; it is a projection)."""
    events_path = run_dir / EVENTS_FILE
    lines = events_path.read_text(encoding="utf-8").splitlines() if events_path.exists() else []
    cues = cues_from_events([json.loads(line) for line in lines if line.strip()])
    target = derived_path(run_dir / CUES_FILE)
    target.write_text("".join(json.dumps(cue, ensure_ascii=False) + "\n" for cue in cues), encoding="utf-8")
    return target
