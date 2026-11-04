"""Cue records of the intent-comparison contract (``outputs/intent_compare/<run_id>/cues_B.jsonl``).

One JSON object per line and per (step, aircraft); keys are fixed by
``configs/intent_compare/compare.yaml``. A skipped target carries ``"skipped": "<reason>"`` and
no ``intent`` / ``evidence``. The file is created exclusively and every line is fsync'ed, so a
crash leaves the translations finished so far.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading.intent_protocol import IntentAnswer

METHOD = "B"
CUES_FILE = f"cues_{METHOD}.jsonl"


@dataclass(frozen=True)
class CueContext:
    seed: int
    model: str
    reasoning_effort: str
    step_seconds: float


def _base(context: CueContext, step: int, aircraft: str, source_run: str) -> dict:
    return {
        "seed": context.seed,
        "step": step,
        "t_s": round(step * context.step_seconds, 6),
        "aircraft": aircraft,
        "method": METHOD,
        "model": context.model,
        "reasoning_effort": context.reasoning_effort,
        "source_run": source_run,
    }


def translated_cue(context: CueContext, step: int, aircraft: str, source_run: str, answer: IntentAnswer) -> dict:
    return {**_base(context, step, aircraft, source_run), "intent": answer.intent, "evidence": list(answer.evidence)}


def skipped_cue(context: CueContext, step: int, aircraft: str, source_run: str, reason: str) -> dict:
    return {**_base(context, step, aircraft, source_run), "skipped": reason}


class CueWriter:
    """Append-only ``cues_B.jsonl``. A new writer refuses an existing file; ``resume=True`` requires one."""

    def __init__(self, path: Path, resume: bool = False) -> None:
        if resume and not path.is_file():
            raise FileNotFoundError(f"{path} does not exist; nothing to resume")
        if not resume:
            derived_path(path).parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8"):
                pass
        self.path = path

    def append(self, cue: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cue, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_cues(path: Path) -> list[dict]:
    """The cue lines of an existing file (a resume reads what an interrupted run wrote)."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
