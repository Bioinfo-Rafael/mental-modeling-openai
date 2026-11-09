"""Judge material: the translation run as the judge reads it, and one event's prompt text.

Every event gets the same human-visible material whatever the translator read: the instructions in
effect, the alarms, the question, the previous translation, the trajectory window up to t (rendered
here from the episode CSV), the output, this target's TIMELINE and, for start and change events, a
FUTURE block of recorded data after t. No trigger source, probability, model, translator input or
request hash reaches the judge (decision 4). Free texts are quoted JSON strings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.config import KINDS, JudgeConfig, criteria_for
from translate.episode import AIRCRAFT, Episode, View, load_episode, render_compact
from translate.inputs import TrajectoryOptions, trajectory
from translate.journal import quoted

EVENT_SCHEMA_VERSION = 3
FUTURE_KINDS = frozenset({"start", "change"})
FUTURE_PRECISION = 5  # decimals of FUTURE floats (1e-5 deg is about 1 m)
EVENT_KEYS = ("event_id", "target", "step", "time_s", "trigger_reasons", "input", "output", "request_sha256")
INPUT_KEYS = ("target_aircraft", "instructions_in_effect", "alarms", "question", "previous_translation")


class RunError(ValueError):
    """The translation run cannot be judged (missing file, wrong schema, malformed record)."""


@dataclass(frozen=True)
class TranslationRun:
    path: Path
    run_json: dict[str, Any]
    records: tuple[dict[str, Any], ...]  # events.jsonl, in file order
    events: tuple[dict[str, Any], ...]  # its translation_event records
    requests: tuple[dict[str, Any], ...]  # requests.jsonl
    file_sha256: dict[str, str]  # run.json and events.jsonl

    @property
    def scenario_id(self) -> str:
        return self.records[0]["scenario_id"] if self.records else ""

    @property
    def episode_id(self) -> str:
        return self.records[0]["episode_id"] if self.records else ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """One JSON object per line, split on "\\n" only (U+2028 is legal inside JSON strings)."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    records = [json.loads(line) for line in lines if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise RunError(f"{path} must hold one JSON object per line")
    return records


def _checked_event(record: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in EVENT_KEYS if key not in record] + [k for k in INPUT_KEYS if k not in record["input"]]
    if missing:
        raise RunError(f"translation_event {record.get('event_id')} has no {missing}")
    kinds = {reason["kind"] for reason in record["trigger_reasons"]}
    if not kinds or not kinds <= set(KINDS):
        raise RunError(f"{record['event_id']}: trigger kinds must be among {list(KINDS)}")
    return record


def load_run(path: Path) -> TranslationRun:
    """A completed (or stopped) translation run of event schema 3; a preview has nothing to judge."""
    path = Path(path)
    if not (path / "events.jsonl").is_file() or not (path / "requests.jsonl").is_file():
        raise RunError(f"{path} has no events.jsonl or requests.jsonl (a preview has nothing to judge)")
    records = read_jsonl(path / "events.jsonl")
    if any(record.get("schema_version") != EVENT_SCHEMA_VERSION for record in records):
        raise RunError(f"{path}: every event must be schema {EVENT_SCHEMA_VERSION} (scripts/upgrade_events_v2.py)")
    events = tuple(_checked_event(r) for r in records if r.get("record_type") == "translation_event")
    sha = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ("run.json", "events.jsonl")}
    run_json = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return TranslationRun(path, run_json, tuple(records), events, tuple(read_jsonl(path / "requests.jsonl")), sha)


def load_frame(run: TranslationRun) -> Episode:
    """The episode the run translated; its CSV must still have the SHA-256 the run recorded."""
    recorded = run.run_json.get("inputs", {}).get("episode", {})
    if not recorded.get("csv_path") or not recorded.get("csv_sha256"):
        raise RunError("run.json inputs.episode needs csv_path and csv_sha256")
    episode = load_episode(Path(recorded["csv_path"]))
    if episode.csv_sha256 != recorded["csv_sha256"]:
        raise RunError("the episode CSV differs from the one the translation run used")
    return episode


def target_key(target: dict[str, str]) -> str:
    return f"{target['kind']}:{target['id']}"


def event_kinds(event: dict[str, Any]) -> tuple[str, ...]:
    """The event's point kinds, in start > change > request order."""
    kinds = {reason["kind"] for reason in event["trigger_reasons"]}
    return tuple(kind for kind in KINDS if kind in kinds)


def _lines(items: list[str]) -> str:
    return "\n".join(items) if items else "- none"


def _previous(previous: dict[str, Any] | None) -> str:
    if previous is None:
        return "- none (first translation of this target)"
    evidence = "".join(f"\n  - {quoted(item)}" for item in previous.get("evidence", []))
    return f"- step {previous['step']}: intent {quoted(previous['intent'])}{evidence}"


def _timeline(events: list[dict[str, Any]], current: dict[str, Any]) -> str:
    return "\n".join(
        f"- step {item['step']} ({item['time_s']} s): {', '.join(event_kinds(item))}; processing_s "
        f"{item.get('processing_s')}" + (" <- this event" if item["event_id"] == current["event_id"] else "")
        for item in events
    )


def _criterion(criterion: Any, has_future: bool) -> str:
    notes = [criterion.aspect]
    if criterion.conditional:
        notes.append("conditional: not_applicable allowed")
    if has_future:
        notes.append("may use FUTURE" if criterion.may_use_future else "must not use FUTURE")
    anchors = "".join(f'\n  "{value}": {criterion.levels[value]}' for value in sorted(criterion.levels, key=int)[::-1])
    bounds = [("cannot_judge", criterion.cannot_judge), ("not_applicable", criterion.not_applicable)]
    boundaries = "".join(f"\n  {name}: {text}" for name, text in bounds if text is not None)
    return f"- [{criterion.id}] ({'; '.join(notes)}) {criterion.text}{anchors}{boundaries}"


def _future_end(event: dict[str, Any], run: TranslationRun, config: JudgeConfig) -> tuple[int, str]:
    """(exclusive end, what cut it): the next instruction or alarm the target's later calls listed,
    its down step, the episode end or ``future_steps``; ties name the external factor."""
    step, target = event["step"], event["target"]
    listed = [record.get("input", {}) for record in run.records if record.get("target") == target]
    instructions = [item["issue_step"] for x in listed for item in x.get("instructions_in_effect", [])]
    messages = [item for x in listed for item in (*x.get("alarms", []), *x.get("messages", []))]
    alarms = [item["step"] for item in messages if item.get("kind", "alarm") == "alarm"]
    closed = [r["down_step"] for r in run.records if r["record_type"] == "target_closed" and r["target"] == target]
    last_step = run.run_json["inputs"]["episode"]["last_complete_step"]
    candidates = [
        ("next_alarm", min((s for s in alarms if s > step), default=None)),
        ("next_instruction", min((s for s in instructions if s > step), default=None)),
        ("target_down", min(closed, default=None)),
        ("episode_end", last_step + 1),
        ("future_steps", step + 1 + config.material.future_steps),
    ]
    end, _, cut = min((value, index, name) for index, (name, value) in enumerate(candidates) if value is not None)
    return end, cut


def _future(event: dict[str, Any], run: TranslationRun, frame: Episode, config: JudgeConfig) -> str:
    if not FUTURE_KINDS & set(event_kinds(event)):
        return ""
    end, cut = _future_end(event, run, config)
    shown = set(range(event["step"] + 1, end, config.material.future_stride))
    rows = []
    if shown:
        own = tuple(event["input"]["target_aircraft"])
        objects = (*own, *(aircraft for aircraft in AIRCRAFT if aircraft not in own))
        text = render_compact(
            frame.frame, event["step"], end - 1, event["step"], objects, config.material.fields, FUTURE_PRECISION
        )
        rows = [line for line in text.split("\n") if json.loads(line)["step"] in shown]
    later = [item for item in run.events if item["target"] == event["target"] and event["step"] < item["step"] < end]
    values = {
        "last_step": end - 1,
        "end_step": end,
        "cut": cut,
        "stride_steps": config.material.future_stride,
        "fields": ", ".join(config.material.fields),
        "rows": "\n".join(rows) if rows else "(no step inside the window)",
        "later": _timeline(later, event) if later else "- none",
    }
    return config.prompt.future.format_map(values)


def window(episode: Episode, event: dict[str, Any], past_steps: int) -> str:
    """The human-visible window ``[max(0, t - past + 1), t]``: the translator's own trajectory input."""
    view = View(event["step"], episode.rows_until(event["step"]), (), episode.teams)
    target = episode.target(f"{event['target']['kind']}:{event['target']['id']}")
    return trajectory(view, target, TrajectoryOptions(past_steps=past_steps))


def rating_rule(config: JudgeConfig) -> str:
    return config.prompt.rating_rule.format_map({"rationale_language": config.rationale_language})


def system_text(config: JudgeConfig) -> str:
    return config.prompt.system.format_map(
        {"rating_rule": rating_rule(config), "rationale_language": config.rationale_language}
    )


def user_text(event: dict[str, Any], run: TranslationRun, frame: Episode, config: JudgeConfig) -> str:
    """The event's user message: the human-visible material, TIMELINE, FUTURE and its criteria."""
    payload, kinds, target = event["input"], event_kinds(event), event["target"]
    instructions = sorted(payload["instructions_in_effect"], key=lambda i: (i["issue_step"], i["instruction_id"]))
    question = payload["question"]
    future = _future(event, run, frame, config)
    own = [item for item in run.events if item["target"] == target and item["step"] <= event["step"]]
    values = {
        "target": f"{target['kind']} {target['id']}",
        "target_aircraft": ", ".join(payload["target_aircraft"]),
        "step": event["step"],
        "time_s": event["time_s"],
        "kinds": ", ".join(kinds),
        "instructions": _lines(
            [
                f"- step {i['issue_step']}, for {i['target']['kind']} {i['target']['id']}: {quoted(i['text'])}"
                for i in instructions
            ]
        ),
        "alarms": _lines([f"- step {a['step']}: {quoted(a['text'])}" for a in payload["alarms"]]),
        "question": f"- step {question['step']}: {quoted(question['text'])}" if question else "- none",
        "previous_translation": _previous(payload["previous_translation"]),
        "trajectory": window(frame, event, config.material.past_steps),
        "intent": quoted(event["output"]["intent"]),
        "evidence": _lines([f"- {quoted(item)}" for item in event["output"]["evidence"]]),
        "timeline": _timeline(own, event),
        "future": future,
        "criteria": "\n".join(_criterion(c, bool(future)) for c in criteria_for(config, kinds)),
        "scale": ", ".join(f'"{value}"' for value in config.scale.values),
        "cannot_judge": config.scale.cannot_judge,
        "not_applicable": config.scale.not_applicable,
        "rating_rule": rating_rule(config),
    }
    return config.prompt.user.format_map(values)
