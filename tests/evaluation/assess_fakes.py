"""Offline helpers for the assessment tests: a hand-written Translator run (event contract v2),
a tiny synthetic episode frame, config overrides and a scripted judge.

Nothing here imports the Translator package (it is migrating to v2 in parallel), touches the
network, a GPU, a real key or a paid API. Texts are placeholders.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.assess_data import AssessConfig, parse_config
from evaluation.assess_judge import JudgeCallError, JudgeReply
from translate import paths
from translate.llm.openai_responses import OpenAIRequest

LEGACY_CONFIG = paths.ROOT / "configs" / "evaluation" / "assess-legacy.yaml"
SCENARIO_ID = "placeholder-episode-01"
EPISODE_ID = "mhi_v1.3_ic100601_win"
ALIAS = "episode_01"
LAST_STEP = 60
TICK = 5
LOOKBACK = 25
TRANSLATION_WINDOW, DETECTOR_WINDOW = 3, 2
FAKE_CSV_SHA256 = "0" * 64
A1 = {"kind": "aircraft", "id": "A0100"}
INSTRUCTIONS = (
    {"instruction_id": "placeholder-aircraft-001", "issue_step": 0, "target": A1, "text": "[PLACEHOLDER] Task one."},
    {"instruction_id": "placeholder-aircraft-002", "issue_step": 25, "target": A1, "text": "[PLACEHOLDER] Task two."},
)
ALARMS = ({"alarm_id": "placeholder-alarm-001", "step": 12, "targets": [A1], "text": "[PLACEHOLDER] Missile warning."},)
# (step, trigger kinds) of the translation events of aircraft A0100.
EVENT_PLAN: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("initial",)),
    (5, ("start",)),
    (10, ("request",)),
    (15, ("change",)),
    (40, ("request",)),
)
PROCESSING_S = {0: 1.5, 5: 2.0, 10: 2.5, 15: 2.2, 40: 2.8}
# The same episode in event schema 3 (2026-11-09 definitions): the start at the episode start, the two
# questions and the change after the alarm at 12; a tactic begun along the plan (step 5) is no point.
V3_PLAN: tuple[tuple[int, tuple[str, ...]], ...] = (
    (0, ("start",)),
    (10, ("request",)),
    (15, ("change",)),
    (40, ("request",)),
)
DETECTOR_LABELS = {5: "start", 15: "change"}  # every other tick answers "none"


def event_id(step: int, target: dict[str, str] = A1) -> str:
    return f"{SCENARIO_ID}:{target['kind']}:{target['id']}:{step:06d}"


def detector_id(step: int, target: dict[str, str] = A1) -> str:
    return f"{SCENARIO_ID}:detector:{target['kind']}:{target['id']}:{step:06d}"


E_INITIAL, E_START, E_REQUEST, E_CHANGE, E_LATE_REQUEST = (event_id(step) for step, _ in EVENT_PLAN)
ONLINE = {"mode": "online", "future_steps": 0}


def question(step: int) -> dict[str, Any]:
    return {"request_id": f"placeholder-request-{step:03d}", "step": step, "text": f"[PLACEHOLDER] Question at {step}?"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def lag17_line(step: int, anchor: int) -> str:
    entity = {"object_id": "A0100", "alive": 1, "action": [1, 2, 1], "missiles_remaining": 4}
    return _line({"step": step, "time_s": round(step * 0.2, 1), "relative_step": step - anchor, "entities": [entity]})


def compact_line(step: int, anchor: int) -> str:
    return _line({"step": step, "relative_step": step - anchor, "objects": [{"object_id": "A0100", "alt_m": 1000.0}]})


def trajectory(step: int, window: int, future: int, line: Any, renderer: str) -> dict[str, Any]:
    start, end = max(0, step - window + 1), min(LAST_STEP, step + future)
    text = "\n".join(line(s, step) for s in range(start, end + 1))
    return {
        "renderer": renderer,
        "start_step": start,
        "end_step": end,
        "anchor_step": step,
        "text": text,
        "sha256": sha256_text(text),
    }


def instructions_at(step: int) -> list[dict[str, Any]]:
    """The target's current instruction (its own only, in this fixture)."""
    return [max((i for i in INSTRUCTIONS if i["issue_step"] <= step), key=lambda i: i["issue_step"])]


def _reason(kind: str, step: int, schema_version: int) -> dict[str, Any]:
    if kind == "start" and schema_version == 3:
        return {"kind": kind, "source": "episode_start", "reference_id": "episode-start", "probability": None}
    if kind == "initial":
        return {
            "kind": kind,
            "source": "first_instruction",
            "reference_id": INSTRUCTIONS[0]["instruction_id"],
            "probability": None,
        }
    if kind == "request":
        return {
            "kind": kind,
            "source": "placeholder-human",
            "reference_id": question(step)["request_id"],
            "probability": None,
        }
    return {"kind": kind, "source": "detector", "reference_id": detector_id(step), "probability": 0.8}


def _translation_text(record: dict[str, Any]) -> str:
    payload, step = record["input"], record["step"]
    lines = [f"Episode: {ALIAS}", f"Current step: {step}", "Translation target: aircraft A0100 (aircraft A0100)", ""]
    lines += [
        "Instructions in effect:",
        *(f"- issued at step {i['issue_step']}:\n  {i['text']}" for i in payload["instructions_in_effect"]),
    ]
    lines += ["", "Alarms:", *(f"- step {a['step']}: {a['text']}" for a in payload["alarms"]), ""]
    if payload["question"]:
        lines += [f"Question from a human (step {payload['question']['step']}):", payload["question"]["text"], ""]
    previous = payload["previous_translation"]
    lines += [
        "Previous translation:",
        f"From step {previous['step']}: {previous['intent']}" if previous else "(none)",
        "",
    ]
    return "\n".join([*lines, "BEGIN WINDOW DATA", payload["trajectory"]["text"], "END WINDOW DATA"])


def _detector_text(record: dict[str, Any]) -> str:
    payload = record["input"]
    lines = [
        "Target: aircraft A0100 (aircraft A0100). Opponents: B0100, B0200.",
        payload["previous_translation"]["intent"],
    ]
    lines += [
        "Instructions in effect:",
        *(f"- step {i['issue_step']}: {i['text']}" for i in payload["instructions_in_effect"]),
    ]
    lines += ["Recent messages:", *(f"- step {m['step']} {m['kind']}: {m['text']}" for m in payload["messages"])]
    return "\n".join([*lines, "Trajectory window:", payload["trajectory"]["text"]])


def payload_of(record: dict[str, Any], extra: str = "") -> dict[str, Any]:
    if record["record_type"] == "translation_event":
        text = _translation_text(record) + extra
        content = [{"type": "input_text", "text": text}]
        return {
            "model": "gpt-6.1-terra",
            "instructions": "[FAKE] system prompt",
            "input": [{"role": "user", "content": content}],
            "store": False,
        }
    return {
        "model": "clef-flash",
        "state": _detector_text(record) + extra,
        "questions": [{"id": "point_kind", "type": "choice"}],
    }


def _common(record_type: str, step: int, target: dict[str, str], schema_version: int = 2) -> dict[str, Any]:
    base = {"schema_version": schema_version, "record_type": record_type, "scenario_id": SCENARIO_ID}
    base = {**base, "episode_id": EPISODE_ID}
    return {**base, "step": step, "time_s": round(step * 0.2, 1), "target": target}


def build_records(
    plan: tuple[tuple[int, tuple[str, ...]], ...] = EVENT_PLAN, future: int = 0, schema_version: int = 2
) -> list[dict[str, Any]]:
    """events.jsonl records (no sequence or request_sha256 yet), in emission order."""
    kinds_at = dict(plan)
    records, previous = [], None
    for step in range(LAST_STEP + 1):
        if previous is not None and step % TICK == 0:
            messages = [{"kind": "alarm", "id": a["alarm_id"], "step": a["step"], "text": a["text"]} for a in ALARMS]
            messages += [
                {"kind": "question", "id": question(s)["request_id"], "step": s, "text": question(s)["text"]}
                for s, k in plan
                if "request" in k
            ]
            messages = sorted((m for m in messages if step - LOOKBACK < m["step"] <= step), key=lambda m: m["step"])
            label = DETECTOR_LABELS.get(step, "none")
            if schema_version == 3:  # the detector's options are none and change
                label = "change" if "change" in kinds_at.get(step, ()) else "none"
            record_input = {
                "trajectory": trajectory(step, DETECTOR_WINDOW, future, compact_line, "lag-detector-compact"),
                "instructions_in_effect": instructions_at(step),
                "messages": messages,
                "previous_translation": {
                    "event_id": previous["event_id"],
                    "step": previous["step"],
                    "intent": previous["intent"],
                },
            }
            output = {
                "label": label,
                "probabilities": {
                    "none": 0.1,
                    **({"start": 0.1} if schema_version == 2 else {}),
                    "change": 0.1,
                    label: 0.8,
                },
                "threshold": 0.65,
            }
            records.append(
                {
                    **_common("detector_call", step, A1, schema_version),
                    "call_id": detector_id(step),
                    "input": record_input,
                    "output": {**output, "triggered": label != "none"},
                }
            )
        if step not in kinds_at:
            continue
        reasons = [_reason(kind, step, schema_version) for kind in kinds_at[step]]
        record_input = {
            "prompt_version": "translator-intent-v2",
            "target_aircraft": ["A0100"],
            "instructions_in_effect": instructions_at(step),
            "alarms": [a for a in ALARMS if a["step"] <= step],
            "question": question(step) if "request" in kinds_at[step] else None,
            "previous_translation": previous,
            "trajectory": trajectory(step, TRANSLATION_WINDOW, future, lag17_line, "lag17-jsonl"),
        }
        output = {
            "intent": f"[PLACEHOLDER] intent at step {step}",
            "evidence": [f"[PLACEHOLDER] evidence at step {step}"],
        }
        body = {
            "event_id": event_id(step),
            "primary_trigger": reasons[0]["kind"],
            "trigger_reasons": reasons,
            "input": record_input,
        }
        records.append(
            {
                **_common("translation_event", step, A1, schema_version),
                **body,
                "output": output,
                "processing_s": PROCESSING_S.get(step, 2.0),
            }
        )
        previous = {
            "event_id": event_id(step),
            "step": step,
            "intent": output["intent"],
            "evidence": output["evidence"],
        }
    return records


def call_id(record: dict[str, Any]) -> str:
    return record["event_id"] if record["record_type"] == "translation_event" else record["call_id"]


def build_run(
    *,
    plan: tuple[tuple[int, tuple[str, ...]], ...] = EVENT_PLAN,
    visibility: dict[str, Any] | None = ONLINE,
    rendered_future: int = 0,
    extra_text: dict[str, str] | None = None,
    after_hash_text: dict[str, str] | None = None,
    input_changes: dict[str, dict[str, Any]] | None = None,
    output_changes: dict[str, dict[str, Any]] | None = None,
    status: str = "completed",
    schema_version: int = 2,
) -> dict[str, Any]:
    """The four files of a Translator run (event schema 2 unless given; schema 3 takes ``V3_PLAN``-style
    kinds). Changes apply before hashing unless named ``after_hash``."""
    extra_text, after_hash_text = extra_text or {}, after_hash_text or {}
    input_changes, output_changes = input_changes or {}, output_changes or {}
    events, requests, responses = [], [], []
    for sequence, record in enumerate(build_records(plan, rendered_future, schema_version), start=1):
        identifier = call_id(record)
        record = {**record, "sequence": sequence, "input": {**record["input"], **input_changes.get(identifier, {})}}
        if record["record_type"] == "translation_event":
            record = {**record, "output": {**record["output"], **output_changes.get(identifier, {})}}
        payload = payload_of(record, extra_text.get(identifier, ""))
        record = {**record, "request_sha256": sha256_text(canonical(payload))}
        sent = payload_of(record, extra_text.get(identifier, "") + after_hash_text.get(identifier, ""))
        layer = "translation" if record["record_type"] == "translation_event" else "detector"
        common = {"schema_version": schema_version, "call_id": identifier, "layer": layer}
        requests.append({**common, "scenario_id": SCENARIO_ID, "step": record["step"], "target": A1, "payload": sent})
        latency = round(record.get("processing_s", 0.5) * 0.8, 3)
        responses.append({**common, "status": "ok", "raw": {}, "error": None, "latency_s": latency})
        events.append(record)
    config = {
        "pipeline": {"tick_interval_steps": TICK, **({"visibility": visibility} if visibility is not None else {})}
    }
    episode = {
        "episode_id": EPISODE_ID,
        "alias": ALIAS,
        "csv_path": "/nonexistent/fake.csv",
        "csv_sha256": FAKE_CSV_SHA256,
    }
    episode = {**episode, "video_path": None, "last_complete_step": LAST_STEP}
    run_json = {
        "schema_version": 1,
        "mode": "execute",
        "status": status,
        "inputs": {"scenario": {"scenario_id": SCENARIO_ID}, "episode": episode},
        "config": config,
    }
    return {"run_json": run_json, "events": events, "requests": requests, "responses": responses}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return path


def write_run(directory: Path, files: dict[str, Any]) -> Path:
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(files["run_json"]), encoding="utf-8")
    for name in ("events", "requests", "responses"):
        write_jsonl(directory / f"{name}.jsonl", files[name])
    return directory


def make_run(root: Path, name: str = "run", **options: Any) -> Path:
    return write_run(root / name, build_run(**options))


# --- config, annotation, ratings --------------------------------------------------------------------


def config_raw() -> dict[str, Any]:
    """The legacy judge config: the hand-written v2 run here is judged with the rubric it was written for."""
    return yaml.safe_load(LEGACY_CONFIG.read_text(encoding="utf-8"))


def with_value(raw: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """A copy of ``raw`` with the dotted key replaced (the original is never mutated)."""
    head, _, rest = dotted.partition(".")
    if not rest:
        return {**raw, head: value}
    return {**raw, head: with_value(raw[head], rest, value)}


def make_config(output_root: Path | None = None, **overrides: Any) -> AssessConfig:
    """The committed assess.yaml with ``a__b=value`` overrides (``__`` separates the levels)."""
    raw = copy.deepcopy(config_raw())
    for dotted, value in overrides.items():
        raw = with_value(raw, dotted.replace("__", "."), value)
    if output_root is not None:
        raw = with_value(raw, "output.root", str(output_root))
    return parse_config(raw, path=LEGACY_CONFIG, sha256="test-config")


def write_config(path: Path, **overrides: Any) -> Path:
    raw = copy.deepcopy(config_raw())
    for dotted, value in overrides.items():
        raw = with_value(raw, dotted.replace("__", "."), value)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def point(point_id: str, kind: str, step: int, onset: int | None = None) -> dict[str, Any]:
    return {"point_id": point_id, "kind": kind, "target": dict(A1), "step": step, "action_onset_step": onset}


def write_points(path: Path, points: list[dict[str, Any]]) -> Path:
    raw = {"schema_version": 1, "scenario_id": SCENARIO_ID, "annotator": "annotator-01", "points": points}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def rating(rating_id: str, event: str, criterion_id: str, judgement: str, **changes: Any) -> dict[str, Any]:
    base = {
        "rating_id": rating_id,
        "rater_id": "rater-01",
        "event_id": event,
        "criterion_id": criterion_id,
        "judgement": judgement,
        "rationale": "[PLACEHOLDER] 根拠",
        "rubric_version": config_raw()["rubric_version"],  # follows the committed assess.yaml
        "rated_at": "2026-11-05T09:00:00+00:00",
        "supersedes": None,
    }
    return {**base, **changes}


# --- synthetic episode frame ------------------------------------------------------------------------

AIRCRAFT_TEAMS = {"A0100": "A", "A0200": "A", "B0100": "B", "B0200": "B"}
FLOAT_FIELDS = (
    "lon_deg",
    "lat_deg",
    "alt_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "tas_mps",
    "vn_mps",
    "ve_mps",
    "vu_mps",
)
ACTION_FIELDS = ("altitude_action_id", "heading_action_id", "speed_action_id")


def make_frame(last_step: int = LAST_STEP) -> pd.DataFrame:
    """Four aircraft, every step complete: enough for ``render_detector_compact``."""
    rows = []
    for step in range(last_step + 1):
        for aircraft, team in AIRCRAFT_TEAMS.items():
            row = {
                "episode_id": EPISODE_ID,
                "step": step,
                "object_id": aircraft,
                "entity_type": "aircraft",
                "team": team,
            }
            row = {**row, "has_csv_record": 1, "state_valid": 1, "alive": 1}
            row = {**row, **{name: float(step) + 0.123456 for name in FLOAT_FIELDS}}
            rows.append({**row, **{name: (1 if team == "A" else None) for name in ACTION_FIELDS}})
    frame = pd.DataFrame(rows)
    for name in ACTION_FIELDS:
        frame[name] = frame[name].astype("Int64")
    frame.attrs.update(csv_sha256=FAKE_CSV_SHA256)
    return frame


# --- scripted judge ---------------------------------------------------------------------------------


def verdict_text(criterion_ids: list[str], judgement: str = "4", **overrides: str) -> str:
    """A judge answer: every criterion ``judgement`` except the ``overrides`` (id with ``.`` -> ``__``)."""
    chosen = {key.replace("__", "."): value for key, value in overrides.items()}
    items = [
        {"criterion_id": cid, "rationale": f"[FAKE] {cid} の根拠", "judgement": chosen.get(cid, judgement)}
        for cid in criterion_ids
    ]
    return json.dumps({"criteria": items}, ensure_ascii=False)


def asked_ids(request: OpenAIRequest) -> list[str]:
    return request.schema["properties"]["criteria"]["items"]["properties"]["criterion_id"]["enum"]


class ScriptedJudge:
    """A ``JudgeModel``: each outcome is a judgement for every asked criterion, a dict of overrides,
    raw output text (``str``) or an exception to raise."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[OpenAIRequest] = []

    def score(self, request: OpenAIRequest) -> JudgeReply:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, dict):
            text = verdict_text(asked_ids(request), **outcome)
        else:
            text = outcome if outcome.startswith("{") else verdict_text(asked_ids(request), outcome)
        usage = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "reasoning_tokens": 1}
        return JudgeReply("gpt-6.1-sol", "resp_fake", text, usage, {"id": "resp_fake"})


def uncertain_error() -> JudgeCallError:
    return JudgeCallError("timed out; outcome uncertain", error_type="OpenAIClientError", uncertain=True)


class MemorySink:
    """Collects the judge journal and results in memory, with a shared order log."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.log: list[tuple[str, str]] = []

    def write_request(self, record: dict[str, Any]) -> None:
        self.requests.append(record)
        self.log.append(("request", record["call_id"]))

    def write_response(self, record: dict[str, Any]) -> None:
        self.responses.append(record)
        self.log.append(("response", record["call_id"]))

    def write_results(self, records: list[dict[str, Any]]) -> None:
        self.results.extend(records)
        self.log.append(("results", records[0]["subject"]["event_id"]))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in (lines[:-1] if lines[-1] == "" else lines)]


def find(records: list[dict[str, Any]], **where: Any) -> list[dict[str, Any]]:
    """Records whose top-level or ``subject`` fields equal ``where`` (``event_id`` / ``reference_point_id``)."""

    def value(record: dict[str, Any], key: str) -> Any:
        return record[key] if key in record else record["subject"].get(key)

    return [record for record in records if all(value(record, key) == expected for key, expected in where.items())]
