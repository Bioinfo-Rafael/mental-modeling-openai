#!/usr/bin/env python3
"""Convert a recorded rollout (npz) plus its replayed ACMI into the 13_LAG episode CSV format.

    python scripts/lag/episode_to_csv.py --rollout-dir <rollout> --seed 100601 --episode-dir <episode dir>

The Translator only reads episodes through ``translate.episode.load_episode`` (64-column
13_LAG CSV + ``manifest.json`` with SHA-256). This converter writes exactly that, so
the renderers and the episode facts work unchanged. Semantics follow
``data/mhi/processed/mhi_v1.3_ic100601_win.csv``:

* aircraft, one row per npz row ``i`` (= state before decision ``i`` = ACMI frame ``#0.2*i``):
  state from the npz (degrees via ``* 180 / pi`` like the ACMI writer, yaw wrapped to [0, 360),
  ``tas = |velocity_neu|``); the first dead step is ``acmi_event=remove``, later ones ``none``,
  and dead rows have ``state_valid=0`` / ``action_status=inactive``; live team-A rows carry the
  recorded action ids and softmax probabilities, live team-B rows are ``not_recorded``;
  ``missile_launched`` sits on the step before the missile's first ACMI frame;
* missiles, from the ACMI (full-precision ``T=`` values): ``update`` rows while live, one
  ``remove`` row (``state_valid=0``, no state) at the frame that removes them; no velocities.

Omitted (not rendered by ``lag17-jsonl``): effect rows, ACMI frames after the last npz row,
``health`` and the ``acmi_*`` copies of aircraft rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The 64-column LAG episode CSV (the 13_LAG definitions; translate.episode reads it with its manifest).
TRANSFORM = ("lon_deg", "lat_deg", "alt_m", "roll_deg", "pitch_deg", "yaw_deg")
VELOCITY = ("tas_mps", "vn_mps", "ve_mps", "vu_mps")
ACTION_LABELS = {
    "altitude_action": ("up", "hold", "down"),
    "heading_action": ("left30", "left15", "hold", "right15", "right30"),
    "speed_action": ("up", "hold", "down"),
    "shoot_action": ("no", "yes"),
}
PROBABILITIES = tuple(
    f"p_{a.removesuffix('_action')}_{label}" for a, labels in ACTION_LABELS.items() for label in labels
)
COLUMNS = (
    *("episode_id", "step", "time_s", "object_id", "entity_type", "team", "color", "name", "acmi_type", "launcher_id"),
    *("controller", "alive", *TRANSFORM, *VELOCITY, "health", "missiles_remaining", *ACTION_LABELS),
    *(f"{x}_id" for x in ACTION_LABELS),
    *PROBABILITIES,
    *("missile_launched", "acmi_event", "radius_m", "has_csv_record", "has_acmi_record", "state_valid", "state_source"),
    *("action_status", "csv_row_number", "acmi_line_number", "acmi_time_s", *(f"acmi_{x}" for x in TRANSFORM)),
    *("acmi_tas_mps", "acmi_properties_json"),
)
INTEGER_COLUMNS = {"step", "alive", "missiles_remaining", "has_csv_record", "has_acmi_record", "state_valid"}
INTEGER_COLUMNS |= {"csv_row_number", "acmi_line_number", *(f"{x}_id" for x in ACTION_LABELS)}
FLOAT_COLUMNS = {"time_s", *TRANSFORM, *VELOCITY, "health", "radius_m", *PROBABILITIES, "acmi_time_s"}
FLOAT_COLUMNS |= {*(f"acmi_{x}" for x in TRANSFORM), "acmi_tas_mps"}
MANIFEST_NAME, SUPPORTED_SCHEMA_VERSION = "manifest.json", "1.0"

STEP_SECONDS = 0.2
TIME_DECIMALS = 1
DEGREES_PER_RADIAN = 180 / np.pi  # the ACMI writer computes get_rpy() * 180 / np.pi
FULL_TURN_DEG = 360.0
TEAM_COLORS = {"A": "Red", "B": "Blue"}
AIRCRAFT_NAME = "F16"
EFFECT_TYPE = "Type=Misc+Explosion"
FRAME_PREFIX = "#"
REMOVE_PREFIX = "-"
GLOBAL_OBJECT = "0"
TRANSFORM_FIELDS = 6
HEAD_ORDER = tuple(ACTION_LABELS)  # altitude, heading, speed, shoot (the npz head order)


@dataclass(frozen=True)
class MissileRecord:
    step: int
    object_id: str
    launcher_id: str
    color: str
    name: str
    event: str  # "update" or "remove"
    transform: tuple[float, ...] | None
    line_number: int
    properties: dict[str, str]


def frame_step(token: str) -> int:
    """``#9.40`` -> 47. ACMI timestamps are ``current_step * 0.2`` printed with two decimals."""
    return round(float(token[len(FRAME_PREFIX) :]) / STEP_SECONDS)


def parse_properties(fields: list[str]) -> dict[str, str]:
    pairs = (field.split("=", 1) for field in fields if "=" in field)
    return {key: value for key, value in pairs}


def launcher_of(missile_id: str, aircraft: tuple[str, ...]) -> str:
    matches = [uid for uid in aircraft if missile_id.startswith(uid) and missile_id != uid]
    if len(matches) != 1:
        raise ValueError(f"cannot attribute missile {missile_id!r} to one launcher")
    return matches[0]


def _missile_update(step: int, line: str, number: int, aircraft: tuple[str, ...]) -> MissileRecord:
    object_id, *rest = line.split(",")
    properties = parse_properties(rest)
    transform = tuple(float(x) for x in properties["T"].split("|"))
    if len(transform) != TRANSFORM_FIELDS:
        raise ValueError(f"ACMI line {number}: expected {TRANSFORM_FIELDS} transform values")
    return MissileRecord(
        step, object_id, launcher_of(object_id, aircraft), properties.get("Color", ""),
        properties.get("Name", ""), "update", transform, number, properties,
    )  # fmt: skip


def parse_acmi_missiles(text: str, aircraft: tuple[str, ...]) -> list[MissileRecord]:
    """Missile update/remove records per frame. Aircraft, effects and the header are ignored."""
    records: list[MissileRecord] = []
    known: dict[str, MissileRecord] = {}
    step: int | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.replace("﻿", "").strip()
        if line.startswith(FRAME_PREFIX):
            step = frame_step(line)
            continue
        if step is None or not line or line.startswith(GLOBAL_OBJECT + ","):
            continue
        if line.startswith(REMOVE_PREFIX):
            object_id = line[len(REMOVE_PREFIX) :]
            if object_id in known:
                last = known[object_id]
                records.append(MissileRecord(step, object_id, last.launcher_id, last.color, last.name,
                                             "remove", None, number, {}))  # fmt: skip
            continue
        object_id = line.split(",", 1)[0]
        if object_id in aircraft or EFFECT_TYPE in line:
            continue
        record = _missile_update(step, line, number, aircraft)
        known = {**known, object_id: record}
        records.append(record)
    return records


def softmax_heads(logits: np.ndarray, sizes: tuple[int, ...]) -> list[np.ndarray]:
    """Per-head softmax of concatenated float32 logits, computed in float64."""
    bounds = np.cumsum((0, *sizes))
    heads = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        values = logits[start:end].astype(np.float64)
        exp = np.exp(values - values.max())
        heads.append(exp / exp.sum())
    return heads


def _empty_row(episode_id: str, step: int, object_id: str) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(COLUMNS)
    return {**row, "episode_id": episode_id, "step": step, "time_s": round(step * STEP_SECONDS, TIME_DECIMALS),
            "object_id": object_id}  # fmt: skip


def _action_fields(actions: np.ndarray, logits: np.ndarray, sizes: tuple[int, ...]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for head, (name, probs) in enumerate(zip(HEAD_ORDER, softmax_heads(logits, sizes), strict=True)):
        labels = ACTION_LABELS[name]
        index = int(actions[head])
        prefix = f"p_{name.removesuffix('_action')}_"
        fields = {**fields, name: labels[index], f"{name}_id": index}
        fields = {**fields, **{prefix + label: float(p) for label, p in zip(labels, probs, strict=True)}}
    return fields


def aircraft_rows(
    episode_id: str, arrays: Any, uids: tuple[str, ...], learners: tuple[str, ...], rule_type: str,
    launches: dict[tuple[int, str], str],
) -> list[dict[str, Any]]:  # fmt: skip
    sizes = tuple(int(s) for s in arrays["head_sizes"])
    rows: list[dict[str, Any]] = []
    for i, step in enumerate(arrays["step"].tolist()):
        for j, uid in enumerate(uids):
            alive = bool(arrays["alive"][i, j])
            first_dead = not alive and (i == 0 or bool(arrays["alive"][i - 1, j]))
            lon, lat, alt = (float(x) for x in arrays["geodetic"][i, j])
            roll, pitch, yaw = (float(x) for x in arrays["rpy"][i, j] * DEGREES_PER_RADIAN)
            vn, ve, vu = (float(x) for x in arrays["velocity_neu"][i, j])
            team = uid[0]
            row = {
                **_empty_row(episode_id, int(step), uid),
                "entity_type": "aircraft", "team": team, "color": TEAM_COLORS[team], "name": AIRCRAFT_NAME,
                "controller": "learned" if uid in learners else f"rule_{rule_type}", "alive": int(alive),
                "lon_deg": lon, "lat_deg": lat, "alt_m": alt, "roll_deg": roll, "pitch_deg": pitch,
                "yaw_deg": yaw % FULL_TURN_DEG, "tas_mps": math.sqrt(vn * vn + ve * ve + vu * vu),
                "vn_mps": vn, "ve_mps": ve, "vu_mps": vu,
                "missiles_remaining": int(arrays["missiles_left"][i, j]) if alive else None,
                "missile_launched": launches.get((int(step), uid)),
                "acmi_event": "update" if alive else ("remove" if first_dead else "none"),
                "has_csv_record": 1, "has_acmi_record": int(alive or first_dead), "state_valid": int(alive),
                "state_source": "csv",
                "action_status": "inactive" if not alive else ("recorded" if uid in learners else "not_recorded"),
            }  # fmt: skip
            if alive and uid in learners:
                k = learners.index(uid)
                row = {**row, **_action_fields(arrays["actions"][i, k], arrays["logits"][i, k], sizes)}
            rows.append(row)
    return rows


def missile_rows(episode_id: str, records: list[MissileRecord], last_step: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record.step > last_step:
            continue
        transform = dict(zip(("lon_deg", "lat_deg", "alt_m", "roll_deg", "pitch_deg", "yaw_deg"),
                             record.transform or (None,) * TRANSFORM_FIELDS, strict=True))  # fmt: skip
        live = record.event == "update"
        rows.append({
            **_empty_row(episode_id, record.step, record.object_id), **transform,
            "entity_type": "missile", "team": record.launcher_id[0], "color": record.color,
            "name": record.name or None, "launcher_id": record.launcher_id, "acmi_event": record.event,
            "has_csv_record": 0, "has_acmi_record": 1, "state_valid": int(live),
            "state_source": "acmi" if live else "none", "action_status": "not_applicable",
            "acmi_line_number": record.line_number, "acmi_time_s": round(record.step * STEP_SECONDS, TIME_DECIMALS),
            **{f"acmi_{k}": v for k, v in transform.items()},
            "acmi_properties_json": json.dumps(record.properties, separators=(",", ":")),
        })  # fmt: skip
    return rows


def launch_steps(records: list[MissileRecord]) -> dict[tuple[int, str], str]:
    """(action step, launcher) -> missile id; the action step precedes the first ACMI frame by one."""
    first: dict[str, MissileRecord] = {}
    for record in records:
        if record.event == "update" and record.object_id not in first:
            first = {**first, record.object_id: record}
    return {(r.step - 1, r.launcher_id): r.object_id for r in first.values()}


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in episode CSV")
        return repr(value)
    return str(value)


def render_csv(rows: list[dict[str, Any]]) -> bytes:
    order = {"aircraft": 0, "missile": 1}
    ordered = sorted(rows, key=lambda r: (r["step"], order[r["entity_type"]], r["object_id"]))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in ordered:
        writer.writerow([format_value(row[name]) for name in COLUMNS])
    return buffer.getvalue().encode("utf-8")


def column_dtypes() -> list[dict[str, str]]:
    def dtype(name: str) -> str:
        if name in INTEGER_COLUMNS:
            return "Int64"
        return "Float64" if name in FLOAT_COLUMNS else "string"

    return [{"name": name, "dtype": dtype(name)} for name in COLUMNS]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def convert(npz: Path, acmi: Path, rollout_manifest: dict[str, Any], episode_id: str) -> tuple[bytes, dict[str, Any]]:
    uids = tuple(rollout_manifest["aircraft_uids"])
    learners = tuple(rollout_manifest["learner_uids"])
    with np.load(npz) as data:
        arrays = {k: data[k] for k in ("step", "alive", "geodetic", "rpy", "velocity_neu", "missiles_left",
                                        "actions", "logits")}  # fmt: skip
    arrays = {**arrays, "head_sizes": np.asarray(rollout_manifest["action_head_sizes"])}
    if not np.array_equal(arrays["step"], np.arange(len(arrays["step"]))):
        raise ValueError("npz rows must be steps 0..T-1")
    last_step = int(arrays["step"][-1])
    records = parse_acmi_missiles(acmi.read_text(encoding="utf-8-sig"), uids)
    rule_type = rollout_manifest["opponent"]["rule_type"]
    rows = aircraft_rows(episode_id, arrays, uids, learners, rule_type, launch_steps(records))
    rows += missile_rows(episode_id, records, last_step)
    return render_csv(rows), {
        "rows": len(rows),
        "last_step": last_step,
        "missiles": sorted({r.object_id for r in records}),
    }


def build_manifest(csv_name: str, data: bytes, info: dict[str, Any], sources: dict[str, Path]) -> dict[str, Any]:
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "dt_s": STEP_SECONDS,
        "delimiter": ",",
        "encoding": "UTF-8",
        "missing_value": "",
        "primary_key": ["episode_id", "step", "object_id"],
        "columns": column_dtypes(),
        "generator": {"scripts/lag/episode_to_csv.py": sha256_bytes(Path(__file__).read_bytes())},
        "episodes": [
            {
                "episode_id": info["episode_id"],
                "output": {"path": csv_name, "sha256": sha256_bytes(data), "bytes": len(data)},
                "rows": info["rows"],
                "csv_last_step": info["last_step"],
                "missiles": info["missiles"],
                "source_files": [
                    {"role": role, "path": str(path), "sha256": sha256_bytes(path.read_bytes())}
                    for role, path in sources.items()
                ],
                "omitted": "effect rows, ACMI frames after csv_last_step, health, acmi_* copies of aircraft rows",
            }
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rollout-dir", type=Path, required=True, help="the recorded rollout (manifest.json + npz)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True, help="where seed<seed>.csv and manifest.json go")
    parser.add_argument("--acmi", type=Path, help="default: <episode-dir>/replay.txt.acmi")
    args = parser.parse_args(argv)
    acmi = args.acmi or args.episode_dir / "replay.txt.acmi"
    manifest_path = args.rollout_dir / "manifest.json"
    rollout_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_id = f"{rollout_manifest['model_alias']}_seed{args.seed}"
    npz_path = args.rollout_dir / f"episode_seed{args.seed}.npz"
    episode_csv = args.episode_dir / f"seed{args.seed}.csv"
    data, info = convert(npz_path, acmi, rollout_manifest, episode_id)
    sources = {"npz": npz_path, "rollout_manifest": manifest_path, "acmi": acmi}
    manifest = build_manifest(episode_csv.name, data, {**info, "episode_id": episode_id}, sources)
    args.episode_dir.mkdir(parents=True, exist_ok=True)
    episode_csv.write_bytes(data)
    (args.episode_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(episode_csv), **info}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
