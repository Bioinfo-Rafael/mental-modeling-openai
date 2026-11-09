"""The episode: one LAG CSV read once (checked against its manifest), its facts, and ``View(t)``.

``View(t)`` is the only thing inputs and change sources receive: the rows at or before step t, the
messages issued at or before t, and the earlier translations. Nothing after t is reachable through
it. The two renderers turn a window of rows into compact JSON lines (``lag17`` for every entity,
``compact`` for chosen objects and fields); both start each line with ``{"step":``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

STEP_SECONDS = 0.2
AIRCRAFT = ("A0100", "A0200", "B0100", "B0200")
STATE_FEATURES = (
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
LAG17_META = ("object_id", "entity_type", "team", "launcher_id", "alive", "state_valid", "acmi_event", "action_status")
ACTION_IDS = ("altitude_action_id", "heading_action_id", "speed_action_id")
MANIFEST, MANIFEST_SCHEMA = "manifest.json", "1.0"


class EpisodeError(ValueError):
    """The episode data, or a window asked of it, is invalid."""


def time_s(step: int) -> float:
    return round(step * STEP_SECONDS, 1)


@dataclass(frozen=True)
class Target:
    """One translation unit (D17): ``aircraft:<id>`` or ``team:<id>`` and the aircraft it covers."""

    id: str
    aircraft: tuple[str, ...]
    opponents: tuple[str, ...]

    @property
    def kind(self) -> str:
        return self.id.split(":")[0]

    @property
    def label(self) -> str:
        return self.id.replace(":", " ")

    def to_json(self) -> dict[str, str]:
        kind, name = self.id.split(":")
        return {"kind": kind, "id": name}


@dataclass(frozen=True)
class Translation:
    """A target's latest valid translation: it feeds its next translation and detector state."""

    event_id: str
    step: int
    intent: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class View:
    """What is observable through step t; inputs and change sources receive nothing else."""

    step: int
    rows: pd.DataFrame  # every row with step <= t
    messages: tuple[Any, ...]  # scenario messages issued at or before t, in scenario order
    teams: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # team -> its aircraft (static)
    translations: Mapping[str, Translation] = field(default_factory=dict)  # target id -> latest, before t

    @property
    def time_s(self) -> float:
        return time_s(self.step)


def load_frame(path: Path) -> tuple[pd.DataFrame, str]:
    """The CSV with nullable types and its SHA-256, which must match the manifest beside it."""
    manifest = json.loads((path.parent / MANIFEST).read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries = [e for e in manifest["episodes"] if e["output"]["path"] == path.name]
    if manifest["schema_version"] != MANIFEST_SCHEMA or len(entries) != 1 or entries[0]["output"]["sha256"] != digest:
        raise EpisodeError(f"{path} differs from its manifest (or the manifest schema is not {MANIFEST_SCHEMA})")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)  # labels and ids are never numbers
    schema = {column["name"]: column["dtype"] for column in manifest["columns"]}
    if list(frame.columns) != list(schema):
        raise EpisodeError(f"{path}: columns differ from the manifest")
    for column, dtype in schema.items():
        frame[column] = frame[column].replace("", pd.NA)
        if dtype != "string":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)
    if frame["episode_id"].nunique() != 1 or frame.duplicated(["step", "object_id"]).any():
        raise EpisodeError(f"{path}: expected one episode with unique step/object keys")
    return frame, digest


def last_complete_step(frame: pd.DataFrame) -> int:
    """The last step at which all four aircraft have a CSV record (later steps exist only in the ACMI)."""
    aircraft = frame[frame["entity_type"].eq("aircraft") & frame["has_csv_record"].eq(1)]
    counts = aircraft.groupby("step")["object_id"].nunique()
    complete = counts[counts == len(AIRCRAFT)]
    if complete.empty:
        raise EpisodeError("no step has CSV records for all four aircraft")
    return int(complete.index.max())


@dataclass(frozen=True)
class Episode:
    frame: pd.DataFrame
    episode_id: str
    csv_path: str
    csv_sha256: str
    last_step: int
    teams: Mapping[str, tuple[str, ...]]  # team -> its aircraft, sorted
    down_steps: Mapping[str, int | None]  # aircraft -> first step with alive == 0

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, csv_path: str = "", csv_sha256: str = "") -> Episode:
        last = last_complete_step(frame)
        aircraft = frame[frame["entity_type"].eq("aircraft") & frame["step"].le(last)]
        teams = {str(team): tuple(sorted(rows["object_id"].unique())) for team, rows in aircraft.groupby("team")}
        down = {
            str(name): (
                int(rows.loc[rows["alive"].eq(0).fillna(False), "step"].min()) if rows["alive"].eq(0).any() else None
            )
            for name, rows in aircraft.groupby("object_id")
        }
        return cls(frame, str(frame["episode_id"].iloc[0]), csv_path, csv_sha256, last, teams, down)

    def target(self, target_id: str) -> Target:
        kind, name = target_id.split(":")
        known = self.down_steps if kind == "aircraft" else self.teams
        if name not in known:
            raise EpisodeError(f"{target_id} is not in episode {self.episode_id}")
        team = next(t for t, members in self.teams.items() if name in members) if kind == "aircraft" else name
        own = (name,) if kind == "aircraft" else self.teams[name]
        opponents = tuple(sorted(a for t, members in self.teams.items() if t != team for a in members))
        return Target(target_id, own, opponents)

    def down_step(self, target: Target) -> int | None:
        """Aircraft: its first ``alive == 0`` step. Team: the step its last aircraft is down, if all are."""
        steps = [self.down_steps[name] for name in target.aircraft]
        return None if None in steps else max(steps)

    def rows_until(self, step: int) -> pd.DataFrame:
        return self.frame[self.frame["step"].le(step)]


def load_episode(path: Path) -> Episode:
    frame, digest = load_frame(Path(path))
    return Episode.from_frame(frame, str(path), digest)


# --- renderers ------------------------------------------------------------------------------------


def _line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _value(value: Any) -> Any:
    return None if pd.isna(value) else value


def _lag17_entity(row: pd.Series) -> dict[str, Any]:
    entity = {key: _value(row[key]) for key in LAG17_META}
    valid = row["state_valid"] == 1
    entity["state"] = [_value(row[name]) if valid else None for name in STATE_FEATURES]
    if row["entity_type"] != "aircraft":
        return entity
    recorded = row["action_status"] == "recorded"
    entity["action"] = [_value(row[name]) if recorded else None for name in ACTION_IDS]
    entity["missiles_remaining"] = _value(row["missiles_remaining"])
    return entity


def render_lag17(rows: pd.DataFrame, start: int, end: int, anchor: int) -> str:
    """Steps ``[start, end]``, one JSON line per step with every aircraft and missile (17 format)."""
    selected = rows[rows["step"].between(start, end) & rows["entity_type"].isin(("aircraft", "missile"))]
    lines = []
    for step in range(start, end + 1):
        at_step = selected[selected["step"].eq(step)].sort_values(["entity_type", "object_id"])
        aircraft = at_step[at_step["entity_type"].eq("aircraft")]
        if set(aircraft["object_id"]) != set(AIRCRAFT) or not aircraft["has_csv_record"].eq(1).all():
            raise EpisodeError(f"incomplete CSV time point: {step}")
        entities = [_lag17_entity(row) for _, row in at_step.iterrows()]
        lines.append(
            _line({"step": step, "time_s": time_s(step), "relative_step": step - anchor, "entities": entities})
        )
    return "\n".join(lines)


def _compact(row: pd.Series, name: str, precision: int) -> Any:
    if name in STATE_FEATURES and row["state_valid"] != 1:
        return None
    value = None if pd.isna(row[name]) else row[name]
    value = value.item() if hasattr(value, "item") else value
    return round(value, precision) if isinstance(value, float) else value


def render_compact(
    rows: pd.DataFrame,
    start: int,
    end: int,
    anchor: int,
    objects: tuple[str, ...],
    fields: tuple[str, ...],
    precision: int,
) -> str:
    """Steps ``[start, end]``: the chosen objects in order with the chosen fields, floats rounded."""
    unknown = set(fields) - set(rows.columns)
    if unknown:
        raise EpisodeError(f"unknown fields {sorted(unknown)}")
    selected = rows[rows["step"].between(start, end) & rows["object_id"].isin(objects)]
    by_key = {(int(row["step"]), row["object_id"]): row for _, row in selected.iterrows()}
    lines = []
    for step in range(start, end + 1):
        found = [(name, by_key.get((step, name))) for name in objects]
        items = [
            {"object_id": name, **{f: _compact(row, f, precision) for f in fields}}
            for name, row in found
            if row is not None
        ]
        lines.append(_line({"step": step, "relative_step": step - anchor, "objects": items}))
    return "\n".join(lines)
