"""Trajectory renderers for LAG step windows.

``render_lag17_jsonl`` reproduces the window body of ``workspace/17_commentary_LAG``
(one JSON line per step, all aircraft and missiles) byte for byte.
``render_detector_compact`` is the short, target-focused view for the change detector (D27).
This module must never import from ``workspace/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .dataset import AIRCRAFT, STATE_FEATURES, last_complete_step

LAG17_RENDERER = "lag17-jsonl"
DETECTOR_RENDERER = "lag-detector-compact"
RENDERER_VERSION = "1"

# Fields of one 17-format entity, in the order the workspace renderer emits them.
LAG17_META_FIELDS = (
    "object_id",
    "entity_type",
    "team",
    "launcher_id",
    "alive",
    "state_valid",
    "acmi_event",
    "action_status",
)
LAG17_ACTION_FIELDS = ("altitude_action", "heading_action", "speed_action")
LAG17_ENTITY_TYPES = ("aircraft", "missile")
LAG17_FIELDS = (*LAG17_META_FIELDS, "state", "action", "missiles_remaining")
STEP_SECONDS = 0.2  # 0.2 s per step; ``round(step * STEP_SECONDS, 1)`` matches the workspace output.
SECONDS_DECIMALS = 1

# Detector defaults (D27); the Translator config may override them.
DEFAULT_DETECTOR_FIELDS = (
    "alive",
    "state_valid",
    "lon_deg",
    "lat_deg",
    "alt_m",
    "yaw_deg",
    "tas_mps",
    "vn_mps",
    "ve_mps",
    "vu_mps",
)
DEFAULT_DETECTOR_WINDOW_STEPS = 10
DEFAULT_FLOAT_PRECISION = 3
RESERVED_COMPACT_KEYS = ("step", "relative_step", "object_id")


class RenderError(ValueError):
    """A trajectory window cannot be rendered from the given frame and arguments."""


@dataclass(frozen=True)
class RenderedTrajectory:
    """Rendered text plus what produced it; start/end are the steps actually rendered (after clipping)."""

    renderer: str
    renderer_version: str
    start_step: int
    end_step: int
    anchor_step: int
    object_ids: tuple[str, ...]
    fields: tuple[str, ...]
    text: str
    sha256: str
    episode_alias: str = ""


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _nullable(value: Any) -> Any:
    return None if pd.isna(value) else value


def _to_python(value: Any) -> Any:
    """Nullable pandas scalar -> JSON-safe Python scalar."""
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise RenderError(f"{name} must be an integer")


def _clip_window(frame: pd.DataFrame, start_step: int, end_step: int, anchor_step: int) -> tuple[int, int]:
    """Validate the request and clip it to ``[0, last_complete_step]``.

    The anchor and the start must lie in the complete range; only the ends may be clipped.
    """
    for name, value in (("start_step", start_step), ("end_step", end_step), ("anchor_step", anchor_step)):
        _require_int(name, value)
    if not start_step <= anchor_step <= end_step:
        raise RenderError("Expected start_step <= anchor_step <= end_step")
    if anchor_step < 0:
        raise RenderError("anchor_step must be nonnegative")
    last_step = last_complete_step(frame)
    if anchor_step > last_step:
        raise RenderError(f"Incomplete CSV time point: {anchor_step} (last complete step is {last_step})")
    return max(0, start_step), min(last_step, end_step)


def _step_rows(frame: pd.DataFrame, start_step: int, end_step: int, entity_types: tuple[str, ...]) -> pd.DataFrame:
    selected = frame[frame["step"].between(start_step, end_step) & frame["entity_type"].isin(entity_types)]
    if selected.duplicated(["step", "object_id"]).any():
        raise RenderError("Duplicate step/object key")
    return selected


def _check_complete(rows: pd.DataFrame, step: int) -> None:
    aircraft = rows[rows["entity_type"].eq("aircraft")]
    if set(aircraft["object_id"]) != set(AIRCRAFT) or not aircraft["has_csv_record"].eq(1).all():
        raise RenderError(f"Incomplete CSV time point: {step}")


def _lag17_entity(row: pd.Series) -> dict[str, Any]:
    entity = {key: _nullable(row[key]) for key in LAG17_META_FIELDS}
    state_valid = row["state_valid"] == 1
    entity["state"] = [_nullable(row[name]) if state_valid else None for name in STATE_FEATURES]
    if row["entity_type"] != "aircraft":
        return entity
    recorded = row["action_status"] == "recorded"
    entity["action"] = [_nullable(row[name + "_id"]) if recorded else None for name in LAG17_ACTION_FIELDS]
    entity["missiles_remaining"] = _nullable(row["missiles_remaining"])
    return entity


def _finish(
    renderer: str,
    start_step: int,
    end_step: int,
    anchor_step: int,
    object_ids: tuple[str, ...],
    fields: tuple[str, ...],
    lines: list[str],
    episode_alias: str = "",
) -> RenderedTrajectory:
    text = "\n".join(lines)
    return RenderedTrajectory(
        renderer=renderer,
        renderer_version=RENDERER_VERSION,
        start_step=start_step,
        end_step=end_step,
        anchor_step=anchor_step,
        object_ids=object_ids,
        fields=fields,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        episode_alias=episode_alias,
    )


def render_lag17_jsonl(
    frame: pd.DataFrame,
    *,
    start_step: int,
    end_step: int,
    anchor_step: int,
    episode_alias: str,
) -> RenderedTrajectory:
    """Render ``[start_step, end_step]`` as 17-format JSON lines (one line per step, no trailing newline).

    ``relative_step`` is ``step - anchor_step``. The alias is metadata only: the workspace
    format does not place it in the lines, so it is stored on the result for provenance.
    Downed aircraft keep their rows (``alive == 0``) and render like any other.
    """
    if not episode_alias:
        raise RenderError("episode_alias must be non-empty")
    start, end = _clip_window(frame, start_step, end_step, anchor_step)
    selected = _step_rows(frame, start, end, LAG17_ENTITY_TYPES)
    lines: list[str] = []
    object_ids: set[str] = set()
    for step in range(start, end + 1):
        rows = selected[selected["step"].eq(step)].sort_values(["entity_type", "object_id"])
        _check_complete(rows, step)
        object_ids.update(rows["object_id"])
        block = {
            "step": step,
            "time_s": round(step * STEP_SECONDS, SECONDS_DECIMALS),
            "relative_step": step - anchor_step,
            "entities": [_lag17_entity(row) for _, row in rows.iterrows()],
        }
        lines.append(_json_line(block))
    return _finish(
        LAG17_RENDERER, start, end, anchor_step, tuple(sorted(object_ids)), LAG17_FIELDS, lines, episode_alias
    )


def _validate_compact_args(
    frame: pd.DataFrame, object_ids: tuple[str, ...], fields: tuple[str, ...], float_precision: int
) -> None:
    if not object_ids or len(set(object_ids)) != len(object_ids):
        raise RenderError("object_ids must be non-empty and unique")
    unknown_objects = set(object_ids) - set(frame["object_id"])
    if unknown_objects:
        raise RenderError(f"Unknown object IDs: {sorted(unknown_objects)}")
    if not fields or len(set(fields)) != len(fields):
        raise RenderError("fields must be non-empty and unique")
    reserved = set(fields) & set(RESERVED_COMPACT_KEYS)
    if reserved:
        raise RenderError(f"Reserved field names: {sorted(reserved)}")
    unknown_fields = set(fields) - set(frame.columns)
    if unknown_fields:
        raise RenderError(f"Unknown fields: {sorted(unknown_fields)}")
    if type(float_precision) is not int or float_precision < 0:
        raise RenderError("float_precision must be a nonnegative integer")


def _compact_value(row: pd.Series, name: str, float_precision: int) -> Any:
    if name in STATE_FEATURES and row["state_valid"] != 1:
        return None
    value = _to_python(row[name])
    return round(value, float_precision) if isinstance(value, float) else value


def render_detector_compact(
    frame: pd.DataFrame,
    *,
    start_step: int,
    end_step: int,
    anchor_step: int,
    object_ids: tuple[str, ...],
    fields: tuple[str, ...],
    float_precision: int,
) -> RenderedTrajectory:
    """Render a short window for the detector: chosen objects, chosen fields, rounded floats.

    One JSON line per step: ``{"step", "relative_step", "objects": [{"object_id", <fields>}]}``.
    Objects follow ``object_ids`` order and fields follow ``fields`` order. State features are
    ``null`` while ``state_valid != 1`` (as in the 17 format). An object with no row at a step
    is omitted from that line. The window is clipped to ``[0, last_complete_step]``.
    """
    _validate_compact_args(frame, object_ids, fields, float_precision)
    start, end = _clip_window(frame, start_step, end_step, anchor_step)
    selected = frame[frame["step"].between(start, end) & frame["object_id"].isin(object_ids)]
    if selected.duplicated(["step", "object_id"]).any():
        raise RenderError("Duplicate step/object key")
    rows_by_key = {(int(row["step"]), row["object_id"]): row for _, row in selected.iterrows()}
    lines: list[str] = []
    for step in range(start, end + 1):
        objects = [
            {"object_id": object_id, **{name: _compact_value(row, name, float_precision) for name in fields}}
            for object_id in object_ids
            if (row := rows_by_key.get((step, object_id))) is not None
        ]
        lines.append(_json_line({"step": step, "relative_step": step - anchor_step, "objects": objects}))
    return _finish(DETECTOR_RENDERER, start, end, anchor_step, object_ids, fields, lines)
