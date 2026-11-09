"""Typed LAG episode loading and record selection, promoted from ``workspace/13_LAG``.

The constants, dtypes, hash verification and helpers behave exactly like
``workspace/13_LAG/lag_data/{definitions,dataset}.py``; ``tests/lag/test_lag_contract.py`` pins that.
This module must never import from ``workspace/``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd

# --- Definitions (13_LAG ``definitions.py``) -------------------------------------------------

DT = Decimal("0.2")
AIRCRAFT = ("A0100", "A0200", "B0100", "B0200")
TRANSFORM = ("lon_deg", "lat_deg", "alt_m", "roll_deg", "pitch_deg", "yaw_deg")
VELOCITY = ("tas_mps", "vn_mps", "ve_mps", "vu_mps")
ACTION_LABELS = {
    "altitude_action": ("up", "hold", "down"),
    "heading_action": ("left30", "left15", "hold", "right15", "right30"),
    "speed_action": ("up", "hold", "down"),
    "shoot_action": ("no", "yes"),
}
PROBABILITIES = tuple(
    f"p_{action.removesuffix('_action')}_{label}" for action, labels in ACTION_LABELS.items() for label in labels
)
CSV_FIELDS = (
    "step",
    "time_s",
    "uid",
    "color",
    "controller",
    "alive",
    *TRANSFORM,
    *VELOCITY,
    *(
        field
        for action, labels in ACTION_LABELS.items()
        for field in (action, *(f"p_{action.removesuffix('_action')}_{x}" for x in labels))
    ),
    "missile_launched",
)
CSV_RENAMES = {"uid": "object_id"}
COLUMNS = (
    "episode_id",
    "step",
    "time_s",
    "object_id",
    "entity_type",
    "team",
    "color",
    "name",
    "acmi_type",
    "launcher_id",
    "controller",
    "alive",
    *TRANSFORM,
    *VELOCITY,
    "health",
    "missiles_remaining",
    *ACTION_LABELS,
    *(f"{x}_id" for x in ACTION_LABELS),
    *PROBABILITIES,
    "missile_launched",
    "acmi_event",
    "radius_m",
    "has_csv_record",
    "has_acmi_record",
    "state_valid",
    "state_source",
    "action_status",
    "csv_row_number",
    "acmi_line_number",
    "acmi_time_s",
    *(f"acmi_{x}" for x in TRANSFORM),
    "acmi_tas_mps",
    "acmi_properties_json",
)
INTEGER_COLUMNS = {
    "step",
    "alive",
    "missiles_remaining",
    "has_csv_record",
    "has_acmi_record",
    "state_valid",
    "csv_row_number",
    "acmi_line_number",
    *(f"{x}_id" for x in ACTION_LABELS),
}
FLOAT_COLUMNS = {
    "time_s",
    *TRANSFORM,
    *VELOCITY,
    "health",
    "radius_m",
    *PROBABILITIES,
    "acmi_time_s",
    *(f"acmi_{x}" for x in TRANSFORM),
    "acmi_tas_mps",
}
ACTIONS = tuple(ACTION_LABELS)
STATE_FEATURES = TRANSFORM + VELOCITY
STATE_UNITS = {
    "lon_deg": "degrees",
    "lat_deg": "degrees",
    "alt_m": "m",
    "roll_deg": "degrees",
    "pitch_deg": "degrees",
    "yaw_deg": "degrees",
    "tas_mps": "m/s",
    "vn_mps": "m/s",
    "ve_mps": "m/s",
    "vu_mps": "m/s",
}

# --- Loader -----------------------------------------------------------------------------------

SUPPORTED_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "manifest.json"


class LagDataError(ValueError):
    """The LAG data or a request against it is invalid.

    Subclasses ``ValueError`` because the workspace loader raised plain ``ValueError``.
    """


@runtime_checkable
class EpisodeLoader(Protocol):
    """Swappable episode loader (D12): the instruction-conditioned recordings plug in here."""

    def __call__(self, path: Path, *, verify_hash: bool = True) -> pd.DataFrame: ...


def load_episode(path: str | Path, *, verify_hash: bool = True) -> pd.DataFrame:
    """Read one episode with nullable numeric types and string identifiers.

    Hash verification uses the manifest beside the CSV. Keep both together.
    Floating-point conversion is for analysis only; the CSV keeps original text.
    """
    path = Path(path)
    manifest = json.loads((path.parent / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise LagDataError("Unsupported schema version")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if verify_hash:
        matching = [e for e in manifest["episodes"] if e["output"]["path"] == path.name]
        if len(matching) != 1 or matching[0]["output"]["sha256"] != digest:
            raise LagDataError("CSV differs from its manifest")
    # First read all fields as text so NA-like labels and IDs are never interpreted as numbers.
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    schema = {x["name"]: x["dtype"] for x in manifest["columns"]}
    if list(frame.columns) != list(schema):
        raise LagDataError("CSV columns differ from manifest")
    for column, dtype in schema.items():
        frame[column] = frame[column].replace("", pd.NA)
        if dtype != "string":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)
    if frame["episode_id"].nunique() != 1 or frame.duplicated(["step", "object_id"]).any():
        raise LagDataError("Expected one episode with unique step/object keys")
    frame.attrs.update(csv_sha256=digest, schema_version=manifest["schema_version"], dt_s=manifest["dt_s"])
    return frame


def last_complete_step(frame: pd.DataFrame) -> int:
    """Return the last step at which all four aircraft have a CSV record.

    Later steps can exist in the ACMI log only (``has_csv_record == 0``); they carry no
    action or state from the CSV and must not be rendered.
    """
    aircraft = frame[frame["entity_type"].eq("aircraft") & frame["has_csv_record"].eq(1)]
    counts = aircraft.groupby("step")["object_id"].nunique()
    complete = counts[counts == len(AIRCRAFT)]
    if complete.empty:
        raise LagDataError("No step has CSV records for all aircraft")
    return int(complete.index.max())


# --- Helpers (13_LAG ``dataset.py``) ----------------------------------------------------------


def history_steps(
    query_step: int,
    *,
    length: int | None = None,
    stride: int = 1,
    offsets: Sequence[int] | None = None,
) -> list[int]:
    """Return chronological past sample steps; q itself is not a history sample.

    length=20, stride=5 at q=100 -> [0, 5, ..., 95].
    offsets=[-20, -10, -5, -1] at q=100 -> [80, 90, 95, 99].
    """
    if type(query_step) is not int or query_step < 0:
        raise LagDataError("query_step must be a nonnegative integer")
    if offsets is not None:
        if length is not None or stride != 1:
            raise LagDataError("Choose offsets OR length/stride")
        offsets = list(offsets)
        if not offsets or any(type(x) is not int or x >= 0 for x in offsets) or len(set(offsets)) != len(offsets):
            raise LagDataError("Offsets must be distinct negative integers")
        selected = sorted(query_step + x for x in offsets)
    else:
        if type(length) is not int or length < 1 or type(stride) is not int or stride < 1:
            raise LagDataError("length and stride must be positive integers")
        selected = list(range(query_step - length * stride, query_step, stride))
    if selected[0] < 0:
        raise LagDataError("History would extend before the episode")
    return selected


def select_records(
    frame: pd.DataFrame,
    *,
    steps: Iterable[int],
    object_ids: Iterable[str],
    columns: Iterable[str],
) -> pd.DataFrame:
    """Select exact recorded keys, without imputation, masking, or task semantics.

    Callers decide whether absent objects are acceptable for their experiment.
    Returned order is chronological, then the requested object order.
    """
    if frame["episode_id"].nunique() != 1:
        raise LagDataError("Select records from one episode at a time")
    steps, object_ids, columns = list(steps), list(object_ids), list(columns)
    if len(set(steps)) != len(steps) or len(set(object_ids)) != len(object_ids):
        raise LagDataError("Duplicate requested step or object ID")
    if not set(columns) <= set(frame.columns):
        raise LagDataError("Unknown columns")
    keys = ["step", "object_id"]
    selected = frame[frame["step"].isin(steps) & frame["object_id"].isin(object_ids)]
    result = selected[list(dict.fromkeys(keys + columns))].copy()
    order = {uid: i for i, uid in enumerate(object_ids)}
    result["_object_order"] = result["object_id"].map(order)
    return result.sort_values(["step", "_object_order"]).drop(columns="_object_order")


def index_records(frame: pd.DataFrame) -> dict[tuple[int, str], dict[str, Any]]:
    """Index a single episode once for repeated temporal lookups, without masking."""
    if frame["episode_id"].nunique() != 1 or frame.duplicated(["step", "object_id"]).any():
        raise LagDataError("Expected one episode with unique step/object keys")
    return frame.set_index(["step", "object_id"]).to_dict(orient="index")
