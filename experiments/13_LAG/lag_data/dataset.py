"""Typed loading and record selection, independent of prediction experiments."""
from pathlib import Path
import hashlib
import json

import pandas as pd

from .definitions import ACTIONS, AIRCRAFT, STATE_FEATURES

HERE = Path(__file__).resolve().parents[1]
PROCESSED = HERE / "data/processed"

def load_episode(path, *, verify_hash=True):
    """Read one episode with nullable numeric types and string identifiers.

    Hash verification uses the manifest beside the CSV. Keep both together.
    Floating-point conversion is for analysis only; the CSV keeps original text.
    """
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "1.0":
        raise ValueError("Unsupported schema version")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if verify_hash:
        matching = [e for e in manifest["episodes"] if e["output"]["path"] == path.name]
        if len(matching) != 1 or matching[0]["output"]["sha256"] != digest:
            raise ValueError("CSV differs from its manifest")
    # First read all fields as text so NA-like labels and IDs are never interpreted as numbers.
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    schema = {x["name"]: x["dtype"] for x in manifest["columns"]}
    if list(frame.columns) != list(schema):
        raise ValueError("CSV columns differ from manifest")
    for column, dtype in schema.items():
        frame[column] = frame[column].replace("", pd.NA)
        if dtype != "string":
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)
    if frame["episode_id"].nunique() != 1 or frame.duplicated(["step", "object_id"]).any():
        raise ValueError("Expected one episode with unique step/object keys")
    frame.attrs.update(csv_sha256=digest, schema_version=manifest["schema_version"], dt_s=manifest["dt_s"])
    return frame


def history_steps(query_step, *, length=None, stride=1, offsets=None):
    """Return chronological past sample steps; q itself is not a history sample.

    length=20, stride=5 at q=100 -> [0, 5, ..., 95].
    offsets=[-20, -10, -5, -1] at q=100 -> [80, 90, 95, 99].
    """
    if type(query_step) is not int or query_step < 0:
        raise ValueError("query_step must be a nonnegative integer")
    if offsets is not None:
        if length is not None or stride != 1:
            raise ValueError("Choose offsets OR length/stride")
        offsets = list(offsets)
        if not offsets or any(type(x) is not int or x >= 0 for x in offsets) or len(set(offsets)) != len(offsets):
            raise ValueError("Offsets must be distinct negative integers")
        selected = sorted(query_step + x for x in offsets)
    else:
        if type(length) is not int or length < 1 or type(stride) is not int or stride < 1:
            raise ValueError("length and stride must be positive integers")
        selected = list(range(query_step - length * stride, query_step, stride))
    if selected[0] < 0:
        raise ValueError("History would extend before the episode")
    return selected


def select_records(frame, *, steps, object_ids, columns):
    """Select exact recorded keys, without imputation, masking, or task semantics.

    Callers decide whether absent objects are acceptable for their experiment.
    Returned order is chronological, then the requested object order.
    """
    if frame["episode_id"].nunique() != 1:
        raise ValueError("Select records from one episode at a time")
    steps, object_ids, columns = list(steps), list(object_ids), list(columns)
    if len(set(steps)) != len(steps) or len(set(object_ids)) != len(object_ids):
        raise ValueError("Duplicate requested step or object ID")
    if not set(columns) <= set(frame.columns):
        raise ValueError("Unknown columns")
    keys = ["step", "object_id"]
    selected = frame[frame["step"].isin(steps) & frame["object_id"].isin(object_ids)]
    result = selected[list(dict.fromkeys(keys + columns))].copy()
    order = {uid: i for i, uid in enumerate(object_ids)}
    result["_object_order"] = result["object_id"].map(order)
    return result.sort_values(["step", "_object_order"]).drop(columns="_object_order")


def index_records(frame):
    """Index a single episode once for repeated temporal lookups, without masking."""
    if frame["episode_id"].nunique() != 1 or frame.duplicated(["step", "object_id"]).any():
        raise ValueError("Expected one episode with unique step/object keys")
    return frame.set_index(["step", "object_id"]).to_dict(orient="index")

