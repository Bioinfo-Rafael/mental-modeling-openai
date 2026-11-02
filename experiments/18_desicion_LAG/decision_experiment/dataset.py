"""Exact causal windows, with evaluation labels kept outside model inputs."""
from importlib import import_module

import pandas as pd

lag = import_module("experiments.13_LAG.lag_data.dataset")
definitions = import_module("experiments.13_LAG.lag_data.definitions")
ACTION_FIELDS = ("altitude_action_id", "heading_action_id", "speed_action_id")
METADATA = ("object_id", "entity_type", "team", "launcher_id", "alive", "state_valid",
            "acmi_event", "action_status")


def nullable(value):
    return None if pd.isna(value) else value


def build_blocks(frame):
    """Mask inactive states; never impute missing ammunition or unrecorded actions."""
    blocks = {}
    selected = frame[frame.entity_type.isin(("aircraft", "missile"))]
    for step, rows in selected.groupby("step", sort=True):
        aircraft = rows[rows.entity_type.eq("aircraft")]
        if set(aircraft.object_id) != set(definitions.AIRCRAFT) or not aircraft.has_csv_record.eq(1).all():
            continue  # ACMI-only terminal state is not a complete action time point.
        entities = []
        for _, row in rows.sort_values(["entity_type", "object_id"]).iterrows():
            obj = {field: nullable(row[field]) for field in METADATA}
            obj["state"] = [nullable(row[field]) if nullable(row.state_valid) == 1 else None
                            for field in definitions.STATE_FEATURES]
            if row.entity_type == "aircraft":
                obj["action"] = [nullable(row[field]) if row.action_status == "recorded" else None
                                 for field in ACTION_FIELDS]
                obj["missiles_remaining"] = nullable(row.missiles_remaining)
            entities.append(obj)
        blocks[int(step)] = {"step": int(step), "time_s": round(int(step) * .2, 1), "entities": entities}
    return blocks


def window(blocks, end, history):
    start = end - history + 1
    steps = list(range(start, end + 1))
    if start < 0 or any(step not in blocks for step in steps):
        raise ValueError(f"Incomplete H={history} window: {start}..{end}")
    return {"episode_alias": "episode_01", "start_step": start, "end_step": end,
            "time_points": history, "start_time_s": round(start * .2, 1),
            "end_time_s": round(end * .2, 1), "elapsed_s": round((history - 1) * .2, 1),
            "blocks": [blocks[step] for step in steps]}


def make_samples(stage, config, annotations, blocks):
    h = config["history"]
    cases = []
    if stage == "develop":
        for i, scene in enumerate(annotations["scenes"], 1):
            cases.extend([(f"case{i*2-1:02d}", scene["positive_end"], scene["events"]),
                          (f"case{i*2:02d}", scene["negative_end"], [])])
    else:
        first, last = min(blocks), max(blocks)
        ends = list(range(first + h - 1, last + 1, config["stride"]))
        if not ends:
            raise ValueError("Episode is shorter than the requested history")
        if ends[-1] != last:
            ends.append(last)
        cases = [(f"end_{end:04d}", end, None) for end in ends]
    samples = {}
    for sid, end, expected_ids in cases:
        data = window(blocks, end, h)
        truth = [event for event in annotations["events"] if data["start_step"] <= event["step"] <= end]
        if expected_ids is not None and set(expected_ids) != {e["id"] for e in truth}:
            raise ValueError(f"{sid}: selected window no longer contains the intended annotation group")
        samples[sid] = {"window": data, "expected_decision": "YES" if truth else "NO",
                        "expected_events": truth,
                        "label_basis": "annotation_present" if truth else "annotation_absent_not_reviewed"}
    return samples
