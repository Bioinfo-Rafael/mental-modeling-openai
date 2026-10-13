"""Lossless field-level join of the supplied ACMI and action CSVs (stdlib only).

Run from any directory: python experiments/13_LAG/run.py
No interpolation, numerical reformatting, or API calls. See data/processed/README.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "data/raw/20261005_LAG_2v2_ACMI"
OUTPUT = HERE / "data/processed"
from .definitions import (
    ACTION_LABELS, AIRCRAFT, COLUMNS, CSV_FIELDS, CSV_RENAMES, DT,
    FLOAT_COLUMNS, INTEGER_COLUMNS, PROBABILITIES, TRANSFORM,
)

def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def time_step(value):
    number = Decimal(value) / DT
    require(number.is_finite() and number >= 0 and number == number.to_integral_value(),
            f"Timestamp is not on the 0.2 s grid: {value}")
    return int(number)


def read_actions(path):
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == CSV_FIELDS, f"Unexpected CSV schema: {path}")
        for row_number, row in enumerate(reader, 2):
            require(None not in row and None not in row.values(), f"Malformed CSV row {row_number}")
            step = int(row["step"])
            require(step == time_step(row["time_s"]), f"CSV step/time mismatch: {row_number}")
            require(row["uid"] in AIRCRAFT, f"Unexpected aircraft: {row['uid']}")
            key = (step, row["uid"])
            require(key not in records, f"Duplicate CSV key: {key}")
            records[key] = (row, row_number)
    require(bool(records), "Empty action CSV")
    steps = sorted({k[0] for k in records})
    require(steps == list(range(steps[-1] + 1)), "Missing CSV timestep")
    require(set(records) == {(s, uid) for s in steps for uid in AIRCRAFT},
            "Each CSV step must contain exactly four aircraft")
    return records


def read_acmi(path):
    records, metadata, registry = {}, {}, {}
    timestamp = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line or line.startswith("//"):
            continue
        if line.startswith(("FileType=", "FileVersion=")):
            key, value = line.split("=", 1)
            metadata[key] = value
            continue
        if line.startswith("#"):
            timestamp = line[1:]
            time_step(timestamp)
            continue
        if line.startswith("-"):
            uid, props, event = line[1:], {}, "remove"
        else:
            uid, *parts = line.split(",")
            props = {}
            for part in parts:
                require("=" in part, f"Unsupported ACMI field at line {line_number}")
                key, value = part.split("=", 1)
                require(key not in props, f"Duplicate ACMI property at line {line_number}")
                props[key] = value
            if uid == "0":
                metadata.update(props)
                continue
            event = "update"
        require(timestamp is not None, "ACMI object before timestamp")
        key = (time_step(timestamp), uid)
        require(key not in records, f"Multiple ACMI records at {key}; revise schema instead of discarding")
        records[key] = {"time_s": timestamp, "line_number": line_number,
                        "event": event, "properties": props}
        identity = registry.setdefault(uid, {})
        for name in ("Name", "Color", "Type"):
            if name in props:
                require(name not in identity or identity[name] == props[name],
                        f"Changing object identity: {uid}/{name}")
                identity[name] = props[name]
    require(bool(records), "Empty ACMI")
    for uid, identity in registry.items():
        name = identity.get("Name")
        kind = ("aircraft" if name == "F16" else "missile" if name == "AIM-9L"
                else "effect" if identity.get("Type") == "Misc+Explosion" else None)
        require(kind is not None, f"Unrecognized object type: {uid}: {identity}")
        require(identity.get("Color") in ("Red", "Blue"), f"Unknown team: {uid}")
        identity["entity_type"] = kind
    return records, metadata, registry


def build_episode(acmi_path):
    csv_path = acmi_path.parent / "actions" / f"{acmi_path.stem}_actions.csv"
    actions = read_actions(csv_path)
    acmi, metadata, registry = read_acmi(acmi_path)
    require({uid for uid, x in registry.items() if x["entity_type"] == "aircraft"} == set(AIRCRAFT),
            "Expected the four known aircraft")
    controllers, launches = {}, {}
    for (step, uid), (record, _) in actions.items():
        controller = record["controller"]
        require(controller in ("learned", "rule_pursue"), f"Unknown controller: {controller}")
        require(controllers.get(uid, controller) == controller, f"Controller changed: {uid}")
        controllers[uid] = controller
        missile = record["missile_launched"]
        if missile:
            require(missile not in launches, f"Repeated missile launch: {missile}")
            launches[missile] = (uid, step)
    require(set(launches) == {uid for uid, x in registry.items() if x["entity_type"] == "missile"},
            "CSV launches and ACMI missile IDs differ")

    rows, max_diff = [], {x: 0.0 for x in (*TRANSFORM, "tas_mps")}
    rank = {"aircraft": 0, "missile": 1, "effect": 2}
    keys = sorted(set(actions) | set(acmi), key=lambda k: (k[0], rank[registry[k[1]]["entity_type"]], k[1]))
    for step, uid in keys:
        identity = registry[uid]
        kind = identity["entity_type"]
        source_csv, csv_number = actions.get((step, uid), ({}, ""))
        source_acmi = acmi.get((step, uid), {})
        props = source_acmi.get("properties", {})
        row = dict.fromkeys(COLUMNS, "")
        row.update(episode_id=acmi_path.stem, step=str(step),
                   time_s=source_csv.get("time_s", source_acmi.get("time_s", "")),
                   object_id=uid, entity_type=kind,
                   color=identity["Color"], team={"Red": "A", "Blue": "B"}[identity["Color"]],
                   name=identity.get("Name", ""), acmi_type=identity.get("Type", ""),
                   launcher_id=launches[uid][0] if uid in launches else "",
                   controller=controllers.get(uid, ""), csv_row_number=str(csv_number),
                   acmi_line_number=str(source_acmi.get("line_number", "")),
                   acmi_time_s=source_acmi.get("time_s", ""),
                   acmi_event=source_acmi.get("event", "none"),
                   has_csv_record=str(int(bool(source_csv))), has_acmi_record=str(int(bool(source_acmi))))
        for key, value in source_csv.items():
            dest = CSV_RENAMES.get(key, key)
            if key == "color":
                require(value == identity["Color"], f"CSV/ACMI team mismatch: {uid}")
            row[dest] = value
        if source_acmi:
            row["acmi_properties_json"] = json.dumps(props, ensure_ascii=False, separators=(",", ":"))
        if "T" in props:
            values = props["T"].split("|")
            require(len(values) == 6, f"Unsupported transform: {uid}/{step}")
            for field, value in zip(TRANSFORM, values):
                row[f"acmi_{field}"] = value
                if not source_csv:
                    row[field] = value
        row["acmi_tas_mps"] = props.get("TAS", "")
        if not source_csv:
            row["tas_mps"] = row["acmi_tas_mps"]
        row.update(health=props.get("Health", ""), missiles_remaining=props.get("Missiles", ""),
                   radius_m=props.get("Radius", ""))
        complete_state = all(row[x] != "" for x in TRANSFORM)
        row["state_source"] = "csv" if source_csv else "acmi" if "T" in props else "none"
        row["state_valid"] = str(int(complete_state and row["alive"] != "0" and row["acmi_event"] != "remove"))
        present_actions = [bool(row[x]) for x in ACTION_LABELS]
        require(all(present_actions) or not any(present_actions), f"Partial action at {step}/{uid}")
        if all(present_actions):
            require(kind == "aircraft" and row["alive"] == "1" and row["controller"] == "learned",
                    f"Unexpected action record: {step}/{uid}")
            row["action_status"] = "recorded"
            for action, labels in ACTION_LABELS.items():
                require(row[action] in labels, f"Unknown action label: {row[action]}")
                row[f"{action}_id"] = str(labels.index(row[action]))
                probabilities = [float(row[f"p_{action.removesuffix('_action')}_{label}"]) for label in labels]
                require(all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities)
                        and abs(sum(probabilities) - 1) <= 1e-6, f"Invalid probabilities: {step}/{uid}")
                require(probabilities[labels.index(row[action])] == max(probabilities),
                        f"Action does not match argmax: {step}/{uid}")
        else:
            require(not any(row[x] for x in PROBABILITIES), f"Probabilities without action: {step}/{uid}")
            row["action_status"] = ("not_applicable" if kind != "aircraft" else "inactive"
                                    if row["alive"] == "0" or row["acmi_event"] == "remove" else
                                    "state_only" if not source_csv else "not_recorded")
        if row["alive"] == "1":
            require(row["acmi_event"] == "update" and all(row[f"acmi_{x}"] for x in TRANSFORM),
                    f"Missing live aircraft ACMI state: {step}/{uid}")
            for field in (*TRANSFORM, "tas_mps"):
                difference = abs(float(row[field]) - float(row[f"acmi_{field}"]))
                max_diff[field] = max(max_diff[field], difference)
                require(difference <= (0.050000001 if field == "tas_mps" else 1e-8),
                        f"CSV/ACMI state mismatch: {step}/{uid}/{field}")
            if row["controller"] == "learned":
                require(row["action_status"] == "recorded", f"Missing learned action: {step}/{uid}")
        for field in FLOAT_COLUMNS:
            require(not row[field] or math.isfinite(float(row[field])), f"Nonfinite value: {field}")
        for field in INTEGER_COLUMNS:
            if row[field]:
                int(row[field])
        rows.append(row)

    # Field-for-field source coverage, including blank cells, decimal text, and deletion records.
    indexed = {(int(r["step"]), r["object_id"]): r for r in rows}
    for key, (source, _) in actions.items():
        require(all(indexed[key][CSV_RENAMES.get(k, k)] == v for k, v in source.items()),
                f"Lost CSV field: {key}")
    for key, source in acmi.items():
        row = indexed[key]
        require(json.loads(row["acmi_properties_json"]) == source["properties"]
                and row["acmi_event"] == source["event"], f"Lost ACMI record: {key}")
    require(max(s for s, _ in acmi) == max(s for s, _ in actions) + 1, "Unexpected final-state alignment")
    launch_report = []
    for missile, (uid, step) in sorted(launches.items()):
        first = min(s for (s, obj), rec in acmi.items() if obj == missile and rec["event"] == "update")
        require(first >= step, f"Missile appears before recorded launch: {missile}")
        require(registry[missile]["Color"] == registry[uid]["Color"], f"Missile team mismatch: {missile}")
        launch_report.append(dict(missile_id=missile, launcher_id=uid, action_step=step,
                                  first_acmi_step=first, offset_steps=first-step))
    summary = dict(
        episode_id=acmi_path.stem, rows=len(rows),
        rows_by_entity_type=dict(sorted(Counter(r["entity_type"] for r in rows).items())),
        object_ids_by_entity_type={kind: sorted(uid for uid, x in registry.items() if x["entity_type"] == kind)
                                   for kind in rank},
        action_status_counts=dict(sorted(Counter(r["action_status"] for r in rows).items())),
        source_csv_rows=len(actions), source_acmi_records=len(acmi),
        acmi_removals=sum(x["event"] == "remove" for x in acmi.values()),
        csv_last_step=max(s for s, _ in actions), acmi_last_step=max(s for s, _ in acmi),
        csv_last_time_s=float(max(s for s, _ in actions)*DT),
        acmi_last_time_s=float(max(s for s, _ in acmi)*DT),
        max_live_state_difference=max_diff, launches=launch_report, acmi_metadata=metadata,
        validation="passed",
        source_files=[dict(path=str(p.relative_to(HERE)), bytes=p.stat().st_size,
                           sha256=sha256(p.read_bytes())) for p in (acmi_path, csv_path)],
    )
    return rows, summary


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_outputs():
    paths = sorted(RAW.glob("*.acmi"))
    require(len(paths) == 3, "Expected exactly three matches")
    outputs, episodes = {}, []
    for path in paths:
        rows, summary = build_episode(path)
        data = csv_bytes(rows)
        name = path.stem + ".csv"
        outputs[name] = data
        summary["output"] = dict(path=name, bytes=len(data), sha256=sha256(data))
        episodes.append(summary)
    manifest = dict(
        schema_version="1.0", dt_s=0.2, encoding="UTF-8", delimiter=",", missing_value="",
        primary_key=["episode_id", "step", "object_id"],
        sort_order=["step", "entity_type (aircraft, missile, effect)", "object_id"],
        columns=[dict(name=x, dtype="Int64" if x in INTEGER_COLUMNS else "Float64"
                      if x in FLOAT_COLUMNS else "string") for x in COLUMNS],
        action_labels_by_id=ACTION_LABELS,
        generator={str(p.relative_to(HERE)): sha256(p.read_bytes())
                   for p in (Path(__file__), Path(__file__).with_name("definitions.py"))},
        episodes=episodes,
    )
    outputs["manifest.json"] = json_text(manifest).encode("utf-8")
    return outputs, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild in memory and compare with saved outputs")
    parser.add_argument("--overwrite", action="store_true", help="Replace changed derived outputs only")
    args = parser.parse_args(argv)
    require(not (args.check and args.overwrite), "--check and --overwrite are mutually exclusive")
    outputs, manifest = build_outputs()
    # Check every destination before writing any of them. Identical files are left untouched.
    for name, data in outputs.items():
        path = OUTPUT / name
        if args.check:
            require(path.exists() and path.read_bytes() == data, f"Missing or changed output: {path}")
        elif path.exists() and path.read_bytes() != data and not args.overwrite:
            raise FileExistsError(f"Changed derived output: {path}; use --overwrite to regenerate")
    if not args.check:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            path = OUTPUT / name
            if not path.exists() or path.read_bytes() != data:
                pending = path.with_suffix(path.suffix + ".tmp")
                pending.write_bytes(data)
                pending.replace(path)
    for item in manifest["episodes"]:
        print(f"{item['episode_id']}: {item['rows']} rows, {item['rows_by_entity_type']}, validation passed")
    print("Saved files match regenerated data." if args.check else f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
