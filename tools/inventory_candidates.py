#!/usr/bin/env python3
"""Read-only, exhaustive candidate inventory. Never imports dataset scripts or models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatlabOpaque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.inventory_llmx_data import _atomic_text, array_spec

PROCESSED = ["frame_number", "aircraft_id", "x_km", "y_km", "z_km", "wind_x_mps", "wind_y_mps"]
CATEGORICAL = re.compile(r"(^id$|aircraft|^date$|^tail$|station|altisgnss|model|method|preprocess|winner|regressor|^type$|^flight$|tactical|status|redformation|question)", re.I)


def fingerprint(schema):
    """Schema excludes row counts, values and missing counts, but retains dtype/order."""
    return hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def stats(values):
    values = list(values)
    return {"count": len(values), "min": min(values), "max": max(values), "mean": statistics.fmean(values), "median": statistics.median(values)} if values else {"count": 0, "min": None, "max": None, "mean": None, "median": None}


def time_stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    d = np.diff(a)
    return {"start": float(a[0]) if a.size else None, "end": float(a[-1]) if a.size else None,
            "duration": float(a[-1] - a[0]) if a.size else None,
            "delta": stats(d.tolist()), "positive_delta": stats(d[d > 0].tolist()),
            "zero_deltas": int((d == 0).sum()), "negative_deltas": int((d < 0).sum())}


def merge_dtype(types):
    types = set(types)
    if not types:
        return "unknown(empty)"
    if any(t in {"object", "str", "string"} for t in types):
        return "str/mixed" if len(types) > 1 else "str"
    return str(np.result_type(*[np.dtype(t) for t in types]))


def inspect_csv(path, *, sep=",", names=None, chunksize=50000, weather=False):
    """Read every record in bounded chunks; report reader dtype, not a CSV native type."""
    types, missing, unique = defaultdict(set), Counter(), defaultdict(set)
    cols, rows, first = [], 0, None
    times = []
    raw_last, raw_track_rows, raw_deltas = {}, Counter(), Counter()
    if path.stat().st_size == 0:
        return {"rows": 0, "columns": 0, "shape": [0, 0], "schema": {"kind": "empty_csv", "fields": []}, "missing": {}, "unique": {}, "example": None}
    reader = pd.read_csv(path, sep=sep, names=names, header=None if names else 0,
                         chunksize=chunksize, skipinitialspace=True, na_values=["M"] if weather else None)
    for chunk in reader:
        cols = [str(c) for c in chunk.columns]
        rows += len(chunk)
        if {"ID", "Time", "Date"}.issubset(chunk.columns):
            ts = pd.to_datetime(chunk["Date"].astype(str) + " " + chunk["Time"].astype(str), format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
            for ident, indexes in chunk.groupby("ID", sort=False).groups.items():
                # pandas resolution can vary; convert explicitly to milliseconds.
                values = ts.loc[indexes].dropna().to_numpy(dtype="datetime64[ms]").astype("int64")
                raw_track_rows[str(ident)] += len(indexes)
                if len(values):
                    if str(ident) in raw_last:
                        raw_deltas[int(values[0] - raw_last[str(ident)])] += 1
                    raw_deltas.update(map(int, np.diff(values)))
                    raw_last[str(ident)] = int(values[-1])
        if first is None and len(chunk):
            first = {str(k): None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v) for k, v in chunk.iloc[0].items()}
        for col in chunk:
            s = chunk[col]
            if s.notna().any():
                types[str(col)].add(str(s.dtype))
            missing[str(col)] += int(s.isna().sum())
            if CATEGORICAL.search(str(col)):
                unique[str(col)].update(str(v) for v in s.dropna().unique())
            if str(col) in {"m-currentPhysicsTime-SEC", "m_currentPhysicsTime_SEC"}:
                times.extend(pd.to_numeric(s, errors="coerce").tolist())
    schema = {"kind": "delimited", "delimiter": sep, "header": names is None,
              "fields": [{"name": c, "dtype": merge_dtype(types[c])} for c in cols]}
    out = {"rows": rows, "columns": len(cols), "shape": [rows, len(cols)], "schema": schema,
           "missing": {c: missing[c] for c in cols}, "unique": {c: len(v) for c, v in unique.items()},
           "categorical_values": {c: sorted(v) for c, v in unique.items()}, "example": first,
           "timestamp_candidates": [c for c in cols if re.search("time|date|frame|valid", c, re.I)],
           "id_candidates": [c for c in cols if re.search(r"(^ID$|aircraft|Case|Tail|episode|trajectory)", c, re.I)]}
    if times:
        out["time"] = time_stats(times)
    if raw_track_rows:
        out["raw_track_rows"] = dict(raw_track_rows)
        out["raw_time_delta_ms_counts"] = {str(k): v for k, v in sorted(raw_deltas.items())}
    return out


def inspect_mat(path):
    try:
        classes = {n: c for n, _, c in whosmat(path)}
    except (TypeError, ValueError):
        classes = {}
    # Avoid materializing enormous MCOS __function_workspace__ payloads (one is >1 GB).
    names = [n for n in classes if not n.startswith("__")] or ["vals", "chirp", "out", "control_IN", "time"]
    data = loadmat(path, struct_as_record=True, squeeze_me=False, variable_names=names)
    variables = {}
    for name, value in data.items():
        if name.startswith("__") and name != "__function_workspace__":
            continue
        spec = array_spec(np.asarray(value))
        spec["python_type"] = type(value).__module__ + "." + type(value).__name__
        spec["matlab_class"] = classes.get(name, "opaque" if isinstance(value, MatlabOpaque) else "unknown")
        spec["struct_fields"] = list(value.dtype.names or [])
        spec["metadata_only"] = name == "__function_workspace__"
        spec["opaque"] = isinstance(value, MatlabOpaque)
        if spec["opaque"]:
            # MATLAB table payload is not exposed as a numeric table by scipy.
            spec["rows_unknown"] = True
            spec["matlab_object_class"] = str(value[0]["_Class"]) if "_Class" in spec["struct_fields"] else "unknown"
        if name == "vals" and value.ndim == 2 and value.shape[1] == 55 and not value.dtype.hasobject:
            spec["time_column_zero_based"] = 37
            spec["time"] = time_stats(value[:, 37])
        variables[name] = spec
    schema = {"kind": "MAT", "variables": {k: {p: v[p] for p in ("dtype", "shape", "matlab_class", "struct_fields", "opaque", "metadata_only")} for k, v in variables.items()}}
    return {"schema": schema, "variables": variables, "rows": None, "columns": None,
            "metadata_policy": "__header__, __version__, __globals__, MCOS __function_workspace__ are not trajectory variables; large internal workspace payload not materialized",
            "variable_discovery": "whosmat" if classes else "whosmat cannot resolve opaque; known downloaded variable names explicitly selected; no internal MCOS schema inferred"}


def inspect_zip(path):
    with zipfile.ZipFile(path) as z:
        entries = [{"path": i.filename, "size_bytes": i.file_size, "compressed_bytes": i.compress_size,
                    "format": Path(i.filename).suffix.lower(), "crc32": i.CRC} for i in z.infolist() if not i.is_dir()]
    return {"schema": {"kind": "archive", "formats": sorted(set(i["format"] for i in entries))},
            "archive": {"file_count": len(entries), "uncompressed_bytes": sum(i["size_bytes"] for i in entries),
                        "format_counts": dict(Counter(i["format"] for i in entries)), "entries": entries}, "rows": None, "columns": None}


def inspect_text(path):
    line_count, blank, lengths = 0, 0, Counter()
    encounters, rule_count, rules, unmatched = [], 0, set(), []
    first = []
    with path.open(errors="strict", encoding="utf-8-sig") as f:
        for line in f:
            line_count += 1
            s = line.strip()
            if len(first) < 4:
                first.append(s[:250])
            if not s:
                blank += 1
                continue
            lengths[len(s.split())] += 1
            if path.name.startswith("scripts-"):
                if m := re.fullmatch(r"Encounter (\d+)", s):
                    encounters.append(int(m[1]))
                elif m := re.fullmatch(r"\[([^]]+)\]\s+\[([^]]+)\]\s+(.+)", s):
                    float(m[1]); int(m[2]); rules.add(m[3]); rule_count += 1
                else:
                    unmatched.append(s[:150])
    out = {"rows": line_count, "columns": None, "physical_lines": line_count, "blank_lines": blank,
           "whitespace_width_counts": dict(lengths), "example_lines": first}
    if path.name.startswith("scripts-"):
        out.update(schema={"kind": "encounter_script", "delimiter": "Encounter N blocks; [weight] [priority] rule", "header": False,
                           "fields": [{"name": n, "dtype": d} for n, d in [("encounter", "int64"), ("weight", "float64"), ("priority", "int64"), ("rule", "str")]]},
                   encounter_count=len(encounters), encounter_min=min(encounters, default=None), encounter_max=max(encounters, default=None),
                   encounter_sequence_contiguous=encounters == list(range(1, len(encounters)+1)), rule_rows=rule_count, rule_names=sorted(rules), unmatched_lines=unmatched)
    elif path.name == "SimulationConfiguration.txt":
        values = dict(re.findall(r"(\w+)=([^,\]]+)", " ".join(first)))
        out.update(schema={"kind": "configuration", "fields": [{"name": k, "dtype": "float64" if "." in v else "int64"} for k, v in sorted(values.items())]}, configuration=values)
    else:
        out["schema"] = {"kind": "unstructured_text", "delimiter": None, "header": None, "fields": []}
    return out


def inspect_ulog(path):
    from pyulog import ULog
    log = ULog(str(path))
    streams = []
    for topic in sorted(log.data_list, key=lambda t: (t.name, t.multi_id)):
        arrays = {name: array_spec(value) for name, value in topic.data.items()}
        rows = len(next(iter(topic.data.values()))) if topic.data else 0
        streams.append({"name": topic.name, "multi_id": topic.multi_id, "rows": rows, "arrays": arrays,
                        "time_seconds": time_stats(topic.data["timestamp"].astype(float)/1e6) if "timestamp" in topic.data else None})
    return {"schema": {"kind": "ULog", "streams": [{"name": s["name"], "multi_id": s["multi_id"], "fields": [{"name": k, "dtype": v["dtype"], "shape_tail": v["shape"][1:]} for k, v in sorted(s["arrays"].items())]} for s in streams]},
            "streams": streams, "rows": sum(s["rows"] for s in streams), "columns": None,
            "duration_seconds": (log.last_timestamp - log.start_timestamp)/1e6}


def inspect_xlsx(path):
    """Read all OOXML cells, including formulas and cached types; do not recalculate."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheets = []
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = {e.attrib["Id"]: e.attrib["Target"] for e in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
        for sheet in wb.find("s:sheets", ns):
            target = rels[sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            cells, types, formulas, labels = 0, Counter(), 0, []
            root = ET.fromstring(z.read(target))
            for cell in root.findall(".//s:c", ns):
                v = cell.find("s:v", ns)
                text = v.text if v is not None else "".join(cell.itertext())
                if not text:
                    continue
                cells += 1
                typ = cell.get("t", "n")
                types[typ] += 1
                formulas += cell.find("s:f", ns) is not None
                if typ in {"s", "inlineStr", "str"}:
                    labels.append({"cell": cell.get("r"), "text": strings[int(text)] if typ == "s" else text})
            dim = root.find("s:dimension", ns)
            sheets.append({"name": sheet.get("name"), "dimension": dim.get("ref") if dim is not None else None,
                           "populated_cells": cells, "cell_types": dict(types), "formula_count": formulas, "text_cells": labels})
    return {"schema": {"kind": "XLSX", "sheets": [{"name": s["name"], "dimension": s["dimension"], "cell_types": sorted(s["cell_types"])} for s in sheets]}, "sheets": sheets, "rows": None, "columns": None}


def inspect_processed(path):
    if path.stat().st_size == 0:
        return {"rows": 0, "columns": 7, "schema": {"kind": "empty_processed", "fields": []}, "track_rows": {}, "frame_deltas": {}, "aircraft_ids": [], "max_simultaneous": 0}
    r = inspect_csv(path, sep=r"\s+", names=PROCESSED)
    # At most one scene (largest 25,386 rows); no dataset-wide concatenation.
    df = pd.read_csv(path, sep=r"\s+", names=PROCESSED)
    tracks, deltas, segments = {}, Counter(), 0
    for ident, track in df.groupby("aircraft_id", sort=False):
        frames = track.frame_number.to_numpy()
        delta = np.diff(frames)
        tracks[str(ident)] = len(track)
        deltas.update(map(float, delta))
        segments += 1 + int((delta != 1).sum())
    r.update(track_rows=tracks, frame_deltas={str(k): v for k, v in sorted(deltas.items())},
             contiguous_1frame_segments=segments, aircraft_ids=sorted(tracks),
             max_simultaneous=int(df.groupby("frame_number").aircraft_id.nunique().max()),
             duplicate_frame_id_rows=int(df.duplicated(["frame_number", "aircraft_id"]).sum()))
    return r


def inspect_one(path, dataset):
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return inspect_zip(path)
    if suffix == ".csv" and path.name == "winHistory.csv":
        with path.open() as f:
            physical = list(csv.reader(f, skipinitialspace=True))
        values = [v.strip() for row in physical for v in row]
        return {"schema": {"kind": "winner_sequence", "delimiter": ",", "header": False, "fields": [{"name": "winner", "dtype": "str"}]},
                "rows": len(physical), "columns": len(physical[0]) if physical else 0,
                "logical_records": len(values), "winner_counts": dict(Counter(values)), "unique": {"winner": len(set(values))}}
    if suffix in {".csv", ".tsv"}:
        with path.open(encoding="utf-8-sig") as f:
            first = f.readline()
        sep = "\t" if suffix == ".tsv" else (";" if first.count(";") > first.count(",") else ",")
        return inspect_csv(path, sep=sep, weather="weather" in str(path))
    if suffix == ".mat":
        return inspect_mat(path)
    if suffix == ".ulg":
        return inspect_ulog(path)
    if suffix == ".xlsx":
        return inspect_xlsx(path)
    if suffix in {".npz", ".npy"}:
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                specs = {k: array_spec(z[k]) for k in z.files}
        else:
            specs = {"array": array_spec(np.load(path, allow_pickle=False))}
        return {"schema": {"kind": suffix, "variables": specs}, "rows": None, "columns": None}
    if suffix == ".txt" and dataset == "trajair" and "processed_data" in path.parts:
        return inspect_processed(path)
    if suffix in {".txt", ".log", ".dat", ""}:
        try:
            return inspect_text(path)
        except UnicodeDecodeError:
            return {"schema": {"kind": "unparsed_binary"}, "rows": None, "columns": None, "limitation": "Binary content; no safe reader selected"}
    return {"schema": {"kind": "support_file", "format": suffix}, "rows": None, "columns": None,
            "inspection": "file metadata; documentation/source/model/media, not loaded or executed"}


def source_files(root):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(p in {".git", ".cache", "__pycache__"} for p in path.parts):
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2 or rel.parts[1] in {"README.md", "SCHEMA.md", "NOT_ACQUIRED.md"}:
            continue  # authored wrappers are not downloaded source data
        yield path


def scan(root):
    records, groups = [], {}
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_files(root)}
    for i, filename in enumerate(before):
        path = Path(filename)
        dataset = path.relative_to(root).parts[0]
        record = {"dataset": dataset, "relative_path": str(path.relative_to(root / dataset)),
                  "format": path.suffix.lower() or "no_extension", "size_bytes": path.stat().st_size}
        try:
            detail = inspect_one(path, dataset)
            schema = detail.pop("schema")
            group_id = dataset + ":" + fingerprint(schema)
            record.update(detail, schema_group=group_id, status="inspected")
            if group_id not in groups:
                groups[group_id] = {"dataset": dataset, "schema": schema, "file_count": 0, "rows": 0, "missing": Counter(), "example_file": record["relative_path"]}
            group = groups[group_id]
            group["file_count"] += 1
            group["rows"] += record.get("rows") or 0
            group["missing"].update(record.get("missing", {}))
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}", rows=None, columns=None, schema_group="ERROR")
        records.append(record)
        if (i + 1) % 1000 == 0:
            print(f"Scanned {i + 1}/{len(before)}", flush=True)
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_files(root)}
    assert before == after, "Source file size/mtime changed during scan"
    totals = {}
    for dataset in sorted({r["dataset"] for r in records}):
        files = [r for r in records if r["dataset"] == dataset]
        archives = [r for r in files if r["format"] == ".zip"]
        totals[dataset] = {"file_count": len(files), "disk_bytes_excluding_git_cache": sum(r["size_bytes"] for r in files),
                           "compressed_bytes_all_zip_including_nested": sum(r["size_bytes"] for r in archives),
                           "compressed_bytes_top_level": sum(r["size_bytes"] for r in archives if "extracted/" not in r["relative_path"]),
                           "extracted_bytes_excluding_nested_zip": sum(r["size_bytes"] for r in files if r not in archives),
                           "format_counts": dict(Counter(r["format"] for r in files)),
                           "schema_group_count": sum(g["dataset"] == dataset for g in groups.values()),
                           "errors": [r["relative_path"] for r in files if r["status"] == "error"]}
    processed = [r for r in records if "track_rows" in r]
    raw = [r for r in records if r["dataset"] == "trajair" and "/raw_data/" in r["relative_path"]]
    weather = [r for r in records if r["dataset"] == "trajair" and r["format"] == ".csv" and "weather" in r["relative_path"]]
    track_rows = [n for r in processed for n in r["track_rows"].values()]
    deltas = Counter()
    for r in processed:
        deltas.update(r["frame_deltas"])
    trajair = {"processed_scenes": len(processed), "empty_files": [r["relative_path"] for r in processed if not r["rows"]],
               "processed_rows": sum(r["rows"] for r in processed), "file_aircraft_tracks": len(track_rows),
               "rows_per_track": stats(track_rows), "rows_per_scene": stats(r["rows"] for r in processed),
               "aircraft_ids_count": len({k for r in processed for k in r["aircraft_ids"]}),
               "one_frame_contiguous_segments": sum(r.get("contiguous_1frame_segments", 0) for r in processed),
               "multi_aircraft_scenes": sum(r["max_simultaneous"] > 1 for r in processed), "max_simultaneous": max((r["max_simultaneous"] for r in processed), default=0),
               "duplicate_frame_id_rows": sum(r.get("duplicate_frame_id_rows", 0) for r in processed), "frame_delta_counts": dict(deltas),
               "raw_files": len(raw), "raw_rows": sum(r["rows"] or 0 for r in raw), "raw_dates": sorted({d for r in raw for d in r.get("categorical_values", {}).get("Date", [])}),
               "raw_aircraft_ids": len({d for r in raw for d in r.get("categorical_values", {}).get("ID", [])}),
               "weather_files": len(weather), "weather_rows": sum(r["rows"] or 0 for r in weather)}
    raw_delta = Counter()
    for r in raw:
        raw_delta.update(r.get("raw_time_delta_ms_counts", {}))
    trajair["raw_time_delta_ms_counts"] = dict(raw_delta)
    trajair["raw_file_aircraft_tracks"] = sum(len(r.get("raw_track_rows", {})) for r in raw)
    trajair["raw_rows_per_file_aircraft_track"] = stats(v for r in raw for v in r.get("raw_track_rows", {}).values())
    return {"schema_version": 1, "source_root": str(root), "scope": "all downloaded files excluding .git/.cache and authored wrapper documents; support files metadata only",
            "source_size_mtime_unchanged": before == after, "files": records, "groups": groups, "datasets": totals, "trajair": trajair}


def write_results(result, output):
    _atomic_text(output / "candidate_schema.json", json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    b = io.StringIO()
    fields = ["dataset", "relative_path", "format", "size_bytes", "rows", "columns", "schema_group", "status"]
    writer = csv.DictWriter(b, fieldnames=fields, extrasaction="ignore")
    writer.writeheader(); writer.writerows(result["files"])
    _atomic_text(output / "candidate_files.csv", b.getvalue())
    b = io.StringIO()
    writer = csv.DictWriter(b, fieldnames=["dataset", "schema_group", "column", "dtype", "rows", "non_null", "missing"])
    writer.writeheader()
    for ident, group in result["groups"].items():
        if group["schema"]["kind"] != "delimited":
            continue  # block/list records have different physical vs logical row units
        for field in group["schema"].get("fields", []):
            miss = group["missing"].get(field["name"], 0)
            writer.writerow({"dataset": group["dataset"], "schema_group": ident, "column": field["name"], "dtype": field["dtype"],
                             "rows": group["rows"], "non_null": group["rows"] - miss, "missing": miss})
    _atomic_text(output / "candidate_columns.csv", b.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/candidate_datasets")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/dataset_inventory")
    args = parser.parse_args()
    result = scan(args.data_root.resolve())
    write_results(result, args.output_dir)
    print(json.dumps({"datasets": result["datasets"], "trajair": {k: v for k, v in result["trajair"].items() if not k.endswith("counts") and k != "raw_dates"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
