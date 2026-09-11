#!/usr/bin/env python3
"""Shared offline orchestration. CLI writes only under the selected experiment's analysis."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True

from .joint_data import HERE, ROOT, BOUNDS, WIDTHS, TASKS, METRICS, COMPONENTS, load_and_parse, required_components, select_subsets
from .joint_metrics import angular_pair
from .joint_aggregate import aggregate_all
from .joint_plots import make_figures


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files(source):
    """Audit every experiment result and existing experiment/upstream Python source."""
    paths = set(source.rglob("*"))
    paths.update((ROOT / "experiments").rglob("*.py"))
    paths.update((ROOT / "upstream/LLM-Xavier").rglob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths) if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    """UTF-8 long tables: missing numeric cells stay empty, structured cells use JSON."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def cell(value):
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite table value: use None and an explicit NA reason")
        return value
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: cell(value) for key, value in row.items()} for row in rows)


def flatten_records(records):
    rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "components"}
        for component, parsed in record["components"].items():
            row.update({f"{component}_{key}": value for key, value in parsed.items()})
        if record["task"] == TASKS[1] and record["metric"].endswith("state"):
            for component in ("state_value", "state_delta"):
                prediction, truth, reason = angular_pair(record, component)
                row.update({f"theta_from_{component}": prediction, "gt_delta_theta": truth if math.isfinite(truth) else None,
                            f"theta_from_{component}_error": reason})
        rows.append(row)
    return rows


def parse_summary(records):
    result = []
    for task in TASKS:
        for metric in METRICS:
            group = [r for r in records if r["task"] == task and r["metric"] == metric]
            if not group:
                continue
            for component in (*required_components(task, metric), "all_required"):
                failures = Counter("one_or_more_components_failed" if component == "all_required" else r["components"][component]["error"]
                                   for r in group if not (r["all_required"] if component == "all_required" else r["components"][component]["ok"]))
                result.append({"task": task, "metric": metric, "component": component,
                               "model": group[0]["model"], "model_alias": group[0]["model_alias"],
                               "selected_n": len(group), "valid_n": len(group)-sum(failures.values()),
                               "failed_n": sum(failures.values()), "failure_reasons": dict(failures)})
    return result


def validate_tables(records, subsets, tables):
    """Independent basic identities on actual output, in addition to hand-case tests."""
    import numpy as np
    for record in records:
        if record["metric"].endswith("state"):
            np.testing.assert_allclose(record["gt_state_delta"], np.subtract(record["late"], record["early"]), rtol=0, atol=0)
    for groups in subsets["conditions"].values():
        sizes = sorted(map(int, groups))
        for n in sizes:
            assert len(groups[str(n)]) == n and groups[str(n)] == groups[str(sizes[-1])][:n]
    for row in tables["accuracy_metrics"]:
        if row["metric_name"] == "accuracy" and row["valid_units"]:
            assert math.isclose(row["value"], 100*row["correct_units"]/row["valid_units"])
    for row in tables["state_macro_metrics"]:
        dimensions = [r for r in tables["state_dimension_metrics"] if all(r[k] == row[k] for k in ("condition_id", "subset_n", "component")) and r["metric_name"] == "nrmse"]
        if row["value"] is not None:
            assert math.isclose(row["value"], sum(r["value"] for r in dimensions)/len(dimensions))
    for row in tables["all_metrics_long"]:
        assert 0 <= row["valid_n"] <= row["selected_n"]
        if row["metric_name"] in {"pearson", "cosine"} and row["value"] is not None:
            assert -1.000000001 <= row["value"] <= 1.000000001
        if row["value"] is None:
            assert row["na_reason"]
    return {"record_count": len(records), "condition_count": len(subsets["conditions"]),
            "source_run_prompt_and_gt_checks": len(records),
            "ground_truth_delta_checks": sum(r["metric"].endswith("state") for r in records), "nested_subsets": True,
            "accuracy_denominators": True, "macro_equals_mean_dimension_nrmse": True,
            "valid_n_bounds": True, "undefined_metrics_have_reason": True}


def validate_saved_outputs(tables, figures, output):
    """Read back every CSV and image, including SVG syntax, before declaring completion."""
    import xml.etree.ElementTree as ET
    from PIL import Image
    for name, rows in tables.items():
        with (output / "tables" / (name + ".csv")).open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(rows)
    for relative in figures:
        path = output / relative
        if path.suffix == ".png":
            with Image.open(path) as picture:
                assert picture.width > 1000 and picture.height > 1000
                picture.verify()
        else:
            assert ET.parse(path).getroot().tag.endswith("svg")


def export_model(records, source, output, episode_hashes, spec, sizes, seed=42,
                 repetitions=1000, tables_only=False):
    """Export a single model with the unchanged Exp.03_1 scoring definitions."""
    if len({r["model"] for r in records}) != 1:
        raise ValueError("Each export must contain exactly one model")
    started = time.perf_counter()
    protected_before = protected_files(source)
    subsets = select_subsets(records, seed, sizes=sizes, expected_n=spec.n)
    tables = aggregate_all(records, subsets, repetitions, seed, sizes=sizes)
    validation = validate_tables(records, subsets, tables)
    tables["parsed_records"] = flatten_records(records)
    tables["parse_summary"] = parse_summary(records)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / "tables" / (name + ".csv"), rows)
    write_json(output / "query_subsets.json", subsets)
    print(f"Parsed {len(records)} records; wrote {len(tables)} CSV tables", flush=True)
    model_label = records[0]["model"] if len(spec.models) > 1 else None
    figures = [] if tables_only else make_figures(tables, output, sizes=sizes, model_label=model_label)
    validate_saved_outputs(tables, figures, output)
    protected_after = protected_files(source)
    if protected_after != protected_before:
        raise RuntimeError("Protected inputs changed during analysis; inspect before using outputs")
    for relative, expected in episode_hashes.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError("Raw episode changed during analysis")
    validation.update(protected_files_unchanged=True, protected_file_count=len(protected_before),
                      raw_episode_hashes_unchanged=True, png_figures=len(figures)//2, svg_figures=len(figures)//2,
                      csv_readback_checks=len(tables), image_readback_checks=len(figures))
    write_json(output / "validation.json", validation)
    write_json(output / "protected_sha256.json", protected_before)
    write_json(output / "figure_index.json", figures)
    metadata = {"source": str(source.relative_to(ROOT)), "source_records_sha256": digest(source / "records.jsonl"),
                "model": records[0]["model"], "model_alias": records[0]["model_alias"], "subset_sizes": sizes,
                "seed": seed, "bootstrap_repetitions": repetitions, "normalization_bounds": BOUNDS,
                "normalization_widths": WIDTHS, "torque_width": 4, "angular_width": math.pi,
                "range_source": "upstream/LLM-Xavier/llm_x/task.py",
                "range_source_sha256": digest(ROOT / "upstream/LLM-Xavier/llm_x/task.py"),
                "raw_episode_sha256": episode_hashes, "accuracy_missing_policy": "valid_only; all_selected alternative in CSV",
                "figures": len(figures), "csv_rows": {k: len(v) for k, v in tables.items()},
                "parse_summary": tables["parse_summary"], "analysis_seconds": time.perf_counter()-started,
                "api_calls": 0, "python": sys.version,
                "analysis_code_sha256": {p.name: digest(p) for p in HERE.glob("*.py")}}
    write_json(output / "analysis_metadata.json", metadata)
    print(json.dumps(validation, indent=2))
    print(f"Finished offline in {metadata['analysis_seconds']:.1f}s. No API calls; protected files unchanged.")
    return tables


def main(experiment_name, *, sizes=(10, 20, 30), tables_only=False, argv=None):
    from importlib import import_module
    from experiments import common
    spec = import_module(f"experiments.{experiment_name}.run").EXPERIMENT
    base = ROOT / "experiments" / spec.name / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=base, help="Optional new subdirectory within this experiment's analysis")
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 2:
        parser.error("--bootstrap must be >= 2")
    output = args.output_dir.absolute()
    if not output.is_relative_to(base) or output.resolve() != output:
        parser.error("Output must stay within this experiment's analysis, without symlinks")
    outputs = [output / alias if len(spec.models) > 1 else output for alias in spec.models]
    for destination in outputs:
        if destination.resolve() != destination:
            parser.error("Model output directories must not be symlinked")
        if (destination / "query_subsets.json").exists() and not args.overwrite_derived:
            parser.error("Derived output exists; use a new --output-dir or --overwrite-derived")
    source = common.latest_results(ROOT / "experiments" / spec.name / "results")
    try:
        if spec.name == "06_new_models_n10" and (source / "batches").exists():
            from experiments.split_execution import merge_results
            source = merge_results(spec)  # Offline, complete grid only; immutable snapshot.
        source, manifest, records, hashes = load_and_parse(source, spec)
        if spec.reuse_from:
            cached = common.check_inputs(spec, manifest)
            reused = {r["query_id"]: r for r in common.read_jsonl(source / "records.jsonl") if r.get("reused")}
            if set(reused) != set(cached):
                raise ValueError("Unexpected Exp.5 reuse set; refusing duplicate/missing source data")
            for qid, row in reused.items():
                if row["api_request_made"] or row["raw_response"] != cached[qid]["raw_response"]:
                    raise ValueError("Reused query was resent or its response changed")
        for alias, destination in zip(spec.models, outputs):
            selected = [r for r in records if r["model_alias"] == alias]
            export_model(selected, source, destination, hashes, spec, sizes, args.seed, args.bootstrap, tables_only)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Offline analysis stopped: {exc}")
