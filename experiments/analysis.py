"""Exp.4: nested-prefix statistics/plots from Exp.3 records, never from an API.

Separated from common.py because this is analysis, not API execution or prompting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import math
import os
import statistics
import sys
import time

from experiments import common

SOURCE = "03_gpt35_history_n30"
DESTINATION = "04_sample_size_analysis"
SAMPLE_SIZES = (30, 20, 10)


def describe(values):
    """Missing values remain missing. One observation has no sample variance."""
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Non-finite observation in source records")
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "variance_population_ddof0": statistics.pvariance(values) if values else None,
        "variance_sample_ddof1": statistics.variance(values) if len(values) > 1 else None,
        "std_population_ddof0": statistics.pstdev(values) if values else None,
        "std_sample_ddof1": statistics.stdev(values) if len(values) > 1 else None,
    }


def load_source():
    """Require a complete, unique, ordered N30 dataset, not a partial run/summary alone."""
    source = common.EXPERIMENTS / SOURCE / "results"
    manifest = common.read_json(source / "manifest.json")
    summary = common.read_json(source / "summary.json")
    records = common.read_jsonl(source / "records.jsonl")
    if manifest["experiment"] != SOURCE or summary["status"] != "complete":
        raise ValueError("Exp.3 must be complete before analysis")
    expected_grid = {
        (common.MODELS["3.5"], task, metric, H)
        for task in common.TASKS for metric in common.METRICS for H in common.H_VALUES
    }
    actual_grid = {(c["model"], c["task"], c["metric"], c["H"]) for c in manifest["conditions"]}
    if actual_grid != expected_grid or len(manifest["conditions"]) != len(expected_grid):
        raise ValueError("Exp.3 condition grid differs from the requested grid")
    if len(records) != len(manifest["queries"]) or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("Source records are incomplete or duplicated")
    grouped = defaultdict(list)
    for record, planned in zip(records, manifest["queries"]):
        if (record["query_id"] != planned["query_id"] or common.query_id(record) != record["query_id"]
                or record["status"] not in common.SCORED_STATUSES or record["status"] != record["score"]["status"]
                or record["ordinal"] != planned["ordinal"] or record["condition_id"] != planned["condition_id"]):
            raise ValueError("Source record identity/order/status differs from manifest")
        grouped[record["condition_id"]].append(record)
    for condition in manifest["conditions"]:
        selected = grouped[condition["condition_id"]]
        if len(selected) != max(SAMPLE_SIZES) or [r["ordinal"] for r in selected] != list(range(max(SAMPLE_SIZES))):
            raise ValueError("Every Exp.3 condition must have the same complete N30 prefix")
    # Private upstream scoring semantics must not silently change during reanalysis.
    previous = manifest["provenance"]["semantics_files_sha256"]
    evaluation_path = common.ROOT / "upstream/LLM-Xavier/llm_x/evaluation.py"
    if common.file_sha256(evaluation_path) != previous[str(evaluation_path.relative_to(common.ROOT))]:
        raise ValueError("Upstream metric implementation changed since Exp.3")
    return manifest, grouped, {
        "experiment": SOURCE,
        "manifest_sha256": common.file_sha256(source / "manifest.json"),
        "records_sha256": common.file_sha256(source / "records.jsonl"),
    }


def calculate(manifest, grouped):
    rows = []
    for condition in manifest["conditions"]:
        ordered = grouped[condition["condition_id"]]
        for N in SAMPLE_SIZES:
            selected = ordered[:N]  # Nested subset, never a fresh sample.
            metrics = common.summarize_scored(selected, condition["metric"])
            row = {
                "condition_id": condition["condition_id"], "model": condition["model"],
                "task": condition["task"], "metric": condition["metric"],
                "question_name": condition["question_name"], "H": condition["H"], "N": N,
                "Accuracy": metrics["legacy_compatible_match_rate"], **metrics,
                "query_ids": [r["query_id"] for r in selected],
            }
            observations = {
                # Correctness over ALL queries: ignored=0. Not the state primary denominator!
                "correct_all_queries": [int(r["status"] == "match") for r in selected],
                "element_accuracy_parsed": [r["score"].get("element_accuracy") for r in selected
                                            if r["status"] != "ignored"],
                "query_elapsed_seconds": [r["query_elapsed_seconds"] for r in selected],
                "request_elapsed_seconds": [r["request_elapsed_seconds"] for r in selected],
                "input_tokens": [r["input_tokens"] for r in selected],
                "output_tokens": [r["output_tokens"] for r in selected],
                "total_tokens": [r["total_tokens"] for r in selected],
            }
            for name, values in observations.items():
                row.update({f"{name}_{key}": value for key, value in describe(values).items()})
            rows.append(row)
    return rows


def figures(rows, directory):
    # Keep matplotlib's font/cache outputs under this experiment, too.
    cache = common.output_path(directory / "cache/matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    previous_cache = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(cache)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        measures = [
            ("accuracy", "Accuracy", "Accuracy (upstream legacy denominator)", False),
            ("elapsed_time", "query_elapsed_seconds", "Query elapsed seconds", True),
            ("input_tokens", "input_tokens", "Input tokens (response usage)", True),
            ("output_tokens", "output_tokens", "Output tokens (response usage)", True),
            ("total_tokens", "total_tokens", "Total tokens (response usage)", True),
        ]
        for filename, field, label, errorbars in measures:
            fig, axes = plt.subplots(len(common.TASKS), len(common.METRICS),
                                     figsize=(16, 7), squeeze=False, sharex=True)
            try:
                for task_index, task in enumerate(common.TASKS):
                    for metric_index, metric in enumerate(common.METRICS):
                        ax = axes[task_index][metric_index]
                        for N in SAMPLE_SIZES:
                            selected = sorted([r for r in rows if r["task"] == task and r["metric"] == metric
                                               and r["N"] == N], key=lambda r: r["H"])
                            ykey = field + "_mean" if errorbars else field
                            ys = [r[ykey] if r[ykey] is not None else float("nan") for r in selected]
                            xs = [r["H"] for r in selected]
                            if errorbars:
                                stds = [r[field + "_std_population_ddof0"] for r in selected]
                                ax.errorbar(xs, ys, yerr=[s if s is not None else float("nan") for s in stds],
                                            marker="o", capsize=2, label=f"N={N}")
                            else:
                                # Binary all-query std is NOT centered on parsed-state Accuracy.
                                ax.plot(xs, ys, marker="o", label=f"N={N}")
                        ax.set_title(f"{task}\n{metric}")
                        ax.set_xticks(common.H_VALUES)
                        ax.set_xlabel("H (actual history steps)")
                        ax.set_ylabel(label)
                        ax.grid(alpha=0.2)
                        if not errorbars:
                            ax.set_ylim(-0.03, 1.03)
                axes[0][0].legend(frameon=False)
                note = "Mean ± population std (ddof=0); not a confidence interval" if errorbars else (
                    "Primary Accuracy: all-query denominator for actions; parsed-query denominator for states. No error bars."
                )
                fig.suptitle(note, fontsize=10)
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                target = common.output_path(directory / "figures" / f"{filename}.png")
                target.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=160)
            finally:
                plt.close(fig)
    finally:
        if previous_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description="API-free nested N30/N20/N10 reanalysis of Exp.3")
    parser.add_argument("--dry-run", action="store_true", help="Validate source and save plan only; no plots/statistics")
    args = parser.parse_args(argv)
    directory = common.EXPERIMENTS / DESTINATION / "results"
    clock, started = time.perf_counter(), common.utc_now()
    try:
        manifest, grouped, source = load_source()
        plan = {"experiment": DESTINATION, "source": source, "sample_sizes": SAMPLE_SIZES,
                "condition_count": len(grouped), "api_requests_made": 0,
                "planned_statistic_rows": len(grouped) * len(SAMPLE_SIZES)}
        if args.dry_run:
            common.write_json(directory / "dry_run/manifest.json", plan)
            print("DRY RUN: source validated; API requests made: 0")
            return 0
        common.output_path(directory).mkdir(parents=True, exist_ok=True)
        if any(p.name not in {".gitkeep", "dry_run"} for p in directory.iterdir()):
            raise FileExistsError("Existing analysis results; archive manually before rerunning")
        common.write_json(directory / ".started.json", {"started_at_utc": started}, exclusive=True)
        common.write_json(directory / "manifest.json", plan)
        rows = calculate(manifest, grouped)
        common.write_csv(directory / "statistics.csv", rows)
        common.write_json(directory / "statistics.json", {
            **plan, "definitions": {
                "Accuracy": "upstream _summarize: legacy_compatible_match_rate",
                "correct_all_queries": "match=1; mismatch/ignored=0, over all N queries, NOT state primary Accuracy",
                "element_accuracy_parsed": "upstream element_accuracy on parsed vectors only; scalar actions have none",
                "variance": "population ddof=0 and sample ddof=1; sample variance is null for count<2",
                "time": "query wall time includes upstream retry delays; request time sums SDK create durations",
                "tokens": "final successful response usage; missing stays null; earlier retry usage remains in raw logs",
                "plots": "Accuracy: no error bars. Time/tokens: mean ± population std (ddof=0), not CI",
            }, "rows": rows,
        })
        figures(rows, directory)
        common.write_json(directory / "timing.json", {
            "started_at_utc": started, "finished_at_utc": common.utc_now(),
            "experiment_elapsed_seconds": time.perf_counter() - clock, "api_requests_made": 0,
        })
        print(f"Analysis complete: {len(rows)} statistic rows; API requests made: 0")
        return 0
    except (Exception, common.SafetyStop) as exc:
        print("Analysis stopped:", common.redact(str(exc)), file=sys.stderr)
        return 2
