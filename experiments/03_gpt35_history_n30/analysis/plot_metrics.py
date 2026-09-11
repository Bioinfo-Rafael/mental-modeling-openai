"""Exp.3 analysis: saved responses in, CSV/figures out, no API calls.

Run from the repository root:
    python experiments/03_gpt35_history_n30/analysis/plot_metrics.py

The main function reads in this order: validate, score, sample, aggregate, plot.
Official response parsing and state-direction rules are reused without editing them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from experiments import common
from experiments.analysis import describe
from llm_x.data import Episode
from llm_x.evaluation import EvaluationConfig, _action_range, _score_response
from llm_x.metrics import element_accuracy, state_directions
from tools.output_safety import derived_path

TASKS = ("MountainCar-v0", "Pendulum-v1")
METRICS = ("next-action", "last-action", "next-state", "last-state")
TITLES = ("Next Action (NA)", "Last Action (LA)", "Next State (NS)", "Last State (LS)")
SUBSET_SIZES = (30, 20, 10)
FIGURE_SPECS = (
    {
        "measure": "accuracy_pct", "stem": "paper_matching_accuracy",
        "ylabel": "Paper Metric Accuracy (%)", "suptitle": "action matching and state-change accuracy",
        "ylim": (0, 100), "yticks": [0, 25, 50, 75, 100],
        "footer": ("Pendulum actions: direct-bin matching. States: 3-class re-analysis of INC/DEC-only responses.\n"
                   "Action parse failures count as incorrect; state parse failures are excluded. Not an exact paper replication."),
    },
    {
        "measure": "query_elapsed_seconds", "stem": "paper_matching_execution_time",
        "ylabel": "Execution Time per Query (seconds)", "suptitle": "query execution time",
        "footer": ("Execution time is query_elapsed_seconds: end-to-end time for each query, including the API request and local CLI processing.\n"
                   "Replayed queries retain their originally recorded execution time."),
    },
    {
        "measure": "total_tokens", "stem": "paper_matching_token_usage",
        "ylabel": "Tokens per Query (input + output)", "suptitle": "total token consumption",
        "footer": ("Token consumption is total_tokens reported by the API for each query: input_tokens + output_tokens.\n"
                   "Replayed queries retain their originally recorded token usage."),
    },
)
SCORING_SEMANTICS = {
    "upstream/LLM-Xavier/llm_x/data.py",
    "upstream/LLM-Xavier/llm_x/evaluation.py",
    "upstream/LLM-Xavier/llm_x/metrics.py",
}


def semantics_drift(manifest: dict) -> dict[str, dict[str, str]]:
    """Report source drift while distinguishing files used by saved-response scoring."""
    drift = {}
    for relative, expected in manifest["provenance"]["semantics_files_sha256"].items():
        actual = common.file_sha256(ROOT / relative)
        if actual != expected:
            drift[relative] = {"expected": expected, "actual": actual}
    return drift


def load_results() -> tuple[Path, dict, list[dict]]:
    """Require one complete, unique grid; read H from its manifest, not constants."""
    source = common.latest_results(HERE.parent / "results")
    manifest = common.read_json(source / "manifest.json")
    summary = common.read_json(source / "summary.json")
    records = common.read_jsonl(source / "records.jsonl")
    if summary["status"] != "complete":
        raise ValueError(f"Latest results are incomplete: {source}")
    if manifest["experiment"] != "03_gpt35_history_n30":
        raise ValueError("Unexpected experiment")
    histories = sorted({c["H"] for c in manifest["conditions"]})
    expected_grid = {(task, metric, h) for task in TASKS for metric in METRICS for h in histories}
    counts = Counter((r["task"], r["metric"], r["H"]) for r in records)
    if len(histories) != 4 or set(counts) != expected_grid or set(counts.values()) != {30}:
        raise ValueError(f"Expected 2 tasks x 4 metrics x 4 H x 30 queries; found {counts}")
    if len(records) != len(manifest["queries"]) or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("Missing/duplicated query identities")
    for record, planned in zip(records, manifest["queries"]):
        for key in (*common.IDENTITY_FIELDS, "query_id", "ordinal", "condition_id"):
            if record[key] != planned[key]:
                raise ValueError(f"Record/manifest mismatch in {key}")
        if common.query_id(record) != record["query_id"]:
            raise ValueError("Invalid query hash")
        if record["status"] not in common.SCORED_STATUSES:
            raise ValueError("Unscored result")
        if record["raw_response"]["choices"][0]["message"]["content"] != record["assistant_text"]:
            raise ValueError("Raw response and assistant text differ")
    blocking_drift = SCORING_SEMANTICS & semantics_drift(manifest).keys()
    if blocking_drift:
        raise ValueError(f"Official scoring semantics changed: {sorted(blocking_drift)}")
    return source, manifest, records


def validate_official_score(record: dict, episode: Episode, config: EvaluationConfig) -> None:
    """Reparse/re-score saved text locally and compare with the saved official score."""
    score = _score_response(
        episode, config, index=record["query_index"], response=record["assistant_text"],
        drop_last_feature=False, presented_action=None, presented_is_correct=None,
    )
    for key, value in score.items():
        if record["score"].get(key) != value:
            raise ValueError(f"Official re-score differs: {record['query_id']} / {key}")
    for key in ("prediction", "ground_truth", "status"):
        if record.get(key) != score.get(key):
            raise ValueError(f"Top-level record differs from official score: {key}")


def score_queries(records: list[dict], manifest: dict) -> tuple[list[dict], list[dict], dict]:
    """One row per query, plus one row per state dimension. Never infer raw predictions."""
    configs = {c["condition_id"]: EvaluationConfig(**c["evaluation_config"])
               for c in manifest["conditions"]}
    episodes: dict[str, Episode] = {}
    computed, dimensions = [], []
    for record in records:
        path = record["episode_path"]
        if path not in episodes:
            episodes[path] = Episode.load(ROOT / path)
        episode = episodes[path]
        if episode.sha256 != record["episode_sha256"]:
            raise ValueError(f"Raw episode changed: {path}")
        config = configs[record["condition_id"]]
        validate_official_score(record, episode, config)
        parsed = record["status"] != "ignored"
        row = {key: record[key] for key in (
            "query_id", "condition_id", "task", "metric", "H", "ordinal",
            "episode_path", "query_index", "status", "question_name",
        )}
        row.update(
            parsed=parsed, parse_error=record["score"].get("parse_error", ""),
            saved_prediction=record.get("prediction"), saved_ground_truth=record["ground_truth"],
            official_exact_match_pct=100.0 * (record["status"] == "match"),
            accuracy_pct=None, original_element_accuracy_pct=None,
            unchanged_dimensions=0, state_dimensions=None, na_reason="",
            query_elapsed_seconds=record["query_elapsed_seconds"],
            request_elapsed_seconds=record["request_elapsed_seconds"],
            input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
            total_tokens=record["total_tokens"],
        )
        if record["metric"].endswith("action"):
            # Official action matching denominator includes ignored responses as failures.
            row["accuracy_pct"] = row["official_exact_match_pct"]
            row["evaluation_kind"] = ("discrete_action_match" if record["task"] == TASKS[0]
                                      else "direct_bin_action_match")
            if record["task"] == TASKS[1] and "no_bins" in record["question_name"]:
                raise ValueError("This analysis expects the recorded direct-bin Pendulum experiment")
        else:
            index = record["query_index"]
            expected = state_directions(
                episode.state_vector(index), episode.state_vector(index + 1),
                threshold=config.state_threshold, allow_unchanged=True,
            )
            row.update(evaluation_kind="three_class_state_change_reanalysis",
                       three_class_ground_truth=expected, state_dimensions=len(expected),
                       unchanged_dimensions=expected.count(2), state_threshold=config.state_threshold,
                       prompt_allows_unchanged="more_options" in record["question_name"])
            if parsed:
                row["accuracy_pct"] = 100 * element_accuracy(expected, record["prediction"])
                row["original_element_accuracy_pct"] = 100 * record["score"]["element_accuracy"]
            else:
                row["na_reason"] = "parse_failure_state_prediction"
            for dimension, truth in enumerate(expected):
                prediction = record["prediction"][dimension] if parsed else None
                dimensions.append({
                    **{k: row[k] for k in ("query_id", "condition_id", "task", "metric", "H", "ordinal")},
                    "dimension": dimension, "parsed": parsed, "prediction_class": prediction,
                    "ground_truth_class": truth,
                    "saved_ground_truth_class": record["ground_truth"][dimension],
                    "accuracy_pct": 100.0 * (prediction == truth) if parsed else None,
                })
        computed.append(row)
    return computed, dimensions, {path: episode.sha256 for path, episode in episodes.items()}


def select_subsets(computed: list[dict], seed: int) -> dict:
    """One stable per-condition shuffle: first 10 subset first 20 subset all 30."""
    grouped = defaultdict(list)
    for row in computed:
        grouped[row["condition_id"]].append(row)
    subsets = {}
    for condition, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: row["ordinal"])
        if [r["ordinal"] for r in ordered] != list(range(30)):
            raise ValueError("Expected unique ordinals 0..29 per condition")
        material = f"{seed}:{condition}".encode()
        condition_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        shuffled = list(range(30))
        random.Random(condition_seed).shuffle(shuffled)
        selection = {}
        for n in SUBSET_SIZES:
            indices = sorted(shuffled[:n])
            selection[str(n)] = {"ordinals": indices,
                                 "query_ids": [ordered[i]["query_id"] for i in indices]}
        assert set(selection["10"]["query_ids"]) <= set(selection["20"]["query_ids"]) <= set(selection["30"]["query_ids"])
        subsets[condition] = {"condition_seed": condition_seed, "subsets": selection}
    return {"seed": seed, "method": "SHA256(seed:condition_id)[:8] -> Random.shuffle; nested prefixes",
            "conditions": subsets}


def summarize_values(rows: list[dict], key: str, n: int) -> dict:
    """N is valid query count, not the requested sample size; SD is population SD."""
    stats = describe([row[key] for row in rows])
    return {"subset_n": n, "N": stats["count"], "missing_count": n - stats["count"],
            "mean": stats["mean"], "std": stats["std_population_ddof0"], "std_ddof": 0}


def aggregate(computed: list[dict], dimensions: list[dict], subsets: dict) -> tuple[list[dict], list[dict]]:
    by_id = {row["query_id"]: row for row in computed}
    by_dimension = defaultdict(list)
    for row in dimensions:
        by_dimension[(row["condition_id"], row["dimension"])].append(row)
    aggregated, dimension_aggregates = [], []
    for condition, entry in subsets["conditions"].items():
        for n in SUBSET_SIZES:
            ids = entry["subsets"][str(n)]["query_ids"]
            selected = [by_id[qid] for qid in ids]
            metadata = {k: selected[0][k] for k in ("condition_id", "task", "metric", "H", "evaluation_kind")}
            for key in ("accuracy_pct", "original_element_accuracy_pct", "official_exact_match_pct",
                        "query_elapsed_seconds", "request_elapsed_seconds", "input_tokens",
                        "output_tokens", "total_tokens"):
                stats = summarize_values(selected, key, n)
                reason = "" if stats["N"] else ("state_only_metric" if key == "original_element_accuracy_pct"
                                                 and selected[0]["metric"].endswith("action") else "no_parsed_predictions")
                aggregated.append({**metadata, "measure": key, **stats,
                                   "parse_failures": sum(not row["parsed"] for row in selected),
                                   "na_reason": reason})
            for (cid, dimension), rows in by_dimension.items():
                if cid == condition:
                    included = [row for row in rows if row["query_id"] in ids]
                    dimension_aggregates.append({**metadata, "dimension": dimension,
                                                 **summarize_values(included, "accuracy_pct", n)})
    return aggregated, dimension_aggregates


def write_csv(path: Path, rows: list[dict]) -> None:
    """Flat UTF-8 CSV; missing numeric values are empty, arrays are JSON strings."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def action_bin_settings(manifest: dict) -> dict:
    """Document actual scorer edges, not newly estimated dataset ranges."""
    import numpy as np
    bins = {}
    for condition in manifest["conditions"]:
        config = EvaluationConfig(**condition["evaluation_config"])
        if "continuous_bins" in config.question_name:
            start, stop = _action_range(config)
            bins[condition["condition_id"]] = {
                "edges": np.linspace(start, stop, config.action_bins + 1).tolist(),
                "source": "llm_x.evaluation._action_range + manifest action_bins",
                "assignment": "np.digitize(value, edges, right=True)-1; clip ground truth to 0..9",
                "prediction": "direct bin ID; not rebinned or clipped",
            }
    return bins


def plot_figures(aggregated: list[dict], output: Path, seed: int) -> list[str]:
    """Use the same 2x4 panel layout for accuracy, execution time, and token usage."""
    os.environ["MPLCONFIGDIR"] = str(output / "cache" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    files = []
    for spec in FIGURE_SPECS:
        folder = output / "figures" / spec["stem"]
        folder.mkdir(parents=True, exist_ok=True)
        for n in SUBSET_SIZES:
            fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
            if "ylim" in spec:
                shared_ylim = spec["ylim"]
            else:
                visible = [r for r in aggregated if r["subset_n"] == n and r["measure"] == spec["measure"]
                           and r["mean"] is not None and r["std"] is not None]
                shared_ylim = (0, max(r["mean"] + r["std"] for r in visible) * 1.05)
            for i, task in enumerate(TASKS):
                for j, metric in enumerate(METRICS):
                    ax = axes[i, j]
                    points = sorted((r for r in aggregated if r["task"] == task and r["metric"] == metric
                                     and r["subset_n"] == n and r["measure"] == spec["measure"]), key=lambda r: r["H"])
                    histories = [r["H"] for r in points]
                    mean = [r["mean"] if r["mean"] is not None else math.nan for r in points]
                    std = [r["std"] if r["std"] is not None else math.nan for r in points]
                    color = "#2463A6" if i == 0 else "#B75628"
                    ax.plot(histories, mean, "o-", color=color, linewidth=1.6, markersize=4)
                    lower = [max(0, m-s) if math.isfinite(m) else math.nan for m, s in zip(mean, std)]
                    upper = [(min(100, m+s) if spec["measure"] == "accuracy_pct" else m+s)
                             if math.isfinite(m) else math.nan for m, s in zip(mean, std)]
                    ax.fill_between(histories, lower, upper, color=color, alpha=0.16)
                    settings = {"xticks": histories, "xlabel": "History size H (timesteps)",
                                "ylim": shared_ylim}
                    if "ylim" in spec:
                        settings.update(yticks=spec["yticks"])
                    ax.set(**settings)
                    ax.grid(axis="y", alpha=0.2, linewidth=0.6)
                    if i == 0:
                        ax.set_title(TITLES[j], pad=14)
                    if j == 0:
                        ax.set_ylabel(f"{task.split('-')[0]}\n{spec['ylabel']}")
                    valid = ", ".join(f"{r['H']}:{r['N']}" for r in points)
                    ax.text(0.02, 1.015, f"Valid N by H = {valid}", transform=ax.transAxes, fontsize=8)
                    if all(r["N"] == 0 for r in points):
                        ax.text(0.5, 0.5, "N/A\nNo valid observations", transform=ax.transAxes, ha="center")
            fig.suptitle(f"GPT-3.5: {spec['suptitle']} | n={n}", fontsize=14, y=0.985)
            band_note = ", clipped to 0–100%" if spec["measure"] == "accuracy_pct" else ""
            fig.text(0.5, 0.925, f"Line: mean; band: query SD (ddof=0){band_note}. Seed={seed}; nested subsets.", ha="center", fontsize=10)
            fig.text(0.5, 0.035, spec["footer"], ha="center", fontsize=9)
            fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.15, hspace=0.45, wspace=0.2)
            for extension in ("png", "svg"):
                path = folder / f"{spec['stem']}_n{n}.{extension}"
                fig.savefig(path, dpi=180)
                files.append(str(path.relative_to(output)))
            plt.close(fig)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated analysis files only, never source results")
    args = parser.parse_args(argv)
    output = derived_path(args.output_dir).resolve()
    if not output.is_relative_to(HERE) or args.output_dir.absolute() != output:
        raise ValueError("Output must be a non-symlink directory under this analysis directory")
    managed = ["computed_metrics.csv", "state_dimension_metrics.csv", "aggregated_metrics.csv",
               "state_dimension_aggregated.csv", "query_subsets.json", "analysis_metadata.json", "validation.json", "bin_edges.json"]
    if not args.overwrite and any((output / name).exists() for name in [*managed, "figures"]):
        raise FileExistsError("Analysis outputs exist. Use --output-dir analysis/<new-name> or --overwrite")

    source, manifest, records = load_results()
    protected = [source / name for name in ("manifest.json", "summary.json", "records.jsonl", "responses.jsonl")]
    before = {str(path): common.file_sha256(path) for path in protected}
    computed, dimensions, raw_hashes = score_queries(records, manifest)
    subsets = select_subsets(computed, args.seed)
    aggregated, dimension_aggregates = aggregate(computed, dimensions, subsets)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in zip(managed[:4], (computed, dimensions, aggregated, dimension_aggregates)):
        write_csv(output / name, rows)
    write_json(output / "query_subsets.json", subsets)
    write_json(output / "bin_edges.json", action_bin_settings(manifest))
    figures = plot_figures(aggregated, output, args.seed)
    after = {str(path): common.file_sha256(path) for path in protected}
    assert before == after, "Source results were modified"
    assert all(common.file_sha256(ROOT / path) == digest for path, digest in raw_hashes.items())
    validation = {
        "queries": len(records), "official_rescores_matched": len(records),
        "unique_queries": len({r["query_id"] for r in records}), "conditions": len(manifest["conditions"]),
        "parse_failures": sum(not row["parsed"] for row in computed),
        "missing_prediction_queries": sum(row["saved_prediction"] is None for row in computed),
        "missing_main_accuracy_queries": sum(row["accuracy_pct"] is None for row in computed),
        "unchanged_state_dimensions": sum(row["ground_truth_class"] == 2 for row in dimensions),
        "state_ground_truth_dimensions_changed_vs_original": sum(row["ground_truth_class"] != row["saved_ground_truth_class"] for row in dimensions),
        "nested_subsets_verified": True, "source_results_unchanged": before == after,
        "raw_episodes_unchanged": True, "figure_count": len(figures), "api_requests_made": 0,
        "non_scoring_semantics_drift": semantics_drift(manifest),
    }
    write_json(output / "validation.json", validation)
    write_json(output / "analysis_metadata.json", {
        "source": str(source.relative_to(ROOT)), "source_sha256": before, "raw_sha256": raw_hashes,
        "script_sha256": common.file_sha256(Path(__file__)), "seed": args.seed,
        "histories": sorted({r["H"] for r in computed}), "subset_sizes": SUBSET_SIZES,
        "std_ddof": 0, "python_version": sys.version, "paper_reference": "https://arxiv.org/html/2406.18505v1",
        "state_definition": "official state_directions(allow_unchanged=True), threshold from manifest, decimals=5",
        "non_scoring_semantics_drift": semantics_drift(manifest),
        "figures": figures,
    })
    print(json.dumps(validation, indent=2))
    print(f"Analysis saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
