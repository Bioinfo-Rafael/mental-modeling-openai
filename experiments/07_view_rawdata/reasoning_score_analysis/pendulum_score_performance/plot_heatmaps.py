"""Join existing reasoning scores to predictions and render 25 PNG heatmaps."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = {"3.5": "GPT-3.5", "luna": "Luna", "terra": "Terra", "sol": "Sol", "all": "All models (equal model weights)"}
METRICS = ["last-action", "next-action"]
MEASURES = {
    "bin_accuracy": ("Exact-bin accuracy", "Higher = better", 0, 100, "YlGnBu"),
    "bin_mae": ("Bin MAE", "Lower = better", 0, 9, "YlGnBu_r"),
    "nrmse": ("Torque NRMSE", "Lower = better", 0, None, "YlGnBu_r"),
    "pearson": ("Pearson correlation", "Higher = better", -1, 1, "RdBu"),
    "cosine": ("Cosine similarity", "Higher = better", -1, 1, "RdBu"),
}


def scalar(text):
    value = ast.literal_eval(text)
    if len(value) != 1 or not np.isfinite(value[0]):
        raise ValueError(f"Invalid scalar vector: {text}")
    return float(value[0])


def calculate(p, g, measure):
    if len(p) == 0:
        return np.nan, "no_valid_predictions"
    if measure == "bin_accuracy":
        return float(100 * np.mean(p == g)), ""
    if measure == "bin_mae":
        return float(np.mean(np.abs(p - g))), ""
    if measure == "nrmse":
        return float(np.sqrt(np.mean((p - g) ** 2)) / 4), ""
    if measure == "pearson":
        if len(p) < 2:
            return np.nan, "fewer_than_two_pairs"
        if np.ptp(p) == 0 or np.ptp(g) == 0:
            return np.nan, "constant_vector"
        return float(np.corrcoef(p, g)[0, 1]), ""
    denominator = np.linalg.norm(p) * np.linalg.norm(g)
    if denominator == 0:
        return np.nan, "zero_norm_vector"
    return float(np.clip(np.dot(p, g) / denominator, -1, 1)), ""


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=here.parents[3])
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    root, out = args.repo_root.resolve(), args.output_dir.resolve()
    outputs = [out / "joined_samples.csv", out / "cell_summary.csv", out / "validation.json"]
    outputs += [out / "figures" / f"{measure}_{model}.png" for measure in MEASURES for model in MODELS]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Output already exists; use a new --output-dir")
    inputs = [root / "experiments/07_view_rawdata/reasoning_for_scoring_scored_astra_high.csv",
              root / "experiments/06_new_models_n10/analysis/comparison_4models/tables/parsed_records.csv",
              root / "experiments/03_1_gpt35_history_n30_joint/analysis/tables/parsed_records.csv"]
    hashes = {}

    def protect(path):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    for path in inputs:
        protect(path)
    scores = pd.read_csv(inputs[0], encoding="utf-8-sig", keep_default_na=False, dtype={"model": str})
    newer = pd.read_csv(inputs[1], keep_default_na=False, dtype=str)
    older = pd.read_csv(inputs[2], keep_default_na=False, dtype=str)
    older["model_alias"] = "3.5"
    parsed = pd.concat([newer[newer.model_alias != "3.5"], older], ignore_index=True).fillna("")
    parsed = parsed.rename(columns={"model": "model_full", "model_alias": "model", "task": "Task", "metric": "Metrics"})
    parsed[["H", "ordinal"]] = parsed[["H", "ordinal"]].astype(int)
    keys = ["model", "Task", "Metrics", "H", "ordinal"]
    scores["ordinal"] = scores.groupby(keys[:-1], sort=False).cumcount()
    scores["score_csv_row"] = np.arange(2, len(scores) + 2)
    if parsed.duplicated(keys).any() or scores.duplicated(keys).any():
        raise ValueError("Duplicate join keys")
    if not scores.score.isin([1, 2, 3, 4]).all():
        raise ValueError("Missing or invalid score")
    joined = scores.merge(parsed, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or len(joined) != 1920:
        raise ValueError("Incomplete 1920-row join")
    sources = {}
    for row in joined.to_dict("records"):
        path = root / row["source_records"]
        if path not in sources:
            protect(path)
            sources[path] = [json.loads(line) for line in path.read_text().splitlines()]
        raw = sources[path][int(row["source_line"]) - 1]
        text = raw.get("assistant_text") or ""
        if raw["query_id"] != row["query_id"] or (row["Reasoning"] and row["Reasoning"] not in text):
            raise ValueError(f"Source identity/text mismatch at CSV row {row['score_csv_row']}")
    data = joined[(joined.Task == "Pendulum-v1") & joined.Metrics.isin(METRICS)].copy()
    if data.Reasoning.eq("").any():
        raise ValueError("Unexpected blank Pendulum action reasoning")
    for component, field, gt in [("bin", "action_bin", "gt_action_bin"), ("torque", "action_value", "gt_action_value")]:
        data[f"{component}_valid"] = data[f"{field}_ok"].eq("True")
        data[f"true_{component}"] = data[gt].map(scalar)
        data[f"pred_{component}"] = [scalar(v) if ok else np.nan for v, ok in zip(data[f"{field}_value"], data[f"{component}_valid"])]
    assert len(data) == 480 and int(data.bin_valid.sum()) == 477 and int(data.torque_valid.sum()) == 470
    assert data.groupby("model").size().to_dict() == {"3.5": 240, "luna": 80, "terra": 80, "sol": 80}
    for values in [data.true_bin, data.loc[data.bin_valid, "pred_bin"]]:
        assert values.between(0, 9).all() and (values == values.astype(int)).all()
    cells = []
    for model in MODELS:
        subset = data if model == "all" else data[data.model == model]
        for measure in MEASURES:
            component = "bin" if measure.startswith("bin_") else "torque"
            for score in range(1, 5):
                for metric in METRICS:
                    cell = subset[(subset.score == score) & (subset.Metrics == metric)]
                    valid = cell[cell[f"{component}_valid"]]
                    value, reason = calculate(valid[f"pred_{component}"].to_numpy(), valid[f"true_{component}"].to_numpy(), measure)
                    cells.append(dict(model=model, score=score, metric=metric, measure=measure, value=value,
                                      n_total=len(cell), n_valid=len(valid), n_excluded=len(cell)-len(valid),
                                      na_reason="no_samples" if len(cell) == 0 else reason))
    summary = pd.DataFrame(cells)
    summary["n_models"] = summary.value.notna().astype(int)
    summary["aggregation"] = "within_model_samples"
    for index, row in summary[summary.model == "all"].iterrows():
        model_values = summary[(summary.model != "all") & (summary.measure == row.measure)
                               & (summary.score == row.score) & (summary.metric == row.metric)].value.dropna()
        summary.loc[index, "value"] = model_values.mean() if len(model_values) else np.nan
        summary.loc[index, "n_models"] = len(model_values)
        summary.loc[index, "aggregation"] = "equal_model_mean"
        summary.loc[index, "na_reason"] = "" if len(model_values) else "no_defined_model_values"
    for model in MODELS:
        expected_n = 480 if model == "all" else (240 if model == "3.5" else 80)
        assert summary[summary.model == model].groupby("measure").n_total.sum().eq(expected_n).all()
    for measure, expected_n in [("bin_accuracy", 477), ("bin_mae", 477), ("nrmse", 470), ("pearson", 470), ("cosine", 470)]:
        assert summary[(summary.model == "all") & (summary.measure == measure)].n_valid.sum() == expected_n
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    data.drop(columns="_merge").to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "savefig.facecolor": "white"})
    scales = {}
    for measure, (title, direction, low, high, cmap_name) in MEASURES.items():
        if high is None:
            high = float(summary.loc[summary.measure == measure, "value"].max())
            if high == 0:
                high = 1.0
        scales[measure] = [low, high]
        for model, name in MODELS.items():
            cells = summary[(summary.model == model) & (summary.measure == measure)].set_index(["score", "metric"])
            matrix = np.array([[cells.loc[(s, m), "value"] for m in METRICS] for s in range(1, 5)])
            fig, ax = plt.subplots(figsize=(7.8, 6.8))
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#e3e3e3")
            im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=low, vmax=high, aspect="auto")
            ax.set_xticks([0, 1], ["Last Action", "Next Action"])
            ax.set_yticks(range(4), ["Score 1", "Score 2", "Score 3", "Score 4"])
            ax.set_ylabel("Reasoning score (1 = strongest)")
            ax.set_title(f"Pendulum · {name}\n{title} · H pooled", pad=14)
            for i, score in enumerate(range(1, 5)):
                for j, metric in enumerate(METRICS):
                    row = cells.loc[(score, metric)]
                    value = row.value
                    display = "N/A" if pd.isna(value) else (f"{value:.1f}%" if measure == "bin_accuracy" else f"{value:.3f}")
                    rgba = im.cmap(im.norm(value)) if pd.notna(value) else (0.89, 0.89, 0.89, 1)
                    luminance = sum(c*w for c,w in zip(rgba[:3], [0.2126, 0.7152, 0.0722]))
                    annotation = f"{display}\nn={int(row.n_valid)} / {int(row.n_total)}"
                    if model == "all":
                        annotation += f"\nmodels={int(row.n_models)} / 4"
                    ax.text(j, i, annotation, ha="center", va="center",
                            fontsize=13, color="white" if luminance < 0.5 else "#17202A", linespacing=1.7)
            ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            fig.colorbar(im, ax=ax, fraction=0.055, pad=0.06, label=direction)
            footer = "n = valid / total samples · Gray = unavailable\nShared color scale across all models for this measure"
            if model == "all":
                footer = "Equal mean of defined model metrics; n is descriptive, not a weight\nmodels = contributing models / 4 · Gray = unavailable"
            fig.text(0.5, 0.03, footer, ha="center", fontsize=10)
            fig.tight_layout(rect=(0, 0.085, 1, 1))
            fig.savefig(out / "figures" / f"{measure}_{model}.png", dpi=300)
            plt.close(fig)
    for path, digest in hashes.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, f"Input modified: {path}"
    report = dict(joined_rows=1920, target_rows=480, valid_bin=477, valid_torque=470,
                  excluded_bin=3, excluded_torque=10, figures=25, scales=scales, input_sha256=hashes,
                  aggregation="Equal arithmetic mean of defined model metrics per cell; no sample-count weighting",
                  na_cells=summary[summary.value.isna()][["model", "score", "metric", "measure", "na_reason"]].to_dict("records"))
    outputs[2].write_text(json.dumps(report, indent=2) + "\n")
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    print(f"Verified 1920 joins; 480 target samples; bin n=477, torque n=470; generated 25 PNGs: {out}")


if __name__ == "__main__":
    main()
