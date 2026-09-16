#!/usr/bin/env python3
"""Descriptive plots of existing human/LLM-assigned scores; no rescoring.

Run: python plot_reasoning_scores.py [--input CSV] [--output-dir NEW_DIRECTORY]
Existing output artifacts are never overwritten.
"""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

MODELS = ["GPT-3.5", "Luna", "Terra", "Sol"]
MODEL_MAP = {"3.5": "GPT-3.5", "luna": "Luna", "terra": "Terra", "sol": "Sol"}
HS = [5, 10, 20, 30]
SCORES = [1, 2, 3, 4]
MODEL_COLORS = dict(zip(MODELS, ["#777777", "#0072B2", "#D55E00", "#009E73"]))
SCORE_COLORS = ["#2166AC", "#67A9CF", "#FDDBC7", "#B2182B"]
SCORE_LABELS = ["Score 1: strongest", "Score 2: specific evidence", "Score 3: general policy", "Score 4: weakest / generic"]
METRICS = ["last-action", "next-action"]
METRIC_NAMES = {"last-action": "Last Action", "next-action": "Next Action"}
STEMS = ["01_model_h_interaction", "02_model_h_heatmap", "03_model_h_stacked_distribution",
         "04_pendulum_last_vs_next_stacked", "05_pendulum_last_vs_next_interaction",
         "06_high_level_reasoning_rate", "07_guessing_rate", "08_model_score_distribution",
         "09_h_score_distribution", "10_metric_score_distribution"]


def summary(df, keys):
    return df.groupby(keys, observed=True).agg(
        n=("score", "size"), mean_reasoning_level=("reasoning_level", "mean"),
        high_level_rate=("high_level_reasoning", "mean"), guessing_rate=("guessing", "mean")
    ).reset_index()


def score_legend(fig, y=0.01):
    fig.legend([Patch(facecolor=c) for c in SCORE_COLORS], SCORE_LABELS,
               loc="lower center", bbox_to_anchor=(0.5, y), ncol=2, frameon=False)


def stacked(ax, df, keys, order, positions=None, labels=None, width=0.8):
    counts = pd.crosstab([df[k] for k in keys], df.score).reindex(index=order, columns=SCORES, fill_value=0)
    n = counts.sum(axis=1).to_numpy()
    if (n == 0).any():
        raise ValueError("A requested plotted condition has no valid scores")
    p = counts.to_numpy() / n[:, None]
    x = np.arange(len(order)) if positions is None else np.asarray(positions)
    bottom = np.zeros(len(order))
    for j, color in enumerate(SCORE_COLORS):
        ax.bar(x, p[:, j], bottom=bottom, width=width, color=color, edgecolor="white", linewidth=0.5)
        for xi, val, lo in zip(x, p[:, j], bottom):
            if val >= 0.12:
                ax.text(xi, lo + val / 2, f"{val:.0%}", ha="center", va="center",
                        color="white" if j in (0, 3) else "#17202A", fontsize=8)
        bottom += p[:, j]
    for xi, ni in zip(x, n):
        ax.text(xi, 1.025, f"n={ni}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels if labels is not None else order)
    ax.set_ylim(0, 1.105)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of samples")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)


def line_axes(ax, rate=False):
    ax.set_xticks(HS)
    ax.set_xlim(3, 32)
    ax.set_xlabel("History length H")
    if rate:
        ax.set_ylim(-0.025, 1.025)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.set_ylim(0.9, 4.1)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylabel("Mean reasoning level (higher = stronger)")
    ax.grid(alpha=0.2)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here.parent / "reasoning_for_scoring_scored_astra_high.csv")
    parser.add_argument("--output-dir", type=Path, default=here)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    figdir = out / "figures"
    names = [figdir / f"{stem}.png" for stem in STEMS]
    names += [out / name for name in ["condition_summary.csv", "model_h_summary.csv", "validation.txt"]]
    collisions = [str(p) for p in names if p.exists()]
    if collisions:
        raise FileExistsError("Refusing to overwrite artifacts. Choose a new --output-dir:\n" + "\n".join(collisions))

    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    raw = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"model": str})
    required = {"model", "Task", "Metrics", "H", "score"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Expected columns {required}; found {list(raw.columns)}")
    report = [f"Input: {args.input.resolve()}", f"SHA256: {digest}", f"Columns: {list(raw.columns)}", f"Total rows: {len(raw)}"]
    for col in ["model", "Task", "Metrics", "H", "score"]:
        report.append(f"Unique {col}: {raw[col].drop_duplicates().tolist()}")
    numeric = pd.to_numeric(raw.score, errors="coerce")
    missing = raw.score.isna() | raw.score.astype(str).str.strip().eq("")
    if ((~missing) & (~numeric.isin(SCORES))).any():
        raise ValueError("Nonmissing scores must be integers 1–4")
    df = raw.loc[~missing].copy()
    df["score"] = numeric.loc[~missing].astype(int)
    if df[["model", "Task", "Metrics", "H"]].isna().any().any():
        raise ValueError("Missing grouping values")
    if set(df.model) != set(MODEL_MAP) or set(df.H) != set(HS):
        raise ValueError("Unexpected model or H values; update explicit mappings")
    df["Model"] = pd.Categorical(df.model.map(MODEL_MAP), MODELS, ordered=True)
    df["reasoning_level"] = 5 - df.score
    df["high_level_reasoning"] = df.score <= 2
    df["guessing"] = df.score == 4
    keys = ["Model", "Task", "Metrics", "H"]
    conditions = summary(df, keys)
    # Include raw counts so exclusions cannot silently hide sparse conditions.
    raw["Model"] = pd.Categorical(raw.model.map(MODEL_MAP), MODELS, ordered=True)
    raw_counts = raw.groupby(keys, observed=True).size().rename("n_input").reset_index()
    conditions = raw_counts.merge(conditions, on=keys, how="left")
    conditions["n"] = conditions.n.fillna(0).astype(int)
    conditions["n_missing_score"] = conditions.n_input - conditions.n
    mh = summary(df, ["Model", "H"])
    pend = df[(df.Task == "Pendulum-v1") & df.Metrics.isin(METRICS)]
    if pend.empty or set(pend.Metrics) != set(METRICS):
        raise ValueError("Missing Pendulum Last/Next Action data")
    expected = pd.MultiIndex.from_product([MODELS, sorted(df.Task.unique()), sorted(df.Metrics.unique()), HS], names=keys)
    absent = expected.difference(pd.MultiIndex.from_frame(conditions[keys]))
    report += [f"Missing scores excluded: {missing.sum()}", f"Valid rows: {len(df)}", f"Absent factorial conditions: {len(absent)}",
               "Condition counts (all Model × Task × Metric × H):", conditions.to_string(index=False),
               "Counts by model (number of conditions, min/max valid n):",
               conditions.groupby("Model", observed=True).n.agg(["count", "min", "max"]).to_string(),
               "Pooling is sample-weighted. Means of ordinal levels are descriptive, not interval-scale measurements."]
    figdir.mkdir(parents=True, exist_ok=True)
    conditions.to_csv(out / "condition_summary.csv", index=False)
    mh.to_csv(out / "model_h_summary.csv", index=False)
    (out / "validation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
                         "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})

    def save(fig, i):
        fig.savefig(figdir / f"{STEMS[i-1]}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for mi, model in enumerate(MODELS):
        subset = conditions[conditions.Model == model]
        for h in HS:
            vals = subset.loc[subset.H == h, "mean_reasoning_level"].dropna().to_numpy()
            jitter = np.linspace(-0.6, 0.6, len(vals)) + (mi - 1.5) * 0.22
            ax.scatter(h + jitter, vals, color=MODEL_COLORS[model], alpha=0.22, s=25, linewidths=0)
        vals = mh[mh.Model == model].set_index("H").reindex(HS)
        ax.plot(HS, vals.mean_reasoning_level, "o-", color=MODEL_COLORS[model], label=model, linewidth=2)
    line_axes(ax)
    ax.set_title("Reasoning level by model and history length")
    fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.055), frameon=False)
    fig.text(0.5, 0.025, "Lines: pooled sample means. Faint dots: Task × Metric means (horizontal offsets for visibility).", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=0.25, top=0.9)
    save(fig, 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    table = mh.pivot(index="Model", columns="H", values="mean_reasoning_level").reindex(index=MODELS, columns=HS)
    im = ax.imshow(table, vmin=1, vmax=4, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), HS)
    ax.set_yticks(range(4), MODELS)
    ax.set_xlabel("History length H")
    ax.set_ylabel("Model")
    ax.set_title("Mean reasoning level · Task and Metric pooled")
    for i in range(4):
        for j in range(4):
            val = table.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > 2.75 else "#17202A", fontsize=13)
    fig.colorbar(im, ax=ax, label="Reasoning level (higher = stronger)", ticks=[1, 2, 3, 4])
    fig.tight_layout()
    save(fig, 2)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    order = pd.MultiIndex.from_product([MODELS, HS], names=["Model", "H"])
    x = [m * 5 + h for m in range(4) for h in range(4)]
    stacked(ax, df, ["Model", "H"], order, x, HS * 4, width=0.86)
    for m, model in enumerate(MODELS):
        ax.text(m * 5 + 1.5, -0.12, model, transform=ax.get_xaxis_transform(), ha="center", fontsize=12)
    ax.set_title("Score distributions by model and H · Task and Metric pooled")
    ax.set_xlabel("History length H", labelpad=35)
    score_legend(fig)
    fig.subplots_adjust(bottom=0.28, top=0.91)
    save(fig, 3)

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharey=True)
    for i, model in enumerate(MODELS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            stacked(ax, pend[(pend.Model == model) & (pend.Metrics == metric)], ["H"], HS)
            ax.set_title(f"{model} · {METRIC_NAMES[metric]}")
            ax.set_xlabel("History length H")
            if j:
                ax.set_ylabel("")
    fig.suptitle("Pendulum · Last Action vs Next Action score distributions", fontsize=15)
    score_legend(fig)
    fig.tight_layout(rect=(0, 0.065, 1, 0.97), h_pad=1.7)
    save(fig, 4)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    pmh = summary(pend, ["Model", "Metrics", "H"])
    for ax, model in zip(axes.flat, MODELS):
        for metric, color, marker in zip(METRICS, ["#0072B2", "#D55E00"], ["o", "s"]):
            vals = pmh[(pmh.Model == model) & (pmh.Metrics == metric)].set_index("H").reindex(HS)
            ax.plot(HS, vals.mean_reasoning_level, marker=marker, color=color, label=METRIC_NAMES[metric], linewidth=2)
        line_axes(ax)
        ax.set_title(model)
        ax.set_ylabel("Mean reasoning level")
        ax.tick_params(labelbottom=True)
    fig.suptitle("Pendulum · Action metric × history length (higher = stronger)")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, 5)

    for i, value, title, ylabel in [
        (6, "high_level_rate", "Concrete / quantitative reasoning rate", "P(score ≤ 2) · higher = stronger"),
        (7, "guessing_rate", "Generic / weak reasoning rate", "P(score = 4) · lower = stronger")]:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for model in MODELS:
            vals = mh[mh.Model == model].set_index("H").reindex(HS)
            ax.plot(HS, vals[value], "o-", color=MODEL_COLORS[model], label=model, linewidth=2)
        line_axes(ax, rate=True)
        ax.set_ylabel(ylabel)
        ax.set_title(title + "\nTask and Metric pooled")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
        fig.tight_layout()
        save(fig, i)

    for i, frame, key, order, labels, title, xlabel in [
        (8, df, "Model", MODELS, MODELS, "Score distribution by model · H, Task and Metric pooled", "Model"),
        (9, df, "H", HS, HS, "Score distribution by H · Model, Task and Metric pooled", "History length H"),
        (10, pend, "Metrics", METRICS, [METRIC_NAMES[m] for m in METRICS], "Pendulum action metrics · Model and H pooled", "Metric")]:
        fig, ax = plt.subplots(figsize=(8.7, 5.5))
        stacked(ax, frame, [key], order, labels=labels, width=0.65)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        score_legend(fig)
        fig.subplots_adjust(bottom=0.23, top=0.9)
        save(fig, i)
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == digest, "Input changed"
    assert all(p.is_file() and p.stat().st_size > 0 for p in names)
    print(f"Generated 10 PNG figures in {figdir}; input unchanged.")


if __name__ == "__main__":
    main()
