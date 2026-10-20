"""PNG-only four-model views of the unchanged Joint long-table metrics.

Every line represents a model, never a state dimension or reconstruction method.
Dimensions/methods get separate figures. Confusion matrices use each model's hue.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
import numpy as np

from .joint_data import TASKS, METRICS, HISTORIES, DIMENSIONS
from .joint_plots import LABELS, grid, select, style

MODEL_ORDER = ("3.5", "terra", "luna", "sol")
MODEL_LABELS = {"3.5": "GPT-3.5 Turbo", "terra": "GPT-5.6 Terra",
                "luna": "GPT-5.6 Luna", "sol": "GPT-5.6 Sol"}
MODEL_COLORS = {"3.5": "#0072B2", "terra": "#009E73", "luna": "#E69F00", "sol": "#CC79A7"}
MODEL_MARKERS = {"3.5": "o", "terra": "s", "luna": "^", "sol": "D"}
CM_GROUPS = (("01", TASKS[0], "last-action"), ("02", TASKS[0], "next-action"),
             ("03", TASKS[0], "last-state"), ("04", TASKS[0], "next-state"),
             ("05", TASKS[1], "last-state"), ("06", TASKS[1], "next-state"))
DIMENSION_SLUGS = {"position": "position", "velocity": "velocity", "cos(theta)": "cos",
                   "sin(theta)": "sin", "angular_velocity": "angular_velocity"}


def draw_models(ax, rows, *, band=False):
    """Four fixed-color series; missing estimates break lines, not zero-fill."""
    for rank, alias in enumerate(MODEL_ORDER):
        selected = select(rows, model_alias=alias)
        points = {r["H"]: r for r in selected}
        if len(points) != len(selected) or set(points) != set(HISTORIES):
            raise ValueError(f"Expected exactly four H values for model {alias}")
        y = np.array([np.nan if points[h]["value"] is None else points[h]["value"] for h in HISTORIES])
        color = MODEL_COLORS[alias]
        ax.plot(HISTORIES, y, label=MODEL_LABELS[alias], color=color,
                marker=MODEL_MARKERS[alias], markersize=5, linewidth=1.7)
        if band:
            std = np.array([np.nan if points[h]["std"] is None else points[h]["std"] for h in HISTORIES])
            ax.fill_between(HISTORIES, np.maximum(0, y-std), y+std, color=color, alpha=.10, linewidth=0)
        for h, value in zip(HISTORIES, y):
            row = points[h]
            if not np.isfinite(value):
                ax.text(h, .025 + .055*rank, "N/A", transform=ax.get_xaxis_transform(),
                        color=color, fontsize=7, ha="center")
            elif row["valid_n"] < row["selected_n"]:
                ax.annotate(f"n={row['valid_n']}", (h, value), xytext=(3, 6+8*rank),
                            textcoords="offset points", color=color, fontsize=7)


def save_png(fig, root, folder, name, inventory, *, legend=True, footnote=True):
    if legend:
        handles = [Line2D([], [], color=MODEL_COLORS[a], marker=MODEL_MARKERS[a],
                          label=MODEL_LABELS[a], linewidth=1.7) for a in MODEL_ORDER]
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=11)
    if footnote:
        fig.supxlabel("N=10 selected per condition; n labels show fewer valid responses; N/A = undefined. "
                      "GPT-3.5 uses the existing seed-42 subset.", fontsize=9)
    target = root / folder / (name + ".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite figure: {target}")
    fig.savefig(target, dpi=170, facecolor="white")
    inventory.append(str(target.relative_to(root)))
    plt.close(fig)


def confusion_figure(rows, task, metric, alias):
    """One model, 2x2 H panels; retain the original count/row-percent definition."""
    color = MODEL_COLORS[alias]
    fig, axes = grid(2, 2, f"{MODEL_LABELS[alias]} | {task} | {metric} | N=10\n"
                           "Counts and row-normalized percentages", (9, 8))
    fig._suptitle.set_color(color)
    labels = ("DEC", "INC", "UNCH") if metric.endswith("state") else ("0", "1", "2")
    cmap = LinearSegmentedColormap.from_list(alias, ["white", color])
    for ax, h in zip(axes.flat, HISTORIES):
        cells = select(rows, task=task, metric=metric, H=h, subset_n=10, model_alias=alias)
        if len(cells) != 9:
            raise ValueError("Expected a complete 3x3 confusion matrix")
        percentages = np.full((3, 3), np.nan)
        for cell in cells:
            g, p, value = cell["true_class"], cell["predicted_class"], cell["row_percentage"]
            percentages[g, p] = np.nan if value is None else value
            rgb = cmap(0 if value is None else value/100)[:3]
            luminance = np.dot(rgb, [.2126, .7152, .0722])
            ax.text(p, g, f"{cell['count']}\n" + ("N/A" if value is None else f"{value:.1f}%"),
                    ha="center", va="center", color="white" if luminance < .5 else "black", fontsize=11)
        im = ax.imshow(np.ma.masked_invalid(percentages), cmap=cmap, vmin=0, vmax=100)
        ax.set(title=f"H={h} | valid queries={cells[0]['valid_n']}/10", xlabel="Predicted class",
               ylabel="True class", xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels)
    fig.colorbar(im, ax=axes, shrink=.85, label="Row percentage (%)")
    return fig


def make_comparison_figures(tables: dict, root: Path) -> list[str]:
    """Keep the original panel layouts, splitting state dimensions and theta methods."""
    if {r["model_alias"] for r in tables["accuracy_metrics"]} != set(MODEL_ORDER):
        raise ValueError("Four-model comparison requires exactly GPT-3.5, Terra, Luna and Sol")
    if {r["subset_n"] for r in tables["accuracy_metrics"]} != {10}:
        raise ValueError("Comparison figures must use N=10 only")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11})
    inventory = []
    fig, axes = grid(2, 4, "Action Matching Rate / State-change Accuracy | N=10 | valid responses only", (21.6, 9))
    for i, task in enumerate(TASKS):
        for j, metric in enumerate(METRICS):
            draw_models(axes[i, j], select(tables["accuracy_metrics"], task=task, metric=metric, metric_name="accuracy"))
            style(axes[i, j], f"{task} | {metric}", "accuracy")
    save_png(fig, root, "cross_task_10", "accuracy", inventory)

    for component, tag in (("state_value", "absolute"), ("state_delta", "delta")):
        fig, axes = grid(2, 2, f"State {tag} | Macro NRMSE | N=10 | band: bootstrap SD", (11, 9))
        for i, task in enumerate(TASKS):
            for j, metric in enumerate(METRICS[2:]):
                draw_models(axes[i, j], select(tables["state_macro_metrics"], task=task, metric=metric, component=component), band=True)
                style(axes[i, j], f"{task} | {metric}", "macro_nrmse")
        save_png(fig, root, "cross_task_10", f"state_{tag}_macro_nrmse", inventory)
        for task in TASKS:
            for dimension in DIMENSIONS[task]:
                for measure in ("nrmse", "pearson", "cosine"):
                    fig, axes = grid(1, 2, f"{task} | State {tag}: {dimension} | {LABELS[measure]} | N=10"
                                          + ("\nBand: bootstrap SD" if measure == "nrmse" else ""), (11, 6.6))
                    for ax, metric in zip(axes.flat, METRICS[2:]):
                        draw_models(ax, select(tables["state_dimension_metrics"], task=task, metric=metric,
                                               component=component, dimension=dimension, metric_name=measure), band=measure == "nrmse")
                        style(ax, metric, measure)
                    save_png(fig, root, "cross_task_10/state_dimensions",
                             f"state_{tag}_{DIMENSION_SLUGS[dimension]}_{measure}", inventory)

    for number, task, metric in CM_GROUPS:
        for alias in MODEL_ORDER:
            token = "35" if alias == "3.5" else alias
            name = f"{number}_{token}_{task.split('-')[0].lower()}_{metric.replace('-', '_')}"
            if metric.endswith("state"):
                name += "_direction"
            save_png(confusion_figure(tables["confusion_matrices"], task, metric, alias), root,
                     "cm", name, inventory, legend=False, footnote=False)

    fig, axes = grid(4, 2, "Pendulum continuous action and bin distance | N=10 | NRMSE band: bootstrap SD", (11, 14))
    for i, measure in enumerate(("nrmse", "pearson", "cosine", "bin_mae")):
        for j, metric in enumerate(METRICS[:2]):
            draw_models(axes[i, j], select(tables["pendulum_action_metrics"], metric=metric, metric_name=measure), band=measure == "nrmse")
            style(axes[i, j], metric, measure)
    save_png(fig, root, "pendulum_10", "pendulum_action_metrics", inventory)

    for component, tag in (("theta_from_state_value", "absolute"), ("theta_from_state_delta", "delta")):
        fig, axes = grid(3, 2, f"Pendulum forward delta theta | From {tag} state | N=10\nNRMSE band: bootstrap SD", (11, 12))
        for i, measure in enumerate(("nrmse", "pearson", "cosine")):
            for j, metric in enumerate(METRICS[2:]):
                draw_models(axes[i, j], select(tables["pendulum_delta_theta_metrics"], metric=metric,
                                               component=component, metric_name=measure), band=measure == "nrmse")
                style(axes[i, j], metric, measure)
        save_png(fig, root, "pendulum_10", f"pendulum_delta_theta_metrics_from_{tag}", inventory)

    for measure in ("parse_success_rate", "input_tokens", "output_tokens", "total_tokens", "query_elapsed_seconds"):
        is_parse = measure == "parse_success_rate"
        title = "Response + parse success (all required components)" if is_parse else LABELS[measure]
        fig, axes = grid(2, 4, title + " | N=10" + ("" if is_parse else " | mean +/- query sample SD"), (21.6, 9))
        for i, task in enumerate(TASKS):
            for j, metric in enumerate(METRICS):
                draw_models(axes[i, j], select(tables["diagnostics"], task=task, metric=metric, metric_name=measure,
                                               component="all_required" if is_parse else "resource"), band=not is_parse)
                style(axes[i, j], f"{task} | {metric}", measure)
        save_png(fig, root, "diagnostics_10", "processing_time" if measure == "query_elapsed_seconds" else measure, inventory)
    assert len(inventory) == 65 and all(name.endswith(".png") for name in inventory)
    return inventory
