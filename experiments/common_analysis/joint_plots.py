"""Paper-style plots from long tables only; no re-selection or re-scoring here."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .joint_data import TASKS, METRICS, HISTORIES, SIZES, DIMENSIONS, required_components

LABELS = {"accuracy": "Accuracy (%)", "parse_success_rate": "Parse success (%)",
          "nrmse": "NRMSE", "macro_nrmse": "Macro NRMSE", "pearson": "Pearson r",
          "cosine": "Cosine similarity", "bin_mae": "Mean absolute bin error",
          "input_tokens": "Input tokens", "output_tokens": "Output tokens",
          "total_tokens": "Total tokens", "query_elapsed_seconds": "Query time (s)"}
COLORS = plt.get_cmap("tab10").colors


def select(rows, **criteria):
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def grid(rows, columns, title, size=None):
    fig, axes = plt.subplots(rows, columns, figsize=size or (5.4*columns, 3.6*rows),
                             squeeze=False, layout="constrained")
    fig.suptitle(title, fontsize=15)
    return fig, axes


def draw_line(ax, rows, label=None, color=COLORS[0], band=False):
    points = {r["H"]: r for r in rows}
    y = np.array([points.get(h, {}).get("value") if points.get(h, {}).get("value") is not None else np.nan for h in HISTORIES])
    ax.plot(HISTORIES, y, marker="o", markersize=4, linewidth=1.5, label=label, color=color)
    if band:
        std = np.array([points.get(h, {}).get("std") if points.get(h, {}).get("std") is not None else np.nan for h in HISTORIES])
        ax.fill_between(HISTORIES, np.maximum(0, y-std), y+std, color=color, alpha=.15, linewidth=0)


def style(ax, title, metric, legend=False):
    ax.set(title=title, xlabel="History length H", ylabel=LABELS[metric], xticks=HISTORIES)
    ax.grid(axis="y", alpha=.22, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    if metric in {"accuracy", "parse_success_rate"}:
        ax.set_ylim(-2, 102)
    elif metric in {"pearson", "cosine"}:
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_ylim(bottom=0)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.25), ncol=2, frameon=False, fontsize=9)


def save(fig, root, folder, name, inventory, model_label=None):
    if model_label:
        fig.suptitle(model_label + "\n" + fig.get_suptitle(), fontsize=15)
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        target = path.with_suffix("." + extension)
        fig.savefig(target, dpi=170, facecolor="white")
        inventory.append(str(target.relative_to(root)))
    plt.close(fig)


def confusion_plot(rows, task, metric, n):
    fig, axes = grid(2, 2, f"{task} | {metric} | N={n}\nCounts and row-normalized percentages", (9, 8))
    labels = ("DEC", "INC", "UNCH") if metric.endswith("state") else ("0", "1", "2")
    for ax, h in zip(axes.flat, HISTORIES):
        cells = select(rows, task=task, metric=metric, H=h, subset_n=n)
        percentages = np.full((3, 3), np.nan)
        for cell in cells:
            g, p = cell["true_class"], cell["predicted_class"]
            percent = cell["row_percentage"]
            percentages[g, p] = np.nan if percent is None else percent
            text = f"{cell['count']}\n" + ("N/A" if percent is None else f"{percent:.1f}%")
            ax.text(p, g, text, ha="center", va="center", color="white" if percent is not None and percent > 60 else "black", fontsize=11)
        im = ax.imshow(np.ma.masked_invalid(percentages), cmap="Blues", vmin=0, vmax=100)
        ax.set(title=f"H={h} | valid queries={cells[0]['valid_n']}/{n}",
               xlabel="Predicted class", ylabel="True class", xticks=range(3), yticks=range(3),
               xticklabels=labels, yticklabels=labels)
    fig.colorbar(im, ax=axes, shrink=.85, label="Row percentage (%)")
    return fig


def make_figures(tables: dict, root: Path, sizes=SIZES, model_label=None) -> list[str]:
    models = {r.get("model") for r in tables["accuracy_metrics"]}
    if len(models) != 1:
        raise ValueError("Plot one model at a time; never pool different models")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    inventory = []
    for n in sizes:
        cross, pend, diag = f"cross_task_{n}", f"pendulum_{n}", f"diagnostics_{n}"
        fig, axes = grid(2, 4, f"Exact action/bin and pooled direction accuracy | N={n} | valid predictions only")
        for i, task in enumerate(TASKS):
            for j, metric in enumerate(METRICS):
                draw_line(axes[i, j], select(tables["accuracy_metrics"], task=task, metric=metric, subset_n=n, metric_name="accuracy"))
                style(axes[i, j], f"{task} | {metric}", "accuracy")
        save(fig, root, cross, "accuracy", inventory, model_label)
        for task in TASKS:
            for metric in METRICS:
                if task == TASKS[1] and metric.endswith("action"):
                    continue
                name = f"cm_{task.split('-')[0].lower()}_{metric.replace('-', '_')}" + ("_direction" if metric.endswith("state") else "")
                save(confusion_plot(tables["confusion_matrices"], task, metric, n), root, cross, name, inventory, model_label)
        for component, tag in (("state_value", "absolute"), ("state_delta", "delta")):
            for metric_name in ("macro_nrmse", "nrmse", "pearson", "cosine"):
                macro = metric_name == "macro_nrmse"
                fig, axes = grid(2, 2, f"State {tag} | {LABELS[metric_name]} | N={n}" + (" | band: bootstrap SD" if "nrmse" in metric_name else ""))
                for i, task in enumerate(TASKS):
                    for j, metric in enumerate(METRICS[2:]):
                        dimensions = ("macro",) if macro else DIMENSIONS[task]
                        for d, name in enumerate(dimensions):
                            rows = select(tables["continuous_metrics"], task=task, metric=metric, subset_n=n,
                                          component=component, dimension=name, metric_name=metric_name)
                            draw_line(axes[i, j], rows, name, COLORS[d], band="nrmse" in metric_name)
                        style(axes[i, j], f"{task} | {metric}", metric_name, legend=not macro)
                suffix = "macro_nrmse" if macro else "dimension_" + metric_name
                save(fig, root, cross, f"state_{tag}_{suffix}", inventory, model_label)
        fig, axes = grid(4, 2, f"Pendulum continuous action and bin distance | N={n}", (11, 13))
        for i, metric_name in enumerate(("nrmse", "pearson", "cosine", "bin_mae")):
            for j, metric in enumerate(METRICS[:2]):
                draw_line(axes[i, j], select(tables["pendulum_action_metrics"], metric=metric, subset_n=n, metric_name=metric_name), band=metric_name == "nrmse")
                style(axes[i, j], metric, metric_name)
        save(fig, root, pend, "pendulum_action_metrics", inventory, model_label)
        fig, axes = grid(3, 2, f"Pendulum forward delta theta | N={n}", (11, 12))
        for i, metric_name in enumerate(("nrmse", "pearson", "cosine")):
            for j, metric in enumerate(METRICS[2:]):
                for k, component in enumerate(("theta_from_state_value", "theta_from_state_delta")):
                    draw_line(axes[i, j], select(tables["pendulum_delta_theta_metrics"], metric=metric, subset_n=n,
                              metric_name=metric_name, component=component), ("From absolute state", "From state delta")[k], COLORS[k], band=metric_name == "nrmse")
                style(axes[i, j], metric, metric_name, legend=True)
        save(fig, root, pend, "pendulum_delta_theta_metrics", inventory, model_label)
        for metric_name in ("parse_success_rate", "input_tokens", "output_tokens", "total_tokens", "query_elapsed_seconds"):
            is_parse = metric_name == "parse_success_rate"
            fig, axes = grid(2, 4, f"{LABELS[metric_name]} | N={n}" + ("" if is_parse else " | mean +/- query sample SD"), (21.6, 9 if is_parse else 7.2))
            for i, task in enumerate(TASKS):
                for j, metric in enumerate(METRICS):
                    components = (*required_components(task, metric), "all_required") if is_parse else ("resource",)
                    for k, component in enumerate(components):
                        draw_line(axes[i, j], select(tables["diagnostics"], task=task, metric=metric, subset_n=n,
                                  metric_name=metric_name, component=component), component, COLORS[k], band=not is_parse)
                    style(axes[i, j], f"{task} | {metric}", metric_name, legend=is_parse)
            save(fig, root, diag, "processing_time" if metric_name == "query_elapsed_seconds" else metric_name, inventory, model_label)
    assert len(inventory) == 44 * len(sizes)
    return inventory
