"""Subset-level long tables. Each metric uses its own independently parsed component."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

from .joint_data import COMPONENTS, DIMENSIONS, SIZES, TASKS, WIDTHS, required_components
from .joint_metrics import angular_pair, cosine, nrmse_statistics, pearson, stable_seed


def base_row(record: dict, n: int, component: str, valid_n: int, dimension="all") -> dict:
    return {**{k: record[k] for k in ("task", "metric", "H", "condition_id")},
            **{k: record[k] for k in ("model", "model_alias") if k in record},
            "subset_n": n, "selected_n": n, "component": component, "dimension": dimension,
            "valid_n": valid_n, "parse_success_rate": 100 * valid_n / n,
            "metric_name": "", "value": None, "std": None, "na_reason": ""}


def continuous_rows(selected: list[dict], component: str, widths: list[float], names: tuple,
                    repetitions: int, seed: int, *, angular=False) -> list[dict]:
    pairs = []
    for row in selected:
        if angular:
            pred, gt, reason = angular_pair(row, component)
            if pred is not None and np.isfinite(gt):
                pairs.append(([pred], [gt]))
        elif row["components"][component]["ok"]:
            pairs.append((row["components"][component]["value"], row["gt_" + component]))
    count = len(pairs)
    prefix = "theta_from_" + component if angular else component
    metadata = base_row(selected[0], len(selected), prefix, count)
    metadata["parse_success_rate"] = 100 * sum(r["components"][component]["ok"] for r in selected) / len(selected)
    metadata["bootstrap_repetitions"] = repetitions
    metadata["bootstrap_seed"] = stable_seed(seed, f"{selected[0]['condition_id']}:{len(selected)}:{prefix}")
    output = []
    estimates = None
    if pairs:
        prediction, truth = np.asarray([p for p, g in pairs]), np.asarray([g for p, g in pairs])
        estimates = nrmse_statistics(prediction, truth, widths, repetitions, metadata["bootstrap_seed"], angular=angular)
    for d, name in enumerate(names):
        for metric in ("nrmse", "pearson", "cosine"):
            row = {**metadata, "dimension": name, "metric_name": metric, "normalization_width": widths[d]}
            if not count:
                row["na_reason"] = "no_valid_pairs"
            elif metric == "nrmse":
                row.update(value=float(estimates["point"][d]), std=float(estimates["bootstrap_std"][d]),
                           bootstrap_mean=float(estimates["bootstrap_mean"][d]), std_kind="paired_bootstrap_ddof1")
            else:
                function = pearson if metric == "pearson" else cosine
                value, reason = function(prediction[:, d], truth[:, d])
                row.update(value=value, na_reason=reason)
            output.append(row)
    if component in {"state_value", "state_delta"} and not angular:
        row = {**metadata, "dimension": "macro", "metric_name": "macro_nrmse"}
        if estimates is None:
            row["na_reason"] = "no_valid_pairs"
        else:
            row.update(value=estimates["macro"], std=estimates["macro_bootstrap_std"],
                       bootstrap_mean=estimates["macro_bootstrap_mean"], std_kind="paired_bootstrap_ddof1")
        output.append(row)
    return output


def accuracy_rows(selected: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    first, n = selected[0], len(selected)
    component = "direction" if first["metric"].endswith("state") else ("action" if first["task"] == TASKS[0] else "action_bin")
    valid = [r for r in selected if r["components"][component]["ok"]]
    dimensions = len(DIMENSIONS[first["task"]]) if component == "direction" else 1
    predictions = np.asarray([r["components"][component]["value"] for r in valid], int).reshape(-1, dimensions)
    truths = np.asarray([r["gt_" + component] for r in valid], int).reshape(-1, dimensions)
    correct = int((predictions == truths).sum())
    row = base_row(first, n, component, len(valid))
    row.update(metric_name="accuracy", value=100 * correct / truths.size if truths.size else None,
               valid_units=int(truths.size), correct_units=correct,
               accuracy_all_selected=100 * correct / (n * dimensions),
               na_reason="" if valid else "no_valid_predictions")
    direction_rows, confusion = [], []
    if component == "direction":
        for d, name in enumerate(DIMENSIONS[first["task"]]):
            direction_rows.append({**base_row(first, n, component, len(valid), name),
                                   "metric_name": "accuracy", "value": float(100 * (predictions[:, d] == truths[:, d]).mean()) if valid else None,
                                   "na_reason": "" if valid else "no_valid_predictions"})
    if component in {"action", "direction"}:
        matrix = np.zeros((3, 3), int)
        for g, p in zip(truths.ravel(), predictions.ravel()):
            matrix[g, p] += 1
        for g in range(3):
            for p in range(3):
                support = int(matrix[g].sum())
                confusion.append({**base_row(first, n, component, len(valid)),
                                  "true_class": g, "predicted_class": p, "count": int(matrix[g, p]),
                                  "row_support": support, "row_percentage": 100 * matrix[g, p] / support if support else None})
        assert np.trace(matrix) == correct and matrix.sum() == truths.size
    rows = [row]
    if component == "action_bin":
        rows.append({**base_row(first, n, component, len(valid)), "metric_name": "bin_mae",
                     "value": float(np.abs(predictions - truths).mean()) if valid else None,
                     "na_reason": "" if valid else "no_valid_predictions"})
    return rows, direction_rows, confusion


def diagnostic_rows(selected: list[dict]) -> list[dict]:
    first, n = selected[0], len(selected)
    result = []
    for component in (*required_components(first["task"], first["metric"]), "all_required"):
        count = sum(r["all_required"] if component == "all_required" else r["components"][component]["ok"] for r in selected)
        result.append({**base_row(first, n, component, count), "metric_name": "parse_success_rate", "value": 100 * count / n})
    for key in ("input_tokens", "output_tokens", "total_tokens", "query_elapsed_seconds", "request_elapsed_seconds"):
        values = [r[key] for r in selected if r[key] is not None]
        result.append({**base_row(first, n, "resource", len(values)), "metric_name": key,
                       "value": float(np.mean(values)) if values else None,
                       "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "std_kind": "query_sample_ddof1", "na_reason": "" if values else "missing_measurement"})
    return result


def aggregate_all(records: list[dict], subsets: dict, repetitions=1000, seed=42, sizes=SIZES) -> dict[str, list[dict]]:
    tables = defaultdict(list)
    by_id = {r["query_id"]: r for r in records}
    for cid, selection in subsets["conditions"].items():
        for n in sizes:
            selected = [by_id[r["query_id"]] for r in selection[str(n)]]
            accuracy, dimensions, confusion = accuracy_rows(selected)
            tables["accuracy_metrics"].extend(accuracy)
            tables["direction_dimension_accuracy"].extend(dimensions)
            tables["confusion_matrices"].extend(confusion)
            tables["diagnostics"].extend(diagnostic_rows(selected))
            if selected[0]["metric"].endswith("state"):
                for component in ("state_value", "state_delta"):
                    rows = continuous_rows(selected, component, WIDTHS[selected[0]["task"]], DIMENSIONS[selected[0]["task"]], repetitions, seed)
                    tables["continuous_metrics"].extend(rows)
                    if selected[0]["task"] == TASKS[1]:
                        tables["continuous_metrics"].extend(continuous_rows(selected, component, [np.pi], ("delta_theta",), repetitions, seed, angular=True))
            elif selected[0]["task"] == TASKS[1]:
                tables["continuous_metrics"].extend(continuous_rows(selected, "action_value", [4.0], ("torque",), repetitions, seed))
    continuous = tables["continuous_metrics"]
    tables["state_dimension_metrics"] = [r for r in continuous if r["component"] in {"state_value", "state_delta"} and r["dimension"] != "macro"]
    tables["state_macro_metrics"] = [r for r in continuous if r["metric_name"] == "macro_nrmse"]
    tables["pendulum_action_metrics"] = [r for r in continuous if r["component"] == "action_value"] + [r for r in tables["accuracy_metrics"] if r["metric_name"] == "bin_mae"]
    tables["pendulum_delta_theta_metrics"] = [r for r in continuous if r["component"].startswith("theta_from_")]
    tables["all_metrics_long"] = [*tables["accuracy_metrics"], *continuous, *tables["direction_dimension_accuracy"], *tables["diagnostics"]]
    return dict(tables)
