#!/usr/bin/env python3
"""Offline inspection of MountainCar-v0 and Pendulum-v1 trajectories.

This experiment never imports an OpenAI client and never invokes another
experiment.  It discovers the original LLM-Xavier NPZ episodes, loads them
with the validated upstream Episode reader, and writes descriptive tables and
figures beneath this experiment's results directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = ROOT / "upstream" / "LLM-Xavier"
for import_root in (ROOT, UPSTREAM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dataset_adapters.llmx import EpisodeSource, discover_episodes  # noqa: E402
from llm_x.data import Episode  # noqa: E402
from llm_x.task import MountainCarTask, PendulumTask  # noqa: E402


TASKS = ("MountainCar-v0", "Pendulum-v1")
N_BINS = 10
UNCHANGED_THRESHOLD = 1e-4
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# These names and finite ranges are transcribed from the corresponding
# observation_space/action_space strings in upstream/LLM-Xavier/llm_x/task.py.
TASK_METADATA = {
    "MountainCar-v0": {
        "short_name": "MountainCar",
        "task_class": MountainCarTask,
        "state_names": ("position", "velocity"),
        "observation_low": (-1.2, -0.07),
        "observation_high": (0.6, 0.07),
        "action_description": "0=accelerate left, 1=no acceleration, 2=accelerate right",
        "action_range": (0.0, 2.0),
        "action_kind": "discrete",
    },
    "Pendulum-v1": {
        "short_name": "Pendulum",
        "task_class": PendulumTask,
        "state_names": ("x = cos(theta)", "y = sin(theta)", "angular velocity"),
        "observation_low": (-1.0, -1.0, -8.0),
        "observation_high": (1.0, 1.0, 8.0),
        "action_description": "torque applied to the free end; positive is counterclockwise",
        "action_range": (-2.0, 2.0),
        "action_kind": "continuous",
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory (default: this experiment's results directory)",
    )
    return parser.parse_args(argv)


def _vectors(array: np.ndarray, *, key: str, path: Path) -> np.ndarray:
    """Return an episode array as (timestep, flattened feature dimension)."""
    value = np.asarray(array)
    if value.ndim < 1:
        raise ValueError(f"{path}: {key} has no timestep dimension: {value.shape}")
    vectors = value.reshape(value.shape[0], -1)
    if vectors.shape[1] == 0 or not np.all(np.isfinite(vectors)):
        raise ValueError(f"{path}: {key} is empty or contains non-finite values")
    return vectors.astype(np.float64, copy=False)


def _shape(value: np.ndarray) -> list[int]:
    return [int(size) for size in np.asarray(value).shape]


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


def validate_upstream_metadata() -> None:
    """Fail if the upstream descriptions no longer support our explicit schema."""
    required_fragments = {
        "MountainCar-v0": ("position", "-1.2", "0.6", "velocity", "-0.07", "0.07"),
        "Pendulum-v1": ("x = cos(theta)", "y = sin(theta)", "-8.0", "8.0", "-2.0", "2.0"),
    }
    for task_name, metadata in TASK_METADATA.items():
        task = metadata["task_class"]()
        description = task.observation_space + task.action_space
        missing = [fragment for fragment in required_fragments[task_name] if fragment not in description]
        if missing:
            raise ValueError(
                f"Upstream {task_name} description changed; missing expected text: {missing}"
            )


def load_data() -> dict[str, list[tuple[EpisodeSource, Episode]]]:
    """Discover and load every available episode for the two target tasks."""
    grouped: dict[str, list[tuple[EpisodeSource, Episode]]] = {task: [] for task in TASKS}
    for source in discover_episodes():
        if source.task in grouped:
            grouped[source.task].append((source, Episode.load(source.path)))
    for task, records in grouped.items():
        records.sort(key=lambda item: str(item[0].path))
        if not records:
            raise FileNotFoundError(f"No episodes discovered for {task}")
    return grouped


def validate_and_stack(
    task: str, records: list[tuple[EpisodeSource, Episode]]
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Validate dimensions and return episode metadata plus stacked values."""
    episode_rows: list[dict[str, object]] = []
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    state_dim: int | None = None
    action_dim: int | None = None

    for source, episode in records:
        arrays = episode.arrays
        states = _vectors(arrays["states"], key="states", path=source.path)
        actions = _vectors(arrays["actions"], key="actions", path=source.path)
        if state_dim is None:
            state_dim, action_dim = states.shape[1], actions.shape[1]
        if states.shape[1] != state_dim or actions.shape[1] != action_dim:
            raise ValueError(f"{task}: inconsistent feature dimensions in {source.path}")
        expected_state_dim = len(TASK_METADATA[task]["state_names"])
        if states.shape[1] != expected_state_dim:
            raise ValueError(
                f"{task}: upstream schema has {expected_state_dim} states, data has {states.shape[1]}"
            )
        episode_rows.append(
            {
                "dataset": source.dataset,
                "episode": source.episode,
                "path": str(source.path.relative_to(ROOT)),
                "sha256": episode.sha256,
                "length": episode.length,
                "state_shape": _shape(arrays["states"]),
                "action_shape": _shape(arrays["actions"]),
                "reward_shape": _shape(arrays["rewards"]),
            }
        )
        state_parts.append(states)
        action_parts.append(actions)

    return episode_rows, np.concatenate(state_parts), np.concatenate(action_parts)


def summarize(
    grouped: dict[str, list[tuple[EpisodeSource, Episode]]]
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build task, episode, and per-dimension descriptive summaries."""
    output: dict[str, object] = {
        "method": {
            "standard_deviation": "population standard deviation (ddof=0)",
            "state_action_reshape": "each episode is reshaped to (timestep, flattened feature dimension)",
        },
        "tasks": {},
    }
    all_states: dict[str, np.ndarray] = {}
    all_actions: dict[str, np.ndarray] = {}

    for task in TASKS:
        episode_rows, states, actions = validate_and_stack(task, grouped[task])
        metadata = TASK_METADATA[task]
        state_stats = []
        for index, name in enumerate(metadata["state_names"]):
            state_stats.append(
                {"state_index": index, "state_label": f"state[{index}]", "state_name": name,
                 **_statistics(states[:, index])}
            )
        action_stats = []
        for index in range(actions.shape[1]):
            action_stats.append(
                {"action_index": index, "action_label": f"action[{index}]", **_statistics(actions[:, index])}
            )
        output["tasks"][task] = {
            "number_of_episodes": len(episode_rows),
            "episode_lengths": [row["length"] for row in episode_rows],
            "state_dimension": int(states.shape[1]),
            "action_dimension": int(actions.shape[1]),
            "state_statistics": state_stats,
            "action_statistics": action_stats,
            "action_kind": metadata["action_kind"],
            "action_range_from_upstream": list(metadata["action_range"]),
            "action_description_from_upstream": metadata["action_description"],
            "episodes": episode_rows,
        }
        all_states[task], all_actions[task] = states, actions

    return output, all_states, all_actions


def save_summary(summary: dict[str, object], results_dir: Path) -> None:
    with (results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    fields = (
        "task", "number_of_episodes", "episode_lengths", "episode_shapes", "state_dimension",
        "action_dimension", "variable", "dimension_index", "dimension_label", "dimension_name",
        "count", "min", "max", "mean", "std",
    )
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            task_summary = summary["tasks"][task]
            common = {
                "task": task,
                "number_of_episodes": task_summary["number_of_episodes"],
                "episode_lengths": json.dumps(task_summary["episode_lengths"], separators=(",", ":")),
                "episode_shapes": json.dumps(
                    [
                        {"episode": row["episode"], "state": row["state_shape"],
                         "action": row["action_shape"], "reward": row["reward_shape"]}
                        for row in task_summary["episodes"]
                    ],
                    separators=(",", ":"),
                ),
                "state_dimension": task_summary["state_dimension"],
                "action_dimension": task_summary["action_dimension"],
            }
            for stats in task_summary["state_statistics"]:
                writer.writerow(
                    {**common, "variable": "state", "dimension_index": stats["state_index"],
                     "dimension_label": stats["state_label"], "dimension_name": stats["state_name"],
                     **{key: stats[key] for key in ("count", "min", "max", "mean", "std")}}
                )
            for stats in task_summary["action_statistics"]:
                writer.writerow(
                    {**common, "variable": "action", "dimension_index": stats["action_index"],
                     "dimension_label": stats["action_label"], "dimension_name": "",
                     **{key: stats[key] for key in ("count", "min", "max", "mean", "std")}}
                )


def bin_edges(task: str, state_index: int, values: np.ndarray) -> dict[str, np.ndarray]:
    metadata = TASK_METADATA[task]
    observed_min, observed_max = float(np.min(values)), float(np.max(values))
    if observed_min == observed_max:
        raise ValueError(f"{task} state[{state_index}] has a degenerate observed range")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, N_BINS + 1), method="linear")
    return {
        "observation_space_equal_width": np.linspace(
            metadata["observation_low"][state_index],
            metadata["observation_high"][state_index],
            N_BINS + 1,
        ),
        "observed_range_equal_width": np.linspace(observed_min, observed_max, N_BINS + 1),
        "quantile": quantiles,
    }


def build_bins(all_states: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], dict[tuple[str, int], dict[str, np.ndarray]]]:
    """Build all three ten-bin candidates and observed occupancy counts."""
    rows: list[dict[str, object]] = []
    edges_by_dimension: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for task in TASKS:
        states = all_states[task]
        for index, state_name in enumerate(TASK_METADATA[task]["state_names"]):
            values = states[:, index]
            candidates = bin_edges(task, index, values)
            edges_by_dimension[(task, index)] = candidates
            for method, edges in candidates.items():
                counts, _ = np.histogram(values, bins=edges)
                for bin_index in range(N_BINS):
                    rows.append(
                        {
                            "task": task,
                            "state_index": f"state[{index}]",
                            "state_name": state_name,
                            "method": method,
                            "bin_index": bin_index,
                            "lower_bound": float(edges[bin_index]),
                            "upper_bound": float(edges[bin_index + 1]),
                            "left_inclusive": "true",
                            "right_inclusive": "true" if bin_index == N_BINS - 1 else "false",
                            "observed_count": int(counts[bin_index]),
                            "observed_fraction": float(counts[bin_index] / values.size),
                        }
                    )
    return rows, edges_by_dimension


def save_bins(rows: list[dict[str, object]], results_dir: Path) -> None:
    with (results_dir / "state_bins.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_deltas(
    grouped: dict[str, list[tuple[EpisodeSource, Episode]]]
) -> tuple[list[dict[str, object]], dict[tuple[str, int], np.ndarray]]:
    """Calculate state changes without crossing episode boundaries."""
    rows: list[dict[str, object]] = []
    delta_values: dict[tuple[str, int], np.ndarray] = {}
    for task in TASKS:
        episode_deltas = [
            np.diff(_vectors(episode.arrays["states"], key="states", path=source.path), axis=0)
            for source, episode in grouped[task]
        ]
        deltas = np.concatenate(episode_deltas)
        for index, state_name in enumerate(TASK_METADATA[task]["state_names"]):
            values = deltas[:, index]
            inc = values > UNCHANGED_THRESHOLD
            dec = values < -UNCHANGED_THRESHOLD
            unch = np.abs(values) <= UNCHANGED_THRESHOLD
            rows.append(
                {
                    "task": task,
                    "state_index": f"state[{index}]",
                    "state_name": state_name,
                    "count": int(values.size),
                    "threshold": UNCHANGED_THRESHOLD,
                    "mean_delta": float(np.mean(values)),
                    "std_delta": float(np.std(values, ddof=0)),
                    "min_delta": float(np.min(values)),
                    "max_delta": float(np.max(values)),
                    "fraction_INC": float(np.mean(inc)),
                    "fraction_DEC": float(np.mean(dec)),
                    "fraction_UNCH": float(np.mean(unch)),
                }
            )
            delta_values[(task, index)] = values
    return rows, delta_values


def save_delta_summary(rows: list[dict[str, object]], results_dir: Path) -> None:
    with (results_dir / "state_delta_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pyplot():
    # Keep matplotlib's font/cache writes outside the repository.
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 120, "savefig.dpi": 160, "axes.spines.top": False,
         "axes.spines.right": False, "font.size": 9}
    )
    return plt


def plot_trajectories(
    grouped: dict[str, list[tuple[EpisodeSource, Episode]]], figures_dir: Path
) -> None:
    plt = _pyplot()
    for task in TASKS:
        source, episode = grouped[task][0]
        metadata = TASK_METADATA[task]
        states = _vectors(episode.arrays["states"], key="states", path=source.path)
        actions = _vectors(episode.arrays["actions"], key="actions", path=source.path)
        short_name = metadata["short_name"]

        fig, axes = plt.subplots(states.shape[1], 1, figsize=(8, 2.5 * states.shape[1]), sharex=True)
        axes = np.atleast_1d(axes)
        for index, axis in enumerate(axes):
            axis.plot(np.arange(states.shape[0]), states[:, index], linewidth=1.2)
            axis.set_ylabel(f"state[{index}]\n{metadata['state_names'][index]}")
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"{task} state trajectory — {source.episode}")
        fig.tight_layout()
        fig.savefig(figures_dir / f"{short_name}_state.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        x = np.arange(actions.shape[0])
        for index in range(actions.shape[1]):
            label = f"action[{index}]" if actions.shape[1] > 1 else None
            if metadata["action_kind"] == "discrete":
                ax.step(x, actions[:, index], where="post", linewidth=1.2, label=label)
                low, high = metadata["action_range"]
                ax.set_yticks(np.arange(int(low), int(high) + 1))
            else:
                ax.plot(x, actions[:, index], linewidth=1.2, label=label)
        if actions.shape[1] > 1:
            ax.legend(frameon=False)
        ax.set(xlabel="timestep", ylabel="action", title=f"{task} action trajectory — {source.episode}")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{short_name}_action.png", bbox_inches="tight")
        plt.close(fig)


def plot_scatter_trajectories(grouped, figures_dir: Path) -> None:
    """Point-only time series from the same first episode as the line plots."""
    plt = _pyplot()
    for task in TASKS:
        source, episode = grouped[task][0]
        metadata = TASK_METADATA[task]
        for key, label in (('states', 'state'), ('actions', 'action'), ('rewards', 'reward')):
            values = _vectors(episode.arrays[key], key=key, path=source.path)
            fig, axes = plt.subplots(values.shape[1], 1,
                                     figsize=(8, 2.5 * values.shape[1] if values.shape[1] > 1 else 3.2),
                                     sharex=True, squeeze=False)
            for index, ax in enumerate(axes[:, 0]):
                ax.scatter(np.arange(len(values)), values[:, index], s=14,
                           color='#59788e', alpha=.85, linewidths=0)
                ylabel = f'state[{index}]\n{metadata["state_names"][index]}' if key == 'states' else label
                ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                ax.set_axisbelow(True)
                if key == 'actions' and metadata['action_kind'] == 'discrete':
                    low, high = metadata['action_range']
                    ax.set_yticks(np.arange(int(low), int(high) + 1))
            axes[-1, 0].set_xlabel('timestep')
            fig.suptitle(f'{task} {label} time series (scatter) — {source.episode}')
            fig.tight_layout()
            fig.savefig(figures_dir / f'{metadata["short_name"]}_{label}_scatter.png', bbox_inches='tight')
            plt.close(fig)


def plot_rewards(grouped, figures_dir: Path) -> None:
    """Match existing plots: first episode for time series, all episodes for distribution."""
    plt = _pyplot()
    report = {}
    for task in TASKS:
        parts = []
        for source, episode in grouped[task]:
            rewards = _vectors(episode.arrays['rewards'], key='rewards', path=source.path)
            if rewards.shape != (episode.length, 1) or not np.isfinite(rewards).all():
                raise ValueError(f'Expected one finite reward per timestep: {source.path}')
            parts.append(rewards[:, 0])
        source, _ = grouped[task][0]
        values = np.concatenate(parts)
        short = TASK_METADATA[task]['short_name']
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(np.arange(len(parts[0])), parts[0], color='#59788e', linewidth=1.2)
        ax.set(xlabel='timestep', ylabel='reward', title=f'{task} reward trajectory — {source.episode}')
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f'{short}_reward.png', bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        constant = np.ptp(values) == 0
        bins = [values[0]-.5, values[0]+.5] if constant else 50
        counts, edges, _ = ax.hist(values, bins=bins, color='#59788e', alpha=.85)
        assert int(counts.sum()) == len(values)
        ax.set(xlabel='reward', ylabel='timestep count',
               title=f'{task} reward distribution — all {len(parts)} episodes (n={len(values)})')
        if constant:
            ax.set_xticks([values[0]])
            ax.text(.98, .94, f'All {len(values)} rewards = {values[0]:g}', transform=ax.transAxes, ha='right', va='top')
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        fig.savefig(figures_dir / f'{short}_reward_distribution.png', bbox_inches='tight')
        plt.close(fig)
        report[task] = dict(trajectory_episode=source.episode, trajectory_count=len(parts[0]),
            distribution_count=len(values), min=float(values.min()), max=float(values.max()),
            mean=float(values.mean()), histogram_counts=counts.astype(int).tolist(), histogram_edges=edges.tolist(),
            sources=[dict(path=str(s.path), sha256=e.sha256) for s,e in grouped[task]])
    (figures_dir.parent / 'reward_summary.json').write_text(json.dumps(report, indent=2)+'\n')


def plot_bin_comparisons(
    all_states: dict[str, np.ndarray],
    edges_by_dimension: dict[tuple[str, int], dict[str, np.ndarray]],
    figures_dir: Path,
) -> None:
    plt = _pyplot()
    method_titles = {
        "observation_space_equal_width": "Observation-space equal-width",
        "observed_range_equal_width": "Observed-range equal-width",
        "quantile": "Quantile",
    }
    colors = {"observation_space_equal_width": "#1f77b4", "observed_range_equal_width": "#ff7f0e", "quantile": "#2ca02c"}
    for task in TASKS:
        for index, state_name in enumerate(TASK_METADATA[task]["state_names"]):
            values = all_states[task][:, index]
            candidates = edges_by_dimension[(task, index)]
            fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True, sharey=True)
            for axis, (method, edges) in zip(axes, candidates.items()):
                axis.hist(values, bins=50, color="#9aa6b2", alpha=0.75)
                for edge in edges:
                    axis.axvline(edge, color=colors[method], linewidth=0.9, alpha=0.9)
                axis.set_ylabel("count")
                axis.set_title(method_titles[method], loc="left", fontsize=9)
                axis.grid(axis="y", alpha=0.18)
            axes[-1].set_xlabel(f"state[{index}] — {state_name}")
            fig.suptitle(f"{task}: state[{index}] 10-bin comparison")
            fig.tight_layout()
            short_name = TASK_METADATA[task]["short_name"]
            fig.savefig(figures_dir / f"{short_name}_state{index}_bins.png", bbox_inches="tight")
            plt.close(fig)


def plot_delta_histograms(
    delta_values: dict[tuple[str, int], np.ndarray], figures_dir: Path
) -> None:
    plt = _pyplot()
    for task in TASKS:
        for index, state_name in enumerate(TASK_METADATA[task]["state_names"]):
            values = delta_values[(task, index)]
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.hist(values, bins=60, color="#59788e", alpha=0.85)
            ax.axvline(-UNCHANGED_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.0)
            ax.axvline(UNCHANGED_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.0,
                       label=f"UNCH bounds ±{UNCHANGED_THRESHOLD:g}")
            ax.set(
                xlabel=f"delta state[{index}] — {state_name}",
                ylabel="count",
                title=f"{task}: state[{index}] one-step change",
            )
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            short_name = TASK_METADATA[task]["short_name"]
            fig.savefig(figures_dir / f"{short_name}_state{index}_delta.png", bbox_inches="tight")
            plt.close(fig)


def _format_boundaries(edges: np.ndarray) -> str:
    return ", ".join(f"{float(value):.10g}" for value in edges)


def save_recommendation(
    all_states: dict[str, np.ndarray],
    edges_by_dimension: dict[tuple[str, int], dict[str, np.ndarray]],
    results_dir: Path,
) -> None:
    lines = [
        "# State 10-bin recommendation",
        "",
        "Primary design: use observation-space equal-width bins. This matches the existing "
        "Pendulum action-bin design: fixed task-level bounds, bins 0–8 are `[lower, upper)`, "
        "and bin 9 is `[lower, upper]`. Quantile bins are included only as a distribution diagnostic.",
        "",
    ]
    for task in TASKS:
        lines.extend((f"## {task}", ""))
        metadata = TASK_METADATA[task]
        for index, state_name in enumerate(metadata["state_names"]):
            values = all_states[task][:, index]
            candidates = edges_by_dimension[(task, index)]
            obs_edges = candidates["observation_space_equal_width"]
            counts, _ = np.histogram(values, bins=obs_edges)
            theoretical_width = obs_edges[-1] - obs_edges[0]
            coverage = (float(np.max(values)) - float(np.min(values))) / theoretical_width
            out_of_range = int(np.sum((values < obs_edges[0]) | (values > obs_edges[-1])))
            occupied = int(np.count_nonzero(counts))
            largest_fraction = float(np.max(counts) / values.size)
            top_two_fraction = float(np.sum(np.sort(counts)[-2:]) / values.size)
            issues = []
            if coverage < 0.25:
                issues.append(f"observed span uses only {coverage:.1%} of the theoretical span")
            if occupied <= 2 or largest_fraction >= 0.8 or top_two_fraction >= 0.8:
                issues.append(
                    "observation-space bins are concentrated "
                    f"(occupied={occupied}/10, largest={largest_fraction:.1%}, top-two={top_two_fraction:.1%})"
                )
            if out_of_range:
                issues.append(f"{out_of_range} observed values lie outside the documented range")
            distribution = (
                f"Observed span covers {coverage:.1%} of the theoretical span; {occupied}/10 "
                "observation-space bins are occupied; "
                f"largest-bin share is {largest_fraction:.1%}; top-two-bin share is {top_two_fraction:.1%}."
            )
            recommendation = (
                "Observation-space binning has a problem and needs review. Keep it as the explicit "
                "baseline; do not silently substitute observed or quantile bounds."
                if issues else
                "Use observation-space equal-width bins."
            )
            reason = "; ".join(issues) if issues else (
                "The documented finite task range contains the data without severe concentration "
                "and preserves a fixed, dataset-independent class definition."
            )
            lines.extend(
                (
                    f"### state[{index}] — {state_name}",
                    "",
                    f"- Observation-space range: `[{obs_edges[0]:.10g}, {obs_edges[-1]:.10g}]`",
                    f"- Observed data range: `[{np.min(values):.10g}, {np.max(values):.10g}]`",
                    f"- Distribution: {distribution}",
                    f"- Recommended method: {recommendation}",
                    f"- Recommended boundaries: `{_format_boundaries(obs_edges)}`",
                    f"- Reason: {reason}",
                    "",
                )
            )
    lines.extend(
        (
            "## Joint prediction interpretation",
            "",
            f"Use `delta = next_state - current_state`: `INC` when delta > {UNCHANGED_THRESHOLD:g}, "
            f"`DEC` when delta < -{UNCHANGED_THRESHOLD:g}, and `UNCH` otherwise. The bin is the "
            "absolute next-state bin, while the real value remains the unrounded next-state value. "
            "See `state_delta_summary.csv` and the delta histograms before fixing this threshold in a prompt.",
            "",
            "Source for names and theoretical ranges: "
            "`upstream/LLM-Xavier/llm_x/task.py` (`MountainCarTask`, `PendulumTask`).",
            "",
        )
    )
    (results_dir / "RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")


def verify_outputs(results_dir: Path, bin_rows: Iterable[dict[str, object]], delta_rows: list[dict[str, object]]) -> None:
    bin_rows = list(bin_rows)
    expected_dimensions = sum(len(TASK_METADATA[task]["state_names"]) for task in TASKS)
    expected_bin_rows = expected_dimensions * 3 * N_BINS
    if len(bin_rows) != expected_bin_rows:
        raise AssertionError(f"Expected {expected_bin_rows} bin rows, got {len(bin_rows)}")
    if len(delta_rows) != expected_dimensions:
        raise AssertionError(f"Expected {expected_dimensions} delta rows, got {len(delta_rows)}")
    for row in delta_rows:
        total = row["fraction_INC"] + row["fraction_DEC"] + row["fraction_UNCH"]
        if not np.isclose(total, 1.0):
            raise AssertionError(f"Direction fractions do not sum to one: {row}")
    expected_files = {
        "summary.csv", "summary.json", "state_bins.csv", "state_delta_summary.csv", "RECOMMENDATION.md"
    }
    expected_files.update(
        f"figures/{TASK_METADATA[task]['short_name']}_{suffix}.png"
        for task in TASKS
        for suffix in ("state", "action", "reward", "reward_distribution", "state_scatter", "action_scatter", "reward_scatter")
    )
    expected_files.update(
        f"figures/{TASK_METADATA[task]['short_name']}_state{index}_{suffix}.png"
        for task in TASKS
        for index in range(len(TASK_METADATA[task]["state_names"]))
        for suffix in ("bins", "delta")
    )
    missing = sorted(path for path in expected_files if not (results_dir / path).is_file())
    if missing:
        raise AssertionError(f"Missing expected outputs: {missing}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    validate_upstream_metadata()
    grouped = load_data()
    summary, all_states, _all_actions = summarize(grouped)
    save_summary(summary, results_dir)
    plot_trajectories(grouped, figures_dir)
    plot_rewards(grouped, figures_dir)
    plot_scatter_trajectories(grouped, figures_dir)
    bin_rows, edges_by_dimension = build_bins(all_states)
    save_bins(bin_rows, results_dir)
    plot_bin_comparisons(all_states, edges_by_dimension, figures_dir)
    delta_rows, delta_values = analyze_deltas(grouped)
    save_delta_summary(delta_rows, results_dir)
    plot_delta_histograms(delta_values, figures_dir)
    save_recommendation(all_states, edges_by_dimension, results_dir)
    verify_outputs(results_dir, bin_rows, delta_rows)

    for task in TASKS:
        print(f"{task}: {len(grouped[task])} episodes")
    print(f"Wrote offline analysis to {results_dir}")
    return 0


if __name__ == "__main__":
    # A temporary matplotlib config directory prevents cache files in the repository.
    with tempfile.TemporaryDirectory(prefix="state-action-inspection-") as matplotlib_config:
        os.environ.setdefault("MPLCONFIGDIR", matplotlib_config)
        raise SystemExit(main())
