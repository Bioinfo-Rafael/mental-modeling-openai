"""Plot every raw Pendulum episode, with all timesteps and feature pairs."""

from itertools import combinations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "data/llmx_data/offline_data/physics_data/raw_transitions/Pendulum-v1"
LABELS = ["State: cos(theta)", "State: sin(theta)", "State: angular velocity", "Action: torque", "Reward"]


def main():
    episodes = []
    for path in sorted(SOURCE.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            n = len(data["states"])
            arrays = [np.asarray(data[key]).reshape(n, -1) for key in ("states", "actions", "rewards")]
            if [a.shape[1] for a in arrays] != [3, 1, 1]:
                raise ValueError(f"Unexpected dimensions: {path}")
            values = np.column_stack(arrays)
            if n == 0 or not np.isfinite(values).all():
                raise ValueError(f"Empty or nonfinite episode: {path}")
            episodes.append((path, values))
    if not episodes:
        raise FileNotFoundError(SOURCE)
    all_values = np.vstack([v for _, v in episodes])
    low, high = all_values.min(axis=0), all_values.max(axis=0)
    padding = np.maximum((high - low) * 0.06, 0.05)
    limits = list(zip(low - padding, high + padding))
    max_steps = max(len(v) for _, v in episodes)
    norm = Normalize(0, max_steps - 1)
    manifest = []
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.facecolor": "white"})
    for path, values in episodes:
        steps = np.arange(len(values))
        fig, axes = plt.subplots(3, 5, figsize=(22, 12), layout="constrained")
        fig.suptitle(f"Pendulum-v1 | {path.stem} | {len(values)} transitions\n"
                     "Top: values by timestep   |   Middle / bottom: all feature pairs   |   Color: timestep",
                     fontsize=17)
        for i, ax in enumerate(axes[0]):
            dots = ax.scatter(steps, values[:, i], c=steps, cmap="viridis", norm=norm, s=15, alpha=0.8, linewidths=0)
            ax.set(xlabel="Timestep (t)", ylabel=LABELS[i], title=LABELS[i], xlim=(-3, max_steps + 2), ylim=limits[i])
        for ax, (i, j) in zip(axes[1:].flat, combinations(range(5), 2)):
            ax.scatter(values[:, i], values[:, j], c=steps, cmap="viridis", norm=norm, s=18, alpha=0.8, linewidths=0)
            ax.set(xlabel=LABELS[i], ylabel=LABELS[j], xlim=limits[i], ylim=limits[j])
        for ax in axes.flat:
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        fig.colorbar(dots, ax=axes, label="Timestep (t)", fraction=0.015, pad=0.015)
        relative = path.relative_to(SOURCE)
        destination = OUT / relative.with_suffix(".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        manifest.append({"source": str(relative), "png": str(destination.relative_to(OUT)), "transitions": len(values)})
        print(destination.name)
    (OUT / "manifest.json").write_text(json.dumps({
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "description": "Stored states[t], actions[t], rewards[t]; no subsampling. Shared axes across episodes.",
        "episodes": manifest,
        "total_transitions": sum(item["transitions"] for item in manifest),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
