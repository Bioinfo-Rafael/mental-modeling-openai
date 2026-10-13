"""LAG experiment entry point. Defaults to a local preview, never API execution."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
# All nonstandard import paths are configured once, at the entry point.
sys.path[:0] = [str(HERE.parent / "13_LAG"), str(ROOT / "upstream/LLM-Xavier")]

from lag_experiment.config import Experiment
from lag_experiment.runner import run

# Edit the experiment grid here. Data definitions stay in 13_LAG/lag_data.
EXPERIMENT = Experiment(
    name="sliding_window_v1",
    models=("sol", "terra", "luna"),
    histories=(2, 5, 10, 20, 30),
    target_ids=("A0100", "A0200"),
    tasks=("next-action", "last-action", "next-state", "last-state"),
    n=5,
    observation_scopes=("self", "all_aircraft_missiles"),
)


if __name__ == "__main__":
    raise SystemExit(run(EXPERIMENT))
