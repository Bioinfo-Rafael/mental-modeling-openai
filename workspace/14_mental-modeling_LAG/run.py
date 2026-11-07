"""LAG experiment entry point. Defaults to a local preview, never API execution."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
# All nonstandard import paths are configured once, at the entry point.
from mtd_rl_xavier_workspace.paths import MENTALMODELING  # noqa: E402
sys.path[:0] = [str(MENTALMODELING)]

from lag_experiment.config import Experiment
from lag_experiment.runner import run

# Edit the experiment grid here. Data definitions come from mtd_rl_xavier_workspace.lag; the CSVs stay in 13_LAG/data.
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
