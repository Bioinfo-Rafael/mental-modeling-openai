"""Joint ten-step LAG forecasts. Default: offline plan and prompt preview."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE.parent / "13_LAG"), str(ROOT / "upstream/LLM-Xavier")]

from dynamics_experiment.config import Experiment
from dynamics_experiment.runner import run

EXPERIMENT = Experiment(
    name="dynamics_batch_v1",
    models=("sol", "terra", "luna"),
    histories=(20,),
    episodes=("mhi_v1.3_ic100601_win",),
    target_ids=("A0100", "A0200"),
    horizon=10,
    n=5,
    observation_scopes=("self", "all_aircraft_missiles"),
)

if __name__ == "__main__":
    raise SystemExit(run(EXPERIMENT))
