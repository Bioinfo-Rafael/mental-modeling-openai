"""Sol/Terra/Luna: Pendulum H20 × 4 metrics × N1."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

EXPERIMENT = common.Experiment(
    name="05_new_models_single", models=common.NEW_MODELS,
    tasks=("Pendulum-v1",), metrics=common.METRICS, histories=(20,), n=1,
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
