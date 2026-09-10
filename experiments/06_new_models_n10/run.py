"""Sol/Terra/Luna full N10 grid, reusing each matching Exp.5 query once."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

EXPERIMENT = common.Experiment(
    name="06_new_models_n10", models=common.NEW_MODELS,
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=10,
    reuse_from="05_new_models_single",
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
