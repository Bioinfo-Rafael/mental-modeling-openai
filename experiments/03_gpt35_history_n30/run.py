"""GPT-3.5: 2 tasks × 4 metrics × 4 history lengths × N30."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

EXPERIMENT = common.Experiment(
    name="03_gpt35_history_n30", models=("3.5",),
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=30,
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
