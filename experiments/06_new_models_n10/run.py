"""Joint N10 grid: reuse all 40 Exp.5 queries/model, send only the remaining 280."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common
from experiments import split_execution
from experiments.joint_questions import JOINT_QUESTIONS

EXPERIMENT = common.Experiment(
    name="06_new_models_n10", models=common.NEW_MODELS,
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=10,
    reuse_from="05_new_models_single",
    reuse_n=10,
    questions=JOINT_QUESTIONS,
)

if __name__ == "__main__":
    raise SystemExit(split_execution.run(EXPERIMENT))
