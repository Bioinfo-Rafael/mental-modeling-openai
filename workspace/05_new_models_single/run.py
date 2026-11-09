"""Sol/Terra/Luna: Joint Pendulum H20 × 4 metrics × N10 (40 queries/model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from mtd_rl_xavier_workspace.experiment import common
from mtd_rl_xavier_workspace.experiment.joint_questions import JOINT_QUESTIONS

EXPERIMENT = common.Experiment(
    name="05_new_models_single", models=common.NEW_MODELS,
    tasks=("Pendulum-v1",), metrics=common.METRICS, histories=(20,), n=10,
    questions=JOINT_QUESTIONS,
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
