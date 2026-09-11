"""GPT-3.5: Exp.3's grid with Joint prompts for continuous outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS

EXPERIMENT = common.Experiment(
    name="03_1_gpt35_history_n30_joint",
    models=("3.5",),
    tasks=common.TASKS,
    metrics=common.METRICS,
    histories=common.H_VALUES,
    n=30,
    questions=JOINT_QUESTIONS,
)


if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
