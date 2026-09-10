"""Eight local previews, using the same pilot H and query identities as Exp.2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

EXPERIMENT = common.Experiment(
    name="01_prompt_preview", models=("3.5",),
    tasks=common.TASKS, metrics=common.METRICS, histories=(common.PILOT_H,), n=1,
    preview=True,
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
