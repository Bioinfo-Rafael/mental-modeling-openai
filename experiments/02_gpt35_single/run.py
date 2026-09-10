"""Eight GPT-3.5 queries; require exact Exp.1 preview identities before sending."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

EXPERIMENT = common.Experiment(
    name="02_gpt35_single", models=("3.5",),
    tasks=common.TASKS, metrics=common.METRICS, histories=(common.PILOT_H,), n=1,
    preview_from="01_prompt_preview",
)

if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
