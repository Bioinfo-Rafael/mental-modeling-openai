"""Pendulum next action: fixed demonstrations versus reference trajectories."""
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runner = import_module('experiments.12_fewshots_episodes.runner')

# 2 models × 2 context methods × N10 = 40 calls (distinct targets, not repeats).
EXPERIMENT = runner.Experiment(
    models=('terra', 'luna'),
    context_methods=('fewshot4', 'episodes_E9'),
    task='Pendulum-v1',
    metric='next-action',
    H=20,
    n=10,
)

if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT))
