"""Pendulum next action: 2 models × 4 methods × 3 contexts × 3 repeats = 72 calls."""
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runner = import_module('experiments.10_ICRL.runner')

EXPERIMENT = runner.Experiment(
    models=('terra', 'luna'),
    methods=('direct', 'ppo_prior', 'icrfqi_k2', 'icrfqi_k5'),
    context_episode_counts=(1, 3, 9),
    repeats=3,
    task='Pendulum-v1',
    metric='next-action',
    H=20,
    start=0,
    target_episode='episode_9_seed3407.npz',
    reward_present=True,
)

if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT))
