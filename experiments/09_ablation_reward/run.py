"""Pendulum reward ablation: episode 9, 32 conditions × 3 repeats = 96 calls."""
import sys
from pathlib import Path
from importlib import import_module

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runner = import_module('experiments.09_ablation_reward.runner.planning')

EXPERIMENT = runner.Experiment(
    models=('terra', 'luna'),
    histories=(5, 20),
    starts=(0, 100),
    shots=(0, 4),
    reward_modes=(True, False),
    repeats=3,
    episode='episode_9_seed3407.npz',
)

if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT))
