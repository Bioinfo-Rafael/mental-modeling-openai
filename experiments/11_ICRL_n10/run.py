"""2 models × 2 reward modes × N10 = 40 calls (distinct target queries)."""
import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
runner = import_module('experiments.11_ICRL_n10.runner')

EXPERIMENT = runner.Experiment(
    models=('terra', 'luna'),
    reward_modes=(True, False),
    task='Pendulum-v1',
    metric='next-action',
    H=20,
    n=10,
    method='icrfqi_k5',
)

if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT))
