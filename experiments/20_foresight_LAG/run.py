"""Experiment 20: an explicit execute command is required for paid API calls."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from foresight_experiment.cli import run

if __name__ == '__main__':
    raise SystemExit(run())
