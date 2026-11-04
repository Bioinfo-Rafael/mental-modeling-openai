"""Refresh the saved run's exports and HTML without API calls."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from foresight_experiment.cli import run

if __name__ == '__main__':
    raise SystemExit(run(['export', *sys.argv[1:]]))
