"""LAG commentary: default invocation only generates local artifacts."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent/'13_LAG'))
from commentary_experiment.runner import run
if __name__ == '__main__':
    raise SystemExit(run())
