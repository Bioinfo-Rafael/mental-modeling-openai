from pathlib import Path
import sys

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(EXPERIMENT.parents[1]))
