"""Exp.06: Joint N10 analysis per model, including Exp.05 reuse."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.common_analysis.runner import main

if __name__ == "__main__":
    main("06_new_models_n10", sizes=(10,))
