"""Exp.06: Joint N10 analysis per model, including Exp.05 reuse."""
from importlib import import_module
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mtd_rl_xavier_workspace.experiment.common_analysis.runner import main

if __name__ == "__main__":
    main(import_module("workspace.06_new_models_n10.run").EXPERIMENT, sizes=(10,))
