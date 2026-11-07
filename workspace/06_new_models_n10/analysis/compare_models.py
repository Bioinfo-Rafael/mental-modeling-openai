"""Compare GPT-3.5 (Exp.03_1 N10 subset) with Terra/Luna/Sol (Exp.05+06)."""
from importlib import import_module
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mtd_rl_xavier_workspace.experiment.common_analysis.comparison import main

if __name__ == "__main__":
    main(import_module("workspace.03_1_gpt35_history_n30_joint.run").EXPERIMENT,
         import_module("workspace.06_new_models_n10.run").EXPERIMENT)
