"""Compare GPT-3.5 (Exp.03_1 N10 subset) with Terra/Luna/Sol (Exp.05+06)."""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.common_analysis.comparison import main

if __name__ == "__main__":
    main()
