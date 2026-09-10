"""API-free analysis of the first N=30/20/10 records in each Exp.3 condition."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common

if __name__ == "__main__":
    raise SystemExit(common.analysis_main())
