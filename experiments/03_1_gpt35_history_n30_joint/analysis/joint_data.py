"""Compatibility import; the implementation lives in experiments/common_analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.common_analysis.joint_data import *  # noqa: F401,F403
