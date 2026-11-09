"""Put the repository root (resolved) on sys.path so `workspace.<experiment>` is importable in tests.

A `pythonpath = [".."]` setting would leave `..` unresolved in module paths, which breaks code that
relates `__file__` to resolved directories (for example workspace/13_LAG/lag_data/prepare.py).
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = str(Path(__file__).resolve().parent.parent)
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)
