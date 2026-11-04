"""Reject derived writes to raw data or src/mentalmodeling/ (prior work), including symlink traversal."""
from pathlib import Path

from mtd_rl_xavier import paths

ROOT = paths.ROOT
PROTECTED_DIRS = (paths.DATA, paths.MENTALMODELING)


def derived_path(path):
    path = Path(path).expanduser()
    resolved = path.resolve()
    for protected in PROTECTED_DIRS:
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(f"derived output cannot be written under read-only source: {path}")
    return path
