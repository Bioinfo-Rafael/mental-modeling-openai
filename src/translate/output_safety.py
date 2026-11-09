"""Reject derived writes into the committed input data (data/), including symlink traversal."""

from pathlib import Path

from translate import paths

ROOT = paths.ROOT
PROTECTED_DIRS = (paths.DATA,)


def derived_path(path):
    path = Path(path).expanduser()
    resolved = path.resolve()
    for protected in PROTECTED_DIRS:
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(f"derived output cannot be written under read-only source: {path}")
    return path
