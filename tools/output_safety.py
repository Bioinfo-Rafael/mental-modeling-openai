"""Reject derived writes to raw data or upstream, including symlink traversal."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def derived_path(path):
    path = Path(path).expanduser()
    resolved = path.resolve()
    for protected in [ROOT / "data", ROOT / "upstream"]:
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(f"derived output cannot be written under read-only source: {path}")
    return path
