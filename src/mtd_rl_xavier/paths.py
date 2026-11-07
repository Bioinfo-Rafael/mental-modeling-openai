"""Repository layout: the one module that knows where the core's top-level directories live.

The legacy LLM-Xavier layout and its saved-result path translation moved to the independent
project in the repository's workspace directory; the core never imports or reads it.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"  # committed inputs (data/mhi/), read-only
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
