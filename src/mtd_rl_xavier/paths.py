"""Repository layout: the one module that knows where top-level directories live.

Saved results from the ``rafael/mental-modeling-openai`` branch record repo-relative
paths in the old layout (``experiments/...``, ``upstream/LLM-Xavier/...``), and results
saved before the vendored code moved under ``src/`` record ``mentalmodeling/...``. Results
are immutable, so readers translate those paths here instead of rewriting files.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MENTALMODELING = ROOT / "src" / "mentalmodeling"  # Prior work: vendored LLM-Xavier (`llm_x`)
WORKSPACE = ROOT / "workspace"  # Exploratory experiments, importable as `workspace.<name>`
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"

# (old prefix, current prefix). Lookups use the longest matching prefix in either direction.
# Several old prefixes may share a current prefix; the inverse then prefers the earliest entry.
LEGACY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("upstream/LLM-Xavier/", "src/mentalmodeling/"),
    ("mentalmodeling/", "src/mentalmodeling/"),
    ("experiments/common_analysis/", "src/mtd_rl_xavier/experiment/common_analysis/"),
    ("experiments/common.py", "src/mtd_rl_xavier/experiment/common.py"),
    ("experiments/analysis.py", "src/mtd_rl_xavier/experiment/analysis.py"),
    ("experiments/split_execution.py", "src/mtd_rl_xavier/experiment/split_execution.py"),
    ("experiments/joint_questions.py", "src/mtd_rl_xavier/experiment/joint_questions.py"),
    ("experiments/README.md", "workspace/EXPERIMENTS.md"),
    ("experiments/", "workspace/"),
    ("dataset_adapters/", "src/mtd_rl_xavier/datasets/"),
    ("preprocessing/", "src/mtd_rl_xavier/preprocessing/"),
    ("tools/", "src/mtd_rl_xavier/tools/"),
    ("notebooks/", "workspace/notebooks/"),
)


def _replace_prefix(text: str, pairs: list[tuple[str, str]]) -> str:
    """Swap the longest matching ``source`` prefix for its ``target``."""
    matches = [(source, target) for source, target in pairs if text.startswith(source)]
    if not matches:
        return text
    source, target = max(matches, key=lambda pair: len(pair[0]))
    return target + text[len(source) :]


def current_relpath(relative: str | Path) -> str:
    """Translate a repo-relative path from the old layout to the current one."""
    return _replace_prefix(Path(relative).as_posix(), list(LEGACY_PREFIXES))


def legacy_relpaths(relative: str | Path) -> tuple[str, ...]:
    """Every key older manifests may have used for a current path, oldest layout first."""
    text = Path(relative).as_posix()
    matches = [(old, new) for old, new in LEGACY_PREFIXES if text.startswith(new)]
    if not matches:
        return ()
    longest = max(len(new) for _, new in matches)
    return tuple(old + text[len(new) :] for old, new in matches if len(new) == longest)


def legacy_relpath(relative: str | Path) -> str:
    """Inverse of `current_relpath`: the oldest-layout key for a current path."""
    candidates = legacy_relpaths(relative)
    return candidates[0] if candidates else Path(relative).as_posix()


def resolve_recorded(root: Path, relative: str | Path) -> Path:
    """Resolve a repo-relative path stored in saved results under ``root``.

    The path is used as recorded when it exists, so current-layout records and test
    fixtures are unaffected; otherwise it is read as an old-layout path.
    """
    recorded = root / relative
    if recorded.exists():
        return recorded
    return root / current_relpath(relative)
