"""Workspace layout: the one module that knows where the workspace's directories live.

``ROOT`` is the ``workspace/`` directory itself: experiments sit directly under it
(``ROOT / "06_new_models_n10"``), next to ``src/``, ``data/``, ``configs/`` and ``outputs/``.

Saved results record repo-relative paths from three earlier layouts, and results are immutable,
so readers translate those paths here instead of rewriting files:

- the ``rafael/mental-modeling-openai`` branch (``experiments/...``, ``upstream/LLM-Xavier/...``);
- the vendored code before it moved under ``src/`` (``mentalmodeling/...``);
- the single-project layout before 2026-11-08, relative to the repository root
  (``workspace/<experiment>/...``, ``src/mtd_rl_xavier/<legacy package>/...``).

``data/...`` and ``src/mentalmodeling/...`` keep their meaning, because both moved together with
the root into ``workspace/``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Experiment scripts still compute the repository root (`HERE.parents[k]`) as their ROOT.
REPOSITORY_ROOT = ROOT.parent
MENTALMODELING = ROOT / "src" / "mentalmodeling"  # Prior work: vendored LLM-Xavier (`llm_x`)
WORKSPACE = ROOT  # Experiments, importable as `workspace.<name>` with the repository root on sys.path
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
LIBRARY = "src/mtd_rl_xavier_workspace/"
EXPERIMENTS_ROOT = ""  # experiments sit at ROOT, so their current prefix is empty

EXPERIMENT_DIR_PATTERNS = ("[0-9][0-9]_*", "notebooks")  # what used to be the whole of workspace/


def experiment_dirs() -> list[Path]:
    """The experiment directories (and notebooks) under ``ROOT``; never ``src/``, ``tests/`` or ``.venv/``."""
    return sorted(path for pattern in EXPERIMENT_DIR_PATTERNS for path in ROOT.glob(pattern) if path.is_dir())


# (old prefix, current prefix). Lookups use the longest matching prefix in either direction.
# Several old prefixes may share a current prefix; the inverse then prefers the earliest entry.
LEGACY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("upstream/LLM-Xavier/", "src/mentalmodeling/"),
    ("mentalmodeling/", "src/mentalmodeling/"),
    ("experiments/common_analysis/", f"{LIBRARY}experiment/common_analysis/"),
    ("experiments/common.py", f"{LIBRARY}experiment/common.py"),
    ("experiments/analysis.py", f"{LIBRARY}experiment/analysis.py"),
    ("experiments/split_execution.py", f"{LIBRARY}experiment/split_execution.py"),
    ("experiments/joint_questions.py", f"{LIBRARY}experiment/joint_questions.py"),
    ("experiments/README.md", "EXPERIMENTS.md"),
    ("experiments/", EXPERIMENTS_ROOT),
    ("dataset_adapters/", f"{LIBRARY}datasets/"),
    ("preprocessing/", f"{LIBRARY}preprocessing/"),
    ("tools/", f"{LIBRARY}tools/"),
    ("src/mtd_rl_xavier/", LIBRARY),
    ("workspace/", EXPERIMENTS_ROOT),
)


def _replace_prefix(text: str, pairs: list[tuple[str, str]]) -> str:
    """Swap the longest matching ``source`` prefix for its ``target``."""
    matches = [(source, target) for source, target in pairs if text.startswith(source)]
    if not matches:
        return text
    source, target = max(matches, key=lambda pair: len(pair[0]))
    return target + text[len(source) :]


def current_relpath(relative: str | Path) -> str:
    """Translate a path recorded in an older layout to one relative to the current ``ROOT``."""
    return _replace_prefix(Path(relative).as_posix(), list(LEGACY_PREFIXES))


def legacy_relpaths(relative: str | Path) -> tuple[str, ...]:
    """Every key older manifests may have used for a current path, oldest layout first.

    Prefixes whose current form is empty (the experiments at ``ROOT``) are not inverted: every
    path would match them, so experiment paths have no derived legacy keys here.
    """
    text = Path(relative).as_posix()
    matches = [(old, new) for old, new in LEGACY_PREFIXES if new and text.startswith(new)]
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
    fixtures are unaffected; otherwise it is read as an old-layout path. A caller that passes the
    repository root (experiment scripts written before the split) always gets the path under ``ROOT``:
    a recorded path that still exists in the core must not resolve into it.
    """
    if Path(root).resolve() == REPOSITORY_ROOT:
        return ROOT / current_relpath(relative)
    recorded = root / relative
    if recorded.exists():
        return recorded
    return root / current_relpath(relative)
