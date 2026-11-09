"""Every repository path the user-facing docs and jobs name exists.

Jobs and root-level docs are read from the repository root, so their paths must exist there; a
package README under ``src/`` may also name a path relative to ``src/`` (``evaluation/cli.py``).

Placeholders (``<run>``, ``$VAR``) end a path; ``*`` is a glob and ``{a,b}`` lists alternatives.
Generated or private trees (``outputs/``, ``private/``, ``logs/``) are not checked.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest

from translate import paths

ROOT = paths.ROOT
DOCS = sorted(
    {
        ROOT / "README.md",
        ROOT / "slurm" / "README.md",
        ROOT / "data" / "mhi" / "README.md",
        *ROOT.joinpath("src").rglob("README.md"),
        *ROOT.joinpath("slurm").glob("*.sbatch"),
    }
)
PATH_TOKEN = re.compile(r"(?<![\w/.$-])((?:src|configs|scripts|slurm|tests|data|evaluation)/[A-Za-z0-9_./*{}<>,$-]*)")
PLACEHOLDER = re.compile(r"[<$]")
TRAILING = ".,:;)`'\""


def _candidates(token: str) -> list[str]:
    path = PLACEHOLDER.split(token)[0].rstrip(TRAILING).rstrip("/")
    if "{" not in path:
        return [path] if path else []
    head, rest = path.split("{", 1)
    options, tail = rest.split("}", 1) if "}" in rest else (rest, "")
    return [head + option + tail for option in options.split(",")]


def _exists(candidate: str, bases: tuple[Path, ...]) -> bool:
    for base in bases:
        if glob.glob(str(base / candidate)) if "*" in candidate else (base / candidate).exists():
            return True
    return False


def _missing(doc: Path) -> list[str]:
    bases = (ROOT, ROOT / "src") if doc.is_relative_to(ROOT / "src") else (ROOT,)
    found = []
    for number, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
        for match in PATH_TOKEN.finditer(line):
            found += [f"{number}: {c}" for c in _candidates(match.group(1)) if not _exists(c, bases)]
    return found


@pytest.mark.parametrize("doc", DOCS, ids=lambda doc: str(doc.relative_to(ROOT)))
def test_documented_repository_paths_exist(doc: Path) -> None:
    assert _missing(doc) == []
