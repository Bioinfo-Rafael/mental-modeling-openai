"""The LLM parts stay independent: no module in ``mtd_rl_xavier.llm`` imports a sibling (D2)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import mtd_rl_xavier.llm as llm_package

PACKAGE = "mtd_rl_xavier.llm"
PACKAGE_DIR = Path(llm_package.__file__).resolve().parent


def _resolve(node: ast.ImportFrom) -> str:
    """Absolute base of a ``from`` import. Every scanned file (``__init__.py`` included) lives
    directly in ``PACKAGE``, so level 1 is ``PACKAGE`` and each further level drops one part."""
    if node.level == 0:
        return node.module or ""
    package_parts = PACKAGE.split(".")
    base = package_parts[: len(package_parts) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _imported_targets(source: str) -> list[str]:
    """Every absolute module or ``package.name`` target imported anywhere in ``source``."""
    targets: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node)
            targets.extend(f"{base}.{alias.name}" for alias in node.names)
    return targets


def sibling_imports(source: str, module_name: str) -> list[str]:
    """Imports of ``PACKAGE`` itself or anything inside it, other than ``module_name`` itself."""
    return [
        target
        for target in _imported_targets(source)
        if (target == PACKAGE or target.startswith(f"{PACKAGE}.")) and target != module_name
    ]


def _llm_modules() -> list[Path]:
    return sorted(PACKAGE_DIR.glob("*.py"))


def test_package_contains_the_phase_one_parts() -> None:
    names = {path.stem for path in _llm_modules()}

    assert {"__init__", "openai_responses", "system_one"} <= names


@pytest.mark.parametrize("path", _llm_modules(), ids=lambda path: path.name)
def test_llm_module_imports_no_sibling(path: Path) -> None:
    module_name = PACKAGE if path.stem == "__init__" else f"{PACKAGE}.{path.stem}"

    assert sibling_imports(path.read_text(encoding="utf-8"), module_name) == []


@pytest.mark.parametrize(
    "source",
    [
        "import mtd_rl_xavier.llm.system_one",
        "import mtd_rl_xavier.llm",
        "from mtd_rl_xavier.llm import system_one",
        "from mtd_rl_xavier.llm.system_one import SystemOneClient",
        "from . import system_one",
        "from .system_one import SystemOneClient",
        "from ..llm import system_one",
        "from ..llm.system_one import SystemOneClient",
        "from .. import llm",
        "def lazy():\n    from .system_one import SystemOneClient\n",
    ],
)
def test_checker_detects_absolute_relative_and_nested_sibling_imports(source: str) -> None:
    assert sibling_imports(source, f"{PACKAGE}.openai_responses") != []


@pytest.mark.parametrize(
    "source",
    [
        "import httpx",
        "from mtd_rl_xavier import paths",
        "from .. import paths",
        "from ..lag import dataset",
        "import mtd_rl_xavier.llmx",
    ],
)
def test_checker_allows_non_sibling_imports(source: str) -> None:
    assert sibling_imports(source, f"{PACKAGE}.openai_responses") == []
