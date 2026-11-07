"""The core (Translator, methods A/B, judge) is independent of the workspace project.

The core never imports the workspace library, the vendored LLM-Xavier package or an experiment
module, never edits ``sys.path``, and installs exactly its own packages.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from mtd_rl_xavier import paths
from mtd_rl_xavier.tools.output_safety import derived_path

CORE_PACKAGES = (paths.ROOT / "src" / "mtd_rl_xavier", paths.ROOT / "src" / "weightreading", paths.ROOT / "evaluation")
CORE_SOURCES = sorted(path for package in CORE_PACKAGES for path in package.rglob("*.py"))
TEST_SOURCES = sorted((paths.ROOT / "tests").rglob("*.py"))
FORBIDDEN_IMPORT_ROOTS = {"workspace", "mtd_rl_xavier_workspace", "llm_x", "mentalmodeling"}


def imported_roots(path: Path) -> set[str]:
    """Top-level names of static imports plus ``import_module("...")`` string literals."""
    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "import_module":
            names += [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
    return {name.split(".")[0] for name in names}


def sys_path_mutations(path: Path) -> list[int]:
    """Line numbers of expressions that touch ``sys.path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "path" and getattr(node.value, "id", None) == "sys"
    ]


def test_core_sources_are_found() -> None:
    assert paths.ROOT / "src" / "mtd_rl_xavier" / "translator" / "cli.py" in CORE_SOURCES
    assert paths.ROOT / "evaluation" / "assess_cli.py" in CORE_SOURCES


@pytest.mark.parametrize("path", CORE_SOURCES + TEST_SOURCES, ids=lambda p: str(p.relative_to(paths.ROOT)))
def test_core_module_does_not_import_the_workspace(path: Path) -> None:
    assert imported_roots(path) & FORBIDDEN_IMPORT_ROOTS == set()


@pytest.mark.parametrize("path", CORE_SOURCES, ids=lambda p: str(p.relative_to(paths.ROOT)))
def test_core_module_does_not_touch_sys_path(path: Path) -> None:
    assert sys_path_mutations(path) == []


def test_vendored_prior_work_is_not_installed_in_the_core_environment(tmp_path: Path) -> None:
    probe = "import importlib.util as u; print(u.find_spec('llm_x'), u.find_spec('mtd_rl_xavier_workspace'))"

    result = subprocess.run(
        [sys.executable, "-I", "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    assert result.stdout.split() == ["None", "None"]


def test_setuptools_package_list_covers_every_package_directory() -> None:
    """The list is explicit (see pyproject.toml), so a new subpackage must be added by hand."""
    tomllib = pytest.importorskip("tomllib")
    declared = set(tomllib.loads((paths.ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    src = paths.ROOT / "src"
    on_disk = {
        ".".join(init.parent.relative_to(src).parts)
        for top in ("mtd_rl_xavier", "weightreading")
        for init in (src / top).rglob("__init__.py")
    }
    judge = {
        ".".join(init.parent.relative_to(paths.ROOT).parts) for init in (paths.ROOT / "evaluation").rglob("__init__.py")
    }

    assert declared == on_disk | judge


def test_core_package_is_importable_without_the_workspace_on_sys_path() -> None:
    assert importlib.util.find_spec("mtd_rl_xavier.translator.cli") is not None
    assert not any(Path(entry).resolve() == paths.ROOT / "workspace" for entry in sys.path if entry)


@pytest.mark.parametrize("relative", ["", "mhi/derived.json", "new_dir/out.csv"])
def test_derived_path_rejects_writes_under_committed_data(relative: str) -> None:
    with pytest.raises(ValueError, match="read-only source"):
        derived_path(paths.DATA / relative)


def test_derived_path_rejects_symlink_into_committed_data(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(paths.DATA)

    with pytest.raises(ValueError, match="read-only source"):
        derived_path(tmp_path / "link/new.json")
