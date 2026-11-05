"""The shared library never depends on an experiment: no ``workspace`` imports, no ``sys.path`` edits.

Experiment entry points under ``workspace/`` import their own ``run.py`` spec and pass it in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

LIBRARY = Path(__file__).resolve().parents[1] / "src" / "mtd_rl_xavier"
SOURCES = sorted(LIBRARY.rglob("*.py"))


def workspace_imports(tree: ast.AST) -> list[str]:
    """Static imports of ``workspace`` plus ``import_module("workspace...")`` string literals."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "import_module":
            names += [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
    return [name for name in names if name.split(".")[0] == "workspace"]


def sys_path_mutations(tree: ast.AST) -> list[int]:
    """Line numbers of calls or assignments that touch ``sys.path``."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path" and getattr(node.value, "id", None) == "sys":
            lines.append(node.lineno)
    return lines


def test_library_sources_are_found() -> None:
    assert LIBRARY / "experiment" / "common_analysis" / "runner.py" in SOURCES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(LIBRARY)))
def test_library_module_does_not_import_workspace(path: Path) -> None:
    assert workspace_imports(ast.parse(path.read_text(encoding="utf-8"))) == []


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(LIBRARY)))
def test_library_module_does_not_touch_sys_path(path: Path) -> None:
    assert sys_path_mutations(ast.parse(path.read_text(encoding="utf-8"))) == []


def test_analysis_runner_given_an_experiment_name_raises_type_error() -> None:
    from mtd_rl_xavier.experiment.common_analysis import runner

    with pytest.raises(TypeError, match="experiment spec"):
        runner.main("06_new_models_n10", argv=[])


def test_load_and_parse_without_spec_raises_type_error() -> None:
    from mtd_rl_xavier.experiment.common_analysis import joint_data

    with pytest.raises(TypeError, match="experiment spec"):
        joint_data.load_and_parse()


def test_analysis_runner_uses_the_passed_spec_for_its_output_root(tmp_path: Path) -> None:
    from importlib import import_module

    from mtd_rl_xavier.experiment.common_analysis import runner

    spec = import_module("workspace.05_new_models_single.run").EXPERIMENT

    with pytest.raises(SystemExit):  # argparse rejects an output outside workspace/05_*/analysis before any I/O
        runner.main(spec, argv=["--output-dir", str(tmp_path)])
