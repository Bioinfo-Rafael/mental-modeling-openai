"""No public function or class in src/ is left without a caller, and shared constants are defined once."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from translate import paths

SRC = paths.ROOT / "src"
SEARCHED = sorted((*SRC.rglob("*.py"), *(paths.ROOT / "scripts").rglob("*.py")))
# Known public names without a caller in src/ or scripts/ (each covered by tests). The lag module is a
# frozen request-path file (tests/fixtures/assess-request-path.sha256); the policy helpers are kept for
# the reading toolkit. Shrink this list; never grow it to make a new leftover pass.
KNOWN_UNCALLED = {
    "translate/lag/dataset.py": {"history_steps", "select_records", "index_records"},
    "translate/policy/bundle.py": {"write_bundle"},
    "translate/policy/tensors.py": {"total_params"},
    "translate/policy/reference.py": {"expected_value"},
}
INVALID_ANSWER_TEXT = "invalid final answer: "


def _public_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]


def _uncalled() -> dict[str, set[str]]:
    texts = {path: path.read_text(encoding="utf-8") for path in SEARCHED}
    found: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for name in _public_names(path):
            pattern = re.compile(rf"\b{re.escape(name)}\b")
            elsewhere = sum(len(pattern.findall(text)) for other, text in texts.items() if other != path)
            own = len(pattern.findall(texts[path])) - 1
            if elsewhere == 0 and own == 0:
                found.setdefault(str(path.relative_to(SRC)), set()).add(name)
    return found


def test_no_public_name_in_src_is_left_without_a_caller() -> None:
    assert _uncalled() == KNOWN_UNCALLED


def test_invalid_answer_prefix_is_defined_once() -> None:
    definitions = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and node.value.value == INVALID_ANSWER_TEXT
    ]

    assert len(definitions) == 1, definitions
