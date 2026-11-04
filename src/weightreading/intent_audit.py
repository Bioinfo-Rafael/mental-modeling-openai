"""Advisory audit of an intent-B session: the weight-reading heuristics plus B-specific flags.

In intent-B, nonlinearities applied to rows taken unchanged from ``A`` are allowed (they verify or
decompose recorded values), so ``nonlinearity:*`` flags are expected and only point the reviewer
at code to read. What the rules forbid is evaluating the network on anything that is not a recorded
row; the extra flags mark the usual ways of building such inputs: writing into ``A`` / ``W``,
synthetic vectors (``np.zeros``/``ones``/``random``/``linspace`` ...) and a recurrence loop.
``human_verdict`` stays null until a reviewer sets it.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from weightreading.audit import AUDIT_FILE, Flag, audit_run
from weightreading.intent_prompt import RULES

MODE = "intent-b"
INPUT_NAMES = {"A", "W"}
SYNTHETIC_CALLS = {"zeros", "ones", "full", "eye", "linspace", "rand", "randn", "normal", "uniform"}
AUTODIFF_MODULES = {"autograd", "jax", "torch", "tensorflow", "mxnet", "chainer"}
EXCERPT_CHARS = 160


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Subscript | ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _writes_input(node: ast.AST) -> bool:
    """``A[...] = ...``, ``A[k][i] += ...`` or ``A.update(...)``-style writes into the given arrays."""
    if isinstance(node, ast.Assign | ast.AugAssign):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(isinstance(t, ast.Subscript) and _root_name(t) in INPUT_NAMES for t in targets)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        mutating = node.func.attr in {"update", "pop", "clear", "setdefault", "fill", "put", "itemset", "setflags"}
        return mutating and _root_name(node.func.value) in INPUT_NAMES
    return False


def _synthetic(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    return node.func.attr in SYNTHETIC_CALLS


def _autodiff_import(node: ast.AST) -> bool:
    names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
    if isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module]
    return any(name.split(".")[0] in AUTODIFF_MODULES for name in names)


def intent_flags(run: int, code: str) -> list[Flag]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # already flagged as "unparseable" by the base audit
    lines = code.splitlines()
    flags = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        excerpt = lines[line - 1].strip()[:EXCERPT_CHARS] if 1 <= line <= len(lines) else ""
        if _writes_input(node):
            flags.append(Flag(run, "intent:writes_recorded_input", line, excerpt))
        if _synthetic(node):
            flags.append(Flag(run, "intent:synthetic_array", line, excerpt))
        if _autodiff_import(node):
            flags.append(Flag(run, "intent:autodiff_import", line, excerpt))
    return sorted(set(flags), key=lambda flag: (flag.line, flag.rule))


def intent_audit(session_dir: Path) -> dict:
    base = audit_run(session_dir)
    path = session_dir / "tool_calls.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    extra = [flag for record in records for flag in intent_flags(int(record["run"]), str(record["code"]))]
    flags = base["flags"] + [asdict(flag) for flag in extra]
    return {
        **base,
        "mode": MODE,
        "flags": flags,
        "flagged_runs": sorted({flag["run"] for flag in flags}),
        "rules": RULES,
        "advisory": (
            "Heuristic flags. In intent-B, nonlinearities on rows taken unchanged from A are allowed; a reviewer "
            "must check that no layer was evaluated on a perturbed or synthetic vector and set human_verdict."
        ),
    }


def write_intent_audit(session_dir: Path) -> Path:
    """Write ``audit.json``; refuses to overwrite one that already carries a human verdict."""
    target = session_dir / AUDIT_FILE
    if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("human_verdict") is not None:
        raise FileExistsError(f"{target} already has a human verdict; not overwriting")
    target.write_text(json.dumps(intent_audit(session_dir), indent=2) + "\n", encoding="utf-8")
    return target
