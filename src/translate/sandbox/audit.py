"""Advisory audit of a code session's ``tool_calls.jsonl``.

The prompt's rules (evaluate the network only on recorded rows) cannot be enforced mechanically,
so this module only *flags* code for a reviewer, who records the verdict in ``audit.json``
(``human_verdict`` stays null until then). Nonlinearities applied to rows taken unchanged from ``A``
are allowed, so ``nonlinearity:*`` flags only point at code to read; the session flags mark the
usual ways of building forbidden inputs: writing into ``A`` / ``W``, synthetic vectors
(``np.zeros``/``ones``/``random``/``linspace`` ...) and autodiff imports.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from translate.sandbox.session import TOOL_CALLS

AUDIT_FILE = "audit.json"
NONLINEAR_CALLS = {
    "tanh",
    "sigmoid",
    "expit",
    "exp",
    "softmax",
    "log_softmax",
    "relu",
    "layer_norm",
    "logistic",
    "erf",
}
FORBIDDEN_MODULES = {"torch", "tensorflow", "jax", "keras", "scipy.special", "sklearn"}
EXCERPT_CHARS = 160
MODE = "code-session"
INPUT_NAMES = {"A", "W"}
SYNTHETIC_CALLS = {"zeros", "ones", "full", "eye", "linspace", "rand", "randn", "normal", "uniform"}
AUTODIFF_MODULES = {"autograd", "jax", "torch", "tensorflow", "mxnet", "chainer"}


@dataclass(frozen=True)
class Flag:
    run: int
    rule: str
    line: int
    excerpt: str


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _excerpt(source_lines: list[str], line: int) -> str:
    if 1 <= line <= len(source_lines):
        return source_lines[line - 1].strip()[:EXCERPT_CHARS]
    return ""


def _is_zero(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _relu_like(node: ast.Call) -> bool:
    """``np.maximum(x, 0)`` / ``np.maximum(0, x)`` / ``np.clip(x, 0, None)``."""
    name = _call_name(node)
    if name == "maximum" and any(_is_zero(arg) for arg in node.args):
        return True
    return name == "clip" and len(node.args) >= 2 and _is_zero(node.args[1])


def _layer_norm_like(node: ast.BinOp) -> bool:
    """``(v - v.mean()) / np.sqrt(v.var() + eps)`` or ``... / v.std()``."""
    if not isinstance(node.op, ast.Div) or not isinstance(node.left, ast.BinOp):
        return False
    if not isinstance(node.left.op, ast.Sub):
        return False
    names = {_call_name(n) for n in ast.walk(node) if isinstance(n, ast.Call)}
    return "mean" in names and bool(names & {"sqrt", "std", "var"})


def _imports(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def _loop_recurrence(loop: ast.For | ast.While) -> bool:
    """A loop body that re-assigns a name it also reads and applies a nonlinearity."""
    assigned = {t.id for n in ast.walk(loop) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    read = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    nonlinear = any(isinstance(n, ast.Call) and _call_name(n) in NONLINEAR_CALLS for n in ast.walk(loop))
    return nonlinear and bool(assigned & read)


def _node_flags(node: ast.AST) -> list[str]:
    rules = []
    if isinstance(node, ast.Call) and _call_name(node) in NONLINEAR_CALLS:
        rules.append(f"nonlinearity:{_call_name(node)}")
    if isinstance(node, ast.Call) and _relu_like(node):
        rules.append("nonlinearity:relu_like")
    if isinstance(node, ast.BinOp) and _layer_norm_like(node):
        rules.append("nonlinearity:layer_norm_like")
    if isinstance(node, ast.For | ast.While) and _loop_recurrence(node):
        rules.append("recurrence_loop")
    for module in _imports(node):
        if module.split(".")[0] in FORBIDDEN_MODULES or module in FORBIDDEN_MODULES:
            rules.append(f"forbidden_import:{module}")
    return rules


def audit_code(run: int, code: str) -> list[Flag]:
    """Flags for one code run. Unparseable code is itself flagged (it cannot be checked)."""
    lines = code.splitlines()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [Flag(run, "unparseable", exc.lineno or 0, _excerpt(lines, exc.lineno or 0))]
    flags = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        flags += [Flag(run, rule, line, _excerpt(lines, line)) for rule in _node_flags(node)]
    return sorted(set(flags), key=lambda flag: (flag.line, flag.rule))


def audit_run(run_dir: Path) -> dict:
    """Audit every recorded code run in ``run_dir``; returns the ``audit.json`` payload."""
    path = run_dir / TOOL_CALLS
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    flags = [flag for record in records for flag in audit_code(int(record["run"]), str(record["code"]))]
    return {
        "run_dir": str(run_dir),
        "runs_audited": len(records),
        "flagged_runs": sorted({flag.run for flag in flags}),
        "flags": [asdict(flag) for flag in flags],
        "advisory": "Flags are heuristic. A reviewer must read every run and set human_verdict.",
        "human_verdict": None,
        "reviewer": None,
        "review_notes": None,
    }


def write_audit(run_dir: Path) -> Path:
    """Write ``audit.json``; refuses to overwrite one that already carries a human verdict."""
    target = run_dir / AUDIT_FILE
    if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("human_verdict") is not None:
        raise FileExistsError(f"{target} already has a human verdict; not overwriting")
    target.write_text(json.dumps(audit_run(run_dir), indent=2) + "\n", encoding="utf-8")
    return target


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


def session_flags(run: int, code: str) -> list[Flag]:
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


def session_audit(session_dir: Path, prompt_version: str) -> dict:
    base = audit_run(session_dir)
    path = session_dir / TOOL_CALLS
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    extra = [flag for record in records for flag in session_flags(int(record["run"]), str(record["code"]))]
    flags = base["flags"] + [asdict(flag) for flag in extra]
    return {
        **base,
        "mode": MODE,
        "flags": flags,
        "flagged_runs": sorted({flag["run"] for flag in flags}),
        "prompt_version": prompt_version,
        "advisory": (
            "Heuristic flags. Nonlinearities on rows taken unchanged from A are allowed; a reviewer must check "
            "that no layer was evaluated on a perturbed or synthetic vector and set human_verdict."
        ),
    }


def write_session_audit(session_dir: Path, prompt_version: str) -> Path:
    """Write ``audit.json``; refuses to overwrite one that already carries a human verdict."""
    target = session_dir / AUDIT_FILE
    if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("human_verdict") is not None:
        raise FileExistsError(f"{target} already has a human verdict; not overwriting")
    target.write_text(json.dumps(session_audit(session_dir, prompt_version), indent=2) + "\n", encoding="utf-8")
    return target
