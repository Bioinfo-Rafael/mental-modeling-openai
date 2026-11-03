"""Advisory forward-pass audit of tool_calls.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weightreading.audit import audit_code, audit_run, write_audit


def _rules(code: str) -> set[str]:
    return {flag.rule for flag in audit_code(1, code)}


@pytest.mark.parametrize(
    ("code", "rule"),
    [
        ("h = np.tanh(W['a'] @ x)", "nonlinearity:tanh"),
        ("z = 1 / (1 + np.exp(-v))", "nonlinearity:exp"),
        ("a = np.maximum(W['w'] @ x + b, 0)", "nonlinearity:relu_like"),
        ("a = np.clip(v, 0, None)", "nonlinearity:relu_like"),
        ("y = (v - v.mean()) / np.sqrt(v.var() + 1e-5)", "nonlinearity:layer_norm_like"),
        ("import torch", "forbidden_import:torch"),
        ("from scipy.special import expit", "forbidden_import:scipy.special"),
        ("for t in range(5):\n    h = np.tanh(U @ h)", "recurrence_loop"),
        ("def f(:\n", "unparseable"),
    ],
)
def test_audit_code_flags_forward_pass_patterns(code: str, rule: str) -> None:
    assert rule in _rules(code)


@pytest.mark.parametrize(
    "code",
    [
        "u, s, vt = np.linalg.svd(W['rnn.gru.weight_hh_l0'][128:256])\nprint(s[:5])",
        "M = W['b.3.weight'] @ (W['b.2.weight'][:, None] * W['b.0.weight'])\nprint(np.abs(M).sum(0))",
        "print(np.argsort(-np.abs(W['h.0.weight']).sum(1))[:5])",
        "for k in W:\n    print(k, W[k].shape)",
    ],
)
def test_audit_code_allows_weight_algebra(code: str) -> None:
    assert _rules(code) == set()


def _write_calls(run_dir: Path, codes: list[str]) -> None:
    lines = [json.dumps({"run": i + 1, "code": code, "out": ""}) for i, code in enumerate(codes)]
    (run_dir / "tool_calls.jsonl").write_text("\n".join(lines) + "\n")


def test_audit_run_lists_flagged_runs_and_leaves_verdict_null(tmp_path: Path) -> None:
    _write_calls(tmp_path, ["print(W.keys())", "print(np.tanh(W['a'] @ x))"])

    payload = audit_run(tmp_path)

    assert payload["runs_audited"] == 2
    assert payload["flagged_runs"] == [2]
    assert payload["human_verdict"] is None


def test_audit_run_without_tool_calls_reports_zero_runs(tmp_path: Path) -> None:
    assert audit_run(tmp_path)["runs_audited"] == 0


def test_write_audit_refuses_to_overwrite_human_verdict(tmp_path: Path) -> None:
    _write_calls(tmp_path, ["print(1)"])
    (tmp_path / "audit.json").write_text(json.dumps({"human_verdict": "pass"}))

    with pytest.raises(FileExistsError):
        write_audit(tmp_path)
