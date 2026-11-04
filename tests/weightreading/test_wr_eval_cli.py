"""``python -m weightreading evaluate``: end to end on the known-sign network, refusals, settings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weightreading.cli import main
from weightreading.evaluate import EvalSettings, EvaluationError, load_settings

from .wr_eval_helpers import eval_spec_dict, write_eval_files
from .wr_helpers import SMALL_SPEC

CLAIMS = [
    {"input": "speed", "head": "climb", "direction": "+", "confidence": 0.9},  # correct
    {"input": "speed", "head": "dive", "direction": "+", "confidence": 0.7},  # wrong sign
    {"input": "noise", "head": "dive", "direction": "none", "confidence": 0.8},  # correct
    {"input": "side", "head": "climb", "direction": "-", "confidence": 0.4},  # overclaim
]
FAST = "samples: 32\nbootstrap_resamples: 10\nwarmup_steps: 2\n"


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def _args(run_dir: Path, checkpoint: Path, spec: Path, out: Path, config: Path | None) -> list[str]:
    argv = ["evaluate", "--run-dir", str(run_dir), "--checkpoint", str(checkpoint), "--spec", str(spec)]
    argv += ["--output-root", str(out)]
    return argv + (["--config", str(config)] if config else [])


@pytest.fixture
def fast_config(tmp_path: Path) -> Path:
    path = tmp_path / "fast.yaml"
    path.write_text(FAST, encoding="utf-8")
    return path


def test_evaluate_writes_new_directory_and_scores_claims(tmp_path: Path, fast_config: Path, capsys) -> None:  # noqa: ANN001
    # Arrange
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    before = _snapshot(run_dir)
    out = tmp_path / "out"

    # Act
    code = main(_args(run_dir, checkpoint, spec, out, fast_config))

    # Assert
    assert code == 0
    assert _snapshot(run_dir) == before
    directory = out / "evaluations" / run_dir.name
    payload = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    claims = payload["conditions"]["h0"]["claims"]
    assert [c["outcome"] for c in claims] == ["correct", "wrong_sign", "correct", "overclaim"]
    assert payload["conditions"]["h0"]["summary"]["overall"]["accuracy"] == 0.5
    assert payload["settings"]["samples"] == 32
    assert payload["primary_condition"] == "h0"
    assert "# Evaluation:" in (directory / "evaluation.md").read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["claims"] == 4


def test_evaluate_refuses_existing_output_directory(tmp_path: Path, fast_config: Path) -> None:
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    argv = _args(run_dir, checkpoint, spec, tmp_path / "out", fast_config)
    assert main(argv) == 0

    assert main(argv) == 1


def test_evaluate_refuses_checkpoint_not_used_by_the_run(tmp_path: Path, fast_config: Path, capsys) -> None:  # noqa: ANN001
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    record["inputs"]["checkpoint_sha256"] = "f" * 64
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")

    code = main(_args(run_dir, checkpoint, spec, tmp_path / "out", fast_config))

    assert code == 1
    assert "checkpoint sha256 differs" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_evaluate_refuses_spec_without_evaluation_section(tmp_path: Path, small_files: tuple[Path, Path]) -> None:
    run_dir, _, _ = write_eval_files(tmp_path, CLAIMS)
    checkpoint, spec = small_files
    assert "evaluation" not in SMALL_SPEC

    assert main(_args(run_dir, checkpoint, spec, tmp_path / "out", None)) == 1


def test_evaluate_refuses_answer_without_claims_block(tmp_path: Path, fast_config: Path) -> None:
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    (run_dir / "final_answer.md").write_text("## Summary\nno json\n", encoding="utf-8")

    assert main(_args(run_dir, checkpoint, spec, tmp_path / "out", fast_config)) == 1


def test_load_settings_defaults_and_overrides(tmp_path: Path, fast_config: Path) -> None:
    assert load_settings(None) == EvalSettings()
    assert load_settings(fast_config).samples == 32


@pytest.mark.parametrize(
    "text",
    [
        "unknown_key: 1\n",
        "samples: 0\n",
        "samples: 1.5\n",
        "agree: 0.4\n",
        "angle_mode: degrees\n",
        "samples: true\n",
        "seed: -1\n",
        "samples: [unclosed\n",
    ],
)
def test_load_settings_rejects_bad_values(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(EvaluationError):
        load_settings(path)


def test_evaluate_with_no_valid_claims_still_writes_report(tmp_path: Path, fast_config: Path) -> None:
    ghost = [{"input": "ghost", "head": "climb", "direction": "+", "confidence": 1}]
    run_dir, checkpoint, spec = write_eval_files(tmp_path, ghost)

    assert main(_args(run_dir, checkpoint, spec, tmp_path / "out", fast_config)) == 0

    report = (tmp_path / "out" / "evaluations" / run_dir.name / "evaluation.md").read_text(encoding="utf-8")
    assert "rejected: 1" in report


def test_evaluate_refuses_input_without_any_valid_state(tmp_path: Path, fast_config: Path, capsys) -> None:  # noqa: ANN001
    # Arrange: the missile group is (practically) never present, so msl_dist has no valid state
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    raw = eval_spec_dict()
    raw["evaluation"]["presence"][0]["p_present"] = 1e-12
    spec.write_text(json.dumps(raw), encoding="utf-8")

    # Act
    code = main(_args(run_dir, checkpoint, spec, tmp_path / "out", fast_config))

    # Assert
    assert code == 1
    assert "no valid base state" in capsys.readouterr().err


def test_evaluate_bad_config_exits_one(tmp_path: Path) -> None:
    run_dir, checkpoint, spec = write_eval_files(tmp_path, CLAIMS)
    bad = tmp_path / "bad.yaml"
    bad.write_text("samples: -3\n", encoding="utf-8")

    assert main(_args(run_dir, checkpoint, spec, tmp_path / "out", bad)) == 1
