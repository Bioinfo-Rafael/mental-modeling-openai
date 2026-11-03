"""``evaluate``: score a run's claims against the policy's measured behaviour (offline numpy).

Reads ``final_answer.md`` and ``run.json`` from the run directory (never writes there), rebuilds the
policy from the checkpoint with ``reference.py``, samples base states from the spec's ``evaluation``
section, measures every input -> head effect under each hidden-state condition and writes
``evaluation.json`` + ``evaluation.md`` into a NEW directory ``<output_root>/evaluations/<run name>/``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading import provenance
from weightreading.bootstrap import BOOTSTRAP_RESAMPLES, bootstrap_scores
from weightreading.claims import ParsedClaims, parse_claims
from weightreading.effects import CONDITIONS, WARMUP_STEPS, EffectTable, measure_effects
from weightreading.eval_report import render_markdown
from weightreading.reference import build_policy
from weightreading.sampling import ANGLE_MODE, ANGLE_MODES, STEP_FRACTION, sample_base_states
from weightreading.scoring import (
    AGREE_FRACTION,
    MIN_EFFECT,
    SIGNED,
    TOP_K,
    PairTruth,
    coverage,
    pair_truths,
    score_claims,
    score_record,
    sensitivity,
    summarize,
    truth_record,
)
from weightreading.spec import Spec, SpecError, check_against_tensors, load_spec
from weightreading.torch_reader import read_checkpoint

DEFAULT_SAMPLES = 512
DEFAULT_SEED = 0
TASK_MODES = ("shown", "blind")
INT_SETTINGS = {"samples", "seed", "warmup_steps", "top_k", "bootstrap_resamples"}
STR_SETTINGS = {"angle_mode"}
# h = 0 is the one hidden state that is not synthetic (every episode starts there); warm and
# sustained use a repeated constant observation and are reported as secondary conditions.
PRIMARY_CONDITION = "h0"


class EvaluationError(ValueError):
    """The run directory, its inputs or the settings cannot be evaluated."""


@dataclass(frozen=True)
class EvalSettings:
    samples: int = DEFAULT_SAMPLES
    seed: int = DEFAULT_SEED
    warmup_steps: int = WARMUP_STEPS
    step_fraction: float = STEP_FRACTION
    agree: float = AGREE_FRACTION
    min_effect: float = MIN_EFFECT
    top_k: int = TOP_K
    angle_mode: str = ANGLE_MODE
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES

    def problems(self) -> list[str]:
        checks = [
            (self.samples >= 1, "samples must be >= 1"),
            (self.seed >= 0, "seed must be >= 0"),
            (self.warmup_steps >= 1, "warmup_steps must be >= 1"),
            (0 < self.step_fraction <= 0.5, "step_fraction must be in (0, 0.5]"),
            (0.5 < self.agree <= 1, "agree must be in (0.5, 1]"),
            (self.min_effect >= 0, "min_effect must be >= 0"),
            (self.top_k >= 1, "top_k must be >= 1"),
            (self.angle_mode in ANGLE_MODES, f"angle_mode must be one of {ANGLE_MODES}"),
            (self.bootstrap_resamples >= 1, "bootstrap_resamples must be >= 1"),
        ]
        return [message for ok, message in checks if not ok]


def _is_setting_value(key: str, value: object) -> bool:
    if isinstance(value, bool):
        return False
    if key in STR_SETTINGS:
        return isinstance(value, str)
    return isinstance(value, int) if key in INT_SETTINGS else isinstance(value, int | float)


def load_settings(path: Path | None) -> EvalSettings:
    """Defaults, overridden by the keys of an optional YAML mapping."""
    if path is None:
        return EvalSettings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError("cannot read the evaluation config as YAML") from exc
    known = {field.name for field in fields(EvalSettings)}
    if not isinstance(raw, dict) or set(raw) - known:
        raise EvaluationError(f"evaluation config must be a mapping with keys from {sorted(known)}")
    wrong = [key for key, value in raw.items() if not _is_setting_value(key, value)]
    if wrong:
        raise EvaluationError(f"evaluation config values have the wrong type: {sorted(wrong)}")
    settings = replace(EvalSettings(), **raw)
    problems = settings.problems()
    if problems:
        raise EvaluationError("; ".join(problems))
    return settings


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {path.name} in the run directory") from exc


def _run_inputs(run_dir: Path, spec: Spec, checkpoint_sha256: str, spec_path: Path) -> dict:
    """Check the run used this checkpoint and alias; report a changed spec file instead of refusing."""
    record = _read_json(run_dir / "run.json")
    if record.get("network_alias") != spec.alias:
        raise EvaluationError("run network_alias differs from the spec's network_alias")
    recorded = record.get("inputs", {})
    if recorded.get("checkpoint_sha256") != checkpoint_sha256:
        raise EvaluationError("checkpoint sha256 differs from the one recorded in run.json")
    if record.get("task_mode") not in TASK_MODES:
        raise EvaluationError("run.json has no valid task_mode")
    spec_sha256 = provenance.sha256_file(spec_path)
    return {
        "task_mode": record["task_mode"],
        "model": record.get("model"),
        "status": record.get("status"),
        "checkpoint_sha256": checkpoint_sha256,
        "spec_path": str(spec_path),
        "spec_sha256": spec_sha256,
        "run_spec_sha256": recorded.get("spec_sha256"),
        "spec_changed_since_run": spec_sha256 != recorded.get("spec_sha256"),
    }


def _audit(run_dir: Path) -> dict | None:
    path = run_dir / "audit.json"
    if not path.exists():
        return None
    audit = _read_json(path)
    return {key: audit.get(key) for key in ("runs_audited", "flagged_runs", "human_verdict")}


def _named(record: dict, spec: Spec) -> dict:
    return {
        "input": spec.inputs[record["input_index"]].name,
        "head": spec.outputs[record["head_index"]].name,
        **record,
    }


def _condition_payload(table: EffectTable, parsed: ParsedClaims, spec: Spec, settings: EvalSettings) -> dict:
    truths = pair_truths(table, settings.agree, settings.min_effect)
    scores = score_claims(parsed.claims, truths)
    counts = {direction: sum(t.truth == direction for t in truths.values()) for direction in (*SIGNED, "none")}
    hits = coverage(truths, parsed.claims, len(spec.outputs), settings.top_k)
    boot = bootstrap_scores(
        table, parsed.claims, settings.agree, settings.min_effect, settings.seed, settings.bootstrap_resamples
    )
    stability = boot.pop("claim_truth_stability")
    records = [
        _named({**score_record(score), "truth_stability": value}, spec)
        for score, value in zip(scores, stability, strict=True)
    ]
    return {
        "summary": summarize(scores, truths),
        "bootstrap": boot,
        "truth_counts": counts,
        "claims": records,
        "coverage": [_named(row, spec) for row in hits],
        "coverage_hit_rate": sum(row["hit"] for row in hits) / len(hits) if hits else None,
        "sensitivity": sensitivity(table, parsed.claims),
        "truths": [_named(truth_record(truth), spec) for truth in truths.values()],
    }


def _truth_agreement(per_condition: dict[str, dict[tuple[int, int], PairTruth]]) -> dict[str, float]:
    names = list(per_condition)
    out = {}
    for position, first in enumerate(names):
        for second in names[position + 1 :]:
            keys = per_condition[first].keys()
            same = [per_condition[first][k].truth == per_condition[second][k].truth for k in keys]
            out[f"{first}_vs_{second}"] = sum(same) / len(same)
    return out


def evaluate_run(run_dir: Path, checkpoint_path: Path, spec_path: Path, settings: EvalSettings) -> dict:
    """The full evaluation payload (no files written)."""
    spec = load_spec(spec_path)
    if spec.evaluation is None:
        raise SpecError(["spec has no evaluation section (base-state ranges); see eval_ranges.py"])
    checkpoint = read_checkpoint(checkpoint_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    run = _run_inputs(run_dir, spec, checkpoint.sha256, spec_path)
    try:
        answer = (run_dir / "final_answer.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError("run directory has no readable final_answer.md") from exc
    parsed = parse_claims(answer, spec, run["task_mode"])
    policy = build_policy(spec, checkpoint.tensors)
    base = sample_base_states(spec.evaluation, len(spec.inputs), settings.samples, np.random.default_rng(settings.seed))
    tables = {
        condition: measure_effects(
            policy, spec.evaluation, base, condition, settings.warmup_steps, settings.step_fraction, settings.angle_mode
        )
        for condition in CONDITIONS
    }
    empty = [name for name, table in tables.items() if (~np.isfinite(table.effects)).all(axis=-1).any()]
    if empty:
        raise EvaluationError("some input has no valid base state (e.g. a presence group never present); raise samples")
    truths = {name: pair_truths(table, settings.agree, settings.min_effect) for name, table in tables.items()}
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "network_alias": spec.alias,
        "run": run,
        "checkpoint_path": str(checkpoint_path),
        "audit": _audit(run_dir),
        "settings": asdict(settings),
        "conditions_order": list(CONDITIONS),
        "primary_condition": PRIMARY_CONDITION,
        "base_states": {"n": settings.samples, "present_fraction": base.present.mean(axis=0).tolist()},
        "claims_parsed": {
            "n": len(parsed.claims),
            "rejected": [asdict(item) for item in parsed.rejected],
            "duplicate_pairs": [list(pair) for pair in parsed.duplicate_pairs],
        },
        "truth_agreement_between_conditions": _truth_agreement(truths),
        "conditions": {name: _condition_payload(table, parsed, spec, settings) for name, table in tables.items()},
    }


def json_safe(value: object) -> object:
    """Replace non-finite floats (e.g. an undefined fraction) by None, recursively."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def write_evaluation(payload: dict, directory: Path) -> Path:
    """Create ``directory`` exclusively and write ``evaluation.json`` + ``evaluation.md``."""
    derived_path(directory).mkdir(parents=True, exist_ok=False)
    text = json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (directory / "evaluation.json").write_text(text, encoding="utf-8")
    (directory / "evaluation.md").write_text(render_markdown(payload), encoding="utf-8")
    return directory
