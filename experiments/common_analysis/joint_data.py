"""Read-only input validation, strict final-marker parsing, and trajectory ground truth."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from experiments import common
from llm_x.data import Episode
from llm_x.evaluation import EvaluationConfig, _action_range
from llm_x.metrics import bin_actions, state_directions

TASKS = ("MountainCar-v0", "Pendulum-v1")
METRICS = ("next-action", "last-action", "next-state", "last-state")
HISTORIES = (5, 10, 20, 30)
SIZES = (10, 20, 30)
DIMENSIONS = {TASKS[0]: ("position", "velocity"),
              TASKS[1]: ("cos(theta)", "sin(theta)", "angular_velocity")}
# Verified against llm_x/task.py observation_space and action_space, not fitted to samples.
BOUNDS = {TASKS[0]: [[-1.2, 0.6], [-0.07, 0.07]], TASKS[1]: [[-1, 1], [-1, 1], [-8, 8]]}
WIDTHS = {task: [high-low for low, high in bounds] for task, bounds in BOUNDS.items()}
COMPONENTS = ("action", "action_value", "action_bin", "direction", "state_value", "state_delta")


def required_components(task: str, metric: str) -> tuple[str, ...]:
    if metric.endswith("state"):
        return ("direction", "state_value", "state_delta")
    return ("action",) if task == TASKS[0] else ("action_value", "action_bin")


def parse_component(text: str, label: str, size: int, kind: str = "number") -> dict:
    """Only anchored output labels. No eval, prose-number extraction, or placeholder repair.

    Optional list numbering, >>, and bold labels are formatting only. Conflicting
    repeated markers fail rather than silently choosing a favorable prediction.
    """
    prefix = r"^[ \t]*(?:(?:\d+[.)]|[-*])[ \t]*)?(?:>>[ \t]*)?(?:\*\*)?(?:\[)?"
    suffix = r"(?:\])?(?:\*\*)?[ \t]*[:=][ \t]*(?:\*\*)?\s*(\[[^\]]*\]|[^\n]+)"
    matches = re.findall(prefix + re.escape(label) + suffix, text, re.M | re.I)
    if not matches:
        return {"value": None, "ok": False, "error": "missing_marker"}
    values = []
    for token in matches:
        try:
            value = ast.literal_eval(token.strip())
        except (SyntaxError, ValueError):
            return {"value": None, "ok": False, "error": "non_literal_or_placeholder"}
        if kind == "action" and not isinstance(value, list):
            value = [value]
        if not isinstance(value, list) or len(value) != size:
            return {"value": None, "ok": False, "error": "wrong_dimension"}
        if kind == "direction":
            if any(not isinstance(v, str) or v.upper() not in {"INC", "DEC", "UNCH"} for v in value):
                return {"value": None, "ok": False, "error": "invalid_direction"}
            value = [{"DEC": 0, "INC": 1, "UNCH": 2}[v.upper()] for v in value]
        else:
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                return {"value": None, "ok": False, "error": "non_finite_or_non_numeric"}
            if kind in {"action", "bin"}:
                maximum = 2 if kind == "action" else 9
                if any(not float(v).is_integer() or not 0 <= v <= maximum for v in value):
                    return {"value": None, "ok": False, "error": f"invalid_{kind}_id"}
                value = [int(v) for v in value]
        values.append(value)
    if any(v != values[0] for v in values[1:]):
        return {"value": None, "ok": False, "error": "conflicting_markers"}
    return {"value": values[0], "ok": True, "error": ""}


def parse_response(text: str, task: str, metric: str) -> dict:
    result = {c: {"value": None, "ok": False, "error": "not_applicable"} for c in COMPONENTS}
    if metric.endswith("state"):
        size = len(DIMENSIONS[task])
        result["direction"] = parse_component(text, "predictions", size, "direction")
        result["state_value"] = parse_component(text, "Final state values", size)
        result["state_delta"] = parse_component(text, "Final state deltas", size)
    elif task == TASKS[0]:
        result["action"] = parse_component(text, "Final action choice", 1, "action")
    else:
        result["action_value"] = parse_component(text, "predictions", 1)
        result["action_bin"] = parse_component(text, "Final action bins", 1, "bin")
    return result


def load_and_parse(source=None, spec=None) -> tuple[Path, dict, list[dict], dict]:
    if spec is None:
        from importlib import import_module
        spec = import_module("experiments.03_1_gpt35_history_n30_joint.run").EXPERIMENT
    source = Path(source) if source is not None else ROOT / "experiments" / spec.name / "results"
    manifest = common.read_json(source / "manifest.json")
    summary = common.read_json(source / "summary.json")
    records = common.read_jsonl(source / "records.jsonl")
    expected = {(common.MODELS[a], t, m, h): spec.n for a in spec.models
                for t in spec.tasks for m in spec.metrics for h in spec.histories}
    count = sum(expected.values())
    if manifest["experiment"] != spec.name or summary["status"] != "complete" or len(records) != count:
        raise ValueError(f"Require completed {spec.name}: {count} queries")
    grid = Counter((r["model"], r["task"], r["metric"], r["H"]) for r in records)
    if grid != Counter(expected):
        raise ValueError("Unexpected Model/Task/Metric/H/count grid")
    if len({r["query_id"] for r in records}) != count or len(manifest["queries"]) != count:
        raise ValueError("Duplicated or missing query identities")
    configs = {c["condition_id"]: EvaluationConfig(**c["evaluation_config"]) for c in manifest["conditions"]}
    episodes, run_records, parsed = {}, {}, []
    for line, (r, q) in enumerate(zip(records, manifest["queries"]), 1):
        if r["model_alias"] not in spec.models or common.MODELS[r["model_alias"]] != r["model"]:
            raise ValueError("Model alias and recorded API model disagree")
        if any(r[k] != q[k] for k in (*common.IDENTITY_FIELDS, "query_id", "ordinal", "condition_id")):
            raise ValueError("Record and manifest differ")
        if common.query_id(r) != r["query_id"]:
            raise ValueError("Query hash mismatch")
        text = r["assistant_text"]
        if text != r["raw_response"]["choices"][0]["message"]["content"]:
            raise ValueError("assistant_text differs from raw_response")
        run_path = ROOT / r["upstream_output"] / "predictions.jsonl"
        if not run_path.resolve().is_relative_to((source / "runs").resolve()):
            raise ValueError("Run source is outside the specified results/runs")
        if str(run_path) not in run_records:
            run_records[str(run_path)] = common.read_jsonl(run_path)
        matching_run_rows = [x for x in run_records[str(run_path)] if x["index"] == r["query_index"]]
        if len(matching_run_rows) != 1:
            raise ValueError("Missing or duplicated source run query")
        run_row = matching_run_rows[0]
        if run_row["prompt"] != r["user_prompt"] or run_row["ground_truth"] != r["ground_truth"]:
            raise ValueError("Run prompt or ground truth differs from records.jsonl")
        path = r["episode_path"]
        if path not in episodes:
            episodes[path] = Episode.load(ROOT / path)
        episode = episodes[path]
        if episode.sha256 != r["episode_sha256"]:
            raise ValueError("Raw episode hash changed")
        task, metric, index = r["task"], r["metric"], r["query_index"]
        if r["question_name"] != spec.questions[task][metric]:
            raise ValueError("Results use different questions; old non-Joint runs are not compatible")
        config = configs[r["condition_id"]]
        components = parse_response(text, task, metric)
        row = {k: r[k] for k in ("query_id", "condition_id", "task", "metric", "H", "ordinal", "query_index", "episode_path", "model", "model_alias")}
        row.update(reused=r.get("reused", False), source_experiment=r.get("source_experiment"),
                   api_request_made=r.get("api_request_made", False), source_record=r.get("source_record"))
        row.update(source_line=line, source_records=str((source / "records.jsonl").relative_to(ROOT)),
                   source_run=str(run_path.relative_to(ROOT)), response_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   components=components, all_required=all(components[c]["ok"] for c in required_components(task, metric)))
        for key in ("input_tokens", "output_tokens", "total_tokens", "query_elapsed_seconds", "request_elapsed_seconds"):
            value = r.get(key)
            row[key] = value if type(value) in (int, float) and math.isfinite(value) else None
        if metric.endswith("state"):
            raw_early, raw_late = episode.state_vector(index), episode.state_vector(index + 1)
            direction = state_directions(raw_early, raw_late, threshold=config.state_threshold, allow_unchanged=True)
            # Retain official dtype/rounding for direction; use float64 for continuous arithmetic.
            early, late = raw_early.astype(float), raw_late.astype(float)
            row.update(early=early.tolist(), late=late.tolist(), gt_direction=direction,
                       gt_state_value=(late if metric == "next-state" else early).tolist(),
                       gt_state_delta=(late - early).tolist())
            if direction != r["ground_truth"]:
                raise ValueError("Trajectory direction disagrees with recorded GT")
        else:
            action = episode.action_vector(index).astype(float).tolist()
            row["gt_action_value"] = action
            if task == TASKS[0]:
                row["gt_action"] = [episode.discrete_action(index)]
                if row["gt_action"][0] != r["ground_truth"]:
                    raise ValueError("Trajectory action disagrees with recorded GT")
            else:
                start, stop = _action_range(config)
                if (start, stop, config.action_bins) != (-2, 2, 10):
                    raise ValueError("Unexpected Pendulum action normalization/bin range")
                row["gt_action_bin"] = bin_actions(action, start=start, stop=stop, bins=config.action_bins)
                if row["gt_action_bin"] != r["ground_truth"]:
                    raise ValueError("Trajectory action bin disagrees with recorded GT")
        parsed.append(row)
    return source, manifest, parsed, {p: e.sha256 for p, e in episodes.items()}


def select_subsets(records: list[dict], seed: int, sizes=SIZES, expected_n=30) -> dict:
    """One shared physical-query ordering per task; overlapping windows share ranks."""
    rank = {}
    for task in TASKS:
        keys = sorted({(r["episode_path"], r["query_index"]) for r in records if r["task"] == task})
        random.Random(seed).shuffle(keys)
        rank[task] = {key: i for i, key in enumerate(keys)}
    groups = {}
    for cid in sorted({r["condition_id"] for r in records}):
        rows = sorted((r for r in records if r["condition_id"] == cid),
                      key=lambda r: rank[r["task"]][(r["episode_path"], r["query_index"])])
        if len(rows) != expected_n or any(n > len(rows) or n < 1 for n in sizes):
            raise ValueError(f"Each condition must contain {expected_n} rows, sufficient for all subsets")
        groups[cid] = {str(n): [{k: r[k] for k in ("query_id", "episode_path", "query_index", "ordinal")}
                                for r in rows[:n]] for n in sizes}
        for small, large in zip(sorted(sizes), sorted(sizes)[1:]):
            assert groups[cid][str(small)] == groups[cid][str(large)][:small]
    return {"seed": seed, "method": "Per-task union of (episode_path, query_index), sorted then Random(seed).shuffle; condition-filtered prefixes",
            "conditions": groups}
