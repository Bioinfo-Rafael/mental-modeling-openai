"""Offline validation and scoring of joint ten-step state/action forecasts."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from .prompts import TRAJECTORY_MARKER
from . import execution

SCORING_VERSION = "lag-trajectory10-action3-state10-v1"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def parse_prediction(text, target_steps):
    """Require the exact ordered future steps, not just ten arbitrary records."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    if text.count(TRAJECTORY_MARKER) != 1:
        return None, "expected_one_final_marker"
    final = text.split(TRAJECTORY_MARKER, 1)[1].lstrip(" \t")
    line = final.splitlines()[0] if final else ""
    try:
        values = json.loads(line, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError):
        return None, "invalid_json_array"
    if not isinstance(values, list) or len(values) != 10 or len(target_steps) != 10:
        return None, "expected_ten_trajectory_steps"
    for row, step in zip(values, target_steps):
        if not isinstance(row, dict) or set(row) != {"step", "state", "action"}:
            return None, "expected_step_state_action_object"
        if type(row["step"]) is not int or row["step"] != step:
            return None, "unexpected_step_order"
        action = row["action"]
        if not isinstance(action, list) or len(action) != 3 or any(
            type(v) is not int or not 0 <= v <= maximum
            for v, maximum in zip(action, (2, 4, 2))
        ):
            return None, "expected_three_action_ids_in_range"
        state = row["state"]
        if not isinstance(state, list) or len(state) != 10 or not all(map(_finite, state)):
            return None, "expected_ten_finite_state_values"
    return values, None


def score(fields, ground_truth, text):
    prediction, error = parse_prediction(text, [row["step"] for row in ground_truth])
    result = dict(scoring_version=SCORING_VERSION, prediction=prediction,
                  parse_success=error is None, parse_error=error)
    if error:
        return result
    per_step = []
    for offset, (actual, expected) in enumerate(zip(prediction, ground_truth), 1):
        matches = dict(zip(fields["action"], (a == b for a, b in zip(actual["action"], expected["action"]))))
        differences = dict(zip(fields["state"], (a - b for a, b in zip(actual["state"], expected["state"]))))
        if not all(map(_finite, differences.values())):
            result.update(parse_success=False, parse_error="nonfinite_state_error")
            return result
        per_step.append(dict(
            offset=offset, step=expected["step"], action_exact_match=all(matches.values()),
            action_component_matches=matches,
            state_absolute_errors={key: abs(value) for key, value in differences.items()},
            circular_angle_errors_deg={key: abs((value + 180) % 360 - 180)
                                       for key, value in differences.items() if key in ("roll_deg", "yaw_deg")},
        ))
    result.update(
        per_step=per_step,
        action_trajectory_exact_match=all(row["action_exact_match"] for row in per_step),
        action_step_accuracy=mean(row["action_exact_match"] for row in per_step),
        action_component_accuracy={key: mean(row["action_component_matches"][key] for row in per_step)
                                   for key in fields["action"]},
        state_mae_by_field={key: mean(row["state_absolute_errors"][key] for row in per_step)
                            for key in fields["state"]},
        circular_angle_mae_deg={key: mean(row["circular_angle_errors_deg"][key] for row in per_step)
                                for key in per_step[0]["circular_angle_errors_deg"]},
    )
    return result


def response_text(raw):
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices") or []
    return choices[0].get("message", {}).get("content") if choices else None


def summarize(records, planned_queries):
    """Per condition: equal weight per window, with failure denominators explicit."""
    groups, planned = defaultdict(list), defaultdict(int)
    for record in records:
        groups[record["condition_id"]].append(record)
    for query in planned_queries:
        planned[query["condition_id"]] += 1
    summary = []
    for condition, count in sorted(planned.items()):
        rows = groups[condition]
        parsed = [row for row in rows if row["parse_success"]]
        item = dict(condition_id=condition, planned=count, responses=len(rows), parsed=len(parsed),
                    parse_failures=sum(not r["parse_success"] and not r.get("api_error") for r in rows),
                    api_errors=sum(bool(r.get("api_error")) for r in rows),
                    unscored=count-len(rows), complete=len(parsed) == count,
                    planned_steps=count * 10, scored_steps=len(parsed) * 10)
        if parsed:
            exact = sum(row["action_trajectory_exact_match"] for row in parsed)
            item.update(action_trajectory_exact_matches=exact,
                        action_trajectory_accuracy_parsed=exact / len(parsed),
                        action_trajectory_match_rate_all_planned=exact / count,
                        action_step_accuracy=mean(row["action_step_accuracy"] for row in parsed))
            for metric in ("action_component_accuracy", "state_mae_by_field", "circular_angle_mae_deg"):
                item[metric] = {key: mean(row[metric][key] for row in parsed) for key in parsed[0][metric]}
            item["by_horizon"] = []
            for index in range(10):
                steps = [row["per_step"][index] for row in parsed]
                item["by_horizon"].append(dict(
                    offset=index + 1, parsed=len(steps), planned=count,
                    action_step_accuracy=mean(step["action_exact_match"] for step in steps),
                    action_component_accuracy={key: mean(step["action_component_matches"][key] for step in steps)
                                               for key in steps[0]["action_component_matches"]},
                    state_mae_by_field={key: mean(step["state_absolute_errors"][key] for step in steps)
                                        for key in steps[0]["state_absolute_errors"]},
                    circular_angle_mae_deg={key: mean(step["circular_angle_errors_deg"][key] for step in steps)
                                            for key in steps[0]["circular_angle_errors_deg"]},
                ))
        summary.append(item)
    return summary


def _write_csv(path, rows):
    flat = []
    for row in rows:
        flattened = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flattened.update({f"{key}.{field}": item for field, item in value.items()})
            elif not isinstance(value, list):
                flattened[key] = value
        flat.append(flattened)
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)


def evaluate_saved(plan, directory):
    """Recompute versioned metrics without changing the plan or raw responses."""
    directory = Path(directory)
    execution.verify_plan(plan)
    api = directory / "api"
    if not api.exists():
        raise ValueError("No API execution to score")
    if execution.read_json(api / "identity.json")["plan_id"] != plan["plan_id"]:
        raise ValueError("Response identity differs from the plan")
    queries = {q["query_id"]: q for q in plan["queries"]}
    records, seen = [], set()
    for response in execution.read_journal(api / "responses.jsonl"):
        qid = response["query_id"]
        if qid not in queries or qid in seen:
            raise ValueError("Unknown or duplicate response query_id")
        seen.add(qid)
        query = queries[qid]
        sample = plan["samples"][query["sample_id"]]
        text = response_text(response["raw_response"])
        record = {key: sample[key] for key in (
            "episode_id", "target_id", "task", "H", "start", "query_step", "target_step",
            "target_steps", "horizon", "sample_number", "problem_id", "ground_truth", "observation_scope",
        )}
        record.update(query)
        record.update(api_error=response.get("error_type"), elapsed_seconds=response.get("elapsed_seconds"),
                      response_text=text, usage=(response["raw_response"] or {}).get("usage"))
        record.update(score(sample["prediction_fields"], sample["ground_truth"], text))
        records.append(record)
    summary = summarize(records, plan["queries"])
    by_condition = {q["condition_id"]: q for q in plan["queries"]}
    horizon_rows = []
    for row in summary:
        query = by_condition[row["condition_id"]]
        sample = plan["samples"][query["sample_id"]]
        metadata = {key: sample[key] for key in ("episode_id", "target_id", "task", "H", "horizon", "observation_scope")}
        metadata.update(model_alias=query["model_alias"], model=query["model"])
        row.update(metadata)
        horizon_rows.extend(dict(condition_id=row["condition_id"], **metadata, **step)
                            for step in row.get("by_horizon", []))
    output = directory / "analysis" / SCORING_VERSION
    output.mkdir(parents=True, exist_ok=True)
    execution.write_json(output / "per_query.json", records)
    execution.write_json(output / "conditions.json", summary)
    _write_csv(output / "conditions.csv", summary)
    _write_csv(output / "horizons.csv", horizon_rows)
    execution.write_json(output / "metadata.json", {
        "plan_id": plan["plan_id"], "scoring_version": SCORING_VERSION,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "response_journal_sha256": hashlib.sha256(
            (api / "responses.jsonl").read_bytes() if (api / "responses.jsonl").exists() else b"").hexdigest(),
        "scored_responses": len(records), "conditions": len(summary),
    })
    return summary
