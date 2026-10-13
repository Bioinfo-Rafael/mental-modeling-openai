"""Offline parsing and scoring; no API client or data loading is required."""
from collections import defaultdict
import csv
import hashlib
import json
import math
import re
from statistics import mean
from pathlib import Path

from .prompts import ACTION_MARKER, STATE_MARKER
from . import execution

SCORING_VERSION = "lag-action3-state10-v1"


def parse_prediction(text, task):
    """Require one final marker and a JSON array of the exact requested shape."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty_response"
    marker = ACTION_MARKER if task.endswith("action") else STATE_MARKER
    matches = re.findall(re.escape(marker) + r"\s*(\[[^\]\n]*\])", text)
    if len(matches) != 1 or text.count(marker) != 1:
        return None, "expected_one_final_marker"
    try:
        values = json.loads(matches[0])
    except (ValueError, TypeError):
        return None, "invalid_json_array"
    if task.endswith("action"):
        if len(values) != 3 or any(type(v) is not int or not 0 <= v <= maximum
                                   for v, maximum in zip(values, (2, 4, 2))):
            return None, "expected_three_action_ids_in_range"
    elif len(values) != 10 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        return None, "expected_ten_finite_state_values"
    return values, None


def score(task, fields, ground_truth, text):
    prediction, error = parse_prediction(text, task)
    result = dict(scoring_version=SCORING_VERSION, prediction=prediction,
                  parse_success=error is None, parse_error=error)
    if error:
        return result
    if task.endswith("action"):
        matches = [a == b for a, b in zip(prediction, ground_truth)]
        result.update(exact_match=all(matches), component_matches=dict(zip(fields, matches)))
    else:
        differences = [actual - expected for actual, expected in zip(prediction, ground_truth)]
        result["absolute_errors"] = dict(zip(fields, map(abs, differences)))
        # Keep raw coordinate errors as primary values; circular errors are explicit diagnostics.
        result["circular_angle_errors_deg"] = {
            field: abs((delta + 180) % 360 - 180)
            for field, delta in zip(fields, differences) if field in ("roll_deg", "yaw_deg")
        }
    return result


def response_text(raw):
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices") or []
    return choices[0].get("message", {}).get("content") if choices else None


def summarize(records, planned_queries):
    """One row per condition; report denominators instead of silently dropping failures."""
    groups = defaultdict(list)
    for record in records:
        groups[record["condition_id"]].append(record)
    planned = defaultdict(int)
    for query in planned_queries:
        planned[query["condition_id"]] += 1
    summary = []
    for condition, count in sorted(planned.items()):
        rows = groups[condition]
        parsed = [r for r in rows if r["parse_success"]]
        item = dict(condition_id=condition, planned=count, responses=len(rows), parsed=len(parsed),
                    parse_failures=sum(not r["parse_success"] and not r.get("api_error") for r in rows),
                    api_errors=sum(bool(r.get("api_error")) for r in rows),
                    unscored=count-len(rows), complete=len(parsed) == count)
        if parsed and "exact_match" in parsed[0]:
            matches = sum(r["exact_match"] for r in parsed)
            item.update(exact_match_accuracy_parsed=matches/len(parsed), exact_matches=matches,
                        exact_match_rate_all_planned=matches/count,
                        component_accuracy={k: mean(r["component_matches"][k] for r in parsed)
                                            for k in parsed[0]["component_matches"]})
        elif parsed:
            item["mae_by_field"] = {k: mean(r["absolute_errors"][k] for r in parsed)
                                    for k in parsed[0]["absolute_errors"]}
            item["circular_angle_mae_deg"] = {k: mean(r["circular_angle_errors_deg"][k] for r in parsed)
                                              for k in parsed[0]["circular_angle_errors_deg"]}
        summary.append(item)
    return summary


def evaluate_saved(plan, directory):
    """Recompute metrics from saved raw responses; never contact a model.

    The versioned analysis is derived output and may be overwritten on rescore.
    manifest.json and api/*.jsonl remain untouched.
    """
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
        record = {k: sample[k] for k in ("episode_id", "target_id", "task", "H", "start",
                                         "query_step", "target_step", "sample_number", "problem_id", "ground_truth")}
        record["observation_scope"] = sample.get("observation_scope", "self")
        record.update(query)
        record.update(api_error=response.get("error_type"), elapsed_seconds=response.get("elapsed_seconds"),
                      response_text=text, usage=(response["raw_response"] or {}).get("usage"))
        record.update(score(sample["task"], sample["prediction_fields"], sample["ground_truth"], text))
        records.append(record)
    summary = summarize(records, plan["queries"])
    by_condition = {q["condition_id"]: q for q in plan["queries"]}
    for row in summary:
        query = by_condition[row["condition_id"]]
        sample = plan["samples"][query["sample_id"]]
        row.update({k: sample[k] for k in ("episode_id", "target_id", "task", "H")})
        row["observation_scope"] = sample.get("observation_scope", "self")
        row.update(model_alias=query["model_alias"], model=query["model"])
    output = directory / "analysis" / SCORING_VERSION
    output.mkdir(parents=True, exist_ok=True)
    execution.write_json(output / "per_query.json", records)
    execution.write_json(output / "conditions.json", summary)
    flat = []
    for row in summary:
        flattened = {}
        for key, value in row.items():
            if isinstance(value, dict):
                flattened.update({f"{key}.{field}": item for field, item in value.items()})
            else:
                flattened[key] = value
        flat.append(flattened)
    columns = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    execution.write_json(output / "metadata.json", {
        "plan_id": plan["plan_id"], "scoring_version": SCORING_VERSION,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "response_journal_sha256": hashlib.sha256(
            (api / "responses.jsonl").read_bytes() if (api / "responses.jsonl").exists() else b"").hexdigest(),
        "scored_responses": len(records), "conditions": len(summary),
    })
    return summary
