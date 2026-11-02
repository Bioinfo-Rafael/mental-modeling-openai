"""Window classification and one-to-one event localization are separate metrics."""
from functools import lru_cache
import statistics

from .outputs import AIRCRAFT_SETS


def ratio(n, d):
    return n / d if d else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def match_events(predicted, expected, tolerance):
    """Maximum matching, then smallest total step error. Each event counts once."""
    @lru_cache(None)
    def search(i, used):
        if i == len(predicted):
            return ()
        best = search(i + 1, used)
        for j, truth in enumerate(expected):
            error = abs(predicted[i]["step"] - truth["step"])
            if used & (1 << j) or predicted[i]["aircraft_id"] != truth["aircraft_id"] or error > tolerance:
                continue
            candidate = ((i, j, error),) + search(i + 1, used | (1 << j))
            if (len(candidate), -sum(x[2] for x in candidate)) > (len(best), -sum(x[2] for x in best)):
                best = candidate
        return best
    return [{"prediction_index": i, "event_id": expected[j]["id"], "absolute_error_steps": error}
            for i, j, error in search(0, 0)]


def summarize(rows):
    valid = [r for r in rows if r["status"] == "ok"]
    attempted = [r for r in rows if r["status"] != "pending"]
    counts = {name: sum(r["classification"] == name for r in valid) for name in ("TP", "TN", "FP", "FN")}
    tp, tn, fp, fn = (counts[x] for x in ("TP", "TN", "FP", "FN"))
    success_latencies = [r["latency_s"] for r in valid]
    attempted_latencies = [r["latency_s"] for r in attempted if r["latency_s"] is not None]
    known_costs = [r["estimated_usd"] for r in attempted if r["estimated_usd"] is not None]
    diagnostic = [r for r in valid if r["predicted_events"] is not None]
    matched = sum(len(r["matches"]) for r in diagnostic)
    expected = sum(len(r["expected_events"]) for r in diagnostic)
    predicted = sum(len(r["predicted_events"]) for r in diagnostic)
    exact = sum(len(r["exact_matches"]) for r in diagnostic)
    return {"planned": len(rows), "attempted": len(attempted), "valid": len(valid),
            "pending": len(rows) - len(attempted), "errors_or_uncertain": len(attempted) - len(valid),
            "valid_fraction_of_attempts": ratio(len(valid), len(attempted)), **counts,
            "accuracy_valid": ratio(tp + tn, len(valid)), "precision_valid": ratio(tp, tp + fp),
            "recall_valid": ratio(tp, tp + fn), "false_negative_rate_valid": ratio(fn, tp + fn),
            "false_positive_rate_valid": ratio(fp, fp + tn), "f2_valid": ratio(5 * tp, 5 * tp + 4 * fn + fp),
            "accuracy_attempted_errors_as_misses": ratio(tp + tn, len(attempted)),
            "positive_recall_attempted_errors_as_misses": ratio(tp, sum(r["expected_decision"] == "YES" for r in attempted)),
            "latency_valid_p50_s": quantile(success_latencies, .5),
            "latency_valid_p95_s": quantile(success_latencies, .95),
            "latency_attempted_p95_s": quantile(attempted_latencies, .95),
            "known_cost_usd": sum(known_costs) if known_costs else None,
            "cost_known_attempts": len(known_costs), "cost_unknown_attempts": len(attempted) - len(known_costs),
            "cost_complete": bool(attempted) and len(known_costs) == len(attempted),
            "cost_per_valid_decision_known_usd": ratio(sum(known_costs), len(valid)) if len(known_costs) == len(attempted) else None,
            "aircraft_selection": aircraft_metrics(rows),
            "event_localization": None if not diagnostic else {
                "expected_event_window_occurrences": expected, "predicted_event_window_occurrences": predicted,
                "matched": matched, "exact_matched": exact, "precision": ratio(matched, predicted),
                "recall": ratio(matched, expected), "exact_recall": ratio(exact, expected),
                "mean_absolute_error_matched_steps": statistics.mean(
                    [m["absolute_error_steps"] for r in diagnostic for m in r["matches"]]) if matched else None}}


def aircraft_metrics(rows):
    supported = [r for r in rows if r.get("aircraft_selection_supported")]
    if not supported:
        return None
    valid = [r for r in supported if r["status"] == "ok" and r.get("selected_aircraft") is not None]
    attempted = [r for r in supported if r["status"] != "pending"]
    matrix = {truth: {prediction: 0 for prediction in AIRCRAFT_SETS} for truth in AIRCRAFT_SETS}
    for row in valid:
        matrix[row["expected_aircraft_choice"]][row["aircraft_choice"]] += 1
    correct = sum(r["aircraft_set_correct"] for r in valid)
    per_aircraft = {}
    for uid in ("A0100", "A0200"):
        tp = sum(uid in r["expected_aircraft"] and uid in r["selected_aircraft"] for r in valid)
        tn = sum(uid not in r["expected_aircraft"] and uid not in r["selected_aircraft"] for r in valid)
        fp = sum(uid not in r["expected_aircraft"] and uid in r["selected_aircraft"] for r in valid)
        fn = sum(uid in r["expected_aircraft"] and uid not in r["selected_aircraft"] for r in valid)
        per_aircraft[uid] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
                             "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
                             "false_negative_rate": ratio(fn, tp + fn),
                             "recall_attempted_errors_as_misses": ratio(tp, sum(uid in r["expected_aircraft"] for r in attempted))}
    return {"valid": len(valid), "attempted": len(attempted), "exact_set_correct": correct,
            "exact_set_accuracy_valid": ratio(correct, len(valid)),
            "exact_set_accuracy_attempted_errors_as_misses": ratio(correct, len(attempted)),
            "confusion_matrix_expected_rows_predicted_columns": matrix, "per_aircraft": per_aircraft}


def stream_metrics(rows, events):
    """Per-repeat event opportunities; binary coverage is explicitly only a proxy."""
    results = []
    for repeat in sorted({r["repeat"] for r in rows}):
        stream = sorted([r for r in rows if r["repeat"] == repeat], key=lambda r: r["end_step"])
        complete = bool(stream) and all(r["status"] == "ok" for r in stream)
        localization_available = any(r["condition"] == "responses_diagnostic" for r in stream)
        yes = [r for r in stream if r["status"] == "ok" and r["decision"] == "YES"]
        event_rows = []
        for event in events:
            opportunities = [r for r in stream if event["id"] in {e["id"] for e in r["expected_events"]}]
            covered = [r for r in yes if r in opportunities]
            aircraft_covered = [r for r in covered if r.get("selected_aircraft") is not None and event["aircraft_id"] in r["selected_aircraft"]]
            located = [r for r in yes if any(m["event_id"] == event["id"] for m in r["matches"])]
            first_end = min((r["end_step"] for r in opportunities), default=None)
            event_rows.append({"event_id": event["id"], "step": event["step"],
                               "first_opportunity_end": first_end,
                               "opportunities": len(opportunities),
                               "binary_coverage_proxy": bool(covered),
                               "binary_first_delay_steps": covered[0]["end_step"] - event["step"] if covered else None,
                               "aircraft_coverage_proxy": bool(aircraft_covered) if any(r.get("aircraft_selection_supported") for r in stream) else None,
                               "aircraft_first_delay_steps_proxy": aircraft_covered[0]["end_step"] - event["step"] if aircraft_covered else None,
                               "localized": bool(located) if localization_available else None,
                               "localized_first_delay_steps": located[0]["end_step"] - event["step"] if located else None,
                               "localized_excess_over_grid_steps": located[0]["end_step"] - first_end if located else None,
                               "localized_duplicate_windows": max(0, len(located) - 1) if localization_available else None})
        cooldowns = []
        for cooldown in (0, 10, 40):
            kept = []
            for row in yes:
                if not kept or row["end_step"] - kept[-1]["end_step"] >= cooldown:
                    kept.append(row)
            covered_ids = {e["id"] for row in kept for e in row["expected_events"]}
            cooldowns.append({"cooldown_steps": cooldown, "trigger_count": len(kept),
                              "covered_event_ids_proxy": sorted(covered_ids),
                              "warning": "Overlap proxy only; no proof that the downstream explanation covers each event."})
        results.append({"repeat": repeat, "complete_valid_stream": complete,
                        "yes_count": len(yes), "events": event_rows, "cooldown_simulations": cooldowns,
                        "localization_available": localization_available})
    return results
