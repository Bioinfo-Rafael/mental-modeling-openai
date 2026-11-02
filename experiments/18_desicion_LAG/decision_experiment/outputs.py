"""Validate responses without turning refusal, truncation, or malformed JSON into NO."""
import json
import math


AIRCRAFT_SETS = {"NONE": [], "A1": ["A0100"], "A2": ["A0200"], "BOTH": ["A0100", "A0200"]}


def aircraft_choice(aircraft):
    selected = sorted(set(aircraft))
    return next(key for key, members in AIRCRAFT_SETS.items() if members == selected)


def parse_output(raw, condition, window, *, decisions_task="binary"):
    if condition == "decisions":
        answers = raw.get("answers", [])
        aircraft_task = decisions_task == "aircraft_set"
        choices = set(AIRCRAFT_SETS) if aircraft_task else {"YES", "NO"}
        question_name = "changed_aircraft" if aircraft_task else "explanation_needed"
        if len(answers) != 1 or answers[0].get("name") != question_name:
            raise ValueError("Expected one named Decisions answer")
        answer = answers[0]
        if answer.get("type") != "choice" or answer.get("choice") not in choices:
            raise ValueError("Decisions refusal or invalid choice")
        probabilities = answer.get("probabilities", [])
        if len(probabilities) != len(choices) or {p.get("value") for p in probabilities} != choices:
            raise ValueError("Invalid choice probabilities")
        values = [p.get("probability") for p in probabilities] + [answer.get("confidence")]
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError("Invalid probability/confidence")
        if not math.isclose(sum(values[:-1]), 1, abs_tol=.001):
            raise ValueError("Choice probabilities do not sum to one")
        if aircraft_task:
            selected = answer["choice"]
            probs = {p["value"]: p["probability"] for p in probabilities}
            return {"decision": "NO" if selected == "NONE" else "YES", "events": None, "reason": None,
                    "aircraft_choice": selected, "selected_aircraft": AIRCRAFT_SETS[selected].copy(),
                    "choice_probabilities": probs, "probability_yes": sum(probs[k] for k in ("A1", "A2", "BOTH")),
                    "aircraft_probabilities": {"A0100": probs["A1"] + probs["BOTH"], "A0200": probs["A2"] + probs["BOTH"]},
                    "confidence": answer["confidence"]}
        return {"decision": answer["choice"], "events": None, "reason": None,
                "probability_yes": next(p["probability"] for p in probabilities if p["value"] == "YES"),
                "confidence": answer["confidence"]}
    if raw.get("status") != "completed":
        raise ValueError(f'Responses status is {raw.get("status")!r}, not completed')
    texts = []
    for item in raw.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "refusal":
                    raise ValueError("Responses refusal")
                if part.get("type") == "output_text":
                    texts.append(part["text"])
    result = json.loads("".join(texts))
    diagnostic = condition == "responses_diagnostic"
    expected_keys = {"decision", "events", "reason"} if diagnostic else {"decision"}
    if not isinstance(result, dict) or set(result) != expected_keys or result["decision"] not in ("YES", "NO"):
        raise ValueError("Output does not match the decision schema")
    if not diagnostic:
        return {**result, "events": None, "reason": None}
    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("Missing decision reason")
    events = result["events"]
    if not isinstance(events, list) or (result["decision"] == "YES") != bool(events):
        raise ValueError("YES requires events; NO requires an empty events list")
    seen = set()
    for event in events:
        if not isinstance(event, dict) or set(event) != {"step", "aircraft_id", "evidence_steps", "reason"}:
            raise ValueError("Invalid event fields")
        if event["aircraft_id"] not in ("A0100", "A0200"):
            raise ValueError("Invalid target aircraft")
        if not isinstance(event["evidence_steps"], list) or not event["evidence_steps"]:
            raise ValueError("Event requires recorded evidence steps")
        for step in [event["step"], *event["evidence_steps"]]:
            if type(step) is not int or not window["start_step"] <= step <= window["end_step"]:
                raise ValueError("Event/evidence step is outside the supplied window")
        if not isinstance(event["reason"], str) or not event["reason"].strip():
            raise ValueError("Missing event reason")
        key = (event["aircraft_id"], event["step"])
        if key in seen:
            raise ValueError("Duplicate event in a response")
        seen.add(key)
    selected = sorted({e["aircraft_id"] for e in events})
    return {**result, "selected_aircraft": selected, "aircraft_choice": aircraft_choice(selected)}


def estimate_cost(usage, condition, pricing):
    """Null means unknown, never free. Cached/write tokens are subsets of input."""
    if not isinstance(usage, dict):
        return None
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    details = usage.get("input_tokens_details") or {}
    cached, writes = details.get("cached_tokens"), details.get("cache_write_tokens")
    # The SDK may materialize absent optional fields as null.
    cached = 0 if cached is None else cached
    writes = 0 if writes is None else writes
    counts = [inputs, outputs, cached, writes]
    if any(type(n) is not int or n < 0 for n in counts) or cached + writes > inputs:
        return None
    input_factor = output_factor = 1
    if inputs > pricing["long_context_threshold"]:
        input_factor = pricing["long_context_input_multiplier"]
        output_factor = pricing["long_context_output_multiplier"]
    rate = pricing["decisions_output" if condition == "decisions" else "responses_output"]
    return ((inputs - cached - writes) * pricing["input"] * input_factor
            + cached * pricing["cached_input"] * input_factor
            + writes * pricing["cache_write"] * input_factor
            + outputs * rate * output_factor) / pricing["per_tokens"]
