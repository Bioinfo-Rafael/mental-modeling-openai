"""Causality audit of a translation run: re-read every journaled request body and check what it shows.

A body passes when every trajectory line in it has ``step <= t`` and ``relative_step <= 0``, no text
of a message issued after t appears in it (unless the same text was issued by t), and neither the
run's episode id nor its scenario id appears. A target with any leaking call is excluded from judging
and reported.
"""

from __future__ import annotations

import json
from typing import Any

from evaluation.material import TranslationRun, target_key

TRAJECTORY_PREFIX = '{"step":'  # every trajectory renderer writes compact JSON lines that start so


def _strings(value: Any) -> list[str]:
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in (str(key), *_strings(item))]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


def messages(run: TranslationRun) -> set[tuple[int, str]]:
    """(issue step, text) of every message the run knows: its scenario and every record's inputs."""
    scenario = run.run_json.get("config", {}).get("scenario", {})
    found = {(m["step"], m["text"]) for m in scenario.get("messages", []) if isinstance(m, dict)}
    for record in run.records:
        listed = record.get("input", {})
        found |= {(i["issue_step"], i["text"]) for i in listed.get("instructions_in_effect", [])}
        found |= {(a["step"], a["text"]) for a in (*listed.get("alarms", []), *listed.get("messages", []))}
        question = listed.get("question")
        found |= {(question["step"], question["text"])} if question else set()
    return {(step, text) for step, text in found if text.strip()}


def _problems(line: dict[str, Any], known: set[tuple[int, str]], run: TranslationRun) -> list[str]:
    step, texts = line["step"], _strings(line.get("body", line.get("payload")))
    joined, problems = "\n".join(texts), []
    for row in (r.strip() for text in texts for r in text.split("\n")):
        if not row.startswith(TRAJECTORY_PREFIX):
            continue
        try:
            parsed = json.loads(row)
        except ValueError:
            parsed = {"step": step + 1}  # an unreadable trajectory line cannot be shown to be causal
        if parsed["step"] > step or parsed.get("relative_step", 0) > 0:
            problems.append(f"trajectory step {parsed['step']} is after t = {step}")
    allowed = {text for issued, text in known if issued <= step}
    later = sorted((issued, text) for issued, text in known if issued > step and text not in allowed)
    problems += [
        f"the text of a message issued at step {issued} > t appears" for issued, text in later if text in joined
    ]
    for name, identifier in (("episode_id", run.episode_id), ("scenario_id", run.scenario_id)):
        if identifier and identifier in joined:
            problems.append(f"the run's own {name} appears")
    return problems


def audit_run(run: TranslationRun) -> list[dict[str, Any]]:
    """One record per journaled request of either layer: ``ok`` or ``leak`` with its problems."""
    known = messages(run)
    records = []
    for line in run.requests:
        problems = _problems(line, known, run)
        target = {"target": line["target"], "step": line["step"], "layer": line.get("layer")}
        records.append(
            {"call_id": line["call_id"], **target, "status": "leak" if problems else "ok", "problems": problems}
        )
    return records


def leaking_targets(records: list[dict[str, Any]]) -> set[str]:
    return {target_key(record["target"]) for record in records if record["status"] != "ok"}
