"""Versioned shared task, two Responses output contracts, and Decisions choices."""
from difflib import unified_diff
import json
from pathlib import Path
import shutil

from .storage import HERE, canonical, digest, safe_name, write_json

FILES = ("system.txt", "user.txt", "question.txt", "choices.json", "diagnostic.txt", "choice.txt")
CONDITIONS = ("responses_diagnostic", "responses_choice", "decisions")


def load_prompt(directory):
    directory = Path(directory)
    prompt = {name: (directory / name).read_text(encoding="utf-8") for name in FILES}
    choices = json.loads(prompt["choices.json"])
    if len(choices) != 2 or {c["value"] for c in choices} != {"YES", "NO"}:
        raise ValueError("This experiment requires exactly YES and NO choices")
    if any(set(c) != {"value", "description"} or not c["description"] for c in choices):
        raise ValueError("Each choice needs value and description")
    return prompt


def schema(diagnostic):
    def obj(properties):
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}
    fields = {"decision": {"type": "string", "enum": ["YES", "NO"]}}
    if diagnostic:
        fields.update(events={"type": "array", "items": obj({
            "aircraft_id": {"type": "string", "enum": ["A0100", "A0200"]},
            "step": {"type": "integer"},
            "evidence_steps": {"type": "array", "items": {"type": "integer"}},
            "reason": {"type": "string"}})}, reason={"type": "string"})
    return obj(fields)


def render_input(prompt, window):
    fields = {**window, "window_data": canonical(window["blocks"])}
    return prompt["user.txt"].format(**fields)


def build_request(condition, prompt, user_text, config):
    choices = json.loads(prompt["choices.json"])
    messages = [{"role": "developer", "content": prompt["system.txt"]},
                {"role": "user", "content": user_text}]
    body = {"model": "gpt-6-luna", "input": messages}
    if condition == "decisions":
        body["questions"] = [{"type": "choice", "name": "explanation_needed",
                              "instructions": prompt["question.txt"].strip(), "choices": choices}]
    elif condition in CONDITIONS:
        diagnostic = condition == "responses_diagnostic"
        question = prompt["question.txt"].strip() + "\n" + "\n".join(
            f'{choice["value"]}: {choice["description"]}' for choice in choices)
        contract = prompt["diagnostic.txt" if diagnostic else "choice.txt"]
        messages[1]["content"] += "\n\n" + question + "\n\n" + contract
        body.update(store=False, reasoning={"effort": config["reasoning_effort"]},
                    max_output_tokens=config["max_output_tokens"],
                    text={"format": {"type": "json_schema", "name": "explanation_detection",
                                     "strict": True, "schema": schema(diagnostic)}})
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return body


def clone_prompt(source, version, note):
    source, version = safe_name(source), safe_name(version)
    original, target = HERE / "prompts" / source, HERE / "prompts" / version
    load_prompt(original)
    if not note.strip():
        raise ValueError("Record the hypothesis/change in --note")
    shutil.copytree(original, target)  # refuses to overwrite an existing version
    write_json(target / "metadata.json", {"version": version, "parent": source,
               "parent_sha256": digest(load_prompt(original)), "description": note})
    return target


def snapshot_prompt(source, target):
    prompt = load_prompt(source)
    shutil.copytree(source, target)
    # Recompute diffs at plan time, including edits made after clone-prompt.
    reference = HERE / "prompts/reference/tactical_short_v1"
    for filename in ("system.txt", "user.txt"):
        diff = unified_diff((reference / filename).read_text().splitlines(True),
                            prompt[filename].splitlines(True),
                            fromfile=f"17/tactical_short_v1/{filename}", tofile=f"18/{filename}")
        (target / filename.replace(".txt", ".diff")).write_text("".join(diff), encoding="utf-8")
    return prompt
