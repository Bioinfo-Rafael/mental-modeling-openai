"""Convert one method-A intent-compare sample into a Translator v2 run directory for evaluation.assess_cli.

Sources (read only): method_a/{run.json,requests.jsonl,responses.jsonl}, cues_A.jsonl, the episode CSV.
The request payload is copied verbatim, so check's request_sha256 / trajectory checks see the exact
input the model received. Run inside the mtd-rl-xavier-cleanup environment (uv run python ...).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mtd_rl_xavier.lag.dataset import last_complete_step, load_episode
from mtd_rl_xavier.translator.schema import canonical_json, sha256_text

EVENT_SCHEMA_VERSION = 2
BEGIN_MARK, END_MARK = "BEGIN WINDOW DATA\n", "\nEND WINDOW DATA"
LAG17_RENDERER, RENDERER_VERSION = "lag17-jsonl", "1"
STEP_SECONDS = 0.2
INITIAL_SOURCE = "first_instruction"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_one(records: list[dict[str, Any]], **match: Any) -> dict[str, Any]:
    found = [r for r in records if all(r.get(k) == v for k, v in match.items())]
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one record matching {match}, found {len(found)}"
        )
    return found[0]


def user_text(payload: dict[str, Any]) -> str:
    return payload["input"][0]["content"][0]["text"]


def window_text(text: str) -> str:
    start = text.index(BEGIN_MARK) + len(BEGIN_MARK)
    return text[start : text.index(END_MARK, start)]


def response_output(raw: dict[str, Any]) -> dict[str, Any]:
    messages = [item for item in raw["output"] if item.get("type") == "message"]
    return json.loads(messages[0]["content"][0]["text"])


def instruction(aircraft: str, text: str) -> dict[str, Any]:
    target = {"kind": "aircraft", "id": aircraft}
    return {
        "instruction_id": f"default-{aircraft}",
        "issue_step": 0,
        "target": target,
        "text": text,
    }


def build(args: argparse.Namespace) -> None:
    method_dir, out = Path(args.method_dir), Path(args.out)
    source_run = json.loads((method_dir / "run.json").read_text(encoding="utf-8"))
    call_id = f"step{args.step:03d}_{args.aircraft}"
    request = find_one(read_jsonl(method_dir / "requests.jsonl"), call_id=call_id)
    response = find_one(read_jsonl(method_dir / "responses.jsonl"), call_id=call_id)
    cue = find_one(read_jsonl(Path(args.cues)), step=args.step, aircraft=args.aircraft)
    output = response_output(response["raw"])
    if (output["intent"], output["evidence"]) != (cue["intent"], cue["evidence"]):
        raise SystemExit("cue intent/evidence differ from the recorded model response")

    payload = request["payload"]
    trajectory_text = window_text(user_text(payload))
    if sha256_text(trajectory_text) != request["trajectory_sha256"]:
        raise SystemExit(
            "extracted window text does not match the recorded trajectory_sha256"
        )
    lines = [json.loads(line) for line in trajectory_text.split("\n")]
    object_ids = sorted({e["object_id"] for line in lines for e in line["entities"]})

    csv_path = Path(source_run["provenance"]["episode_csv"]["path"])
    frame = load_episode(csv_path)
    episode_id = str(frame["episode_id"].iloc[0])
    for forbidden in (episode_id, args.scenario_id):
        if forbidden in canonical_json(payload):
            raise SystemExit(
                f"{forbidden!r} appears in the payload (would be a run_identifier leak)"
            )

    target = {"kind": "aircraft", "id": args.aircraft}
    event_id = f"{args.scenario_id}:aircraft:{args.aircraft}:{args.step:06d}"
    instructions = [instruction(a, args.instruction_text) for a in ("A0100", "A0200")]
    raw = response["raw"]
    record = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "sequence": 1,
        "record_type": "translation_event",
        "scenario_id": args.scenario_id,
        "episode_id": episode_id,
        "step": args.step,
        "time_s": round(args.step * STEP_SECONDS, 1),
        "target": target,
        "event_id": event_id,
        "episode": {
            "episode_id": episode_id,
            "alias": source_run["provenance"]["episode_alias"],
            "csv_sha256": frame.attrs["csv_sha256"],
        },
        "primary_trigger": "initial",
        "trigger_reasons": [
            {
                "kind": "initial",
                "source": INITIAL_SOURCE,
                "reference_id": f"default-{args.aircraft}",
                "probability": None,
            }
        ],
        "input": {
            "prompt_version": source_run["provenance"]["prompt_version"],
            "target_aircraft": [args.aircraft],
            "instructions_in_effect": instructions,
            "primary_instruction_id": f"default-{args.aircraft}",
            "alarms": [],
            "question": None,
            "trajectory": {
                "renderer": LAG17_RENDERER,
                "renderer_version": RENDERER_VERSION,
                "start_step": lines[0]["step"],
                "end_step": lines[-1]["step"],
                "anchor_step": args.step,
                "object_ids": object_ids,
                "fields": [],
                "text": trajectory_text,
                "sha256": sha256_text(trajectory_text),
            },
            "previous_translation": None,
            "output_language": "Japanese",
        },
        "output": {"intent": output["intent"], "evidence": output["evidence"]},
        "model": {
            "backend": "openai",
            "model": response["model"],
            "reasoning_effort": cue["reasoning_effort"],
            "max_output_tokens": payload["max_output_tokens"],
            "response_model": raw.get("model"),
            "response_id": response["response_id"],
        },
        "usage": response["usage"],
        "request_sha256": sha256_text(canonical_json(payload)),
        "processing_s": float(raw["completed_at"] - raw["created_at"]),
    }
    run_json = {
        "mode": "execute",
        "status": "completed",
        "inputs": {
            "scenario": {"scenario_id": args.scenario_id},
            "episode": {
                "episode_id": episode_id,
                "alias": source_run["provenance"]["episode_alias"],
                "csv_path": str(csv_path),
                "csv_sha256": frame.attrs["csv_sha256"],
                "manifest_sha256": sha256_file(csv_path.parent / "manifest.json"),
                "video_path": args.video,
                "last_complete_step": last_complete_step(frame),
            },
        },
        "config": {"pipeline": {"visibility": {"mode": "online", "future_steps": 0}}},
        "provenance": {
            "converted_from": {
                "method_dir": str(method_dir.resolve()),
                "call_id": call_id,
                "cues": str(Path(args.cues).resolve()),
                "run_json_sha256": sha256_file(method_dir / "run.json"),
            },
            "converter_sha256": sha256_file(Path(__file__)),
            "notes": "instruction ids, trigger 'initial', fields=[] and processing_s (API completed_at - created_at) "
            "are converter assumptions; payload/intent/evidence/trajectory are verbatim",
        },
        "api_requests_made": 1,
    }
    out.mkdir(parents=True, exist_ok=False)
    dump = lambda obj: json.dumps(obj, ensure_ascii=False)
    (out / "run.json").write_text(
        json.dumps(run_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "events.jsonl").write_text(dump(record) + "\n", encoding="utf-8")
    req_line = {
        "call_id": event_id,
        "layer": "translation",
        "step": args.step,
        "target": target,
        "payload": payload,
    }
    (out / "requests.jsonl").write_text(dump(req_line) + "\n", encoding="utf-8")
    resp_line = {
        "call_id": event_id,
        "latency_s": record["processing_s"],
        "status": response["status"],
        "response_id": response["response_id"],
        "usage": response["usage"],
    }
    (out / "responses.jsonl").write_text(dump(resp_line) + "\n", encoding="utf-8")
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--cues", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--aircraft", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--instruction-text", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
