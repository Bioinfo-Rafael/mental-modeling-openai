#!/usr/bin/env python3
"""Method A of the intent-translation comparison: the trajectory-based Translator at fixed steps.

Preview by default (no client, no credential read, no network): writes every prompt and request
body plus a token / cost estimate under ``<output_root>/<run_id>/method_a/preview/``.
``--execute --confirm-paid-api`` sends one OpenAI Responses call per (step, aircraft) in
chronological order; each target chains its own previous translation. Every request is journaled
before it is sent and nothing is retried. An answer that violates the length limits (or the schema) gets a
``skipped`` cue like in method B, the chain keeps the last valid answer, and the run goes on; the first
API failure stops the run. The key comes only
from the ``OPENAI_API_KEY`` environment variable (read by the Translator's ``make_openai_client``).
Exit codes: 0 ok; 1 failure; 2 usage / gate misuse; 3 a call's outcome is uncertain (may be billed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
from intent_compare_a_requests import (
    Slot,
    TranslationSettings,
    answer_problem,
    build_request,
    cost_summary,
    estimate_call,
    json_line,
    load_frame,
    load_pricing,
    load_settings,
    plan_slots,
    prompt_for,
    provenance,
    request_body,
    sha256_text,
)
from intent_compare_config import DEFAULT_CONFIG, CompareConfig, load_config

from mtd_rl_xavier.translator.schema import IntentTranslation, ModelCallError, PreviousTranslation
from mtd_rl_xavier.translator.translation import (
    IntentModel,
    IntentModelOutput,
    TranslationLayer,
    TranslationOutputError,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_UNCERTAIN = 0, 1, 2, 3
METHOD = "A"
METHOD_DIR = "method_a"
# Reasoning effort "high" needs room: the Translator default (4000, tuned for "medium") risks an
# incomplete response, which fails the call after it is billed. 25000 is the repo's gpt-6.1-sol
# judge setting (evaluation/config/judge-text.yaml). Only actual usage is billed.
DEFAULT_MAX_OUTPUT_TOKENS = 25_000
RUN_DIR_MODE = 0o700
INVALID_ANSWER_PREFIX = "invalid final answer: "  # the wording of method B's skipped cues


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    method_dir: Path
    cues: Path

    @property
    def source_run(self) -> str:
        return f"{self.run_dir.name}/{self.method_dir.name}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def slot_name(slot: Slot) -> str:
    return f"step{slot.step:03d}_{slot.aircraft}"


def cue_record(config: CompareConfig, slot: Slot, source_run: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "step": slot.step,
        "t_s": config.t_s(slot.step),
        "aircraft": slot.aircraft,
        "method": METHOD,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        **output,
        "source_run": source_run,
    }


def preview(config: CompareConfig, frame: pd.DataFrame, slots: list[Slot], paths: RunPaths, settings: TranslationSettings) -> int:  # fmt: skip
    out = paths.method_dir / "preview"
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    estimates, calls = [], []
    for slot in slots:
        if slot.skipped:
            calls.append({"slot": slot_name(slot), "skipped": slot.skipped})
            continue
        prompt = prompt_for(build_request(config, frame, slot, previous=None), settings)
        body = request_body(prompt, config, settings)
        estimate = estimate_call(prompt, settings, chained=slot.aircraft in seen)
        seen.add(slot.aircraft)
        write_json(out / f"request_{slot_name(slot)}.json", body)
        estimates.append(estimate)
        calls.append({"slot": slot_name(slot), "request_sha256": sha256_text(json.dumps(body, sort_keys=True)),
                      "user_text_sha256": sha256_text(prompt.user_text), **estimate})  # fmt: skip
    summary = {
        "mode": "preview",
        "api_requests_made": 0,
        "created_at": utc_now(),
        "note": "previous translation is unknown in preview: prompts are rendered with previous=None and chained "
        "calls carry a fixed allowance instead",
        "calls": calls,
        "estimate": cost_summary(estimates, load_pricing(config.model), settings.max_output_tokens),
        "provenance": provenance(config, frame, settings),
    }
    write_json(out / "preview.json", summary)
    print(json.dumps({"preview_dir": str(out), **summary["estimate"]}, indent=2))
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def _append(handle: TextIO, record: dict[str, Any]) -> None:
    handle.write(json_line(record) + "\n")
    handle.flush()


def make_model(config: CompareConfig, settings: TranslationSettings) -> IntentModel:
    """The real OpenAI Responses model; reads ``OPENAI_API_KEY`` (only called for a paid run)."""
    from mtd_rl_xavier.llm.openai_responses import OpenAIResponsesClient, make_openai_client
    from mtd_rl_xavier.translator.translation import OpenAIIntentModel

    return OpenAIIntentModel(
        OpenAIResponsesClient(make_openai_client(timeout_s=settings.timeout_s)),
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
    )


def invalid_answer_reason(layer: TranslationLayer, output: IntentModelOutput, settings: TranslationSettings) -> tuple[IntentTranslation | None, str | None]:  # fmt: skip
    """(translation, None) for a valid answer, else (None, reason) in method B's wording.

    The Translator's own parser (schema, strict JSON, intent length) and method B's rules (intent and
    evidence length / count, characters not bytes) both reject; either way the run continues.
    """
    try:
        translation = layer.parse(output)
    except TranslationOutputError as exc:
        return None, f"{INVALID_ANSWER_PREFIX}{exc}"
    problem = answer_problem(translation.intent, translation.evidence, settings)
    if problem is not None:
        return None, f"{INVALID_ANSWER_PREFIX}{problem}"
    return translation, None


def execute(config: CompareConfig, frame: pd.DataFrame, slots: list[Slot], paths: RunPaths, settings: TranslationSettings, model: IntentModel) -> int:  # fmt: skip
    if paths.cues.exists():
        print(f"refusing to overwrite {paths.cues}; choose a new --run-id", file=sys.stderr)
        return EXIT_USAGE
    paths.method_dir.mkdir(parents=True, exist_ok=True, mode=RUN_DIR_MODE)
    record = {"mode": "execute", "status": "running", "started_at": utc_now(),
              "provenance": provenance(config, frame, settings)}  # fmt: skip
    write_json(paths.method_dir / "run.json", record)
    layer = TranslationLayer(
        model, output_language=settings.output_language, max_intent_chars=settings.max_intent_chars
    )
    status, error, sent = "completed", None, 0
    rejected: list[dict[str, str]] = []
    previous: dict[str, PreviousTranslation] = {}
    files = ("requests.jsonl", "responses.jsonl")
    with (paths.method_dir / files[0]).open("x") as requests, (paths.method_dir / files[1]).open("x") as responses:
        with paths.cues.open("x", encoding="utf-8") as cues:
            for slot in slots:
                if slot.skipped:
                    _append(cues, cue_record(config, slot, paths.source_run, {"skipped": slot.skipped}))
                    continue
                request = build_request(config, frame, slot, previous.get(slot.aircraft))
                prompt = prompt_for(request, settings)
                body = layer.request_payload(prompt)
                call_id = slot_name(slot)
                _append(requests, {"call_id": call_id, "step": slot.step, "aircraft": slot.aircraft,
                                   "trajectory_sha256": request.trajectory.sha256, "payload": body})  # fmt: skip
                sent += 1
                try:
                    output = layer.generate(prompt)
                except ModelCallError as exc:
                    status = "uncertain" if exc.uncertain else "failed"
                    error = {"call_id": call_id, "type": exc.error_type, "message": str(exc)}
                    _append(responses, {"call_id": call_id, "status": status, "raw": exc.raw, "error": error})
                    break
                except Exception as exc:  # unknown outcome: the request may have reached the server
                    status = "uncertain"
                    error = {"call_id": call_id, "type": type(exc).__name__, "message": "details withheld"}
                    _append(responses, {"call_id": call_id, "status": status, "raw": None, "error": error})
                    break
                _append(responses, {"call_id": call_id, "status": "ok", "model": output.model,
                                    "response_id": output.response_id, "usage": output.usage, "raw": output.raw})  # fmt: skip
                translation, reason = invalid_answer_reason(layer, output, settings)
                if translation is None:  # like method B: skipped cue, the chain keeps the last valid answer
                    rejected.append({"call_id": call_id, "reason": str(reason)})
                    _append(cues, cue_record(config, slot, paths.source_run, {"skipped": reason}))
                    continue
                previous = {**previous, slot.aircraft: PreviousTranslation(call_id, slot.step, translation)}
                _append(cues, cue_record(config, slot, paths.source_run, translation.to_json()))
    final = {**record, "status": status, "error": error, "api_requests_made": sent, "ended_at": utc_now(),
             "rejected_answers": rejected}  # fmt: skip
    write_json(paths.method_dir / "run.json", final)
    print(json.dumps({"run_dir": str(paths.run_dir), "status": status, "api_requests_made": sent, "error": error,
                      "rejected_answers": rejected}))  # fmt: skip
    if status == "uncertain":
        return EXIT_UNCERTAIN
    return EXIT_OK if status == "completed" else EXIT_FAILED


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", help="output sub-directory (default: seed<seed>)")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--execute", action="store_true", help="make real (paid) API calls")
    parser.add_argument("--confirm-paid-api", action="store_true", help="required together with --execute")
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("paid calls need BOTH --execute and --confirm-paid-api")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    run_id = args.run_id or f"seed{config.seed}"
    run_dir = config.run_dir(run_id)
    paths = RunPaths(run_dir, run_dir / METHOD_DIR, config.cues_path(run_id, METHOD))
    settings = load_settings(config, args.max_output_tokens)
    frame, facts = load_frame(config)
    slots = plan_slots(config, facts)
    if not args.execute:
        return preview(config, frame, slots, paths, settings)
    return execute(config, frame, slots, paths, settings, make_model(config, settings))


if __name__ == "__main__":
    sys.exit(main())
