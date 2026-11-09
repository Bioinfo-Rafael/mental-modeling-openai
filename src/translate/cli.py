"""``python -m translate --config C [--scenario S] [--run-name N] [--execute --confirm-paid-api]``.

Preview by default: no backend, no detector client, no key, no network; ``<run>-preview/`` gets
``run.json`` and ``preview.json`` (the schedule, the caps, the exact first request, the detector
question and state template). A paid run needs both flags; its run directory (default
``<config stem>-<sha256(config)[:8]>``) is created once, before any key is read, so a resubmitted job
stops there. Exit codes: 0 preview or completed; 1 failed; 2 usage or gate; 3 uncertain (a call may
have been billed; never resend it blindly); 4 a cap stopped the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from translate.backends import BACKENDS, BackendError
from translate.config import ConfigError, Loaded, load_config
from translate.detector import DetectorError, SystemOneClient, check_budget, request_body, state_text
from translate.episode import Episode, Translation, View, load_episode
from translate.journal import SAFE_NAME, Journal, git_state, new_run_dir, repo_path, utc_now, write_json
from translate.pipeline import Pipeline, base_values, prompt_sha256, translation_prompt
from translate.triggers import Schedule, schedule

EXIT_CODES = {"completed": 0, "failed": 1, "uncertain": 3, "cap_reached": 4}
CHARS_PER_TOKEN = 4  # approximate tokens = ceil(chars / 4), labelled approximate


def _name(value: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a name (letters, digits, . _ -), never a path")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m translate", description="Translate an episode with one run config."
    )
    parser.add_argument("--config", type=Path, required=True, help="run config (configs/translate/*.yaml)")
    parser.add_argument(
        "--scenario", type=Path, help="scenario YAML; overrides the config's (real ones live in private/)"
    )
    parser.add_argument("--run-name", type=_name, help="run directory name (default: <config stem>-<sha256[:8]>)")
    parser.add_argument("--execute", action="store_true", help="make the paid calls (needs --confirm-paid-api)")
    parser.add_argument("--confirm-paid-api", action="store_true", help="confirm paid calls (needs --execute)")
    return parser


def check_scenario(loaded: Loaded, episode: Episode) -> None:
    """Steps within ``[0, last_complete_step]``, known targets, preset steps that can fire."""
    for target_id in loaded.scenario.targets:
        episode.target(target_id)
    late = sorted({m.step for m in loaded.scenario.messages} | set(getattr(loaded.config.change, "steps", ())))
    if late and late[-1] > episode.last_step:
        raise ConfigError(
            f"steps {[s for s in late if s > episode.last_step]} are after the last complete step {episode.last_step}"
        )


def first_body(loaded: Loaded, episode: Episode, step: int, target_id: str) -> dict[str, Any]:
    """The exact request of the first translation (no previous translation exists yet)."""
    messages = tuple(m for m in loaded.scenario.messages if m.step <= step)
    view = View(step, episode.rows_until(step), messages, episode.teams)
    prompt = translation_prompt(loaded.config, view, episode.target(target_id))
    return BACKENDS[loaded.config.model.backend].request_body(loaded.config.model, prompt)


def detector_preflight(loaded: Loaded, episode: Episode, plan: Schedule) -> int:
    """The worst state bound over every scheduled tick, with a worst-case previous intent; none may reach the budget."""
    change, options = loaded.config.change, loaded.detector_inputs()
    if getattr(options.get("previous_translation"), "evidence", False):
        raise ConfigError("change.inputs.previous_translation needs evidence: false (evidence has no length bound)")
    worst = 0
    for step, target_id in plan.ticks:
        target = episode.target(target_id)
        empty = {target_id: Translation("", 0, "", ())}
        messages = tuple(m for m in loaded.scenario.messages if m.step <= step)
        view = View(step, episode.rows_until(step), messages, episode.teams, empty)
        base = base_values(view, target, loaded.config.answer.max_intent_chars)
        state = state_text(change, options, view, target, base)
        worst = max(worst, check_budget(change, request_body(change, state), loaded.config.answer.max_intent_chars))
    return worst


def run_record(loaded: Loaded, episode: Episode, mode: str, status: str) -> dict[str, Any]:
    config, change = loaded.config, loaded.config.change
    resolved = {"path": str(loaded.path), "sha256": loaded.sha256, "resolved": config.model_dump(mode="json")}
    scenario = {
        "path": str(loaded.scenario_path),
        "sha256": hashlib.sha256(loaded.scenario_path.read_bytes()).hexdigest(),
    }
    episode_facts = {"csv_path": episode.csv_path, "csv_sha256": episode.csv_sha256, "episode_id": episode.episode_id}
    detector = {"model": change.model, "revision": change.revision} if change.source == "clef_flash" else None
    return {
        "mode": mode,
        "status": status,
        "started_at": utc_now(),
        "git": git_state(),
        "config": {**resolved, "scenario": loaded.scenario.model_dump(mode="json")},
        "prompt": {"name": loaded.prompt_name, "sha256": prompt_sha256(config)},
        "inputs": {"scenario": scenario, "episode": {**episode_facts, "last_complete_step": episode.last_step}},
        "detector": detector,
    }


def preview(loaded: Loaded, episode: Episode, plan: Schedule, run_dir: Path, worst_bound: int | None) -> int:
    limits = loaded.config.limits
    body = first_body(loaded, episode, *plan.mandatory[0]) if plan.mandatory else None
    warnings = (
        [f"{len(plan.ticks)} detector calls exceed max_detector_calls"]
        if len(plan.ticks) > limits.max_detector_calls
        else []
    )
    if plan.translation_bound > limits.max_translations:
        warnings.append(f"up to {plan.translation_bound} translations exceed max_translations")
    change = loaded.config.change
    detector = (
        None
        if change.source != "clef_flash"
        else {"question": change.question.model_dump(), "state": change.state, "worst_bound": worst_bound}
    )
    summary = {
        "forced": [list(item) for item in plan.forced],
        "mandatory_translations": len(plan.mandatory),
        "detector_calls": len(plan.ticks),
        "translation_bound": plan.translation_bound,
        "suppressed": [list(item) for item in plan.suppressed],
        "caps": limits.model_dump(),
        "warnings": warnings,
        "approx_input_tokens_first_request": None if body is None else math.ceil(len(str(body)) / CHARS_PER_TOKEN),
    }
    write_json(run_dir / "preview.json", {**summary, "first_request": body, "detector": detector})
    write_json(run_dir / "run.json", {**run_record(loaded, episode, "preview", "preview"), "ended_at": utc_now()})
    shown = {key: summary[key] for key in ("mandatory_translations", "detector_calls", "warnings")}
    print(json.dumps({"run_dir": str(run_dir), "status": "preview", **shown}, indent=2, ensure_ascii=False))
    print("Preview only. API requests made: 0", file=sys.stderr)
    return 0


def execute(
    loaded: Loaded, episode: Episode, run_dir: Path, backend_factory: Callable, detector_factory: Callable
) -> int:
    """Run directory first (a resubmitted job stops at mkdir), then the backend (it reads the key), then the loop."""
    record = run_record(loaded, episode, "execute", "running")
    write_json(run_dir / "run.json", record)
    try:
        detector = detector_factory(loaded.config.change) if loaded.config.change.source == "clef_flash" else None
        if detector is not None:
            detector.health()  # before any paid client exists
        summary = Pipeline(loaded, episode, Journal(run_dir), backend_factory(loaded.config.model), detector).run()
    except (Exception, KeyboardInterrupt) as exc:  # nothing was sent: the pipeline handles its own calls
        message = (
            str(exc) if isinstance(exc, (ValueError, OSError, BackendError, DetectorError)) else "details withheld"
        )
        write_json(
            run_dir / "run.json",
            {
                **record,
                "status": "failed",
                "ended_at": utc_now(),
                "error": {"type": type(exc).__name__, "message": message},
            },
        )
        print(f"error: {type(exc).__name__}: {message} ({run_dir})", file=sys.stderr)
        return 1
    counts = {layer: asdict(value) for layer, value in summary.counts.items()}
    final = {**record, "status": summary.status, "ended_at": utc_now(), "counts": {**counts, "events": summary.events}}
    write_json(run_dir / "run.json", {**final, "error": summary.error, "unsent": summary.unsent})
    print(json.dumps({"run_dir": str(run_dir), "status": summary.status, "counts": final["counts"]}, indent=2))
    return EXIT_CODES[summary.status]


def main(
    argv: list[str] | None = None, *, backend_factory: Callable | None = None, detector_factory: Callable | None = None
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("a paid run needs BOTH --execute and --confirm-paid-api")
    try:
        loaded = load_config(args.config, args.scenario)
        episode = load_episode(loaded.scenario.episode)
        check_scenario(loaded, episode)
        plan = schedule(loaded.scenario, episode, loaded.config.change)
        worst = detector_preflight(loaded, episode, plan) if loaded.config.change.source == "clef_flash" else None
        name = args.run_name or f"{args.config.stem}-{loaded.sha256[:8]}"
        root = repo_path(loaded.config.output)
        if not args.execute:
            return preview(loaded, episode, plan, new_run_dir(root, f"{name}-preview", reuse=True), worst)
        if name.endswith("-preview"):
            raise ConfigError("a paid run name must not end in -preview (that suffix is the preview's)")
        run_dir = new_run_dir(root, name, reuse=False)
        backends = backend_factory or (lambda model: BACKENDS[model.backend](model))
        return execute(loaded, episode, run_dir, backends, detector_factory or SystemOneClient)
    except (ValueError, OSError, BackendError, DetectorError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: interrupted before any call", file=sys.stderr)
        return 1
