"""``python -m mtd_rl_xavier.translator.cli``: run the two-layer Translator on one scenario.

Preview by default: no client is built and no credential is read. ``--execute`` runs the layers;
a config with a paid (OpenAI) layer also needs ``--confirm-paid-api``. This module is the only
composition root: it alone constructs ``SystemOneClient`` and the OpenAI SDK client.
Exit codes: 0 preview written or run completed; 1 deterministic failure; 2 usage or gate misuse;
3 an external call has an uncertain outcome (may have been billed; never resent), also after
Ctrl-C during a call and even when the final ``run.json`` cannot be written;
4 a call cap stopped the run (``cap_reached``; the event log is partial).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import httpx
import pandas as pd

from mtd_rl_xavier import paths
from mtd_rl_xavier.lag.dataset import MANIFEST_NAME, load_episode
from mtd_rl_xavier.lag.rendering import (
    DETECTOR_RENDERER,
    LAG17_RENDERER,
    RENDERER_VERSION,
    render_detector_compact,
    render_lag17_jsonl,
)
from mtd_rl_xavier.llm.openai_responses import (
    API_BASE_URL,
    OpenAIClientError,
    OpenAIResponsesClient,
    build_responses_body,
    make_openai_client,
    set_provenance_env_names,
)
from mtd_rl_xavier.llm.system_one import SystemOneClient, SystemOneError, question_set_sha256
from mtd_rl_xavier.tools.output_safety import derived_path

from .config import DetectorConfig, TranslatorConfig, load_config, load_scenario
from .detector import DETECTOR_STATE_VERSION, DetectorLayer, SystemOneDetectionModel, noul_question
from .pipeline import (
    INTERRUPTIONS,
    STATUS_CAP_REACHED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    BudgetFit,
    EpisodeFacts,
    PipelineError,
    SchedulePlan,
    Translator,
    TranslatorSummary,
    check_detector_budget,
    episode_facts,
    instructions_in_effect,
    plan_schedule,
    validate_scenario,
)
from .schema import SAFE_ID_PATTERN, Scenario, canonical_json, redacted_error, sha256_text
from .translation import (
    PROMPT_VERSION,
    OpenAIIntentModel,
    TranslationLayer,
    TranslationModelRequest,
    TranslationPrompt,
    build_openai_request,
    build_prompt,
    output_schema,
    system_prompt,
)

DEFAULT_CONFIG = paths.CONFIGS / "translator" / "run.yaml"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNCERTAIN = 3
EXIT_CAP_REACHED = 4
EXIT_BY_STATUS = {
    STATUS_COMPLETED: EXIT_OK,
    STATUS_FAILED: EXIT_FAILED,
    STATUS_UNCERTAIN: EXIT_UNCERTAIN,
    STATUS_CAP_REACHED: EXIT_CAP_REACHED,
}
RUN_SCHEMA_VERSION = 1
MODE_PREVIEW = "preview"
MODE_EXECUTE = "execute"
STATUS_PREVIEW = "preview"
STATUS_RUNNING = "running"
EVENTS_FILE = "events.jsonl"
REQUESTS_FILE = "requests.jsonl"
RESPONSES_FILE = "responses.jsonl"
CHARS_PER_TOKEN = 4  # approximate text tokens = ceil(chars / 4); labelled approximate everywhere
TOKENS_PER_MILLION = 1_000_000
REPRESENTATIVE_WINDOWS = 3  # full-length translation windows rendered for the per-call size estimate
GIT_TIMEOUT_S = 10
RUN_DIR_MODE = 0o700  # run outputs hold model-visible scenario text: owner-only
PROVENANCE_PACKAGES = ("translator", "lag", "llm")
GIT_TRACKED_PATHS = (
    "src/mtd_rl_xavier/translator",
    "src/mtd_rl_xavier/lag",
    "src/mtd_rl_xavier/llm",
    "configs/translator",
)

DetectorClientFactory = Callable[[DetectorConfig], SystemOneClient]
OpenAIClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class Inputs:
    config_path: Path
    scenario_path: Path
    config: TranslatorConfig
    scenario: Scenario
    frame: pd.DataFrame
    facts: EpisodeFacts
    plan: SchedulePlan
    budget: BudgetFit


def run_name(value: str) -> str:
    """argparse type: a run directory name, never a path. A bad one is a usage error (exit 2)."""
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {SAFE_ID_PATTERN.pattern} (a name, not a path)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mtd_rl_xavier.translator.cli",
        description="Two-layer LLM Translator. Writes a preview unless --execute is given.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="run config YAML")
    parser.add_argument("--scenario", type=Path, required=True, help="scenario YAML (real ones: private/translator/)")
    parser.add_argument("--output-root", type=Path, help="override output.root of the config")
    parser.add_argument(
        "--run-name", type=run_name, help="run directory name (default: <UTC stamp>_<scenario_id>_<mode>)"
    )
    parser.add_argument("--execute", action="store_true", help="call the layers (local and/or paid)")
    parser.add_argument("--confirm-paid-api", action="store_true", help="required with --execute for paid layers")
    return parser


def build_system_one_client(config: DetectorConfig, *, transport: httpx.BaseTransport | None = None) -> SystemOneClient:
    """The only place a detector client is built. Reads the key variable named in the config."""
    return SystemOneClient(
        base_url=config.connection.base_url,
        uds_path=config.connection.uds_path,
        model=config.served_model_name,
        questions=(noul_question(config.question),),
        input_token_budget=config.input_token_budget,
        timeout_s=config.timeout_s,
        api_key_env=config.connection.api_key_env,
        transport=transport,
    )


# --- inputs, schedule and preflight ---------------------------------------------------------------


def load_inputs(config_path: Path, scenario_path: Path, config: TranslatorConfig) -> Inputs:
    """Scenario + hash-verified episode + schedule + detector budget fit. No client, no credential."""
    scenario = load_scenario(scenario_path)
    frame = load_episode(scenario.episode.csv_path)
    facts = episode_facts(frame)
    validate_scenario(scenario, facts)
    plan = plan_schedule(scenario, facts, config.pipeline.tick_interval_steps)
    ticks = tuple(
        (target_plan.target, step) for target_plan in plan.targets for step in target_plan.detector_tick_steps
    )
    budget = check_detector_budget(
        frame,
        ticks=ticks,
        facts=facts,
        config=config,
        questions=(noul_question(config.detector.question),),
        renderer=render_detector_compact,
    )
    if not budget.fits:
        raise PipelineError(
            f"detector state bound {budget.worst_bound} >= input_token_budget {budget.input_token_budget} "
            f"({budget.worst_target.label if budget.worst_target else '-'} at step {budget.worst_step}); "
            "shorten detector.window or its fields, or lower translation.max_intent_chars"
        )
    return Inputs(config_path, scenario_path, config, scenario, frame, facts, plan, budget)


def _mandatory_requests(inputs: Inputs) -> list[TranslationModelRequest]:
    """The scenario-triggered translations, rendered exactly (previous translation unknown in preview)."""
    window = inputs.config.translation.window
    requests = []
    for target_plan in inputs.plan.targets:
        for step in target_plan.mandatory_translation_steps:
            in_effect = instructions_in_effect(inputs.scenario, step)
            primary = next(item for item in in_effect if item.target == target_plan.target)
            trajectory = render_lag17_jsonl(
                inputs.frame,
                start_step=max(0, step - window.past_steps + 1),
                end_step=step,
                anchor_step=step,
                episode_alias=inputs.scenario.episode.alias,
            )
            request = TranslationModelRequest(
                target=target_plan.target,
                target_aircraft=inputs.facts.target_aircraft(target_plan.target),
                step=step,
                episode_alias=inputs.scenario.episode.alias,
                instructions_in_effect=in_effect,
                primary_instruction_id=primary.instruction_id,
                trajectory=trajectory,
                previous=None,
            )
            requests.append(request)
    return sorted(requests, key=lambda request: (request.step, request.target))


def _representative_chars(inputs: Inputs) -> int:
    """Largest trajectory text among a few full-length windows (approximate per-call size)."""
    last, past = inputs.facts.last_step, inputs.config.translation.window.past_steps
    first_full = min(last, past - 1)
    ends = sorted(
        {first_full + (last - first_full) * i // (REPRESENTATIVE_WINDOWS - 1) for i in range(REPRESENTATIVE_WINDOWS)}
    )
    alias = inputs.scenario.episode.alias
    texts = [
        render_lag17_jsonl(
            inputs.frame, start_step=max(0, end - past + 1), end_step=end, anchor_step=end, episode_alias=alias
        ).text
        for end in ends
    ]
    return max(len(text) for text in texts)


def _approx_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def _translation_estimate(inputs: Inputs, prompts: list[TranslationPrompt]) -> dict[str, Any]:
    config = inputs.config
    instructions_chars = len(
        system_prompt(
            output_language=config.translation.output_language, max_intent_chars=config.translation.max_intent_chars
        )
    )
    mandatory_tokens = sum(_approx_tokens(instructions_chars + len(prompt.user_text)) for prompt in prompts)
    extra_calls = inputs.plan.translation_upper_bound - inputs.plan.mandatory_translations
    per_extra = _approx_tokens(instructions_chars + _representative_chars(inputs))
    upper_tokens = mandatory_tokens + extra_calls * per_extra
    price = config.price(config.translation.model)
    max_out = config.translation.max_output_tokens

    def cost(input_tokens: int, calls: int) -> float | None:
        if price is None:
            return None
        return round((input_tokens * price.input + calls * max_out * price.output) / TOKENS_PER_MILLION, 4)

    return {
        "approximate": f"text tokens ~= ceil(chars / {CHARS_PER_TOKEN}); excludes previous translations; "
        "output priced at max_output_tokens per call (upper bound)",
        "mandatory_input_tokens": mandatory_tokens,
        "per_call_input_tokens_full_window": per_extra,
        "upper_bound_input_tokens": upper_tokens,
        "cost_usd_mandatory": cost(mandatory_tokens, inputs.plan.mandatory_translations),
        "cost_usd_upper_bound": cost(upper_tokens, inputs.plan.translation_upper_bound),
    }


def _plan_json(inputs: Inputs) -> dict[str, Any]:
    plan, limits = inputs.plan, inputs.config.limits
    warnings = []
    if plan.detector_calls > limits.max_detector_calls:
        warnings.append(
            f"scheduled detector calls {plan.detector_calls} exceed max_detector_calls {limits.max_detector_calls}"
        )
    if plan.translation_upper_bound > limits.max_translation_calls:
        warnings.append(
            f"translation upper bound {plan.translation_upper_bound} exceeds max_translation_calls {limits.max_translation_calls}"
        )
    targets = [
        {
            "target": item.target.to_json(),
            "first_instruction_step": item.first_instruction_step,
            "down_step": item.down_step,
            "mandatory_translation_steps": list(item.mandatory_translation_steps),
            "detector_ticks": len(item.detector_tick_steps),
            "suppressed_trigger_steps": list(item.suppressed_trigger_steps),
        }
        for item in plan.targets
    ]
    return {
        "last_complete_step": inputs.facts.last_step,
        "targets": targets,
        "mandatory_translations": plan.mandatory_translations,
        "detector_calls": plan.detector_calls,
        "translation_upper_bound": plan.translation_upper_bound,
        "suppressed_triggers": plan.suppressed_triggers,
        "caps": asdict(limits),
        "headroom": {
            "detector": limits.max_detector_calls - plan.detector_calls,
            "translation": limits.max_translation_calls - plan.translation_upper_bound,
        },
        "warnings": warnings,
    }


def _budget_json(budget: BudgetFit) -> dict[str, Any]:
    return {
        "input_token_budget": budget.input_token_budget,
        "worst_bound": budget.worst_bound,
        "margin": budget.margin,
        "worst_target": budget.worst_target.to_json() if budget.worst_target else None,
        "worst_step": budget.worst_step,
        "checked_states": budget.checked_states,
        "intent_reserve_bytes": budget.intent_reserve_bytes,
    }


# --- provenance and artifacts ---------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )

    try:
        commit = git("rev-parse", "HEAD")
        status = git("--no-optional-locks", "status", "--porcelain", "--", *GIT_TRACKED_PATHS)
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}
    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _code_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    return {
        f"mtd_rl_xavier/{package}/{path.name}": _sha256_file(path)
        for package in PROVENANCE_PACKAGES
        for path in sorted((package_root / package).glob("*.py"))
    }


def _run_record(inputs: Inputs, *, mode: str, status: str) -> dict[str, Any]:
    config, scenario = inputs.config, inputs.scenario
    csv_path = scenario.episode.csv_path
    questions = (noul_question(config.detector.question),)
    language, max_chars = config.translation.output_language, config.translation.max_intent_chars
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "started_at": _utc_now(),
        "ended_at": None,
        "inputs": {
            "config": {"path": str(inputs.config_path), "sha256": _sha256_file(inputs.config_path)},
            "scenario": {
                "path": str(inputs.scenario_path),
                "sha256": _sha256_file(inputs.scenario_path),
                "scenario_id": scenario.scenario_id,
            },
            "episode": {
                "episode_id": scenario.episode.episode_id,
                "alias": scenario.episode.alias,
                "csv_path": str(csv_path),
                "csv_sha256": inputs.frame.attrs.get("csv_sha256"),
                "manifest_sha256": _sha256_file(csv_path.parent / MANIFEST_NAME),
                "video_path": str(scenario.episode.video_path) if scenario.episode.video_path else None,
                "last_complete_step": inputs.facts.last_step,
            },
        },
        "provenance": {
            "git": _git_state(),
            "code_sha256": _code_hashes(),
            "python": sys.version.split()[0],
            "renderers": {LAG17_RENDERER: RENDERER_VERSION, DETECTOR_RENDERER: RENDERER_VERSION},
            "detector_state_version": DETECTOR_STATE_VERSION,
            "question_set_sha256": question_set_sha256(questions),
            "prompt_version": PROMPT_VERSION,
            "openai_base_url": API_BASE_URL,  # the only endpoint make_openai_client targets
            "sdk_environment_set": set_provenance_env_names(),  # names only; proxy URLs can hold credentials
            "system_prompt_sha256": sha256_text(system_prompt(output_language=language, max_intent_chars=max_chars)),
            "output_schema_sha256": sha256_text(canonical_json(output_schema())),
        },
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "plan": _plan_json(inputs),
        "detector_budget": _budget_json(inputs.budget),
        "api_requests_made": 0,
    }


def _write_json(path: Path, payload: object) -> None:
    derived_path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _new_run_dir(args: argparse.Namespace, inputs: Inputs, mode: str) -> Path:
    """A new, immutable run directory; an existing one is refused (``FileExistsError``)."""
    name = (
        args.run_name or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{inputs.scenario.scenario_id}_{mode}"
    )
    if not SAFE_ID_PATTERN.fullmatch(name):
        raise PipelineError(f"--run-name must match {SAFE_ID_PATTERN.pattern}")
    root = args.output_root if args.output_root is not None else inputs.config.output_root
    run_dir = derived_path(Path(root) / name)
    run_dir.mkdir(parents=True, exist_ok=False, mode=RUN_DIR_MODE)
    return run_dir


def finite_json(value: Any) -> Any:
    """Standard JSON only: a non-finite float (a lax server's NaN) becomes the string "NaN" / "Infinity"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


class JsonlRunSink:
    """Append-only JSONL files of one run. Each line is flushed and fsynced before the call proceeds.

    A lone surrogate (e.g. ``"\\ud800"`` decoded from a response body) cannot be encoded as UTF-8; it
    is written as the JSON escape ``\\ud800`` instead (``backslashreplace``). The line stays valid JSON
    and reads back as the same text (a lone high + low pair reads back as the one character it
    encodes). So a fully received response is always journaled and never turns ``uncertain``.
    """

    def __init__(self, run_dir: Path) -> None:
        names = (EVENTS_FILE, REQUESTS_FILE, RESPONSES_FILE)
        self._files: dict[str, TextIO] = {
            name: (run_dir / name).open("x", encoding="utf-8", errors="backslashreplace") for name in names
        }

    def _write(self, name: str, record: dict[str, Any]) -> None:
        handle = self._files[name]
        handle.write(json.dumps(finite_json(record), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def write_event(self, record: dict[str, Any]) -> None:
        self._write(EVENTS_FILE, record)

    def write_request(self, record: dict[str, Any]) -> None:
        self._write(REQUESTS_FILE, {**record, "written_at": _utc_now()})

    def write_response(self, record: dict[str, Any]) -> None:
        self._write(RESPONSES_FILE, {**record, "written_at": _utc_now()})

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()

    def __enter__(self) -> JsonlRunSink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# --- modes ----------------------------------------------------------------------------------------


def _redact(exc: BaseException) -> dict[str, str]:
    """The LLM parts' errors are sanitised by design; everything else goes through ``redacted_error``."""
    if isinstance(exc, (SystemOneError, OpenAIClientError)):
        return {"type": type(exc).__name__, "message": str(exc)}
    return redacted_error(exc)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _preview(args: argparse.Namespace, inputs: Inputs) -> int:
    config = inputs.config.translation
    prompts = [
        build_prompt(request, output_language=config.output_language, max_intent_chars=config.max_intent_chars)
        for request in _mandatory_requests(inputs)
    ]
    run_dir = _new_run_dir(args, inputs, MODE_PREVIEW)
    record = {**_run_record(inputs, mode=MODE_PREVIEW, status=STATUS_PREVIEW), "ended_at": _utc_now()}
    preview = {
        "api_requests_made": 0,
        "plan": record["plan"],
        "detector_budget": record["detector_budget"],
        "translation_estimate": _translation_estimate(inputs, prompts),
    }
    _write_json(run_dir / "run.json", record)
    _write_json(run_dir / "preview.json", preview)
    if prompts:
        request = build_openai_request(
            prompts[0],
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            max_output_tokens=config.max_output_tokens,
        )
        _write_json(run_dir / "example_translation_request.json", build_responses_body(request))
    plan = record["plan"]
    _print(
        {
            "run_dir": str(run_dir),
            "status": STATUS_PREVIEW,
            "mandatory_translations": plan["mandatory_translations"],
            "detector_calls": plan["detector_calls"],
            "translation_upper_bound": plan["translation_upper_bound"],
            "detector_budget_margin": inputs.budget.margin,
            "warnings": plan["warnings"],
        }
    )
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def _build_translator(inputs: Inputs, detector_client: SystemOneClient, openai_client: Any) -> Translator:
    config = inputs.config
    detection = SystemOneDetectionModel(detector_client, question_id=config.detector.question.question_id)
    intent = OpenAIIntentModel(
        OpenAIResponsesClient(openai_client),
        model=config.translation.model,
        reasoning_effort=config.translation.reasoning_effort,
        max_output_tokens=config.translation.max_output_tokens,
    )
    return Translator(
        detector=DetectorLayer(detection, threshold=config.detector.threshold),
        translation=TranslationLayer(
            intent,
            output_language=config.translation.output_language,
            max_intent_chars=config.translation.max_intent_chars,
        ),
        detector_renderer=render_detector_compact,
        translation_renderer=render_lag17_jsonl,
    )


def _final_record(record: dict[str, Any], summary: TranslatorSummary) -> dict[str, Any]:
    counts = {"detector": asdict(summary.detector), "translation": asdict(summary.translation)}
    return {
        **record,
        "status": summary.status,
        "ended_at": _utc_now(),
        "counts": {**counts, "events_written": summary.events_written},
        "unsent": summary.unsent,
        "error": summary.error,
        "api_requests_made": summary.detector.attempted + summary.translation.attempted,
    }


def _finish(run_dir: Path, record: dict[str, Any], summary: TranslatorSummary) -> int:
    """Final ``run.json``. If it cannot be written, an uncertain or capped run keeps its exit code."""
    final = _final_record(record, summary)
    exit_code = EXIT_BY_STATUS[summary.status]
    try:
        _write_json(run_dir / "run.json", final)
    except INTERRUPTIONS as exc:
        print(f"error: final run.json not written: {type(exc).__name__} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED if exit_code == EXIT_OK else exit_code
    _print({"run_dir": str(run_dir), "status": summary.status, "counts": final["counts"], "unsent": summary.unsent})
    return exit_code


def _close_detector(client: SystemOneClient) -> None:
    """Close the local client. Runs after the outcome is decided, so a failure here never replaces it."""
    try:
        client.close()
    except INTERRUPTIONS as exc:
        print(f"error: closing the detector client failed: {type(exc).__name__}", file=sys.stderr)


def _execute(
    args: argparse.Namespace,
    inputs: Inputs,
    detector_client_factory: DetectorClientFactory,
    openai_client_factory: OpenAIClientFactory,
) -> int:
    """Order: local client -> /v1/models health -> run dir -> paid client -> pipeline (write-before-send)."""
    detector_client = detector_client_factory(inputs.config.detector)
    try:
        return _execute_with(args, inputs, detector_client, openai_client_factory)
    finally:
        _close_detector(detector_client)  # an uncertain or capped run keeps its exit code


def _execute_with(
    args: argparse.Namespace,
    inputs: Inputs,
    detector_client: SystemOneClient,
    openai_client_factory: OpenAIClientFactory,
) -> int:
    detector_client.health()  # before any paid client exists
    run_dir = _new_run_dir(args, inputs, MODE_EXECUTE)
    record: dict[str, Any] = {"schema_version": RUN_SCHEMA_VERSION, "mode": MODE_EXECUTE, "started_at": _utc_now()}
    summary: TranslatorSummary | None = None
    try:
        record = _run_record(inputs, mode=MODE_EXECUTE, status=STATUS_RUNNING)
        _write_json(run_dir / "run.json", record)
        openai_client = openai_client_factory(timeout_s=inputs.config.translation.timeout_s)
        translator = _build_translator(inputs, detector_client, openai_client)
        with JsonlRunSink(run_dir) as sink:
            # never raises: call failures, journal failures and Ctrl-C become a summary status
            summary = translator.run(frame=inputs.frame, scenario=inputs.scenario, config=inputs.config, sink=sink)
    except INTERRUPTIONS as exc:
        error = _redact(exc)
        if summary is not None:  # closing the journals failed after every line was fsynced
            print(f"error: closing the journals failed: {error['type']} ({run_dir})", file=sys.stderr)
            return _finish(run_dir, record, summary)
        # before the pipeline (or Ctrl-C there): no model call is in flight; recorded, then a failure exit
        failed = {**record, "status": STATUS_FAILED, "error": error, "ended_at": _utc_now()}
        _write_json(run_dir / "run.json", failed)
        print(f"error: {error['type']}: {error['message']} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED
    return _finish(run_dir, record, summary)


def main(
    argv: list[str] | None = None,
    *,
    detector_client_factory: DetectorClientFactory = build_system_one_client,
    openai_client_factory: OpenAIClientFactory = make_openai_client,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.confirm_paid_api and not args.execute:
        parser.error("--confirm-paid-api is only valid together with --execute")
    try:
        config = load_config(args.config)
        paid = config.paid_layers()
        if args.execute and paid and not args.confirm_paid_api:
            parser.error(f"layers {list(paid)} call a paid API: use BOTH --execute and --confirm-paid-api")
        inputs = load_inputs(args.config, args.scenario, config)
        if not args.execute:
            return _preview(args, inputs)
        return _execute(args, inputs, detector_client_factory, openai_client_factory)
    except INTERRUPTIONS as exc:  # reported without traceback (Ctrl-C included); unknown text is withheld
        error = _redact(exc)
        print(f"error: {error['type']}: {error['message']}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
