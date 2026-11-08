"""``python -m translate --config <run config>``: run one translation configuration.

Preview by default: no client is built and no credential is read; the run directory gets
``run.json``, ``preview.json`` and ``example_translation_request.json`` (the exact first request).
``--execute --confirm-paid-api`` runs the pipeline (every configuration has a paid translation
call). This module is the only composition root: it alone builds the detector client, the OpenAI
SDK client and the sandbox executors, and a code-session run passes the sandbox preflight before
any key is read.
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
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from translate import paths
from translate.call_one_shot import OneShotCall, OpenAIIntentModel, RetryJournal, build_openai_request, build_prompt
from translate.call_session import SessionCall, SessionSettings, sandbox_executor
from translate.config import ConfigError, DetectorConfig, RunConfig, load_config
from translate.detector import DETECTOR_STATE_VERSION, DetectorLayer, SystemOneDetectionModel, choice_question
from translate.events import SAFE_ID_PATTERN, Scenario, canonical_json, redacted_error, sha256_text
from translate.inputs import RunInputs, SlotState, slot_values, trajectory_window
from translate.lag.dataset import MANIFEST_NAME, load_episode
from translate.lag.rendering import (
    DETECTOR_RENDERER,
    LAG17_RENDERER,
    RENDERER_VERSION,
    render_detector_compact,
    render_lag17_jsonl,
)
from translate.llm.openai_responses import (
    API_BASE_URL,
    OpenAIClientError,
    OpenAIResponsesClient,
    build_responses_body,
    make_openai_client,
    set_provenance_env_names,
)
from translate.llm.retry import RetryingClient
from translate.llm.system_one import SystemOneClient, SystemOneError, question_set_sha256
from translate.output_safety import derived_path
from translate.pipeline import (
    INTERRUPTIONS,
    SESSIONS_DIR,
    STATUS_CAP_REACHED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    Translator,
    TranslatorSummary,
)
from translate.policy.material import PolicyMaterial, load_policy_material, write_session_inputs
from translate.run_files import JsonlRunSink, utc_now, write_cues, write_json
from translate.sandbox.executor import SandboxLimits, discover_interpreter
from translate.sandbox.preflight import run_preflight
from translate.sandbox.session import SessionLimits
from translate.scenario import (
    EpisodeFacts,
    PipelineError,
    episode_facts,
    instructions_in_effect,
    load_scenario,
    validate_scenario,
)
from translate.session_backend import OpenAIResponsesBackend, user_item
from translate.triggers import (
    BudgetFit,
    SchedulePlan,
    check_detector_budget,
    check_preset_points,
    plan_schedule,
    run_triggers,
    tick_interval,
)

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
RUN_SCHEMA_VERSION = 2
MODE_PREVIEW = "preview"
MODE_EXECUTE = "execute"
STATUS_PREVIEW = "preview"
STATUS_RUNNING = "running"
CHARS_PER_TOKEN = 4  # approximate text tokens = ceil(chars / 4); labelled approximate everywhere
TOKENS_PER_MILLION = 1_000_000
GIT_TIMEOUT_S = 10
RUN_DIR_MODE = 0o700  # run outputs hold model-visible scenario text: owner-only
GIT_TRACKED_PATHS = ("src/translate", "configs/translate")
EXAMPLE_REQUEST_FILE = "example_translation_request.json"

DetectorClientFactory = Callable[[DetectorConfig], SystemOneClient]
OpenAIClientFactory = Callable[..., Any]
Preflight = Callable[..., Any]


@dataclass(frozen=True)
class Inputs:
    config: RunConfig
    scenario_path: Path
    scenario: Scenario
    frame: pd.DataFrame
    facts: EpisodeFacts
    plan: SchedulePlan
    budget: BudgetFit | None  # online runs only
    policy: PolicyMaterial | None  # the policy input only


def run_name(value: str) -> str:
    """argparse type: a run directory name, never a path. A bad one is a usage error (exit 2)."""
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {SAFE_ID_PATTERN.pattern} (a name, not a path)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m translate",
        description="Translate an episode with one run configuration. Writes a preview unless --execute is given.",
    )
    parser.add_argument("--config", type=Path, required=True, help="run config YAML (configs/translate/)")
    parser.add_argument("--scenario", type=Path, help="scenario YAML; overrides the config's scenario")
    parser.add_argument("--output-root", type=Path, help="override output.root of the config")
    parser.add_argument("--run-name", type=run_name, help="run directory name (default: <UTC>_<scenario_id>_<mode>)")
    parser.add_argument("--execute", action="store_true", help="call the models (paid)")
    parser.add_argument("--confirm-paid-api", action="store_true", help="required together with --execute")
    return parser


def build_system_one_client(config: DetectorConfig, *, transport: httpx.BaseTransport | None = None) -> SystemOneClient:
    """The only place a detector client is built. Reads the key variable named in the config."""
    return SystemOneClient(
        base_url=config.connection.base_url,
        uds_path=config.connection.uds_path,
        model=config.served_model_name,
        questions=(choice_question(config.question),),
        input_token_budget=config.input_token_budget,
        timeout_s=config.timeout_s,
        api_key_env=config.connection.api_key_env,
        transport=transport,
    )


# --- inputs, schedule and preflight ---------------------------------------------------------------


def load_inputs(config: RunConfig, scenario_path: Path | None) -> Inputs:
    """Scenario + hash-verified episode + schedule + detector budget + policy. No client, no credential."""
    path = scenario_path or config.scenario_path
    if path is None:
        raise ConfigError("no scenario: set scenario in the config or pass --scenario")
    scenario = load_scenario(path, limits=config.scenario_limits)
    frame = load_episode(scenario.episode.csv_path)
    facts = episode_facts(frame)
    validate_scenario(scenario, facts, config.scenario_limits)
    check_preset_points(config, facts)  # preview and execute fail alike, before any directory or client
    plan = plan_schedule(scenario, facts, run_triggers(scenario, config), tick_interval(config))
    budget = _budget(config, scenario, frame, facts, plan) if config.online else None
    policy = None
    if config.policy is not None:
        settings = config.policy
        policy = load_policy_material(settings.rollout_dir, settings.seed, settings.spec, settings.checkpoint)
    return Inputs(config, Path(path), scenario, frame, facts, plan, budget, policy)


def _budget(
    config: RunConfig, scenario: Scenario, frame: pd.DataFrame, facts: EpisodeFacts, plan: SchedulePlan
) -> BudgetFit:
    ticks = tuple(
        (target_plan.target, step) for target_plan in plan.targets for step in target_plan.detector_tick_steps
    )
    budget = check_detector_budget(
        frame,
        ticks=ticks,
        scenario=scenario,
        facts=facts,
        config=config,
        questions=(choice_question(config.detector.question),),
        renderer=render_detector_compact,
    )
    if not budget.fits:
        raise PipelineError(
            f"detector state bound {budget.worst_bound} >= input_token_budget {budget.input_token_budget} "
            f"({budget.worst_target.label if budget.worst_target else '-'} at step {budget.worst_step}); "
            "shorten detector.window or its fields, the scenario's instruction, alarm or question texts, "
            "or lower detector.messages_lookback_steps or answer.max_intent_chars"
        )
    return budget


def _run_inputs(inputs: Inputs) -> RunInputs:
    return RunInputs(inputs.frame, inputs.scenario, inputs.facts, inputs.config, render_lag17_jsonl, inputs.policy)


def mandatory_bodies(inputs: Inputs) -> list[dict[str, Any]]:
    """The exact first request of every trigger known in advance (previous translation unknown: none)."""
    config, run_inputs = inputs.config, _run_inputs(inputs)
    bodies = []
    for target_plan in inputs.plan.targets:
        for step in target_plan.mandatory_translation_steps:
            in_effect = instructions_in_effect(inputs.scenario, step)
            instruction = next(i for i in in_effect if i.target == target_plan.target)
            slot = SlotState(step, target_plan.target, instruction, None)
            values = slot_values(run_inputs, slot, trajectory_window(run_inputs, step))
            bodies.append((step, target_plan.target, _request_body(config, values)))
    return [body for _, _, body in sorted(bodies, key=lambda item: (item[0], item[1]))]


def _request_body(config: RunConfig, values: dict[str, object]) -> dict[str, Any]:
    translation = config.translation
    if config.code_session:
        _, message = config.prompt.render(values)
        options = _session_options(config)
        return {"input": [user_item(message)], **options}
    prompt = build_prompt(config.prompt, values)
    request = build_openai_request(
        prompt,
        model=translation.model,
        reasoning_effort=translation.reasoning_effort,
        max_output_tokens=translation.max_output_tokens,
    )
    return build_responses_body(request)


def _session_options(config: RunConfig) -> dict[str, Any]:
    translation = config.translation
    return OpenAIResponsesBackend(
        None, translation.model, translation.reasoning_effort, translation.max_output_tokens
    ).options


def _approx_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def _estimate(inputs: Inputs, bodies: list[dict[str, Any]]) -> dict[str, Any]:
    config = inputs.config
    price = config.price(config.translation.model)
    calls = len(bodies)
    first_tokens = sum(_approx_tokens(len(canonical_json(body))) for body in bodies)
    if config.code_session and config.session is not None:
        per_call = config.session.max_total_tokens
        bound = (
            None if price is None else round(calls * per_call * max(price.input, price.output) / TOKENS_PER_MILLION, 4)
        )
        return {
            "approximate": "a code session is capped at session.max_total_tokens (input + output); "
            "the bound prices every capped token at the higher of the input and output rates",
            "known_calls": calls,
            "first_message_tokens": first_tokens,
            "cost_usd_upper_bound_known_calls": bound,
        }
    max_out = config.translation.max_output_tokens
    cost = (
        None
        if price is None
        else round((first_tokens * price.input + calls * max_out * price.output) / TOKENS_PER_MILLION, 4)
    )
    return {
        "approximate": f"text tokens ~= ceil(chars / {CHARS_PER_TOKEN}) of the request body; excludes previous "
        "translations; output priced at max_output_tokens per call (upper bound)",
        "known_calls": calls,
        "input_tokens_known_calls": first_tokens,
        "cost_usd_known_calls": cost,
        "translation_upper_bound": inputs.plan.translation_upper_bound,
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
        "trigger_source": inputs.config.triggers.source,
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


def _budget_json(budget: BudgetFit | None) -> dict[str, Any] | None:
    if budget is None:
        return None
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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    package = Path(__file__).resolve().parent
    return {f"translate/{path.relative_to(package)}": _file_digest(path) for path in sorted(package.rglob("*.py"))}


def _resolved_config(config: RunConfig) -> dict[str, Any]:
    """The resolved config; ``pipeline.visibility`` is where the judges read the visibility."""
    resolved = json.loads(json.dumps(asdict(config), default=str))
    return {**resolved, "pipeline": {"visibility": asdict(config.visibility)}}


def _run_record(inputs: Inputs, *, mode: str, status: str) -> dict[str, Any]:
    config, scenario = inputs.config, inputs.scenario
    csv_path = scenario.episode.csv_path
    record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "started_at": utc_now(),
        "ended_at": None,
        "inputs": {
            "config": {"path": str(config.path), "sha256": config.sha256},
            "prompt": {
                "path": str(config.prompt.path),
                "sha256": config.prompt.sha256,
                "version": config.prompt.version,
            },
            "scenario": {
                "path": str(inputs.scenario_path),
                "sha256": _file_digest(inputs.scenario_path),
                "scenario_id": scenario.scenario_id,
            },
            "episode": {
                "episode_id": scenario.episode.episode_id,
                "alias": scenario.episode.alias,
                "csv_path": str(csv_path),
                "csv_sha256": inputs.frame.attrs.get("csv_sha256"),
                "manifest_sha256": _file_digest(csv_path.parent / MANIFEST_NAME),
                "video_path": str(scenario.episode.video_path) if scenario.episode.video_path else None,
                "last_complete_step": inputs.facts.last_step,
            },
            "policy": inputs.policy.provenance() if inputs.policy is not None else None,
        },
        "provenance": {
            "git": _git_state(),
            "code_sha256": _code_hashes(),
            "python": sys.version.split()[0],
            "renderers": {LAG17_RENDERER: RENDERER_VERSION, DETECTOR_RENDERER: RENDERER_VERSION},
            "prompt_version": config.prompt.version,
            "openai_base_url": API_BASE_URL,  # the only endpoint make_openai_client targets
            "sdk_environment_set": set_provenance_env_names(),  # names only; proxy URLs can hold credentials
        },
        "config": _resolved_config(config),
        "plan": _plan_json(inputs),
        "detector_budget": _budget_json(inputs.budget),
        "api_requests_made": 0,
    }
    if config.detector is None:
        return record
    questions = (choice_question(config.detector.question),)
    detector = {"detector_state_version": DETECTOR_STATE_VERSION, "question_set_sha256": question_set_sha256(questions)}
    return {**record, "provenance": {**record["provenance"], **detector}}


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


# --- modes ----------------------------------------------------------------------------------------


def _redact(exc: BaseException) -> dict[str, str]:
    """The LLM parts' errors are sanitised by design; everything else goes through ``redacted_error``."""
    if isinstance(exc, (SystemOneError, OpenAIClientError)):
        return {"type": type(exc).__name__, "message": str(exc)}
    return redacted_error(exc)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _preview(args: argparse.Namespace, inputs: Inputs) -> int:
    bodies = mandatory_bodies(inputs)
    run_dir = _new_run_dir(args, inputs, MODE_PREVIEW)
    record = {**_run_record(inputs, mode=MODE_PREVIEW, status=STATUS_PREVIEW), "ended_at": utc_now()}
    preview = {
        "api_requests_made": 0,
        "plan": record["plan"],
        "detector_budget": record["detector_budget"],
        "known_requests_sha256": [sha256_text(canonical_json(body)) for body in bodies],
        "translation_estimate": _estimate(inputs, bodies),
    }
    write_json(run_dir / "run.json", record)
    write_json(run_dir / "preview.json", preview)
    if bodies:
        write_json(run_dir / EXAMPLE_REQUEST_FILE, bodies[0])
    plan = record["plan"]
    _print(
        {
            "run_dir": str(run_dir),
            "status": STATUS_PREVIEW,
            "mandatory_translations": plan["mandatory_translations"],
            "detector_calls": plan["detector_calls"],
            "translation_upper_bound": plan["translation_upper_bound"],
            "detector_budget_margin": inputs.budget.margin if inputs.budget else None,
            "warnings": plan["warnings"],
        }
    )
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def _one_shot_call(config: RunConfig, openai_client: Any, journal: RetryJournal) -> OneShotCall:
    translation = config.translation
    client = openai_client
    if translation.retry.max_retries > 0:  # retry: none keeps the bare client (a 5xx is not resent)
        client = RetryingClient(openai_client, translation.retry, log=journal.record)
    model = OpenAIIntentModel(
        OpenAIResponsesClient(client),
        model=translation.model,
        reasoning_effort=translation.reasoning_effort,
        max_output_tokens=translation.max_output_tokens,
    )
    return OneShotCall(model, config.prompt, config.answer.limits)


def session_settings(config: RunConfig, policy: PolicyMaterial) -> SessionSettings:
    session = config.session
    sandbox = session.sandbox
    limits = SessionLimits(
        session.max_runs,
        session.max_total_tokens,
        session.max_protocol_errors,
        int(session.max_total_tokens * session.answer_at_fraction),
    )
    sandbox_limits = SandboxLimits(
        sandbox.timeout_s,
        sandbox.memory_bytes,
        sandbox.file_size_bytes,
        sandbox.stdout_cap_chars,
        sandbox.stderr_tail_chars,
        sandbox.tmp_size_mb,
        sandbox.threads,
        sandbox.work_max_bytes,
    )
    hidden = (paths.ROOT, Path.home(), policy.checkpoint.path.parent, policy.rollout.path.parent)
    return SessionSettings(limits, sandbox_limits, config.policy.past_steps, hidden)


def _session_call(inputs: Inputs, openai_client: Any, preflight: Preflight) -> SessionCall:
    config, policy = inputs.config, inputs.policy
    translation = config.translation
    backend = OpenAIResponsesBackend(
        openai_client, translation.model, translation.reasoning_effort, translation.max_output_tokens, translation.retry
    )
    settings = session_settings(config, policy)
    executors = sandbox_executor(discover_interpreter(config.session.sandbox.python), settings.sandbox)
    return SessionCall(
        backend, backend.options, config.prompt, settings, policy, executors, config.answer.limits, preflight
    )


def sandbox_selftest(inputs: Inputs, preflight: Preflight = run_preflight) -> bool:
    """Fail-closed preflight in a throwaway directory, with the first slot's real session inputs."""
    plan = next((item for item in inputs.plan.targets if item.mandatory_translation_steps), None)
    if plan is None or inputs.policy is None:
        return True
    settings = session_settings(inputs.config, inputs.policy)
    scratch = Path(tempfile.mkdtemp(prefix="translate-preflight-"))
    try:
        step = plan.mandatory_translation_steps[0]
        session_inputs = write_session_inputs(
            inputs.policy, plan.target.target_id, step, settings.past_steps, scratch / "inputs"
        )
        executors = sandbox_executor(discover_interpreter(inputs.config.session.sandbox.python), settings.sandbox)
        executor = executors(scratch / "work", session_inputs.bundle.weights)
        result = preflight(executor, [*settings.hidden, session_inputs.bundle.path])
        print(json.dumps({"sandbox_preflight": result.to_dict()}), file=sys.stderr)
        return bool(result.ok)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


class CountingClient:
    """The OpenAI SDK client with every ``responses.create`` counted.

    It sits below the retry and session layers, so the count is every request sent to the paid API:
    each one-shot call, each code-session turn and each retried attempt.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self.responses = self
        self.created = 0

    def create(self, **kwargs: Any) -> Any:
        self.created += 1
        return self._client.responses.create(**kwargs)


def _final_record(record: dict[str, Any], summary: TranslatorSummary, openai_requests: int) -> dict[str, Any]:
    counts = {"detector": asdict(summary.detector), "translation": asdict(summary.translation)}
    return {
        **record,
        "status": summary.status,
        "ended_at": utc_now(),
        "counts": {**counts, "events_written": summary.events_written, "openai_requests": openai_requests},
        "unsent": summary.unsent,
        "error": summary.error,
        "api_requests_made": summary.detector.attempted + openai_requests,
    }


def _finish(run_dir: Path, record: dict[str, Any], summary: TranslatorSummary, openai_requests: int) -> int:
    """Final ``run.json`` and ``cues.jsonl``. If they cannot be written, an uncertain or capped run keeps its exit code."""
    final = _final_record(record, summary, openai_requests)
    exit_code = EXIT_BY_STATUS[summary.status]
    try:
        write_json(run_dir / "run.json", final)
        write_cues(run_dir)
    except INTERRUPTIONS as exc:
        print(f"error: final run.json not written: {type(exc).__name__} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED if exit_code == EXIT_OK else exit_code
    _print({"run_dir": str(run_dir), "status": summary.status, "counts": final["counts"], "unsent": summary.unsent})
    return exit_code


def _close_detector(client: SystemOneClient | None) -> None:
    """Close the local client. Runs after the outcome is decided, so a failure here never replaces it."""
    if client is None:
        return
    try:
        client.close()
    except INTERRUPTIONS as exc:
        print(f"error: closing the detector client failed: {type(exc).__name__}", file=sys.stderr)


@dataclass(frozen=True)
class Factories:
    detector_client: DetectorClientFactory = build_system_one_client
    openai_client: OpenAIClientFactory = make_openai_client
    preflight: Preflight = run_preflight


def _execute(args: argparse.Namespace, inputs: Inputs, factories: Factories) -> int:
    """Order: sandbox preflight -> local client -> /v1/models health -> run dir -> paid client -> pipeline."""
    if inputs.config.code_session and not sandbox_selftest(inputs, factories.preflight):
        print("Refused: sandbox preflight failed. API requests made: 0", file=sys.stderr)
        return EXIT_FAILED
    detector_client = factories.detector_client(inputs.config.detector) if inputs.config.online else None
    try:
        return _execute_with(args, inputs, detector_client, factories)
    finally:
        _close_detector(detector_client)  # an uncertain or capped run keeps its exit code


def _build_translator(
    inputs: Inputs, run_dir: Path, detector_client: SystemOneClient | None, factories: Factories, openai_client: Any
) -> Translator:
    config = inputs.config
    journal = RetryJournal()
    if config.code_session:
        call = _session_call(inputs, openai_client, factories.preflight)
    else:
        call = _one_shot_call(config, openai_client, journal)
    detector = None
    if detector_client is not None:
        detection = SystemOneDetectionModel(detector_client, question_id=config.detector.question.question_id)
        detector = DetectorLayer(detection, threshold=config.detector.threshold)
    sessions = run_dir / SESSIONS_DIR if config.code_session else None
    return Translator(call=call, detector=detector, policy=inputs.policy, sessions_dir=sessions, retries=journal)


def _execute_with(
    args: argparse.Namespace, inputs: Inputs, detector_client: SystemOneClient | None, factories: Factories
) -> int:
    if detector_client is not None:
        detector_client.health()  # before any paid client exists
    run_dir = _new_run_dir(args, inputs, MODE_EXECUTE)
    record: dict[str, Any] = {"schema_version": RUN_SCHEMA_VERSION, "mode": MODE_EXECUTE, "started_at": utc_now()}
    summary: TranslatorSummary | None = None
    counter = CountingClient(None)
    try:
        record = _run_record(inputs, mode=MODE_EXECUTE, status=STATUS_RUNNING)
        write_json(run_dir / "run.json", record)
        counter = CountingClient(factories.openai_client(timeout_s=inputs.config.translation.timeout_s))
        translator = _build_translator(inputs, run_dir, detector_client, factories, counter)
        with JsonlRunSink(run_dir) as sink:
            # never raises: call failures, journal failures and Ctrl-C become a summary status
            summary = translator.run(frame=inputs.frame, scenario=inputs.scenario, config=inputs.config, sink=sink)
    except INTERRUPTIONS as exc:
        error = _redact(exc)
        if summary is not None:  # closing the journals failed after every line was fsynced
            print(f"error: closing the journals failed: {error['type']} ({run_dir})", file=sys.stderr)
            return _finish(run_dir, record, summary, counter.created)
        # before the pipeline (or Ctrl-C there): no model call is in flight; recorded, then a failure exit
        failed = {**record, "status": STATUS_FAILED, "error": error, "ended_at": utc_now()}
        write_json(run_dir / "run.json", failed)
        print(f"error: {error['type']}: {error['message']} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED
    return _finish(run_dir, record, summary, counter.created)


def main(argv: list[str] | None = None, *, factories: Factories | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("a paid run needs BOTH --execute and --confirm-paid-api")
    try:
        inputs = load_inputs(load_config(args.config), args.scenario)
        if not args.execute:
            return _preview(args, inputs)
        return _execute(args, inputs, factories or Factories())
    except INTERRUPTIONS as exc:  # reported without traceback (Ctrl-C included); unknown text is withheld
        error = _redact(exc)
        print(f"error: {error['type']}: {error['message']}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
