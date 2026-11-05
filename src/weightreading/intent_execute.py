"""Paid sequential run of method B, with an explicit resume after a crash.

Order of a fresh run: run dir (exclusive) + ``cues_B.jsonl`` (exclusive) -> sandbox preflight in a
throwaway directory (fail closed) -> API client (reads the key) -> sessions by (step, aircraft).
Every session also runs its own preflight. A session that ends without a valid answer still gets a
cue line (``"skipped": "<reason>"``), and the aircraft's chain keeps its latest valid translation.

``resume=True`` continues an existing run: ``run.json`` must describe the same inputs, the cue file
must hold a prefix of the session plan, the chain is rebuilt from it, and the directory of a session
that was interrupted (no cue line) is moved aside to ``<name>.aborted-<NN>`` before it is redone.
Its requests may already have been billed; that is why resuming is never automatic.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mtd_rl_xavier import paths
from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading import provenance
from weightreading.backends import ChatBackend, OpenAIResponsesBackend
from weightreading.bundle import verify_bundle
from weightreading.commands import hidden_paths, make_executor, utc_stamp, write_json
from weightreading.config import ConfigError
from weightreading.intent_audit import write_intent_audit
from weightreading.intent_cues import METHOD, CueContext, CueWriter, read_cues, skipped_cue, translated_cue
from weightreading.intent_episode import PlannedSession
from weightreading.intent_inputs import PRELOAD, SessionInputs
from weightreading.intent_prompt import PreviousTranslation
from weightreading.intent_protocol import (
    AnswerLimits,
    IntentAnswer,
    IntentAnswerError,
    parse_answer,
    parse_intent_reply,
)
from weightreading.intent_run import (
    EXIT_OK,
    EXIT_REFUSED,
    INTENT_ROOT,
    IntentPrepared,
    common_record,
    message_for,
    session_limits_for,
    write_session_inputs,
)
from weightreading.preflight import run_preflight
from weightreading.sandbox import SandboxExecutor
from weightreading.session import FINAL_ANSWER, RunLog, run_session

RESUME_KEYS = ("inputs", "prompt_version", "model", "reasoning_effort", "limits", "steps", "targets", "window_steps")


class PreflightRefused(RuntimeError):
    """The sandbox preflight failed; no (further) request is sent."""


@dataclass(frozen=True)
class _RunContext:
    prepared: IntentPrepared
    run_dir: Path
    cues: CueWriter
    cue_context: CueContext


def _session_executor(prepared: IntentPrepared, inputs: SessionInputs, work: Path) -> SandboxExecutor:
    base = make_executor(prepared.config, work, inputs.bundle.weights)
    return SandboxExecutor(base.interp, work, inputs.bundle.weights, base.limits, PRELOAD)


def _hidden(prepared: IntentPrepared, inputs: SessionInputs) -> list[Path]:
    return hidden_paths(inputs.bundle.path, prepared.checkpoint.path.parent, prepared.episode.path.parent)


def _answer(session_dir: Path, stop_reason: str | None, limits: AnswerLimits) -> tuple[IntentAnswer | None, str | None]:
    """(answer, None) or (None, why there is no answer)."""
    if stop_reason != "final_answer":
        return None, f"session ended without a final answer ({stop_reason})"
    try:
        return parse_answer((session_dir / FINAL_ANSWER).read_text(encoding="utf-8"), limits), None
    except IntentAnswerError as exc:
        return None, f"invalid final answer: {exc}"


def _run_one(
    context: _RunContext, backend: ChatBackend, planned: PlannedSession, previous: PreviousTranslation | None
) -> tuple[IntentAnswer | None, str | None, dict]:
    prepared = context.prepared
    session_dir = context.run_dir / "sessions" / planned.name
    inputs = write_session_inputs(prepared, planned, session_dir / "inputs")
    message = message_for(prepared, planned, previous)
    (session_dir / "message.txt").write_text(message, encoding="utf-8")
    executor = _session_executor(prepared, inputs, session_dir / "work")
    preflight = run_preflight(executor, _hidden(prepared, inputs))
    record = {
        "step": planned.step,
        "aircraft": planned.aircraft,
        "learner_row": planned.row,
        "window": list(planned.window),
        "previous_step": None if previous is None else previous.step,
        "message_sha256": provenance.sha256_text(message),
        "inputs_manifest_sha256": inputs.bundle.manifest_sha256,
        "inputs_files_sha256": inputs.files_sha256,
        "sandbox_preflight": preflight.to_dict(),
        "started_utc": utc_stamp(),
    }
    if not preflight.ok:
        write_json(session_dir / "session.json", {**record, "status": "refused: sandbox preflight failed"})
        raise PreflightRefused(planned.name)
    state = run_session(
        backend, executor, message, session_limits_for(prepared.config), RunLog(session_dir), parse_intent_reply
    )
    verify_bundle(inputs.bundle.path, inputs.bundle.manifest_sha256)
    audit = write_intent_audit(session_dir)
    answer, reason = _answer(session_dir, state.stop_reason, prepared.compare.answer_limits)
    summary = {
        **record,
        "status": "complete",
        "finished_utc": utc_stamp(),
        "session": {**asdict(state), "total_tokens": state.total_tokens},
        "answered": answer is not None,
        "no_answer_reason": reason,
        "audit": str(audit),
    }
    write_json(session_dir / "session.json", summary)
    return answer, reason, summary


def _source_run(run_dir: Path, name: str) -> str:
    target = run_dir / "sessions" / name
    return str(target.relative_to(paths.ROOT)) if target.is_relative_to(paths.ROOT) else str(target)


def _cue(context: _RunContext, planned: PlannedSession, answer: IntentAnswer | None, reason: str | None) -> dict:
    source = _source_run(context.run_dir, planned.name)
    if answer is None:
        return skipped_cue(context.cue_context, planned.step, planned.aircraft, source, reason or "no answer")
    return translated_cue(context.cue_context, planned.step, planned.aircraft, source, answer)


def _chain_from_cues(cues: list[dict]) -> dict[str, PreviousTranslation]:
    chain: dict[str, PreviousTranslation] = {}
    for cue in cues:
        if "intent" in cue:
            chain[cue["aircraft"]] = PreviousTranslation(
                cue["step"], IntentAnswer(cue["intent"], tuple(cue["evidence"]))
            )
    return chain


def _sessions(context: _RunContext, backend: ChatBackend, done: list[dict]) -> list[dict]:
    chain = _chain_from_cues(done)
    records: list[dict] = [{"step": cue["step"], "aircraft": cue["aircraft"], "resumed": True} for cue in done]
    for planned in context.prepared.sessions[len(done) :]:
        if planned.skipped:
            context.cues.append(_cue(context, planned, None, planned.skipped))
            records.append({"step": planned.step, "aircraft": planned.aircraft, "skipped": planned.skipped})
            continue
        answer, reason, record = _run_one(context, backend, planned, chain.get(planned.aircraft))
        context.cues.append(_cue(context, planned, answer, reason))
        if answer is not None:
            chain[planned.aircraft] = PreviousTranslation(planned.step, answer)
        records.append(record)
    return records


def _resume_state(prepared: IntentPrepared, run_dir: Path, record: dict, cues_path: Path) -> list[dict]:
    """Check that ``run_dir`` belongs to these inputs; return the cue lines already written."""
    previous = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    changed = [key for key in RESUME_KEYS if previous.get(key) != record.get(key)]
    if changed:
        raise ConfigError(f"cannot resume {run_dir}: {changed} differ from the current inputs")
    done = read_cues(cues_path)
    planned = [(p.step, p.aircraft) for p in prepared.sessions]
    if [(cue["step"], cue["aircraft"]) for cue in done] != planned[: len(done)]:
        raise ConfigError(f"cannot resume: {cues_path} is not a prefix of the session plan")
    previous_runs = len(list(run_dir.glob("run.previous-*.json"))) + 1
    (run_dir / "run.json").rename(run_dir / f"run.previous-{previous_runs:02d}.json")
    if len(done) < len(planned):
        interrupted = run_dir / "sessions" / prepared.sessions[len(done)].name
        if interrupted.exists():
            aborted = len(list(interrupted.parent.glob(f"{interrupted.name}.aborted-*"))) + 1
            interrupted.rename(interrupted.with_name(f"{interrupted.name}.aborted-{aborted:02d}"))
    return done


def _first_preflight_ok(context: _RunContext, planned: PlannedSession) -> bool:
    """Preflight in a throwaway directory before the API client (and the key) is touched."""
    attempt = len(list(context.run_dir.glob("preflight-*"))) + 1  # one per start or resume
    probe = context.run_dir / f"preflight-{attempt:02d}"
    inputs = write_session_inputs(context.prepared, planned, probe / "inputs")
    executor = _session_executor(context.prepared, inputs, probe / "work")
    result = run_preflight(executor, _hidden(context.prepared, inputs))
    write_json(probe / "preflight.json", result.to_dict())
    return result.ok


def execute_run(
    prepared: IntentPrepared,
    run_id: str,
    client_factory: Callable[[], Any],
    intent_root: Path = INTENT_ROOT,
    resume: bool = False,
) -> tuple[int, dict]:
    run_dir = intent_root / "runs" / run_id
    cues_path = prepared.compare.cues_path(run_id, METHOD)
    record = {
        **common_record(prepared),
        "run_id": run_id,
        "cues_path": str(cues_path),
        "code_sha256": provenance.code_hashes(),
        "git": provenance.git_state(paths.ROOT),
        "environment": {**provenance.environment(), "openai_sdk": _sdk_version()},
        "status": "started",
    }
    if resume:
        done = _resume_state(prepared, run_dir, record, cues_path)
        cues = CueWriter(cues_path, resume=True)
    else:
        derived_path(run_dir).mkdir(parents=True, exist_ok=False)
        done, cues = [], CueWriter(cues_path)
    record = {**record, "resumed_from_cues": len(done)}
    write_json(run_dir / "run.json", record)
    compare = prepared.compare
    cue_context = CueContext(
        compare.seed, prepared.config.model, prepared.config.reasoning_effort, compare.step_seconds
    )
    context = _RunContext(prepared, run_dir, cues, cue_context)
    remaining = [planned for planned in prepared.sessions[len(done) :] if not planned.skipped]
    if remaining and not _first_preflight_ok(context, remaining[0]):
        write_json(run_dir / "run.json", {**record, "status": "refused: sandbox preflight failed"})
        return EXIT_REFUSED, record
    config = prepared.config
    backend = OpenAIResponsesBackend(
        client_factory(), config.model, config.reasoning_effort, config.limits.max_output_tokens, config.retry
    )
    try:
        sessions = _sessions(context, backend, done)
    except Exception as exc:
        write_json(run_dir / "run.json", {**record, "status": f"error: {type(exc).__name__}: {exc}"[:500]})
        raise
    failed = sum(1 for item in sessions if item.get("answered") is False)
    status = "complete" if failed == 0 else f"complete: {failed} session(s) without a valid answer"
    final = {**record, "status": status, "sessions": sessions}
    write_json(run_dir / "run.json", final)
    return EXIT_OK, final


def _sdk_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version("openai")
    except metadata.PackageNotFoundError:
        return None
