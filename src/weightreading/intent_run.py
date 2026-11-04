"""Method B of the intent comparison: preparation and local preview (paid run: ``intent_execute.py``).

Layout (everything with weights or activations stays under the gitignored ``outputs/weightreading``):

* preview: ``outputs/weightreading/intent/previews/<UTC>_seed<seed>/`` with ``preview.json`` and
  ``sessions/<s0045_A0100>/{message.txt, inputs/}``;
* run: ``outputs/weightreading/intent/runs/<run_id>/`` with ``run.json`` and per session
  ``message.txt``, ``inputs/``, ``work/``, ``transcript.jsonl``, ``tool_calls.jsonl``,
  ``state.json``, ``final_answer.md``, ``audit.json``, ``session.json``;
* cues: ``<compare.output_root>/<run_id>/cues_B.jsonl`` (the shared contract).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from mtd_rl_xavier import paths
from mtd_rl_xavier.tools.output_safety import derived_path
from weightreading import provenance
from weightreading.commands import DEFAULT_OUTPUT_ROOT, executor_limits, write_json
from weightreading.config import ConfigError, RunConfig, load_config
from weightreading.intent_config import CompareConfig, load_compare
from weightreading.intent_episode import Episode, PlannedSession, load_episode, plan_sessions
from weightreading.intent_estimate import session_estimate
from weightreading.intent_inputs import SessionInputs, write_inputs
from weightreading.intent_prompt import PROMPT_VERSION, IntentRequest, PreviousTranslation, intent_message
from weightreading.intent_protocol import MAX_INTENT_CHARS, IntentAnswer
from weightreading.intent_trace import TraceError, activation_docs, trace_episode, validate_trace, window_arrays
from weightreading.reference import ReferencePolicy, build_policy
from weightreading.session import SessionLimits
from weightreading.spec import Spec, check_against_tensors, load_spec
from weightreading.torch_reader import Checkpoint, read_checkpoint

INTENT_ROOT = DEFAULT_OUTPUT_ROOT / "intent"
DEFAULT_COMPARE = paths.CONFIGS / "intent_compare" / "compare.yaml"
DEFAULT_CONFIG = paths.CONFIGS / "weightreading" / "intent_b.yaml"
DEFAULT_SPEC = paths.CONFIGS / "weightreading" / "lag_mappo_2v2_33.spec.json"
# Preview stand-in for a previous translation (filled at run time): a full-length intent plus this
# much evidence text, so the token estimate is an upper bound for chained sessions.
PREVIOUS_EVIDENCE_RESERVE_CHARS = 2000
# Share of max_total_tokens after which no more code is run and the final answer is requested, so
# a session ends with an answer instead of hitting the hard cap.
ANSWER_AT_FRACTION = 0.75
EXIT_OK = 0
EXIT_REFUSED = 1


@dataclass(frozen=True)
class IntentPrepared:
    compare: CompareConfig
    config: RunConfig
    config_path: Path
    spec: Spec
    spec_path: Path
    checkpoint: Checkpoint
    episode: Episode
    policy: ReferencePolicy
    trace: dict
    validation: dict
    sessions: tuple[PlannedSession, ...]


def prepare(
    compare_path: Path, config_path: Path, spec_path: Path, checkpoint_path: Path | None = None
) -> IntentPrepared:
    """Load and cross-check every input, replay the episode and validate it against the recording."""
    compare = load_compare(compare_path, paths.ROOT)
    config = load_config(config_path, paths.ROOT)
    if (config.model, config.reasoning_effort) != (compare.model, compare.reasoning_effort):
        raise ConfigError(
            f"{config_path} model/reasoning_effort ({config.model}, {config.reasoning_effort}) must equal "
            f"compare.yaml ({compare.model}, {compare.reasoning_effort})"
        )
    episode = load_episode(compare)
    checkpoint = read_checkpoint(checkpoint_path or episode.checkpoint_path)
    if checkpoint.sha256 != episode.checkpoint_sha256:
        raise ConfigError("checkpoint sha256 differs from the actor recorded in the rollout manifest")
    spec = load_spec(spec_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    policy = build_policy(spec, checkpoint.tensors)
    arrays = episode.arrays
    trace = trace_episode(policy, arrays["obs"])
    validation = validate_trace(trace, policy, arrays["logits"], arrays["gru_out"], arrays["actions"], arrays["rnn_in"])
    if not validation["ok"]:
        raise TraceError(f"replayed activations do not reproduce the recording: {validation}")
    sessions = plan_sessions(episode, compare)
    return IntentPrepared(
        compare, config, config_path, spec, spec_path, checkpoint, episode, policy, trace, validation, sessions
    )


def answer_at_tokens(config: RunConfig) -> int:
    return int(config.limits.max_total_tokens * ANSWER_AT_FRACTION)


def session_limits_for(config: RunConfig) -> SessionLimits:
    limits = config.limits
    return SessionLimits(limits.max_runs, limits.max_total_tokens, limits.max_protocol_errors, answer_at_tokens(config))


def _request(prepared: IntentPrepared, planned: PlannedSession, previous: PreviousTranslation | None) -> IntentRequest:
    compare = prepared.compare
    return IntentRequest(
        aircraft=planned.aircraft,
        step=planned.step,
        t_s=round(planned.step * compare.step_seconds, 6),
        window=planned.window,
        step_seconds=compare.step_seconds,
        instruction=compare.instruction,
        previous=previous,
    )


def message_for(prepared: IntentPrepared, planned: PlannedSession, previous: PreviousTranslation | None) -> str:
    request = _request(prepared, planned, previous)
    docs = activation_docs(prepared.policy)
    return intent_message(prepared.spec, prepared.checkpoint.tensors, docs, request, executor_limits(prepared.config))


def write_session_inputs(prepared: IntentPrepared, planned: PlannedSession, directory: Path) -> SessionInputs:
    start, end = planned.window
    arrays = window_arrays(prepared.trace, planned.row, start, end)
    metadata = {
        "network_alias": prepared.spec.alias,
        "checkpoint_sha256": prepared.checkpoint.sha256,
        "rollout_sha256": prepared.episode.sha256,
        "seed": prepared.compare.seed,
        "aircraft": planned.aircraft,
        "learner_row": planned.row,
        "step": planned.step,
        "window": [start, end],
    }
    return write_inputs(directory, prepared.checkpoint.tensors, arrays, metadata)


def input_record(prepared: IntentPrepared) -> dict:
    return {
        "compare_path": str(prepared.compare.path),
        "compare_sha256": prepared.compare.sha256,
        "config_path": str(prepared.config_path),
        "config_sha256": provenance.sha256_file(prepared.config_path),
        "spec_path": str(prepared.spec_path),
        "spec_sha256": provenance.sha256_file(prepared.spec_path),
        "checkpoint_path": str(prepared.checkpoint.path),
        "checkpoint_sha256": prepared.checkpoint.sha256,
        "rollout_path": str(prepared.episode.path),
        "rollout_sha256": prepared.episode.sha256,
        "rollout_manifest_sha256": prepared.episode.manifest_sha256,
    }


def common_record(prepared: IntentPrepared) -> dict:
    compare = prepared.compare
    return {
        "method": "B",
        "prompt_version": PROMPT_VERSION,
        "network_alias": prepared.spec.alias,
        "seed": compare.seed,
        "steps": list(compare.steps),
        "targets": list(compare.targets),
        "window_steps": compare.window_steps,
        "model": prepared.config.model,
        "reasoning_effort": prepared.config.reasoning_effort,
        "inputs": input_record(prepared),
        "trace_validation": prepared.validation,
        "limits": asdict(prepared.config.limits),
    }


def _placeholder(label: str, chars: int) -> str:
    return label.ljust(chars, ".")


def _preview_previous(planned: PlannedSession, chained: dict[str, int]) -> PreviousTranslation | None:
    """Run-time stand-in: a full-length intent and evidence reserve, so the estimate is an upper bound."""
    if planned.aircraft not in chained:
        return None
    placeholder = IntentAnswer(
        intent=_placeholder("[PREVIEW: method B's previous intent for this aircraft goes here]", MAX_INTENT_CHARS),
        evidence=(_placeholder("[PREVIEW: its evidence goes here]", PREVIOUS_EVIDENCE_RESERVE_CHARS),),
    )
    return PreviousTranslation(step=chained[planned.aircraft], answer=placeholder)


def write_preview(prepared: IntentPrepared, directory: Path) -> dict:
    """Messages, session inputs and a token/cost bound for every planned session. No API call."""
    derived_path(directory).mkdir(parents=True, exist_ok=False)
    chained: dict[str, int] = {}
    entries = []
    for planned in prepared.sessions:
        entry = {"step": planned.step, "aircraft": planned.aircraft, "window": list(planned.window)}
        if planned.skipped:
            entries.append({**entry, "skipped": planned.skipped})
            continue
        session_dir = directory / "sessions" / planned.name
        inputs = write_session_inputs(prepared, planned, session_dir / "inputs")
        previous = _preview_previous(planned, chained)
        message = message_for(prepared, planned, previous)
        (session_dir / "message.txt").write_text(message, encoding="utf-8")
        chained[planned.aircraft] = planned.step
        estimate = session_estimate(len(message), prepared.config, answer_at_tokens(prepared.config))
        entries.append(
            {
                **entry,
                "skipped": None,
                "previous_placeholder": previous is not None,
                "message_sha256": provenance.sha256_text(message),
                "inputs_manifest_sha256": inputs.bundle.manifest_sha256,
                "estimate": estimate,
            }
        )
    payload = {**common_record(prepared), "sessions": entries, "totals": _totals(entries), "api_requests_made": 0}
    write_json(directory / "preview.json", payload)
    return payload


def _sum_cost(entries: list[dict], case: str) -> float | None:
    costs = [entry["estimate"][case]["cost_usd"] for entry in entries if entry.get("estimate")]
    if not costs or any(cost is None for cost in costs):
        return None
    return round(sum(costs), 2)


def _totals(entries: list[dict]) -> dict:
    return {
        "sessions": sum(1 for entry in entries if not entry["skipped"]),
        "skipped": sum(1 for entry in entries if entry["skipped"]),
        "hard_cap_cost_usd": _sum_cost(entries, "hard_cap"),
        "expected_cost_usd": _sum_cost(entries, "expected"),
    }
