#!/usr/bin/env python3
"""Offline replay of recorded translation requests (restructure guard; never calls an API).

For every recorded trajectory-input run (``outputs/intent_compare/<run>/method_a``) this rebuilds
each request body from the current code, chaining the recorded answers, and compares its canonical
SHA-256 with the recorded payload. For every recorded policy-input run
(``outputs/weightreading/intent/runs/<run>``) it rebuilds each session's first message and replays the
whole session with the recorded model replies and sandbox results, comparing the SHA-256 of the
conversation history sent at every turn with the recorded ``input_sha256`` (so the first message
and every follow-up turn are covered).

    python scripts/check_request_replay.py --source-root <checkout>

The baseline ``tests/fixtures/request-replay-baseline.json`` lists every recorded hash, taken from
the recordings themselves (the check first confirms that the recordings still hold exactly these), plus ``known_unreproduced``: runs the code cannot rebuild, with the reason.
The check passes only when every hash of every other run is reproduced; that list may only shrink.
Exit codes: 0 ok; 1 a hash outside ``known_unreproduced`` was not reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtd_rl_xavier.intent_compare import load_compare
from mtd_rl_xavier.intent_compare.method_a_requests import Slot, build_request, load_settings, prompt_for, request_body
from translate import paths
from translate.events import IntentTranslation, PreviousTranslation, canonical_json, sha256_text
from translate.lag.dataset import load_episode as load_csv_episode
from weightreading.backends import BackendReply, Usage, history_sha256
from weightreading.config import load_config
from weightreading.intent_episode import load_episode, plan_sessions
from weightreading.intent_prompt import PreviousTranslation as PolicyPrevious
from weightreading.intent_protocol import IntentAnswer, parse_intent_reply
from weightreading.intent_run import IntentPrepared, message_for, session_limits_for
from weightreading.intent_trace import trace_episode, validate_trace
from weightreading.reference import build_policy
from weightreading.sandbox import ExecResult
from weightreading.session import RunLog, run_session
from weightreading.spec import check_against_tensors, load_spec
from weightreading.torch_reader import read_checkpoint

EXIT_OK, EXIT_MISMATCH = 0, 1
RUN_IDS = ("seed100601", "seed100601-ja", "seed100601-ja-short")
BASELINE = paths.ROOT / "tests" / "fixtures" / "request-replay-baseline.json"
CONFIG_DIR = paths.CONFIGS / "intent_compare"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repo_file(recorded: str, source_root: Path) -> Path:
    """A recorded repository path (relative, or absolute under the source checkout) in this checkout."""
    path = Path(recorded)
    return paths.ROOT / (path.relative_to(source_root) if path.is_absolute() else path)


def latest_answers(cues: list[dict[str, Any]], before_step: int) -> dict[str, tuple[int, str, list[str]]]:
    """Per aircraft, its latest answered cue before ``before_step``: the chain both methods keep."""
    chain: dict[str, tuple[int, str, list[str]]] = {}
    for cue in cues:
        if cue["step"] < before_step and "intent" in cue:
            chain = {**chain, cue["aircraft"]: (cue["step"], cue["intent"], cue["evidence"])}
    return chain


# --- trajectory-input runs (recorded by method A) -------------------------------------------------


def trajectory_run(source_root: Path, run_id: str) -> dict[str, str]:
    """call_id -> rebuilt request sha256, for every recorded request of one run."""
    run_dir = source_root / "outputs" / "intent_compare" / run_id
    record = json.loads((run_dir / "method_a" / "run.json").read_text(encoding="utf-8"))
    provenance = record["provenance"]
    config_path = repo_file(provenance["compare_config"]["path"], source_root)
    config = load_compare(config_path, source_root)
    frame = load_csv_episode(Path(provenance["episode_csv"]["path"]))
    requests = read_jsonl(run_dir / "method_a" / "requests.jsonl")
    settings = load_settings(config, requests[0]["payload"]["max_output_tokens"])
    cues = read_jsonl(run_dir / "cues_A.jsonl")
    rebuilt = {}
    for line in requests:
        slot = Slot(line["step"], line["aircraft"], None)
        known = latest_answers(cues, slot.step).get(slot.aircraft)
        previous = None
        if known is not None:
            step, intent, evidence = known
            previous = PreviousTranslation(
                f"step{step:03d}_{slot.aircraft}", step, IntentTranslation(intent, tuple(evidence))
            )
        body = request_body(prompt_for(build_request(config, frame, slot, previous), settings), config, settings)
        rebuilt[line["call_id"]] = sha256_text(canonical_json(body))
    return rebuilt


def trajectory_recorded(source_root: Path, run_id: str) -> dict[str, str]:
    lines = read_jsonl(source_root / "outputs" / "intent_compare" / run_id / "method_a" / "requests.jsonl")
    return {line["call_id"]: sha256_text(canonical_json(line["payload"])) for line in lines}


# --- policy-input runs (recorded by method B) -----------------------------------------------------


@dataclass
class RecordedBackend:
    """Serves the recorded replies in order and keeps the history hash of every request."""

    replies: list[dict[str, Any]]
    model: str = "recorded"

    def __post_init__(self) -> None:
        self.sent: list[str] = []

    def respond(self, history: tuple[dict, ...]) -> BackendReply:
        self.sent.append(history_sha256(history))
        line = self.replies[len(self.sent) - 1]
        usage = line["usage"]
        return BackendReply(
            text=line["text"],
            output_items=tuple(line["output_items"]),
            usage=Usage(
                usage["input_tokens"], usage["output_tokens"], usage["reasoning_tokens"], usage["cached_input_tokens"]
            ),
            status=line["status"],
            response_id=line["response_id"],
            request_record=line["request"],
        )


@dataclass
class RecordedExecutor:
    runs: list[dict[str, Any]]

    def run(self, code: str, run_index: int) -> ExecResult:
        line = self.runs[run_index - 1]
        if line["code"] != code:
            raise ValueError(f"run {run_index}: replayed code differs from the recorded code")
        return ExecResult(line["out"], line["truncated"], line["err"], line["exit"], line["timed_out"], line["s"])


def policy_prepared(source_root: Path, record: dict[str, Any]) -> IntentPrepared:
    inputs = record["inputs"]
    compare_path = repo_file(inputs["compare_path"], source_root)
    config_path = repo_file(inputs["config_path"], source_root)
    spec_path = repo_file(inputs["spec_path"], source_root)
    compare = load_compare(compare_path, source_root)
    config = load_config(config_path, paths.ROOT)
    episode = load_episode(compare)
    checkpoint = read_checkpoint(episode.checkpoint_path)
    spec = load_spec(spec_path)
    check_against_tensors(spec, checkpoint.tensors, checkpoint.sha256)
    policy = build_policy(spec, checkpoint.tensors)
    arrays = episode.arrays
    trace = trace_episode(policy, arrays["obs"])
    validation = validate_trace(trace, policy, arrays["logits"], arrays["gru_out"], arrays["actions"], arrays["rnn_in"])
    sessions = plan_sessions(episode, compare)
    return IntentPrepared(
        compare, config, config_path, spec, spec_path, checkpoint, episode, policy, trace, validation, sessions
    )


def policy_run(source_root: Path, run_id: str) -> dict[str, str]:
    """``<session>/message`` and ``<session>/turn-NN`` -> rebuilt sha256, for every recorded session."""
    run_dir = source_root / "outputs" / "weightreading" / "intent" / "runs" / run_id
    prepared = policy_prepared(source_root, json.loads((run_dir / "run.json").read_text(encoding="utf-8")))
    cues = read_jsonl(source_root / "outputs" / "intent_compare" / run_id / "cues_B.jsonl")
    rebuilt = {}
    for planned in prepared.sessions:
        if planned.skipped:
            continue
        known = latest_answers(cues, planned.step).get(planned.aircraft)
        previous = None if known is None else PolicyPrevious(known[0], IntentAnswer(known[1], tuple(known[2])))
        message = message_for(prepared, planned, previous)
        rebuilt[f"{planned.name}/message"] = sha256_text(message)
        session_dir = run_dir / "sessions" / planned.name
        backend = RecordedBackend(read_jsonl(session_dir / "transcript.jsonl"))
        tool_calls = session_dir / "tool_calls.jsonl"
        executor = RecordedExecutor(read_jsonl(tool_calls) if tool_calls.exists() else [])
        with tempfile.TemporaryDirectory() as scratch:
            run_session(
                backend,
                executor,
                message,
                session_limits_for(prepared.config),
                RunLog(Path(scratch)),
                parse_intent_reply,
            )
        rebuilt.update({f"{planned.name}/turn-{index:02d}": digest for index, digest in enumerate(backend.sent, 1)})
    return rebuilt


def policy_recorded(source_root: Path, run_id: str) -> dict[str, str]:
    run_dir = source_root / "outputs" / "weightreading" / "intent" / "runs" / run_id
    recorded = {}
    for session_json in sorted(run_dir.glob("sessions/s[0-9]*_*/session.json")):
        if ".aborted-" in session_json.parent.name:
            continue
        name = session_json.parent.name
        recorded[f"{name}/message"] = json.loads(session_json.read_text(encoding="utf-8"))["message_sha256"]
        for index, line in enumerate(read_jsonl(session_json.parent / "transcript.jsonl"), 1):
            recorded[f"{name}/turn-{index:02d}"] = line["request"]["input_sha256"]
    return recorded


# --- comparison -----------------------------------------------------------------------------------


def recorded_hashes(source_root: Path) -> dict[str, dict[str, str]]:
    found = {f"trajectory/{run}": trajectory_recorded(source_root, run) for run in RUN_IDS}
    return {**found, **{f"policy/{run}": policy_recorded(source_root, run) for run in RUN_IDS}}


def rebuilt_hashes(source_root: Path) -> dict[str, dict[str, str]]:
    found = {f"trajectory/{run}": trajectory_run(source_root, run) for run in RUN_IDS}
    return {**found, **{f"policy/{run}": policy_run(source_root, run) for run in RUN_IDS}}


def compare(baseline: dict[str, dict[str, str]], rebuilt: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    return {
        run: [key for key, digest in expected.items() if rebuilt.get(run, {}).get(key) != digest]
        for run, expected in baseline.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True, help="the checkout whose outputs/ hold the runs")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline["hashes"] != recorded_hashes(source_root):
        print(json.dumps({"ok": False, "error": "the baseline hashes differ from the recordings"}))
        return EXIT_MISMATCH
    known = baseline["known_unreproduced"]
    mismatches = compare(baseline["hashes"], rebuilt_hashes(source_root))
    checked = sum(len(hashes) for run, hashes in baseline["hashes"].items() if run not in known)
    failed = {run: keys for run, keys in mismatches.items() if keys and run not in known}
    ok = checked > 0 and not failed
    report = {"ok": ok, "checked": checked, "failed": failed, "known_unreproduced": sorted(known)}
    print(json.dumps({**report, "mismatches": mismatches}))
    return EXIT_OK if ok else EXIT_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
