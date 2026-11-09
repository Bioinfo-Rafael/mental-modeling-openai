#!/usr/bin/env python3
"""Offline replay of recorded translation runs through the current pipeline (restructure guard; no API call).

Each recorded comparison run is re-run end to end with the shipped run config of its kind, the
model replaced by its recorded replies:

- trajectory input (``outputs/intent_compare/<run>/method_a``): the ``seed100601-trajectory*``
  config with the prompt it was recorded with (``translator-intent-v1``, without the alarm and
  question inputs that v1 did not have); the recorded responses answer in order, and the SHA-256
  of every journaled request body must equal the recorded one;
- policy input (``outputs/weightreading/intent/runs/<run>``): the ``seed100601-policy*`` config;
  every code session is driven by its recorded replies and sandbox results, and the SHA-256 of the
  first message and of the conversation history sent at every turn must equal the recorded ones.

    python scripts/check_request_replay.py --source-root <checkout>

The baseline ``tests/fixtures/request-replay-baseline.json`` holds the recorded hashes (the check
first confirms the recordings still hold exactly these) and ``known_unreproduced``, which may
only shrink. Exit codes: 0 ok; 1 a hash outside ``known_unreproduced`` was not reproduced.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from translate import paths
from translate.call_one_shot import IntentModelOutput, OneShotCall, TranslationPrompt, build_openai_request
from translate.call_session import SessionCall
from translate.cli import session_settings
from translate.config import RunConfig, parse_config
from translate.events import canonical_json, sha256_text
from translate.lag.dataset import load_episode
from translate.llm.openai_responses import build_responses_body
from translate.pipeline import Translator
from translate.policy.material import load_policy_material
from translate.sandbox.executor import ExecResult
from translate.scenario import load_scenario
from translate.session_backend import BackendReply, Usage, history_sha256

EXIT_OK, EXIT_MISMATCH = 0, 1
RUN_IDS = ("seed100601", "seed100601-ja", "seed100601-ja-short")
CONFIG_SUFFIX = {"seed100601": "", "seed100601-ja": "-ja", "seed100601-ja-short": "-ja-short"}
BASELINE = paths.ROOT / "tests" / "fixtures" / "request-replay-baseline.json"
TRANSLATE = paths.CONFIGS / "translate"
SCENARIO = TRANSLATE / "scenarios" / "seed100601-neutral.yaml"
V1_PROMPT = TRANSLATE / "prompts" / "translator-intent-v1.yaml"
V1_INPUTS = ["instructions", "previous_translation", "trajectory"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MemorySink:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def write_event(self, record: dict[str, Any]) -> None:
        return None

    def write_request(self, record: dict[str, Any]) -> None:
        self.requests.append(record)

    def write_response(self, record: dict[str, Any]) -> None:
        return None


def scenario_file(source_root: Path, scratch: Path) -> Path:
    """The shipped scenario with its episode paths pointing into the source checkout."""
    raw = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    episode_dir = source_root / "outputs" / "intent_compare" / "episode" / "seed100601"
    episode = {**raw["episode"], "csv_path": str(episode_dir / "seed100601.csv"), "video_path": None}
    path = scratch / "scenario.yaml"
    path.write_text(yaml.safe_dump({**raw, "episode": episode}, sort_keys=False), encoding="utf-8")
    return path


def run_config(name: str, scratch: Path, source_root: Path, **sections: Any) -> RunConfig:
    path = TRANSLATE / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = {**raw, "scenario": str(scenario_file(source_root, scratch)), "output": {"root": str(scratch)}, **sections}
    return parse_config(raw, path=path, sha256="replay")


def translate(config: RunConfig, translator: Translator) -> MemorySink:
    scenario = load_scenario(config.scenario_path, limits=config.scenario_limits)
    sink = MemorySink()
    summary = translator.run(frame=load_episode(scenario.episode.csv_path), scenario=scenario, config=config, sink=sink)
    if summary.status != "completed":
        raise RuntimeError(f"replay run ended {summary.status}: {summary.error}")
    return sink


# --- trajectory input -----------------------------------------------------------------------------


@dataclass
class RecordedIntentModel:
    """Builds the real Responses body and answers with the recorded responses, in order."""

    config: RunConfig
    responses: list[dict[str, Any]]
    sent: int = 0

    def request_payload(self, prompt: TranslationPrompt) -> dict[str, Any]:
        translation = self.config.translation
        request = build_openai_request(
            prompt,
            model=translation.model,
            reasoning_effort=translation.reasoning_effort,
            max_output_tokens=translation.max_output_tokens,
        )
        return build_responses_body(request)

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        line = self.responses[self.sent]
        self.sent += 1
        messages = [item for item in line["raw"]["output"] if item.get("type") == "message"]
        text = messages[0]["content"][0]["text"]
        return IntentModelOutput(line["model"], line["response_id"], text, line["usage"], line["raw"])


def trajectory_run(source_root: Path, run_id: str) -> dict[str, str]:
    method_dir = source_root / "outputs" / "intent_compare" / run_id / "method_a"
    recorded = read_jsonl(method_dir / "requests.jsonl")
    with tempfile.TemporaryDirectory() as scratch:
        name = f"seed100601-trajectory{CONFIG_SUFFIX[run_id]}.yaml"
        base = yaml.safe_load((TRANSLATE / name).read_text(encoding="utf-8"))
        prompt = {"file": str(V1_PROMPT), "vars": base["prompt"]["vars"]}
        config = run_config(name, Path(scratch), source_root, prompt=prompt, inputs=V1_INPUTS)
        model = RecordedIntentModel(config, read_jsonl(method_dir / "responses.jsonl"))
        sink = translate(config, Translator(call=OneShotCall(model, config.prompt, config.answer.limits)))
    journaled = [sha256_text(canonical_json(line["payload"])) for line in sink.requests]
    return {line["call_id"]: digest for line, digest in zip(recorded, journaled, strict=False)}


def trajectory_recorded(source_root: Path, run_id: str) -> dict[str, str]:
    lines = read_jsonl(source_root / "outputs" / "intent_compare" / run_id / "method_a" / "requests.jsonl")
    return {line["call_id"]: sha256_text(canonical_json(line["payload"])) for line in lines}


# --- policy input ---------------------------------------------------------------------------------


@dataclass
class RecordedExecutor:
    runs: list[dict[str, Any]]

    def run(self, code: str, run_index: int) -> ExecResult:
        line = self.runs[run_index - 1]
        if line["code"] != code:
            raise ValueError(f"run {run_index}: replayed code differs from the recorded code")
        return ExecResult(line["out"], line["truncated"], line["err"], line["exit"], line["timed_out"], line["s"])


@dataclass
class RecordedSessions:
    """Serves every session's recorded replies and sandbox results, session by session, in run order."""

    session_dirs: list[Path]
    history: list[tuple[str, str]] = field(default_factory=list)  # (session name, history sha256)
    current: int = -1
    replies: list[dict[str, Any]] = field(default_factory=list)
    model: str = "recorded"

    def executor(self, work: Path, weights: Path) -> RecordedExecutor:
        self.current += 1
        directory = self.session_dirs[self.current]
        self.replies = read_jsonl(directory / "transcript.jsonl")
        tool_calls = directory / "tool_calls.jsonl"
        return RecordedExecutor(read_jsonl(tool_calls) if tool_calls.exists() else [])

    def respond(self, history: tuple[dict, ...]) -> BackendReply:
        name = self.session_dirs[self.current].name
        self.history.append((name, history_sha256(history)))
        line = self.replies[sum(1 for session, _ in self.history if session == name) - 1]
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


@dataclass(frozen=True)
class PassedPreflight:
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"ok": True, "replay": "preflight not run (no code is executed)"}


def policy_dirs(run_dir: Path) -> list[Path]:
    return sorted(p for p in (run_dir / "sessions").glob("s[0-9]*_*") if ".aborted-" not in p.name)


def policy_run(source_root: Path, run_id: str) -> dict[str, str]:
    run_dir = source_root / "outputs" / "weightreading" / "intent" / "runs" / run_id
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    sessions = RecordedSessions(policy_dirs(run_dir))
    with tempfile.TemporaryDirectory() as scratch:
        name = f"seed100601-policy{CONFIG_SUFFIX[run_id]}.yaml"
        base = yaml.safe_load((TRANSLATE / name).read_text(encoding="utf-8"))
        rollout = Path(record["inputs"]["rollout_path"]).parent
        config = run_config(name, Path(scratch), source_root, policy={**base["policy"], "rollout_dir": str(rollout)})
        policy = config.policy
        material = load_policy_material(policy.rollout_dir, policy.seed, policy.spec, policy.checkpoint)
        options = {"model": config.translation.model}  # only the history hash is compared
        call = SessionCall(
            sessions, options, config.prompt, session_settings(config, material), material, sessions.executor,
            config.answer.limits, lambda executor, hidden: PassedPreflight(),
        )  # fmt: skip
        sink = translate(config, Translator(call=call, policy=material, sessions_dir=Path(scratch) / "sessions"))
    rebuilt: dict[str, str] = {}
    for request in sink.requests:
        message = request["payload"]["input"][0]["content"][0]["text"]
        rebuilt[f"s{request['step']:04d}_{request['target']['id']}/message"] = sha256_text(message)
    for index, (session, digest) in enumerate(sessions.history):
        turn = sum(1 for name, _ in sessions.history[: index + 1] if name == session)
        rebuilt[f"{session}/turn-{turn:02d}"] = digest
    return rebuilt


def policy_recorded(source_root: Path, run_id: str) -> dict[str, str]:
    run_dir = source_root / "outputs" / "weightreading" / "intent" / "runs" / run_id
    recorded = {}
    for directory in policy_dirs(run_dir):
        session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        recorded[f"{directory.name}/message"] = session["message_sha256"]
        for index, line in enumerate(read_jsonl(directory / "transcript.jsonl"), 1):
            recorded[f"{directory.name}/turn-{index:02d}"] = line["request"]["input_sha256"]
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
