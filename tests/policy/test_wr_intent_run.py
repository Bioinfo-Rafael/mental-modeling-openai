"""Policy-input pipeline: preview, paid gate, sandbox refusal, skips, and chaining."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.sandbox.test_wr_sandbox import LIMITS as SANDBOX_LIMITS
from tests.sandbox.test_wr_session import CODE, FakeBackend, FakeExecutor
from tests.translate.tr_fakes import fake_translation_renderer
from translate import cli
from translate.events import EpisodeRef
from translate.inputs import RunInputs
from translate.policy.bundle import verify_bundle
from translate.policy.material import load_policy_material
from translate.policy.session_inputs import PRELOAD, write_inputs
from translate.prompts import ConfiguredTurns
from translate.sandbox.audit import session_flags
from translate.sandbox.executor import wrap_code
from translate.sandbox.preflight import PreflightResult
from translate.sandbox.session import RunLog, SessionLimits, SessionState, run_session
from translate.scenario import episode_facts
from translate.triggers import plan_schedule, run_triggers, tick_interval

from .policy_fakes import AIRCRAFT, SEED, policy_config, synthetic_scenario, write_rollout
from .test_wr_cli_gate import ScriptedClient

ANSWER = '```json\n{"intent": "Hold course and close in.", "evidence": ["heads.logits_1 favours hold"]}\n```'
INVALID = '```json\n{"intent": "x", "evidence": []}\n```'
PASSED = PreflightResult(True, {"all": True}, "2.0", 0, "")
FAILED = PreflightResult(False, {"network_blocked": False}, None, 0, "")


class Factory:
    def __init__(self, client: object = None) -> None:
        self.calls = 0
        self.client = client

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        return self.client


@dataclass(frozen=True)
class PolicyHarness:
    inputs: cli.Inputs
    config_path: Path


def _frame() -> pd.DataFrame:
    rows = []
    for step in range(8):
        for aircraft in AIRCRAFT:
            rows.append(
                {
                    "episode_id": "synthetic",
                    "entity_type": "aircraft",
                    "has_csv_record": 1,
                    "step": step,
                    "object_id": aircraft,
                    "team": aircraft[0],
                    "alive": 0 if aircraft == "A0200" and step >= 5 else 1,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["csv_sha256"] = "synthetic-csv-sha256"
    return frame


@pytest.fixture
def policy_harness(small_files: tuple[Path, Path], tmp_path: Path) -> PolicyHarness:
    checkpoint, spec = small_files
    rollout = tmp_path / "rollout"
    write_rollout(rollout, checkpoint, spec)
    triggers = {
        "source": "preset",
        "tick_interval_steps": None,
        "points": [
            {"step": 3, "kind": "initial", "targets": None},
            {"step": 6, "kind": "change", "targets": None},
            {"step": 7, "kind": "change", "targets": [{"kind": "aircraft", "id": "A0100"}]},
        ],
    }
    config = policy_config(rollout, spec, triggers=triggers)
    material = load_policy_material(rollout, SEED, spec)
    episode_csv = tmp_path / "episode.csv"
    episode_csv.write_text("synthetic\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text("schema_version: 2\n", encoding="utf-8")
    scenario = replace(
        synthetic_scenario(),
        episode=EpisodeRef("synthetic", "synthetic", episode_csv, None),
    )
    frame = _frame()
    facts = episode_facts(frame)
    plan = plan_schedule(scenario, facts, run_triggers(scenario, config), tick_interval(config))
    inputs = cli.Inputs(config, scenario_path, scenario, frame, facts, plan, None, material)
    return PolicyHarness(inputs, config.path)


def _args(harness: PolicyHarness, output_root: Path, *extra: str, name: str = "run1") -> list[str]:
    return [
        "--config",
        str(harness.config_path),
        "--output-root",
        str(output_root),
        "--run-name",
        name,
        *extra,
    ]


def _patch_inputs(monkeypatch: pytest.MonkeyPatch, harness: PolicyHarness) -> None:
    translator = cli.Translator
    run_inputs = RunInputs(
        harness.inputs.frame,
        harness.inputs.scenario,
        harness.inputs.facts,
        harness.inputs.config,
        fake_translation_renderer,
        harness.inputs.policy,
    )
    monkeypatch.setattr(cli, "load_config", lambda path: harness.inputs.config)
    monkeypatch.setattr(cli, "load_inputs", lambda config, scenario: harness.inputs)
    monkeypatch.setattr(cli, "_run_inputs", lambda inputs: run_inputs)
    monkeypatch.setattr(
        cli,
        "Translator",
        lambda **kwargs: translator(**kwargs, translation_renderer=fake_translation_renderer),
    )


def test_policy_preview_writes_no_request_or_response_journals(
    policy_harness: PolicyHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, policy_harness)
    output = tmp_path / "preview"

    assert cli.main(_args(policy_harness, output)) == cli.EXIT_OK

    run_dir = output / "run1"
    assert (run_dir / "preview.json").is_file()
    assert not (run_dir / "requests.jsonl").exists()
    assert not (run_dir / "responses.jsonl").exists()
    preview = json.loads((run_dir / "preview.json").read_text(encoding="utf-8"))
    assert preview["api_requests_made"] == 0
    assert preview["plan"]["mandatory_translations"] == 4
    assert preview["plan"]["suppressed_triggers"] == 1


@pytest.mark.parametrize("flags", [["--execute"], ["--confirm-paid-api"]])
def test_intent_run_with_one_paid_flag_is_an_argparse_error(
    policy_harness: PolicyHarness, tmp_path: Path, flags: list[str]
) -> None:
    factory = Factory()

    with pytest.raises(SystemExit) as raised:
        cli.main(
            _args(policy_harness, tmp_path / "out", *flags),
            factories=cli.Factories(openai_client=factory),
        )

    assert raised.value.code == cli.EXIT_USAGE
    assert factory.calls == 0


def test_intent_run_without_flags_is_a_preview(
    policy_harness: PolicyHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, policy_harness)
    factory = Factory()

    assert (
        cli.main(_args(policy_harness, tmp_path / "out"), factories=cli.Factories(openai_client=factory)) == cli.EXIT_OK
    )
    assert factory.calls == 0


def test_intent_run_with_failed_preflight_refuses_before_client(
    policy_harness: PolicyHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, policy_harness)
    factory = Factory()

    code = cli.main(
        _args(policy_harness, tmp_path / "out", "--execute", "--confirm-paid-api"),
        factories=cli.Factories(openai_client=factory, preflight=lambda *_: FAILED),
    )

    assert code == cli.EXIT_FAILED
    assert factory.calls == 0
    assert not (tmp_path / "out").exists()


def test_intent_run_with_fakes_chains_previous_and_writes_cues(
    policy_harness: PolicyHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inputs(monkeypatch, policy_harness)
    client = ScriptedClient([ANSWER, "no answer", "still none", "nope", INVALID, ANSWER])
    factory = Factory(client)
    output = tmp_path / "execute"

    code = cli.main(
        _args(policy_harness, output, "--execute", "--confirm-paid-api"),
        factories=cli.Factories(openai_client=factory, preflight=lambda *_: PASSED),
    )

    run_dir = output / "run1"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (code, factory.calls, record.get("error")) == (cli.EXIT_OK, 1, None)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    translated = [event for event in events if event["record_type"] == "translation_event"]
    assert [(event["step"], event["target"]["id"]) for event in translated] == [
        (3, "A0100"),
        (3, "A0200"),
        (6, "A0100"),
        (7, "A0100"),
    ]
    assert translated[1]["output"] == {"intent": "", "evidence": []}
    assert "session ended without a final answer" in translated[1]["answer_problem"]
    assert translated[2]["output"] == {"intent": "", "evidence": []}
    assert "invalid final answer" in translated[2]["answer_problem"]
    chained = (run_dir / "sessions" / "s0007_A0100" / "message.txt").read_text(encoding="utf-8")
    assert "From step 3:" in chained and "Hold course and close in." in chained
    cues = [json.loads(line) for line in (run_dir / "cues.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum("skipped" in cue for cue in cues) == 3
    assert all(request["model"] == "fake-model" for request in client.requests)
    assert all(request["reasoning"] == {"effort": "high"} for request in client.requests)
    assert all(request["max_output_tokens"] == 25_000 for request in client.requests)
    assert all("tools" not in request for request in client.requests)


def test_invalid_final_answer_is_terminal_and_keeps_the_last_valid_chain(
    policy_harness: PolicyHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_intent_run_with_fakes_chains_previous_and_writes_cues(policy_harness, tmp_path, monkeypatch)


def test_soft_token_budget_asks_for_the_answer_instead_of_running_code(tmp_path: Path) -> None:
    backend = FakeBackend([CODE, CODE, ANSWER], tokens_per_turn=100)
    executor = FakeExecutor()
    limits = SessionLimits(max_runs=10, max_total_tokens=10_000, max_protocol_errors=2, answer_at_total_tokens=300)
    turns = ConfiguredTurns(policy_config(Path("rollout"), Path("spec")).prompt.turns)

    state = run_session(backend, executor, "go", limits, RunLog(tmp_path), turns)

    assert state.stop_reason == "final_answer" and state.runs == 1 and state.budget_notice_sent
    assert not limits.answer_now(SessionState(runs=9)) and limits.answer_now(SessionState(runs=10))


def test_wrap_code_preloads_activations_only_when_asked() -> None:
    plain = wrap_code("print(1)", "<run 1>", SANDBOX_LIMITS)
    with_a = wrap_code("print(1)", "<run 1>", SANDBOX_LIMITS, PRELOAD)

    assert "'/work/weights/activations/index.json'" in with_a and "'A': A" in with_a
    assert "activations" not in plain and "'A'" not in plain


def test_write_inputs_is_exclusive_and_verifiable(small_files, tmp_path: Path) -> None:
    from translate.policy.torch_reader import read_checkpoint

    tensors = read_checkpoint(small_files[0]).tensors
    arrays = {"step": np.arange(3, dtype=np.int64), "x": np.zeros((3, 2), dtype=np.float32)}

    inputs = write_inputs(tmp_path / "inputs", tensors, arrays, {"aircraft": "A0100"})

    assert verify_bundle(inputs.bundle.path, inputs.bundle.manifest_sha256)
    assert json.loads((tmp_path / "inputs" / "weights" / "activations" / "index.json").read_text()) == [
        "step",
        "x",
    ]
    with pytest.raises(FileExistsError):
        write_inputs(tmp_path / "inputs", tensors, arrays, {})


def test_intent_flags_mark_writes_into_a_and_synthetic_vectors() -> None:
    code = "A['x'][0] = 0\nv = np.zeros(33)\nW.update({})\ns = A['x'].mean(0)\nimport autograd.numpy\n"

    rules = [(flag.line, flag.rule) for flag in session_flags(1, code)]

    assert rules == [
        (1, "intent:writes_recorded_input"),
        (2, "intent:synthetic_array"),
        (3, "intent:writes_recorded_input"),
        (5, "intent:autodiff_import"),
    ]
