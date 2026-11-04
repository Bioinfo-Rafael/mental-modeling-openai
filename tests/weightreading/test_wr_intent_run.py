"""Method B end to end on a synthetic recording: preview, paid gate, skips, chaining, cues, sandbox prelude."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from weightreading import cli, intent_execute
from weightreading.bundle import verify_bundle
from weightreading.intent_audit import intent_flags
from weightreading.intent_inputs import PRELOAD, write_inputs
from weightreading.preflight import PreflightResult
from weightreading.sandbox import wrap_code
from weightreading.session import RunLog, SessionLimits, SessionState, run_session

from .test_wr_cli_gate import ScriptedClient, _fake_executor
from .test_wr_sandbox import LIMITS as SANDBOX_LIMITS
from .test_wr_session import CODE, FakeBackend, FakeExecutor
from .wr_intent_helpers import DEAD_FROM, STEPS, IntentFiles, write_intent_files

ANSWER = '```json\n{"intent": "Hold course and close in.", "evidence": ["heads.logits_1 favours hold"]}\n```'
PASSED = PreflightResult(True, {"all": True}, "2.0", 0, "")
FAILED = PreflightResult(False, {"network_blocked": False}, None, 0, "")


class Factory:
    def __init__(self, client: object = None) -> None:
        self.calls = 0
        self.client = client

    def __call__(self) -> object:
        self.calls += 1
        return self.client


@pytest.fixture
def intent_files(small_files, tmp_path: Path) -> IntentFiles:
    checkpoint, spec = small_files
    return write_intent_files(tmp_path, checkpoint, spec)


def _args(files: IntentFiles, command: str, *extra: str) -> list[str]:
    return [
        command,
        "--compare", str(files.compare),
        "--config", str(files.config),
        "--spec", str(files.spec),
        "--output-root", str(files.output_root),
        *extra,
    ]  # fmt: skip


def _fake_sandbox(monkeypatch, preflight: PreflightResult) -> FakeExecutor:
    executor = _fake_executor()
    monkeypatch.setattr(intent_execute, "_session_executor", lambda *_: executor)
    monkeypatch.setattr(intent_execute, "run_preflight", lambda *_: preflight)
    return executor


def test_intent_preview_writes_messages_inputs_and_skips_dead_target(intent_files: IntentFiles, capsys) -> None:
    code = cli.main(_args(intent_files, "intent-preview"), client_factory=Factory())

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    preview = Path(printed["preview_dir"])
    payload = json.loads((preview / "preview.json").read_text())
    assert payload["api_requests_made"] == 0 and payload["trace_validation"]["ok"]
    assert payload["totals"] == {**payload["totals"], "sessions": 3, "skipped": 1}
    skipped = [entry for entry in payload["sessions"] if entry["skipped"]]
    assert [(entry["step"], entry["aircraft"]) for entry in skipped] == [(STEPS[1], "A0200")]
    assert f"not alive at step {STEPS[1]}" in skipped[0]["skipped"] and STEPS[1] >= DEAD_FROM
    later = (preview / "sessions" / f"s{STEPS[1]:04d}_A0100" / "message.txt").read_text()
    assert "[PREVIEW: method B's previous intent" in later
    inputs = preview / "sessions" / f"s{STEPS[0]:04d}_A0100" / "inputs"
    verify_bundle(inputs)
    window = np.load(inputs / "weights" / "activations" / "step.npy")
    np.testing.assert_array_equal(window, [0, 1, 2, 3])
    assert payload["inputs"]["compare_sha256"] and payload["inputs"]["rollout_sha256"]
    assert not (intent_files.cues_root).exists()


@pytest.mark.parametrize("flags", [["--execute"], ["--confirm-paid-api"]])
def test_intent_run_with_one_paid_flag_is_an_argparse_error(intent_files: IntentFiles, flags: list[str]) -> None:
    factory = Factory()

    with pytest.raises(SystemExit) as raised:
        cli.main(_args(intent_files, "intent-run", *flags), client_factory=factory)

    assert raised.value.code == 2 and factory.calls == 0


def test_intent_run_without_flags_is_a_preview(intent_files: IntentFiles) -> None:
    factory = Factory()

    assert cli.main(_args(intent_files, "intent-run"), client_factory=factory) == 0
    assert factory.calls == 0 and not (intent_files.output_root / "intent" / "runs").exists()


def test_intent_run_with_failed_preflight_refuses_before_client(intent_files: IntentFiles, monkeypatch) -> None:
    _fake_sandbox(monkeypatch, FAILED)
    factory = Factory()

    code = cli.main(_args(intent_files, "intent-run", "--run-id", "r1", "--execute", "--confirm-paid-api"), factory)

    assert code == 1 and factory.calls == 0
    record = json.loads((intent_files.output_root / "intent" / "runs" / "r1" / "run.json").read_text())
    assert "refused" in record["status"]


def test_intent_run_with_fakes_chains_previous_and_writes_cues(intent_files: IntentFiles, monkeypatch) -> None:
    executor = _fake_sandbox(monkeypatch, PASSED)
    client = ScriptedClient([CODE, ANSWER, "no answer at all", "still none", "nope", ANSWER])
    factory = Factory(client)

    code = cli.main(_args(intent_files, "intent-run", "--run-id", "r2", "--execute", "--confirm-paid-api"), factory)

    assert code == 0 and factory.calls == 1
    cues = [json.loads(line) for line in (intent_files.cues_root / "r2" / "cues_B.jsonl").read_text().splitlines()]
    assert [(cue["step"], cue["aircraft"]) for cue in cues] == [(3, "A0100"), (3, "A0200"), (6, "A0100"), (6, "A0200")]
    assert cues[0]["intent"] == "Hold course and close in." and cues[0]["model"] == "fake-model"
    assert "without a final answer (protocol_errors)" in cues[1]["skipped"] and "intent" not in cues[1]
    assert cues[2]["intent"] and "not alive" in cues[3]["skipped"]
    run_dir = intent_files.output_root / "intent" / "runs" / "r2"
    chained = (run_dir / "sessions" / "s0006_A0100" / "message.txt").read_text()
    assert "From step 3:" in chained and "Hold course and close in." in chained
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"].startswith("complete: 1 session(s)") and record["inputs"]["checkpoint_sha256"]
    assert "weightreading/intent_run.py" in record["code_sha256"]
    audit = json.loads((run_dir / "sessions" / "s0003_A0100" / "audit.json").read_text())
    assert audit["mode"] == "intent-b" and audit["human_verdict"] is None
    assert executor.calls == [("print(len(W))\n", 1)]
    assert all("tools" not in request for request in client.requests)


def test_intent_run_refuses_model_mismatch_with_compare(intent_files: IntentFiles, capsys) -> None:
    text = intent_files.config.read_text().replace("model: fake-model", "model: other-model")
    intent_files.config.write_text(text)

    assert cli.main(_args(intent_files, "intent-preview"), client_factory=Factory()) == 1
    assert "must equal" in capsys.readouterr().err


def test_soft_token_budget_asks_for_the_answer_instead_of_running_code(tmp_path: Path) -> None:
    from weightreading.intent_protocol import parse_intent_reply

    backend = FakeBackend([CODE, CODE, ANSWER], tokens_per_turn=100)
    executor = FakeExecutor()
    limits = SessionLimits(max_runs=10, max_total_tokens=10_000, max_protocol_errors=2, answer_at_total_tokens=300)

    state = run_session(backend, executor, "go", limits, RunLog(tmp_path), parse_intent_reply)

    assert state.stop_reason == "final_answer" and state.runs == 1 and state.budget_notice_sent
    assert not limits.answer_now(SessionState(runs=9)) and limits.answer_now(SessionState(runs=10))


def test_wrap_code_preloads_activations_only_when_asked() -> None:
    plain = wrap_code("print(1)", "<run 1>", SANDBOX_LIMITS)
    with_a = wrap_code("print(1)", "<run 1>", SANDBOX_LIMITS, PRELOAD)

    assert "'/work/weights/activations/index.json'" in with_a and "'A': A" in with_a
    assert "activations" not in plain and "'A'" not in plain


def test_write_inputs_is_exclusive_and_verifiable(small_files, tmp_path: Path) -> None:
    from weightreading.torch_reader import read_checkpoint

    tensors = read_checkpoint(small_files[0]).tensors
    arrays = {"step": np.arange(3, dtype=np.int64), "x": np.zeros((3, 2), dtype=np.float32)}

    inputs = write_inputs(tmp_path / "inputs", tensors, arrays, {"aircraft": "A0100"})

    assert verify_bundle(inputs.bundle.path, inputs.bundle.manifest_sha256)
    assert json.loads((tmp_path / "inputs" / "weights" / "activations" / "index.json").read_text()) == ["step", "x"]
    with pytest.raises(FileExistsError):
        write_inputs(tmp_path / "inputs", tensors, arrays, {})


def test_intent_flags_mark_writes_into_a_and_synthetic_vectors() -> None:
    code = "A['x'][0] = 0\nv = np.zeros(33)\nW.update({})\ns = A['x'].mean(0)\nimport autograd.numpy\n"

    rules = [(flag.line, flag.rule) for flag in intent_flags(1, code)]

    assert rules == [
        (1, "intent:writes_recorded_input"),
        (2, "intent:synthetic_array"),
        (3, "intent:writes_recorded_input"),
        (5, "intent:autodiff_import"),
    ]


def test_intent_run_default_run_id_matches_method_a_and_refuses_existing_cues(intent_files, monkeypatch) -> None:
    _fake_sandbox(monkeypatch, PASSED)
    (intent_files.cues_root / "seed7").mkdir(parents=True)
    (intent_files.cues_root / "seed7" / "cues_B.jsonl").write_text("")
    factory = Factory(ScriptedClient([]))

    code = cli.main(_args(intent_files, "intent-run", "--execute", "--confirm-paid-api"), factory)

    assert code == 1 and factory.calls == 0
    assert (intent_files.output_root / "intent" / "runs" / "seed7").is_dir()


def test_invalid_final_answer_is_terminal_and_keeps_the_last_valid_chain(intent_files, monkeypatch) -> None:
    _fake_sandbox(monkeypatch, PASSED)
    invalid = '```json\n{"intent": "x", "evidence": []}\n```'
    factory = Factory(ScriptedClient([ANSWER, invalid, invalid]))

    assert (
        cli.main(_args(intent_files, "intent-run", "--run-id", "r3", "--execute", "--confirm-paid-api"), factory) == 0
    )

    cues = [json.loads(line) for line in (intent_files.cues_root / "r3" / "cues_B.jsonl").read_text().splitlines()]
    assert "invalid final answer" in cues[1]["skipped"] and "invalid final answer" in cues[2]["skipped"]
    later = (
        intent_files.output_root / "intent" / "runs" / "r3" / "sessions" / "s0006_A0100" / "message.txt"
    ).read_text()
    assert "From step 3:" in later
    record = json.loads((intent_files.output_root / "intent" / "runs" / "r3" / "run.json").read_text())
    assert record["status"].startswith("complete: 2 session(s)")


def test_resume_redoes_the_interrupted_session_and_rebuilds_the_chain(intent_files, monkeypatch) -> None:
    _fake_sandbox(monkeypatch, PASSED)

    class Exploding(ScriptedClient):
        def create(self, **kwargs: object) -> object:
            if not self.texts:
                raise RuntimeError("network down")
            return super().create(**kwargs)

    args = _args(intent_files, "intent-run", "--run-id", "r4", "--execute", "--confirm-paid-api")
    with pytest.raises(RuntimeError):
        cli.main(args, Factory(Exploding([ANSWER])))
    run_dir = intent_files.output_root / "intent" / "runs" / "r4"
    assert "error: RuntimeError" in json.loads((run_dir / "run.json").read_text())["status"]

    client = ScriptedClient([ANSWER, ANSWER])
    assert cli.main([*args, "--resume"], Factory(client)) == 0

    cues = [json.loads(line) for line in (intent_files.cues_root / "r4" / "cues_B.jsonl").read_text().splitlines()]
    assert [(cue["step"], cue["aircraft"]) for cue in cues] == [(3, "A0100"), (3, "A0200"), (6, "A0100"), (6, "A0200")]
    assert any(path.name.startswith("s0003_A0200.aborted-") for path in (run_dir / "sessions").iterdir())
    assert "From step 3:" in (run_dir / "sessions" / "s0006_A0100" / "message.txt").read_text()
    assert len(client.requests) == 2 and list(run_dir.glob("run.previous-*.json"))


def test_resume_refuses_changed_inputs(intent_files, monkeypatch) -> None:
    _fake_sandbox(monkeypatch, PASSED)
    args = _args(intent_files, "intent-run", "--run-id", "r5", "--execute", "--confirm-paid-api")
    assert cli.main(args, Factory(ScriptedClient([ANSWER, ANSWER, ANSWER]))) == 0
    intent_files.compare.write_text(intent_files.compare.read_text().replace("window_steps: 4", "window_steps: 3"))

    assert cli.main([*args, "--resume"], Factory(ScriptedClient([]))) == 1


def test_intent_preview_with_ja_compare_requests_japanese_and_records_language(
    intent_files: IntentFiles, capsys
) -> None:
    raw = yaml.safe_load(intent_files.compare.read_text(encoding="utf-8"))
    ja_compare = intent_files.compare.with_name("compare.ja.yaml")
    ja_compare.write_text(yaml.safe_dump({**raw, "output_language": "ja"}), encoding="utf-8")

    assert cli.main(_args(intent_files, "intent-preview"), client_factory=Factory()) == 0
    english_dir = Path(json.loads(capsys.readouterr().out)["preview_dir"])
    # A separate output root: preview directories are named with a one-second timestamp.
    ja_files = replace(intent_files, compare=ja_compare, output_root=intent_files.output_root.with_name("wr_ja"))
    assert cli.main(_args(ja_files, "intent-preview"), client_factory=Factory()) == 0
    japanese_dir = Path(json.loads(capsys.readouterr().out)["preview_dir"])

    session = f"s{STEPS[0]:04d}_A0100"
    english = (english_dir / "sessions" / session / "message.txt").read_text()
    japanese = (japanese_dir / "sessions" / session / "message.txt").read_text()
    assert "Write both fields in English." in english and "natural Japanese" not in english
    assert "Write both fields in natural Japanese" in japanese
    assert json.loads((japanese_dir / "preview.json").read_text()).get("output_language") == "ja"
    assert json.loads((english_dir / "preview.json").read_text()).get("output_language") == "en"
