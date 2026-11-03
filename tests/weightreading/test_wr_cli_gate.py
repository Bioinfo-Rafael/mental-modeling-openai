"""CLI: paid-call gate, preview path, refusal on failed preflight, full run with fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weightreading import cli, commands
from weightreading.backends import make_openai_client
from weightreading.preflight import PreflightResult
from weightreading.sandbox import Interpreter

from .test_wr_session import CODE, FINAL, FakeExecutor

DUMMY_INTERPRETER = Interpreter("/opt/python", "bin/python3", "/opt/site/numpy", "", "2.0", "3.12.0")


class ScriptedClient:
    """Stands in for ``openai.OpenAI``: ``responses.create`` returns scripted replies, no network."""

    def __init__(self, texts: list[str]) -> None:
        self.responses = self
        self.texts = texts
        self.requests: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        text = self.texts.pop(0)
        usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 5})()
        return type(
            "Response", (), {"id": "r", "status": "completed", "output_text": text, "output": [], "usage": usage}
        )()


def _fake_executor() -> FakeExecutor:
    executor = FakeExecutor()
    executor.interp = DUMMY_INTERPRETER
    return executor


class RecordingFactory:
    def __init__(self, client: object = None) -> None:
        self.calls = 0
        self.client = client

    def __call__(self) -> object:
        self.calls += 1
        return self.client


def _args(small_files: tuple[Path, Path], out: Path, *extra: str) -> list[str]:
    checkpoint, spec = small_files
    return ["run", "--checkpoint", str(checkpoint), "--spec", str(spec), "--output-root", str(out), *extra]


@pytest.mark.parametrize("flags", [["--execute"], ["--confirm-paid-api"]])
def test_run_with_only_one_paid_flag_is_an_argparse_error(small_files, tmp_path: Path, flags: list[str]) -> None:
    factory = RecordingFactory()

    with pytest.raises(SystemExit) as raised:
        cli.main(_args(small_files, tmp_path / "out", *flags), client_factory=factory)

    assert raised.value.code == 2
    assert factory.calls == 0
    assert not (tmp_path / "out").exists()


def test_run_without_flags_writes_preview_and_builds_no_client(small_files, tmp_path: Path) -> None:
    factory = RecordingFactory()

    code = cli.main(_args(small_files, tmp_path / "out"), client_factory=factory)

    assert code == 0 and factory.calls == 0
    preview_dir = next((tmp_path / "out" / "previews").iterdir())
    assert (preview_dir / "message.txt").exists()
    assert json.loads((preview_dir / "preview.json").read_text())["api_requests_made"] == 0
    assert len(list((tmp_path / "out" / "bundles" / "net-0123abcd" / "weights").glob("*.npy"))) == 18
    assert not (tmp_path / "out" / "runs").exists()


def test_run_execute_with_failed_preflight_refuses_before_client(small_files, tmp_path, monkeypatch) -> None:
    factory = RecordingFactory()
    failed = PreflightResult(False, {"network_blocked": False}, None, 0, "")
    monkeypatch.setattr(commands, "make_executor", lambda *_: _fake_executor())
    monkeypatch.setattr(commands, "run_preflight", lambda *_: failed)

    code = cli.main(_args(small_files, tmp_path / "out", "--execute", "--confirm-paid-api"), client_factory=factory)

    assert code == 1 and factory.calls == 0
    run_dir = next((tmp_path / "out" / "runs").iterdir())
    assert "refused" in json.loads((run_dir / "run.json").read_text())["status"]


def test_run_execute_with_fakes_produces_complete_run_dir(small_files, tmp_path, monkeypatch) -> None:
    executor = _fake_executor()
    passed = PreflightResult(True, {"all": True}, "2.0", 0, "")
    monkeypatch.setattr(commands, "make_executor", lambda *_: executor)
    monkeypatch.setattr(commands, "run_preflight", lambda *_: passed)
    client = ScriptedClient([CODE, FINAL])
    factory = RecordingFactory(client)

    code = cli.main(
        _args(small_files, tmp_path / "out", "--execute", "--confirm-paid-api", "--task-mode", "blind"),
        client_factory=factory,
    )

    assert code == 0 and factory.calls == 1
    run_dir = next((tmp_path / "out" / "runs").iterdir())
    assert run_dir.name.endswith("_net-0123abcd_blind")
    for name in ["message.txt", "transcript.jsonl", "tool_calls.jsonl", "state.json", "final_answer.md", "audit.json"]:
        assert (run_dir / name).exists(), name
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "complete"
    assert record["session"]["stop_reason"] == "final_answer"
    assert record["inputs"]["checkpoint_sha256"]
    assert "weightreading/torch_reader.py" in record["code_sha256"]
    assert json.loads((run_dir / "audit.json").read_text())["human_verdict"] is None
    assert all("tools" not in request for request in client.requests)
    assert executor.calls == [("print(len(W))\n", 1)]


def test_make_openai_client_without_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_openai_client()


def test_inspect_with_tensor_hashes_prints_hash_map(small_files, capsys) -> None:
    code = cli.main(["inspect", str(small_files[0]), "--tensor-hashes"])

    hashes = json.loads(capsys.readouterr().out)
    assert code == 0
    assert set(hashes) >= {"base.mlp.fc.0.weight", "rnn.gru.weight_hh_l0"}


def test_validate_spec_with_mismatching_checkpoint_exits_nonzero(small_files, tmp_path, capsys) -> None:
    spec = json.loads(small_files[1].read_text())
    spec["inputs"].append({"index": 3, "name": "extra", "definition": "d"})
    spec["architecture"][0]["input_size"] = 4
    bad = tmp_path / "bad.spec.json"
    bad.write_text(json.dumps(spec))

    code = cli.main(["validate-spec", "--checkpoint", str(small_files[0]), "--spec", str(bad)])

    assert code == 1
    assert "base.mlp.fc.0.weight" in capsys.readouterr().err
