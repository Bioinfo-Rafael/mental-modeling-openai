"""File-backed prompts preserve requests, validate before sending, and record the text actually read."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from evaluation import cli
from evaluation.config import load_config
from evaluation.material import load_run
from translate.config import ConfigError
from translate.journal import request_sha256

from .fakes import JUDGE_CONFIG, Factory, event, judge_config, write_run


@pytest.fixture
def file_config(tmp_path: Path) -> Path:
    path = tmp_path / "config" / "judge.yaml"
    path.parent.mkdir()
    shutil.copytree(JUDGE_CONFIG.parent / "prompts", path.parent / "prompts")
    raw = yaml.safe_load(JUDGE_CONFIG.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump({**raw, "output": str(tmp_path / "evaluation")}), encoding="utf-8")
    return path


def _replace_prompt(path: Path, prompt: dict) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump({**raw, "prompt": prompt}), encoding="utf-8")


def _refuse_client(**_: object) -> None:
    raise AssertionError("preview/invalid config constructed an API client")


def test_files_and_inline_configs_build_identical_requests_from_any_cwd(
    tmp_path: Path, file_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inline = load_config(judge_config(tmp_path / "inline"))
    monkeypatch.chdir(tmp_path)
    external = load_config(file_config)
    records = [event(0), event(45, ("change",)), event(70, ("request",)), event(80, ("start", "request"))]
    records[-1]["input"]["previous_translation"] = {"step": 45, "intent": 'quoted "{value}"\ntext', "evidence": []}
    records[-1]["output"]["intent"] = "{rationale_language}\n## CRITERIA"
    run = load_run(write_run(tmp_path / "run", records))

    old, _ = cli._calls(run, inline, run.path)
    new, _ = cli._calls(run, external, run.path)

    assert external.prompt == inline.prompt
    assert [call.body for call in new] == [call.body for call in old]
    assert [call.sha256 for call in new] == [call.sha256 for call in old]
    assert external.prompt.future.startswith("\n\n")
    assert all(not text.endswith("\n") for text in external.prompt.model_dump().values())


def test_loaded_text_and_provenance_do_not_change_when_the_source_file_changes(
    tmp_path: Path, file_config: Path
) -> None:
    config = load_config(file_config)
    source = Path(config.prompt_sources["system"]["path"])
    original = source.read_bytes()
    source.write_bytes(original + b"\nAn experimental instruction.\n")
    changed = load_config(file_config)
    run = load_run(write_run(tmp_path / "run", [event(0)]))
    args = cli.build_parser().parse_args(["judge", "--run", str(run.path), "--config", str(file_config)])

    assert cli.run_judge(args, config, run, _refuse_client) == 0
    record = json.loads((tmp_path / "evaluation" / "run" / "judge" / "run.json").read_text())
    assert record["resolved"]["prompt"]["system"].encode("utf-8") == original
    assert record["prompt_sources"]["system"] == {"path": str(source), "sha256": hashlib.sha256(original).hexdigest()}
    before = cli._calls(run, config, run.path)[0][0]
    after = cli._calls(run, changed, run.path)[0][0]
    assert before.sha256 != after.sha256
    assert after.body["instructions"].endswith("\nAn experimental instruction.\n")


@pytest.mark.parametrize("form", ["partial", "mixed", "unknown", "non_string"])
def test_file_references_require_one_complete_form(file_config: Path, form: str) -> None:
    prompt = yaml.safe_load(file_config.read_text())["prompt"]
    if form == "partial":
        del prompt["future_file"]
    elif form == "mixed":
        prompt["system"] = "inline"
    elif form == "unknown":
        prompt["extra_file"] = "extra.txt"
    else:
        prompt["system_file"] = 123
    _replace_prompt(file_config, prompt)

    with pytest.raises(ConfigError, match="prompt"):
        load_config(file_config)


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "system_file"), (b"\xff", "utf-8"), (b"", "system"), (b"{", "malformed"), (b"{oops}", "unknown")],
)
def test_bad_template_names_the_file_and_fails_before_client_creation(
    tmp_path: Path, file_config: Path, content: bytes | None, message: str, capsys: pytest.CaptureFixture
) -> None:
    source = file_config.parent / "prompts" / "judge" / "system.txt"
    if content is None:
        source.unlink()
    else:
        source.write_bytes(content)
    with pytest.raises(ConfigError, match=message) as error:
        load_config(file_config)
    assert str(source) in str(error.value)
    code = cli.main(
        [
            "judge",
            "--run",
            str(tmp_path / "unused-run"),
            "--config",
            str(file_config),
            "--execute",
            "--confirm-paid-api",
        ],
        client_factory=_refuse_client,
    )
    assert code == 1
    assert str(source) in capsys.readouterr().err


def test_preview_sdk_and_journal_use_the_same_body_and_hash(tmp_path: Path, file_config: Path) -> None:
    run = write_run(tmp_path / "run", [event(0), event(45, ("change",))])
    args = ["judge", "--run", str(run), "--config", str(file_config)]
    assert cli.main(args, client_factory=_refuse_client) == 0
    directory = tmp_path / "evaluation" / "run" / "judge"
    preview = json.loads((directory / "preview.json").read_text())["requests"]
    factory = Factory()

    assert cli.main([*args, "--execute", "--confirm-paid-api"], client_factory=factory) == 0
    journal = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    assert [item["body"] for item in preview] == factory.sdk.bodies == [item["body"] for item in journal]
    assert [item["request_sha256"] for item in journal] == [request_sha256(body) for body in factory.sdk.bodies]
