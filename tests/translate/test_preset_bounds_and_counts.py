"""Preset points must lie within the episode in every mode; run.json counts every request sent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import openai
import pandas as pd
import pytest

from tests.llm.llm_fakes import DumpableResponse, ScriptedResponsesClient, message_item, responses_body
from translate import cli

from .tr_fakes import EXAMPLE_SCENARIO, run_config_raw, with_value, write_yaml

LAST_COMPLETE_STEP = 340  # of the example scenario's episode, mhi_v1.3_ic100601_win
SHORT_LAST_STEP = 30  # the short_frame fixture
REQUEST = httpx.Request("POST", "https://api.openai.com/v1/responses")
ANSWER = json.dumps({"intent": "[FAKE] intent", "evidence": ["[FAKE] evidence"]})


class RecordingFactory:
    def __init__(self, result: object = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def _preset_config(tmp_path: Path, late_step: int, retry: str = "none") -> Path:
    """The online example as preset points: initial at 0, the scenario's request at 10, a change at ``late_step``."""
    points = [
        {"step": 0, "kind": "initial", "targets": None},
        {"step": 10, "kind": "request", "targets": [{"kind": "aircraft", "id": "A0100"}]},
        {"step": late_step, "kind": "change", "targets": [{"kind": "aircraft", "id": "A0100"}]},
    ]
    raw = with_value(run_config_raw(), "triggers", {"source": "preset", "tick_interval_steps": None, "points": points})
    raw = with_value(with_value(raw, "detector", None), "translation.retry", retry)
    return write_yaml(tmp_path / "config.yaml", raw)


def _args(config: Path, out: Path, name: str, *extra: str) -> list[str]:
    return [
        "--config",
        str(config),
        "--scenario",
        str(EXAMPLE_SCENARIO),
        "--output-root",
        str(out),
        "--run-name",
        name,
        *extra,
    ]


def test_preset_point_after_the_last_complete_step_fails_preview_and_execute_alike(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, out, openai_factory = _preset_config(tmp_path, LAST_COMPLETE_STEP + 1), tmp_path / "out", RecordingFactory()
    factories = cli.Factories(openai_client=openai_factory)

    preview = cli.main(_args(config, out, "preview"), factories=factories)
    preview_error = capsys.readouterr().err
    execute = cli.main(_args(config, out, "execute", "--execute", "--confirm-paid-api"), factories=factories)
    execute_error = capsys.readouterr().err

    assert preview == execute == cli.EXIT_FAILED
    assert f"after the last complete step {LAST_COMPLETE_STEP}" in preview_error
    assert preview_error == execute_error
    assert openai_factory.calls == [] and not out.exists()


def test_preset_point_at_the_last_complete_step_is_accepted(tmp_path: Path) -> None:
    config = _preset_config(tmp_path, LAST_COMPLETE_STEP)

    assert cli.main(_args(config, tmp_path / "out", "preview")) == cli.EXIT_OK


def _ok() -> DumpableResponse:
    return DumpableResponse(responses_body(output=[message_item(ANSWER)]))


def test_api_requests_made_counts_a_retried_one_shot_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_frame: pd.DataFrame
) -> None:
    monkeypatch.setattr(cli, "load_episode", lambda path: short_frame)
    monkeypatch.setattr("translate.llm.retry.time.sleep", lambda _: None)
    server_error = openai.InternalServerError("boom", response=httpx.Response(503, request=REQUEST), body=None)
    # initial for aircraft A0100 and team A at 0, the request at 10, the change at 20: four calls, one retried
    client = ScriptedResponsesClient([server_error, _ok(), _ok(), _ok(), _ok()])
    config = _preset_config(tmp_path, SHORT_LAST_STEP - 10, retry="default")

    code = cli.main(
        _args(config, tmp_path / "out", "run", "--execute", "--confirm-paid-api"),
        factories=cli.Factories(openai_client=RecordingFactory(client)),
    )

    record = json.loads((tmp_path / "out" / "run" / "run.json").read_text(encoding="utf-8"))
    assert code == cli.EXIT_OK
    assert record["counts"]["translation"]["attempted"] == 4
    assert record["api_requests_made"] == len(client.requests) == 5
