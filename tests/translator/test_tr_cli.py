"""CLI: paid gate, preview without clients, local preflight first, artifacts, provenance, exit codes.

Execute tests use the production adapters over fakes: ``SystemOneClient`` over
``httpx.MockTransport`` and ``OpenAIResponsesClient`` over a scripted SDK stand-in.
"""

from __future__ import annotations

import getpass
import json
import math
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mtd_rl_xavier.lag.rendering import render_lag17_jsonl
from mtd_rl_xavier.llm.openai_responses import PROVENANCE_ENV
from mtd_rl_xavier.translator import cli
from mtd_rl_xavier.translator.config import DetectorConfig, load_config
from mtd_rl_xavier.translator.schema import canonical_json, sha256_text
from tests.llm.llm_fakes import (
    APITimeoutError,
    DumpableResponse,
    ScriptedResponsesClient,
    SystemOneServer,
    json_response,
    message_item,
    models_body,
    responses_body,
)

from .tr_fakes import (
    CHANGE,
    EXAMPLE_SCENARIO,
    RUN_CONFIG,
    START,
    Answer,
    over_tcp,
    read_jsonl,
    run_config_raw,
    systemone_responder,
    write_yaml,
)

KEY_ENV = "VLLM_API_KEY"  # the variable NAME in run.yaml; tests set a fake value
SENTINEL_KEY = "local-test-key-0123"
TRANSLATION_ANSWER = json.dumps({"intent": "[FAKE] intent", "evidence": ["[FAKE] evidence"]})
# Detector answers on the example scenario: A0100 starts a tactic at tick 5 and changes it at tick 15.
EXAMPLE_ANSWERS = {("aircraft A0100", 5): START, ("aircraft A0100", 15): CHANGE}


class RecordingFactory:
    """Counts calls; returns a fixed object or raises."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result, self.error = result, error

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the preview must not touch the network")


def _detector_factory(server: SystemOneServer, order: list[str]) -> Any:
    def build(config: DetectorConfig) -> Any:
        order.append("detector")
        return cli.build_system_one_client(over_tcp(config), transport=server.transport)

    return build


def _openai_factory(client: object, order: list[str]) -> Any:
    def build(*, timeout_s: float) -> object:
        order.append("openai")
        return client

    return build


def _args(tmp_path: Path, *extra: str, name: str = "run1") -> list[str]:
    return ["--scenario", str(EXAMPLE_SCENARIO), "--output-root", str(tmp_path / "out"), "--run-name", name, *extra]


def _systemone_ok(calls: int, answers: dict[tuple[str, int], Answer] | None = None) -> list[Any]:
    """The health check, then ``calls`` answers chosen by state (``none`` unless listed in ``answers``)."""
    responder = systemone_responder(answers or {})
    return [json_response(200, models_body("clef-flash")), *[responder for _ in range(calls)]]


def _openai_ok(calls: int) -> ScriptedResponsesClient:
    return ScriptedResponsesClient(
        [DumpableResponse(responses_body(output=[message_item(TRANSLATION_ANSWER)])) for _ in range(calls)]
    )


@pytest.fixture
def short_episode(monkeypatch: pytest.MonkeyPatch, short_frame: pd.DataFrame) -> pd.DataFrame:
    """Execute tests run the real episode cut at SHORT_LAST_STEP (real rows, fast rendering)."""
    monkeypatch.setattr(cli, "load_episode", lambda path: short_frame)
    monkeypatch.setenv(KEY_ENV, SENTINEL_KEY)
    return short_frame


def test_preview_builds_no_client_reads_no_credential_and_writes_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    detector, openai = RecordingFactory(), RecordingFactory()

    code = cli.main(_args(tmp_path), detector_client_factory=detector, openai_client_factory=openai)

    run_dir = tmp_path / "out" / "run1"
    assert code == cli.EXIT_OK and detector.calls == [] and openai.calls == []
    assert sorted(p.name for p in run_dir.iterdir()) == ["example_translation_request.json", "preview.json", "run.json"]
    assert run_dir.stat().st_mode & 0o777 == 0o700
    preview = json.loads((run_dir / "preview.json").read_text())
    assert preview["api_requests_made"] == 0
    # initial x {A0100, team A} at step 0 + the request at 10; team A's instruction at 20 forces nothing (D50)
    assert preview["plan"]["mandatory_translations"] == 3 and preview["plan"]["detector_calls"] == 136
    assert preview["detector_budget"]["margin"] > 0 and preview["detector_budget"]["intent_reserve_bytes"] == 2400
    record = json.loads((run_dir / "run.json").read_text())
    assert (record["mode"], record["status"], record["api_requests_made"]) == ("preview", "preview", 0)
    assert record["config"]["pipeline"]["visibility"] == {"mode": "online", "future_steps": 0}
    assert record["provenance"]["prompt_version"] == "translator-intent-v2"
    assert record["provenance"]["detector_state_version"] == "2"
    example = json.loads((run_dir / "example_translation_request.json").read_text())
    assert example["model"] == "gpt-6.1-terra" and example["store"] is False


def test_existing_run_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], short_episode: pd.DataFrame
) -> None:
    (tmp_path / "out" / "run1").mkdir(parents=True)

    code = cli.main(_args(tmp_path))

    assert code == cli.EXIT_FAILED and "FileExistsError" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--confirm-paid-api"], ["--execute"]])
def test_one_paid_flag_alone_is_a_usage_error(tmp_path: Path, flags: list[str]) -> None:
    detector, openai = RecordingFactory(), RecordingFactory()

    with pytest.raises(SystemExit) as raised:
        cli.main(_args(tmp_path, *flags), detector_client_factory=detector, openai_client_factory=openai)

    assert raised.value.code == cli.EXIT_USAGE
    assert detector.calls == [] and openai.calls == [] and not (tmp_path / "out").exists()


def test_a_local_only_config_passes_the_gate_with_execute_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_episode: pd.DataFrame
) -> None:
    local_only = replace(
        load_config(RUN_CONFIG), translation=replace(load_config(RUN_CONFIG).translation, backend="local")
    )
    monkeypatch.setattr(cli, "load_config", lambda path: local_only)
    detector = RecordingFactory(error=cli.SystemOneError("stop after the gate"))

    code = cli.main(_args(tmp_path, "--execute"), detector_client_factory=detector)

    assert local_only.paid_layers() == ()
    assert code == cli.EXIT_FAILED and len(detector.calls) == 1


def test_execute_runs_both_layers_and_records_provenance(tmp_path: Path, short_episode: pd.DataFrame) -> None:
    order: list[str] = []
    server = SystemOneServer(_systemone_ok(12, EXAMPLE_ANSWERS))  # 6 ticks (5..30) x A0100 and team A
    client = _openai_ok(5)  # initial at step 0 x 2, start 5, request 10, change 15

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(server, order),
        openai_client_factory=_openai_factory(client, order),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    events = read_jsonl(run_dir / "events.jsonl")
    requests = read_jsonl(run_dir / "requests.jsonl")
    responses = read_jsonl(run_dir / "responses.jsonl")
    assert code == cli.EXIT_OK and order == ["detector", "openai"]
    assert server.requests[0].path == "/v1/models"
    assert record["status"] == "completed" and record["api_requests_made"] == 17
    assert record["counts"]["translation"] == {"attempted": 5, "completed": 5, "failed": 0, "uncertain": 0}
    assert record["inputs"]["scenario"]["sha256"] and record["inputs"]["config"]["sha256"]
    assert record["inputs"]["episode"]["csv_sha256"] == short_episode.attrs["csv_sha256"]
    assert record["provenance"]["question_set_sha256"] and record["provenance"]["renderers"] == {
        "lag17-jsonl": "1",
        "lag-detector-compact": "1",
    }
    assert "mtd_rl_xavier/translator/pipeline.py" in record["provenance"]["code_sha256"]
    assert len(requests) == len(responses) == 17 and all(r["status"] == "ok" for r in responses)
    assert all(
        isinstance(r["latency_s"], float) and math.isfinite(r["latency_s"]) and r["latency_s"] >= 0 for r in responses
    )
    assert record["config"]["pipeline"]["visibility"] == {"mode": "online", "future_steps": 0}
    assert SENTINEL_KEY not in (run_dir / "run.json").read_text() + (run_dir / "requests.jsonl").read_text()
    translations = [e for e in events if e["record_type"] == "translation_event"]
    assert [(e["step"], e["target"]["id"], e["primary_trigger"]) for e in translations] == [
        (0, "A0100", "initial"),
        (0, "A", "initial"),
        (5, "A0100", "start"),
        (10, "A0100", "request"),
        (15, "A0100", "change"),
    ]
    assert all(e["schema_version"] == 2 and e["processing_s"] >= 0 for e in events if "processing_s" in e)
    assert len([e for e in events if "processing_s" in e]) == len(translations)
    _assert_exact_model_inputs(events, server, client, short_episode)


def _assert_exact_model_inputs(
    events: list[dict[str, Any]], server: SystemOneServer, client: ScriptedResponsesClient, frame: pd.DataFrame
) -> None:
    """The logged inputs are byte-identical to what the fake server and the fake SDK received."""
    detector_events = [e for e in events if e["record_type"] == "detector_call"]
    assert [r.body["state"] for r in server.requests[1:]] == [e["input"]["state_text"] for e in detector_events]
    assert [sha256_text(canonical_json(r.body)) for r in server.requests[1:]] == [
        e["request_sha256"] for e in detector_events
    ]
    translation_events = [e for e in events if e["record_type"] == "translation_event"]
    for event, sent in zip(translation_events, client.requests, strict=True):
        user_text = sent["input"][0]["content"][0]["text"]
        trajectory = event["input"]["trajectory"]
        assert event["request_sha256"] == sha256_text(canonical_json(sent))
        assert f"BEGIN WINDOW DATA\n{trajectory['text']}\nEND WINDOW DATA" in user_text
        target = event["target"]
        aircraft = ", ".join(event["input"]["target_aircraft"])
        assert f"Translation target: {target['kind']} {target['id']} (aircraft {aircraft})" in user_text
        assert all(item["text"] in user_text for item in event["input"]["instructions_in_effect"])
        assert all(f"  {item['text']}" in user_text for item in event["input"]["alarms"])
        assert all(item["step"] <= event["step"] for item in event["input"]["alarms"])
        question = event["input"]["question"]
        assert (question is not None) == ("Question from a human" in user_text)
        assert (
            question is None or f"Question from a human (step {question['step']}):\n  {question['text']}" in user_text
        )
        previous = event["input"]["previous_translation"]
        assert previous is None or f"intent: {previous['intent']}" in user_text
        assert f"Write both in {event['input']['output_language']}." in sent["instructions"]
        rendered = render_lag17_jsonl(
            frame,
            start_step=trajectory["start_step"],
            end_step=trajectory["end_step"],
            anchor_step=event["step"],
            episode_alias="episode_01",
        )
        assert trajectory["text"] == rendered.text and trajectory["sha256"] == rendered.sha256
        assert "mhi_v1.3" not in user_text + sent["instructions"]  # the episode id never reaches the model


def test_failed_local_preflight_happens_before_any_paid_client(tmp_path: Path, short_episode: pd.DataFrame) -> None:
    order: list[str] = []
    server = SystemOneServer([json_response(401, {"error": "unauthorized"})])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(server, order),
        openai_client_factory=_openai_factory(_openai_ok(0), order),
    )

    assert code == cli.EXIT_FAILED and order == ["detector"] and not (tmp_path / "out").exists()


def test_translation_timeout_is_uncertain_exit_three(tmp_path: Path, short_episode: pd.DataFrame) -> None:
    client = ScriptedResponsesClient([APITimeoutError("slow")])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=_openai_factory(client, []),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    requests = read_jsonl(run_dir / "requests.jsonl")
    (response,) = read_jsonl(run_dir / "responses.jsonl")
    assert code == cli.EXIT_UNCERTAIN and record["status"] == "uncertain"
    assert len(requests) == 1 and response["status"] == "uncertain" and response["error"]["type"] == "OpenAIClientError"
    assert record["error"]["uncertain"] is True and record["api_requests_made"] == 1


def test_translation_cap_stops_with_exit_four(tmp_path: Path, short_episode: pd.DataFrame) -> None:
    raw = run_config_raw()
    config_path = write_yaml(tmp_path / "run.yaml", {**raw, "limits": {**raw["limits"], "max_translation_calls": 2}})

    code = cli.main(
        ["--config", str(config_path), *_args(tmp_path, "--execute", "--confirm-paid-api")],
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(12)), []),
        openai_client_factory=_openai_factory(_openai_ok(2), []),
    )

    record = json.loads((tmp_path / "out" / "run1" / "run.json").read_text())
    assert code == cli.EXIT_CAP_REACHED and record["status"] == "cap_reached"
    assert record["unsent"] == {
        "layer": "translation",
        "target": {"kind": "aircraft", "id": "A0100"},
        "step": 10,
        "cap": 2,
    }


def test_missing_openai_key_after_the_run_dir_exists_records_failure(
    tmp_path: Path, short_episode: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
    )

    record = json.loads((tmp_path / "out" / "run1" / "run.json").read_text())
    assert code == cli.EXIT_FAILED
    assert record["status"] == "failed" and record["error"]["type"] == "OpenAIClientError"
    assert not (tmp_path / "out" / "run1" / "requests.jsonl").exists()


def test_help_lists_the_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert all(flag in out for flag in ("--execute", "--confirm-paid-api", "--scenario", "--run-name"))


def test_unexpected_factory_exception_is_recorded_without_its_text(
    tmp_path: Path, short_episode: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-SECRET-from-an-unexpected-exception"

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=RecordingFactory(error=RuntimeError(secret)),
    )

    run_json = (tmp_path / "out" / "run1" / "run.json").read_text()
    assert code == cli.EXIT_FAILED
    assert json.loads(run_json)["status"] == "failed" and json.loads(run_json)["error"]["type"] == "RuntimeError"
    assert secret not in run_json + capsys.readouterr().err


def test_jsonl_sink_writes_standard_json_for_non_finite_raw_values(tmp_path: Path) -> None:
    raw = {"answers": {"q": {"noul": 0.5, "confidence": float("nan")}}, "spread": [float("inf"), float("-inf")]}

    with cli.JsonlRunSink(tmp_path) as sink:
        sink.write_response({"call_id": "c", "raw": raw})

    def reject(constant: str) -> None:
        raise AssertionError(f"non-standard JSON constant {constant}")

    line = (tmp_path / "responses.jsonl").read_text().strip()
    record = json.loads(line, parse_constant=reject)
    assert record["raw"] == {"answers": {"q": {"noul": 0.5, "confidence": "NaN"}}, "spread": ["Infinity", "-Infinity"]}


# --- review fixes (D38/D39) -----------------------------------------------------------------------

SECRET = "SECRETPW-canary-7f3a"  # fake password canary; must never reach stderr, stdout or a file


@pytest.mark.parametrize("mode", [[], ["--execute", "--confirm-paid-api"]])
def test_detector_budget_that_does_not_fit_fails_preflight_before_any_directory_or_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], short_episode: pd.DataFrame, mode: list[str]
) -> None:
    raw = run_config_raw()  # 4096 (wev serve's cap) cannot hold the 2400-byte intent reserve plus state and window
    config_path = write_yaml(
        tmp_path / "run.yaml", {**raw, "detector": {**raw["detector"], "input_token_budget": 4096}}
    )
    detector, openai = RecordingFactory(), RecordingFactory()

    code = cli.main(
        ["--config", str(config_path), *_args(tmp_path, *mode)],
        detector_client_factory=detector,
        openai_client_factory=openai,
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILED and detector.calls == [] and openai.calls == []
    assert not (tmp_path / "out").exists()
    assert "PipelineError" in err and ">= input_token_budget 4096" in err


@pytest.mark.parametrize("name", ["../x", "a/b", "", "x y", ".", "..", "ABSOLUTE", "-x"])
def test_run_name_that_is_not_a_plain_name_is_a_usage_error_and_creates_nothing(tmp_path: Path, name: str) -> None:
    value = str(tmp_path / "abs") if name == "ABSOLUTE" else name
    detector, openai = RecordingFactory(), RecordingFactory()

    with pytest.raises(SystemExit) as raised:
        cli.main(_args(tmp_path, name=value), detector_client_factory=detector, openai_client_factory=openai)

    assert raised.value.code == cli.EXIT_USAGE and detector.calls == [] and openai.calls == []
    assert sorted(tmp_path.iterdir()) == []


HEX_KEY = "deadbeefcafe7f3a"  # a pasted hex key that starts with a letter
FULLWIDTH_COLON, FULLWIDTH_AT = chr(0xFF1A), chr(0xFF20)  # NFKC turns them into ":" and "@" inside urlsplit


@pytest.mark.parametrize(
    ("connection", "canary"),
    [
        ({"base_url": f"http://user:{SECRET}@127.0.0.1:8000", "uds_path": None, "api_key_env": "K"}, SECRET),
        (
            {"base_url": f"http://user{FULLWIDTH_COLON}{SECRET}@127.0.0.1:8000", "uds_path": None, "api_key_env": "K"},
            SECRET,
        ),
        ({"base_url": f"http://{SECRET}{FULLWIDTH_AT}127.0.0.1:8000", "uds_path": None, "api_key_env": "K"}, SECRET),
        ({"base_url": None, "uds_path": "/tmp/{user}-vllm/clef.sock", "api_key_env": SECRET}, SECRET),
        ({"base_url": "http://127.0.0.1:8000", "uds_path": None, "api_key_env": HEX_KEY}, HEX_KEY),
    ],
)
def test_secret_in_the_detector_connection_fails_the_preview_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], connection: dict[str, Any], canary: str
) -> None:
    raw = run_config_raw()
    config_path = write_yaml(tmp_path / "run.yaml", {**raw, "detector": {**raw["detector"], "connection": connection}})

    code = cli.main(["--config", str(config_path), *_args(tmp_path)])

    captured = capsys.readouterr()
    assert code == cli.EXIT_FAILED and not (tmp_path / "out").exists()
    assert "TranslatorConfigError" in captured.err and canary not in captured.err + captured.out


def test_default_unix_socket_with_user_placeholder_runs_and_records_the_endpoints(
    tmp_path: Path, short_episode: pd.DataFrame
) -> None:
    socket_dir = tmp_path / f"{getpass.getuser()}-vllm"
    socket_dir.mkdir()
    socket_dir.chmod(0o700)
    raw = run_config_raw()
    connection = {**raw["detector"]["connection"], "uds_path": str(tmp_path / "{user}-vllm" / "clef-flash.sock")}
    config_path = write_yaml(tmp_path / "run.yaml", {**raw, "detector": {**raw["detector"], "connection": connection}})
    server = SystemOneServer(_systemone_ok(12, EXAMPLE_ANSWERS))

    code = cli.main(
        ["--config", str(config_path), *_args(tmp_path, "--execute", "--confirm-paid-api")],
        detector_client_factory=lambda config: cli.build_system_one_client(config, transport=server.transport),
        openai_client_factory=_openai_factory(_openai_ok(5), []),
    )

    record = json.loads((tmp_path / "out" / "run1" / "run.json").read_text())
    assert raw["detector"]["connection"]["uds_path"] == "/tmp/{user}-vllm/clef-flash.sock"  # committed default
    assert code == cli.EXIT_OK and record["status"] == "completed"
    assert record["config"]["detector"]["connection"]["uds_path"] == str(socket_dir / "clef-flash.sock")
    assert record["provenance"]["openai_base_url"] == "https://api.openai.com/v1"


def test_ctrl_c_during_a_translation_call_is_uncertain_exit_three_without_traceback(
    tmp_path: Path, short_episode: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    client = ScriptedResponsesClient([KeyboardInterrupt()])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=_openai_factory(client, []),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    (response,) = read_jsonl(run_dir / "responses.jsonl")
    assert code == cli.EXIT_UNCERTAIN and len(client.requests) == 1 and "Traceback" not in capsys.readouterr().err
    assert record["status"] == "uncertain" and record["error"]["type"] == "unexpected:KeyboardInterrupt"
    assert response["status"] == "uncertain" and record["counts"]["translation"]["uncertain"] == 1


def _interrupt_first_event(monkeypatch: pytest.MonkeyPatch) -> None:
    def write_event(sink: Any, record: dict[str, Any]) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.JsonlRunSink, "write_event", write_event)


@pytest.mark.parametrize("where", ["before_the_pipeline", "between_calls"])
def test_ctrl_c_with_no_call_in_flight_ends_with_a_terminal_failed_run_json(
    tmp_path: Path,
    short_episode: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    where: str,
) -> None:
    client = _openai_ok(5)
    factory: Any = RecordingFactory(error=KeyboardInterrupt())
    if where == "between_calls":  # the first call is answered and journaled; Ctrl-C hits the event write
        _interrupt_first_event(monkeypatch)
        factory = _openai_factory(client, [])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=factory,
    )

    record = json.loads((tmp_path / "out" / "run1" / "run.json").read_text())
    assert code == cli.EXIT_FAILED and "Traceback" not in capsys.readouterr().err
    assert record["status"] == "failed" and record["ended_at"] is not None
    assert record["error"]["type"] == "KeyboardInterrupt"
    if where == "between_calls":
        assert len(client.requests) == 1 and record["counts"]["translation"]["completed"] == 1
        assert record["error"]["uncertain"] is False and record["counts"]["events_written"] == 0


def test_ctrl_c_before_the_run_directory_exists_prints_no_traceback(
    tmp_path: Path, short_episode: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=RecordingFactory(error=KeyboardInterrupt()),
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILED and "KeyboardInterrupt" in err and "Traceback" not in err
    assert not (tmp_path / "out").exists()


def test_failures_after_an_uncertain_run_keep_exit_three(
    tmp_path: Path, short_episode: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_close = cli.JsonlRunSink.close
    writes: list[Path] = []

    def close_then_fail(sink: Any) -> None:
        original_close(sink)
        raise OSError("simulated close failure")

    def write_json(path: Path, payload: object) -> None:
        writes.append(path)
        if len(writes) > 1:  # the final run.json
            raise OSError("simulated disk full")
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(cli.JsonlRunSink, "close", close_then_fail)
    monkeypatch.setattr(cli, "_write_json", write_json)
    client = ScriptedResponsesClient([APITimeoutError("slow")])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=_openai_factory(client, []),
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_UNCERTAIN and len(client.requests) == 1
    assert "closing the journals failed: OSError" in err and "final run.json not written: OSError" in err
    assert read_jsonl(tmp_path / "out" / "run1" / "responses.jsonl")[0]["status"] == "uncertain"


def test_lone_surrogate_in_a_received_response_is_journaled_and_fails_instead_of_uncertain(
    tmp_path: Path, short_episode: pd.DataFrame
) -> None:
    answer = '{"intent": "\ud800 climb", "evidence": ["[FAKE] evidence"]}'  # a decoded "\\ud800" escape
    client = ScriptedResponsesClient([DumpableResponse(responses_body(output=[message_item(answer)]))])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=_openai_factory(client, []),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    (response,) = read_jsonl(run_dir / "responses.jsonl")
    assert code == cli.EXIT_FAILED and record["status"] == "failed"
    assert record["error"]["type"] == "TranslationOutputError" and record["error"]["uncertain"] is False
    assert record["counts"]["translation"] == {"attempted": 1, "completed": 1, "failed": 0, "uncertain": 0}
    assert response["status"] == "ok" and response["raw"]["output"][1]["content"][0]["text"] == answer
    assert b"\\ud800" in (run_dir / "responses.jsonl").read_bytes()  # stored as its JSON escape


@pytest.mark.parametrize("error", [OSError("simulated close failure"), KeyboardInterrupt()])
def test_detector_client_close_failure_after_an_uncertain_run_keeps_exit_three(
    tmp_path: Path,
    short_episode: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: BaseException,
) -> None:
    def close_then_fail(client: Any) -> None:
        raise error

    monkeypatch.setattr(cli.SystemOneClient, "close", close_then_fail)
    client = ScriptedResponsesClient([APITimeoutError("slow")])

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
        openai_client_factory=_openai_factory(client, []),
    )

    err = capsys.readouterr().err
    record = json.loads((tmp_path / "out" / "run1" / "run.json").read_text())
    assert code == cli.EXIT_UNCERTAIN and record["status"] == "uncertain"
    assert f"closing the detector client failed: {type(error).__name__}" in err and "Traceback" not in err


# --- SDK environment (D44) ------------------------------------------------------------------------

PROXY_CANARY = "http://user:PW-canary-9f3a@proxy.example:3128"


def _clean_sdk_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*PROVENANCE_ENV, "OPENAI_LOG"):
        monkeypatch.delenv(name, raising=False)


def test_preview_records_sdk_environment_names_only_and_ignores_openai_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", PROXY_CANARY)
    monkeypatch.setenv("no_proxy", "internal.example")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-PW-canary-9f3a")
    monkeypatch.setenv("OPENAI_LOG", "debug")
    detector, openai = RecordingFactory(), RecordingFactory()

    code = cli.main(_args(tmp_path), detector_client_factory=detector, openai_client_factory=openai)

    run_dir = tmp_path / "out" / "run1"
    text = (run_dir / "run.json").read_text()
    assert code == cli.EXIT_OK and detector.calls == [] and openai.calls == []  # preview: no client, no refusal
    assert json.loads(text)["provenance"]["sdk_environment_set"] == ["HTTPS_PROXY", "no_proxy", "OPENAI_ORG_ID"]
    assert all("PW-canary" not in path.read_text() for path in run_dir.iterdir())


def test_execute_records_sdk_environment_names_only(
    tmp_path: Path, short_episode: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("ALL_PROXY", PROXY_CANARY)
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/PW-canary-9f3a.pem")

    code = cli.main(
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(12, EXAMPLE_ANSWERS)), []),
        openai_client_factory=_openai_factory(_openai_ok(5), []),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_OK and record["provenance"]["sdk_environment_set"] == ["ALL_PROXY", "SSL_CERT_FILE"]
    assert all("PW-canary" not in path.read_text() for path in run_dir.iterdir())


def test_execute_with_openai_log_set_fails_before_any_call_without_echoing_the_value(
    tmp_path: Path, short_episode: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_LOG", "debug-PW-canary")

    code = cli.main(  # the real make_openai_client refuses before it imports the SDK
        _args(tmp_path, "--execute", "--confirm-paid-api"),
        detector_client_factory=_detector_factory(SystemOneServer(_systemone_ok(0)), []),
    )

    run_dir = tmp_path / "out" / "run1"
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_FAILED and record["status"] == "failed" and record["api_requests_made"] == 0
    assert record["error"]["type"] == "OpenAIClientError" and "OPENAI_LOG" in record["error"]["message"]
    assert "PW-canary" not in record["error"]["message"] + capsys.readouterr().err


# --- contract v2: exact budget preflight ----------------------------------------------------------


@pytest.mark.parametrize("mode", [[], ["--execute", "--confirm-paid-api"]])
def test_scenario_texts_that_push_a_detector_state_over_budget_fail_preflight_before_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], short_episode: pd.DataFrame, mode: list[str]
) -> None:
    raw = yaml.safe_load(EXAMPLE_SCENARIO.read_text(encoding="utf-8"))
    csv = (EXAMPLE_SCENARIO.parent / raw["episode"]["csv_path"]).resolve()
    alarms = [  # each within max_alarm_chars; together they exceed the budget at tick 25 (lookback [1, 25])
        {"alarm_id": f"w{step}", "step": step, "targets": [{"kind": "team", "id": "A"}], "text": "\U0001f600" * 200}
        for step in range(1, 21)
    ]
    scenario = write_yaml(
        tmp_path / "scenario.yaml", {**raw, "episode": {**raw["episode"], "csv_path": str(csv)}, "alarms": alarms}
    )
    detector, openai = RecordingFactory(), RecordingFactory()
    args = ["--scenario", str(scenario), "--output-root", str(tmp_path / "out"), "--run-name", "run1", *mode]

    code = cli.main(args, detector_client_factory=detector, openai_client_factory=openai)

    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILED and detector.calls == [] and openai.calls == []
    assert not (tmp_path / "out").exists()
    assert "PipelineError" in err and ">= input_token_budget 16384" in err and "alarm" in err
