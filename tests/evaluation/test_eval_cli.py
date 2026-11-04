"""Judge CLI: paired paid gate, preview without a client, partial judging, artifacts, exit codes, E2E.

Execute tests use the production ``OpenAIResponsesClient`` over a scripted SDK stand-in. The two
E2E tests run a real 13_LAG episode through the Translator (fake layers; then the production
adapters over ``httpx.MockTransport`` and a scripted SDK client) and preview the judge on the
resulting event log. No real network, paid API, GPU or decoder is used.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evaluation import cli
from evaluation.rendering import HEADING_TRAJECTORY, WINDOW_BEGIN, WINDOW_END
from mtd_rl_xavier.llm.openai_responses import PROVENANCE_ENV
from mtd_rl_xavier.translator import cli as translator_cli
from tests.llm.llm_fakes import (
    APITimeoutError,
    ScriptedResponsesClient,
    SystemOneServer,
    json_response,
    models_body,
    noul_answer,
    systemone_body,
)
from tests.translator.tr_fakes import EXAMPLE_SCENARIO, FakeIntentModel, over_tcp

from .eval_fakes import (
    A1_REQUEST,
    A1_START,
    EVENT_IDS,
    EXAMPLE_JUDGED,
    EXAMPLE_KEY_POINTS,
    JUDGE_CONFIG,
    RecordingFactory,
    judge_config_raw,
    judge_response,
    key_points_raw,
    read_jsonl,
    tree_snapshot,
    verdict_response,
    with_value,
    write_translator_run,
    write_yaml,
)

RUN_NAME = "judge1"
DETECTOR_TICKS = 136  # 68 ticks (5..340) x {aircraft A0100, team A} on the full example episode
TRANSLATIONS = 5


def _no_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("no test may touch the network")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)


def _args(tmp_path: Path, translator_run: Path, key_points: Path = EXAMPLE_KEY_POINTS, *extra: str) -> list[str]:
    return [
        "--translator-run",
        str(translator_run),
        "--key-points",
        str(key_points),
        "--output-root",
        str(tmp_path / "out"),
        "--run-name",
        RUN_NAME,
        *extra,
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scripted(*key_point_ids: tuple[str, ...]) -> ScriptedResponsesClient:
    return ScriptedResponsesClient([verdict_response(ids) for ids in key_point_ids])


def test_help_lists_the_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert all(flag in out for flag in ("--translator-run", "--key-points", "--execute", "--confirm-paid-api"))


@pytest.mark.parametrize("flag", ["--execute", "--confirm-paid-api"])
def test_one_paid_flag_alone_is_a_usage_error(tmp_path: Path, translator_run: Path, flag: str) -> None:
    factory = RecordingFactory()

    with pytest.raises(SystemExit) as raised:
        cli.main(_args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, flag), openai_client_factory=factory)

    assert raised.value.code == cli.EXIT_USAGE and factory.calls == [] and not (tmp_path / "out").exists()


def test_preview_builds_no_client_and_writes_only_its_run_directory(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch, offline: None
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = RecordingFactory()
    before = tree_snapshot(tmp_path) | tree_snapshot(translator_run)

    code = cli.main(_args(tmp_path, translator_run), openai_client_factory=factory)

    run_dir = tmp_path / "out" / RUN_NAME
    assert code == cli.EXIT_OK and factory.calls == []
    created = (tree_snapshot(tmp_path) | tree_snapshot(translator_run)) - before
    assert created == {tmp_path / "out", run_dir} | set(run_dir.iterdir())
    assert sorted(p.name for p in run_dir.iterdir()) == ["example_judge_request.json", "preview.json", "run.json"]
    assert run_dir.stat().st_mode & 0o777 == 0o700
    preview = json.loads((run_dir / "preview.json").read_text())
    assert preview["api_requests_made"] == 0 and preview["warnings"] == []
    assert (preview["plan"]["judged"], preview["plan"]["not_judged"]) == (2, 3)
    assert [e["planned"] for e in preview["plan"]["events"]] == [
        "judged",
        "not_judged",
        "judged",
        "not_judged",
        "not_judged",
    ]
    assert preview["estimate"]["judge_calls"] == 2 and preview["estimate"]["cost_usd_upper_bound"] > 0
    record = json.loads((run_dir / "run.json").read_text())
    assert (record["mode"], record["status"], record["api_requests_made"]) == ("preview", "preview", 0)
    inputs = record["inputs"]
    assert inputs["translator_run"]["run_json_sha256"] == _sha256(translator_run / "run.json")
    assert inputs["translator_run"]["events_jsonl_sha256"] == _sha256(translator_run / "events.jsonl")
    assert inputs["key_points"]["sha256"] == _sha256(EXAMPLE_KEY_POINTS)
    assert inputs["config"]["sha256"] == _sha256(JUDGE_CONFIG)
    assert set(record["provenance"]["git"]) == {"commit", "dirty"}
    assert "evaluation/judge.py" in record["provenance"]["code_sha256"]
    example = json.loads((run_dir / "example_judge_request.json").read_text())
    assert example["model"] == "gpt-6.1-sol" and example["store"] is False and "temperature" not in example
    every_artifact = "".join(path.read_text() for path in run_dir.iterdir())
    assert "base64" not in every_artifact and "data:image" not in every_artifact


def test_preview_with_no_key_points_writes_no_example_and_warns(tmp_path: Path, translator_run: Path) -> None:
    key_points = write_yaml(tmp_path / "kp.yaml", {"schema_version": 1, "events": []})

    code = cli.main(_args(tmp_path, translator_run, key_points), openai_client_factory=RecordingFactory())

    run_dir = tmp_path / "out" / RUN_NAME
    preview = json.loads((run_dir / "preview.json").read_text())
    assert code == cli.EXIT_OK and not (run_dir / "example_judge_request.json").exists()
    assert preview["plan"]["not_judged"] == len(EVENT_IDS) and "nothing will be judged" in preview["warnings"][0]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (key_points_raw({"placeholder-episode-01:aircraft:A0100:000001": ["a"]}), "name no translation event"),
        ({"schema_version": 1, "events": key_points_raw({A1_START: ["a"]})["events"] * 2}, "duplicate event_id"),
        (key_points_raw({A1_START: ["   "]}), "non-blank"),
        ({"schema_version": 1, "events": [{"event_id": A1_START, "key_points": []}]}, "non-empty list"),
    ],
)
@pytest.mark.parametrize("mode", [[], ["--execute", "--confirm-paid-api"]])
def test_key_point_errors_fail_before_any_directory_or_client(
    tmp_path: Path,
    translator_run: Path,
    capsys: pytest.CaptureFixture[str],
    raw: dict[str, Any],
    message: str,
    mode: list[str],
) -> None:
    factory = RecordingFactory()

    code = cli.main(
        _args(tmp_path, translator_run, write_yaml(tmp_path / "kp.yaml", raw), *mode), openai_client_factory=factory
    )

    assert code == cli.EXIT_FAILED and factory.calls == [] and not (tmp_path / "out").exists()
    assert "KeyPointsError" in (err := capsys.readouterr().err) and message in err


@pytest.mark.parametrize("mode", [[], ["--execute", "--confirm-paid-api"]])
def test_more_judged_events_than_the_cap_fail_in_preflight(
    tmp_path: Path, translator_run: Path, mode: list[str]
) -> None:
    config = write_yaml(tmp_path / "judge.yaml", with_value(judge_config_raw(), "limits.max_judged_events", 1))
    factory = RecordingFactory()

    code = cli.main(
        ["--config", str(config), *_args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, *mode)],
        openai_client_factory=factory,
    )

    assert code == cli.EXIT_FAILED and factory.calls == [] and not (tmp_path / "out").exists()


def test_a_translator_preview_directory_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    preview_dir = tmp_path / "translator-preview"
    preview_dir.mkdir()
    (preview_dir / "run.json").write_text(json.dumps({"mode": "preview"}))

    code = cli.main(_args(tmp_path, preview_dir))

    assert code == cli.EXIT_FAILED and "has no events.jsonl" in capsys.readouterr().err


def test_existing_run_directory_is_refused(
    tmp_path: Path, translator_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "out" / RUN_NAME).mkdir(parents=True)

    code = cli.main(_args(tmp_path, translator_run))

    assert code == cli.EXIT_FAILED and "FileExistsError" in capsys.readouterr().err


# --- execute --------------------------------------------------------------------------------------


def test_execute_judges_the_subset_once_each_and_keeps_raw_responses(tmp_path: Path, translator_run: Path) -> None:
    client = _scripted(("kp-001", "kp-002"), ("kp-001",))
    factory = RecordingFactory(result=client)

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=factory,
    )

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    requests, responses = read_jsonl(run_dir / "requests.jsonl"), read_jsonl(run_dir / "responses.jsonl")
    results = read_jsonl(run_dir / "results.jsonl")
    assert code == cli.EXIT_OK and factory.calls == [{"timeout_s": 1800.0}]
    assert len(client.requests) == len(requests) == len(responses) == 2
    assert [r["payload"] for r in requests] == client.requests and [r["call_id"] for r in requests] == list(
        EXAMPLE_JUDGED
    )
    assert all(r["status"] == "ok" and r["raw"]["model"] for r in responses)
    assert [r["event_id"] for r in results] == list(EVENT_IDS)
    assert [r["status"] for r in results] == ["judged", "not_judged", "judged", "not_judged", "not_judged"]
    assert [kp["key_point_id"] for kp in results[2]["key_points"]] == ["kp-001"]
    assert set(results[0]) == set(results[1])  # every status has the same keys
    assert record["status"] == "completed" and record["api_requests_made"] == 2
    assert record["counts"] == {
        "attempted": 2,
        "judged": 2,
        "not_judged": 3,
        "failed": 0,
        "uncertain": 0,
        "results_written": 5,
    }
    assert record["usage"]["input_tokens"] == 240 and record["approx_cost_usd"] > 0
    assert record["inputs"]["translator_run"]["status"] == "completed"
    assert (run_dir / "requests.jsonl").stat().st_size > 0 and run_dir.stat().st_mode & 0o777 == 0o700


def test_execute_with_no_overlap_makes_no_call_and_builds_no_client(tmp_path: Path, translator_run: Path) -> None:
    key_points = write_yaml(tmp_path / "kp.yaml", {"schema_version": 1, "events": []})
    factory = RecordingFactory(error=AssertionError("no client may be built"))

    code = cli.main(
        _args(tmp_path, translator_run, key_points, "--execute", "--confirm-paid-api"), openai_client_factory=factory
    )

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_OK and factory.calls == [] and record["api_requests_made"] == 0
    assert [r["status"] for r in read_jsonl(run_dir / "results.jsonl")] == ["not_judged"] * len(EVENT_IDS)
    assert (run_dir / "requests.jsonl").read_text() == ""


def test_execute_timeout_is_uncertain_exit_three(tmp_path: Path, translator_run: Path) -> None:
    client = ScriptedResponsesClient([APITimeoutError("slow")])

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(result=client),
    )

    record = json.loads((tmp_path / "out" / RUN_NAME / "run.json").read_text())
    assert code == cli.EXIT_UNCERTAIN and record["status"] == "uncertain" and len(client.requests) == 1
    assert record["error"]["uncertain"] is True and record["api_requests_made"] == 1


def test_execute_invalid_verdict_exits_one_after_keeping_the_raw_response(tmp_path: Path, translator_run: Path) -> None:
    client = ScriptedResponsesClient([judge_response('{"overall": {"rationale": "r", "score": 3}}')])

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(result=client),
    )

    run_dir = tmp_path / "out" / RUN_NAME
    (response,) = read_jsonl(run_dir / "responses.jsonl")
    assert code == cli.EXIT_FAILED and response["status"] == "ok" and response["raw"]["id"]
    assert read_jsonl(run_dir / "results.jsonl")[-1]["error"]["type"] == "JudgeSchemaError"
    assert json.loads((run_dir / "run.json").read_text())["status"] == "failed"


def test_missing_openai_key_is_recorded_after_the_run_directory_exists(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch, offline: None
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = cli.main(_args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"))

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_FAILED and record["status"] == "failed" and record["error"]["type"] == "OpenAIClientError"
    assert not (run_dir / "requests.jsonl").exists()


def test_unexpected_factory_exception_text_is_withheld(
    tmp_path: Path, translator_run: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-SECRET-from-an-unexpected-exception"

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(error=RuntimeError(secret)),
    )

    run_json = (tmp_path / "out" / RUN_NAME / "run.json").read_text()
    assert code == cli.EXIT_FAILED and json.loads(run_json)["error"]["type"] == "RuntimeError"
    assert secret not in run_json + capsys.readouterr().err


# --- end to end -----------------------------------------------------------------------------------


def _judge_preview_trajectory(run_dir: Path) -> str:
    """The trajectory text inside the judge's example request (between the window markers)."""
    example = json.loads((run_dir / "example_judge_request.json").read_text())
    (block,) = [item["text"] for item in example["input"][0]["content"] if item["text"].startswith(HEADING_TRAJECTORY)]
    return block[block.index(WINDOW_BEGIN) + len(WINDOW_BEGIN) + 1 : block.rindex(WINDOW_END) - 1]


def _event(run_dir: Path, event_id: str) -> dict[str, Any]:
    return next(e for e in read_jsonl(run_dir / "events.jsonl") if e.get("event_id") == event_id)


def test_e2e_real_episode_fake_layers_event_log_judge_preview(
    tmp_path: Path, episode_frame: pd.DataFrame, offline: None
) -> None:
    intent = FakeIntentModel()
    summary = write_translator_run(tmp_path / "translator", episode_frame, intent=intent)
    factory = RecordingFactory()

    code = cli.main(_args(tmp_path, tmp_path / "translator"), openai_client_factory=factory)

    run_dir = tmp_path / "out" / RUN_NAME
    preview = json.loads((run_dir / "preview.json").read_text())
    assert summary.status == "completed" and len(intent.prompts) == TRANSLATIONS
    assert code == cli.EXIT_OK and factory.calls == [] and preview["api_requests_made"] == 0
    assert (preview["plan"]["translation_events"], preview["plan"]["judged"]) == (TRANSLATIONS, 2)
    trajectory = _judge_preview_trajectory(run_dir)
    assert trajectory == _event(tmp_path / "translator", A1_START)["input"]["trajectory"]["text"]
    assert f"{WINDOW_BEGIN}\n{trajectory}\n{WINDOW_END}" in intent.prompts[0].user_text  # what the layer saw


def test_e2e_real_episode_production_adapters_event_log_judge_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: None
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "local-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = systemone_body({"unexplained_change": noul_answer(0.1)})
    server = SystemOneServer(
        [json_response(200, models_body("clef-flash"))] + [json_response(200, answer)] * DETECTOR_TICKS
    )
    translation_answer = json.dumps({"intent": "[FAKE] intent", "evidence": ["[FAKE] evidence"]})
    translation_client = ScriptedResponsesClient([judge_response(translation_answer) for _ in range(TRANSLATIONS)])
    translator_dir = tmp_path / "translator"

    translator_code = translator_cli.main(
        ["--scenario", str(EXAMPLE_SCENARIO), "--output-root", str(tmp_path), "--run-name", "translator"]
        + ["--execute", "--confirm-paid-api"],
        detector_client_factory=lambda config: translator_cli.build_system_one_client(
            over_tcp(config), transport=server.transport
        ),
        openai_client_factory=lambda *, timeout_s: translation_client,
    )
    judge_factory = RecordingFactory()
    code = cli.main(_args(tmp_path, translator_dir, EXAMPLE_KEY_POINTS), openai_client_factory=judge_factory)

    assert translator_code == translator_cli.EXIT_OK and len(server.requests) == 1 + DETECTOR_TICKS
    assert len(translation_client.requests) == TRANSLATIONS
    run_dir = tmp_path / "out" / RUN_NAME
    preview = json.loads((run_dir / "preview.json").read_text())
    assert code == cli.EXIT_OK and judge_factory.calls == [] and preview["api_requests_made"] == 0
    assert (preview["plan"]["translation_events"], preview["plan"]["judged"]) == (TRANSLATIONS, 2)
    trajectory = _judge_preview_trajectory(run_dir)
    sent_user_text = translation_client.requests[0]["input"][0]["content"][0]["text"]
    assert f"{WINDOW_BEGIN}\n{trajectory}\n{WINDOW_END}" in sent_user_text  # byte-identical to what the layer saw
    assert _event(translator_dir, A1_REQUEST)["output"]["intent"] == "[FAKE] intent"


def test_failures_after_an_uncertain_run_keep_exit_three(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original_close = cli.JsonlJudgeSink.close
    writes: list[Path] = []

    def close_then_fail(sink: Any) -> None:
        original_close(sink)
        raise OSError("simulated close failure")

    def write_json(path: Path, payload: object) -> None:
        writes.append(path)
        if len(writes) > 1:  # the final run.json
            raise OSError("simulated disk full")
        path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(cli.JsonlJudgeSink, "close", close_then_fail)
    monkeypatch.setattr(cli, "_write_json", write_json)
    client = ScriptedResponsesClient([APITimeoutError("slow")])

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(result=client),
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_UNCERTAIN and len(client.requests) == 1
    assert "closing the journals failed: OSError" in err and "final run.json not written: OSError" in err
    assert read_jsonl(tmp_path / "out" / RUN_NAME / "results.jsonl")[0]["status"] == "uncertain"


# --- review fixes (D38) ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../x", "a/b", "", "x y", ".", "..", "ABSOLUTE", "-x"])
def test_run_name_that_is_not_a_plain_name_is_a_usage_error_and_creates_nothing(
    tmp_path: Path, translator_run: Path, name: str
) -> None:
    value = str(tmp_path / "abs") if name == "ABSOLUTE" else name
    argv = ["--translator-run", str(translator_run), "--key-points", str(EXAMPLE_KEY_POINTS)]
    argv += ["--output-root", str(tmp_path / "out"), "--run-name", value]
    factory = RecordingFactory()

    with pytest.raises(SystemExit) as raised:
        cli.main(argv, openai_client_factory=factory)

    assert raised.value.code == cli.EXIT_USAGE and factory.calls == [] and sorted(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode", [[], ["--execute", "--confirm-paid-api"]])
def test_cap_equal_to_the_judged_events_but_below_all_events_judges_exactly_them(
    tmp_path: Path, translator_run: Path, mode: list[str]
) -> None:
    cap = len(EXAMPLE_JUDGED)
    config = write_yaml(tmp_path / "judge.yaml", with_value(judge_config_raw(), "limits.max_judged_events", cap))
    client = _scripted(("kp-001", "kp-002"), ("kp-001",))
    factory = RecordingFactory(result=client)

    code = cli.main(
        ["--config", str(config), *_args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, *mode)],
        openai_client_factory=factory,
    )

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert cap < len(EVENT_IDS) and code == cli.EXIT_OK
    assert record["plan"]["judged"] == cap and record["provenance"]["openai_base_url"] == "https://api.openai.com/v1"
    if not mode:
        assert factory.calls == [] and client.requests == []
        return
    results = read_jsonl(run_dir / "results.jsonl")
    assert record["status"] == "completed" and len(client.requests) == cap
    assert [r["event_id"] for r in results if r["status"] == "judged"] == list(EXAMPLE_JUDGED)
    assert [r["status"] for r in results].count("not_judged") == len(EVENT_IDS) - cap


# --- SDK environment (D44) and Ctrl-C outside a call ----------------------------------------------

PROXY_CANARY = "http://user:PW-canary-9f3a@proxy.example:3128"


def _clean_sdk_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*PROVENANCE_ENV, "OPENAI_LOG"):
        monkeypatch.delenv(name, raising=False)


def test_preview_records_sdk_environment_names_only_and_ignores_openai_log(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", PROXY_CANARY)
    monkeypatch.setenv("SSL_CERT_DIR", "/etc/ssl/PW-canary-9f3a")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-PW-canary-9f3a")
    monkeypatch.setenv("OPENAI_LOG", "debug")
    factory = RecordingFactory()

    code = cli.main(_args(tmp_path, translator_run), openai_client_factory=factory)

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_OK and factory.calls == []  # preview: no client, no refusal
    assert record["provenance"]["sdk_environment_set"] == ["HTTP_PROXY", "SSL_CERT_DIR", "OPENAI_PROJECT_ID"]
    assert all("PW-canary" not in path.read_text() for path in run_dir.iterdir())


def test_execute_records_sdk_environment_names_only(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("https_proxy", PROXY_CANARY)
    monkeypatch.setenv("NO_PROXY", "internal.example")

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=RecordingFactory(result=_scripted(("kp-001", "kp-002"), ("kp-001",))),
    )

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_OK and record["provenance"]["sdk_environment_set"] == ["NO_PROXY", "https_proxy"]
    assert all("PW-canary" not in path.read_text() for path in run_dir.iterdir())


def test_execute_with_openai_log_set_fails_before_any_call_without_echoing_the_value(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _clean_sdk_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("OPENAI_LOG", "debug-PW-canary")

    code = cli.main(  # the real make_openai_client refuses before it imports the SDK
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api")
    )

    run_dir = tmp_path / "out" / RUN_NAME
    record = json.loads((run_dir / "run.json").read_text())
    assert code == cli.EXIT_FAILED and record["status"] == "failed" and record["api_requests_made"] == 0
    assert record["error"]["type"] == "OpenAIClientError" and "OPENAI_LOG" in record["error"]["message"]
    assert "PW-canary" not in record["error"]["message"] + capsys.readouterr().err
    assert not (run_dir / "requests.jsonl").exists()


def _interrupting_sink(*args: Any, **kwargs: Any) -> Any:
    raise KeyboardInterrupt


@pytest.mark.parametrize("where", ["building_the_client", "opening_the_journals"])
def test_ctrl_c_outside_run_judge_ends_with_a_terminal_failed_run_json_and_no_traceback(
    tmp_path: Path,
    translator_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    where: str,
) -> None:
    factory = RecordingFactory(error=KeyboardInterrupt())
    if where == "opening_the_journals":
        factory = RecordingFactory(result=_scripted(("kp-001", "kp-002"), ("kp-001",)))
        monkeypatch.setattr(cli, "JsonlJudgeSink", _interrupting_sink)

    code = cli.main(
        _args(tmp_path, translator_run, EXAMPLE_KEY_POINTS, "--execute", "--confirm-paid-api"),
        openai_client_factory=factory,
    )

    record = json.loads((tmp_path / "out" / RUN_NAME / "run.json").read_text())
    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILED and "Traceback" not in err and "KeyboardInterrupt" in err
    assert record["status"] == "failed" and record["ended_at"] is not None
    assert record["error"]["type"] == "KeyboardInterrupt" and record["api_requests_made"] == 0


def test_ctrl_c_before_the_run_directory_exists_prints_no_traceback(
    tmp_path: Path, translator_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "load_inputs", interrupt)

    code = cli.main(_args(tmp_path, translator_run))

    err = capsys.readouterr().err
    assert code == cli.EXIT_FAILED and "KeyboardInterrupt" in err and "Traceback" not in err
    assert not (tmp_path / "out").exists()
