"""The CLI (paid gate, preview, run directory), the config rules, the OpenAI backend and the detector client."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest
import yaml

from translate import cli
from translate.backends import BackendError, Prompt, input_text
from translate.backends.openai import OpenAIBackend, make_client, parse_reply
from translate.config import ConfigError, load_config
from translate.detector import DetectorError, SystemOneClient, check_tcp, parse_decision, request_body
from translate.journal import request_sha256

from .fakes import INSTRUCTION, ONLINE, PRESET, QUESTION, FakeBackend, clef, config, lines, run, scenario


def _refuse(*_: object) -> object:
    raise AssertionError("a preview built a backend or a detector client")


def test_a_preview_builds_nothing_paid_and_writes_the_exact_first_request(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = config(tmp_path, scenario(tmp_path, [INSTRUCTION]))

    code = cli.main(["--config", str(path), "--run-name", "run"], backend_factory=_refuse, detector_factory=_refuse)

    preview = json.loads((tmp_path / "out" / "run-preview" / "preview.json").read_text())
    assert code == 0 and "API requests made: 0" in capsys.readouterr().err
    assert preview["mandatory_translations"] == 1 and preview["first_request"]["store"] is False
    assert not (tmp_path / "out" / "run-preview" / "requests.jsonl").exists()


@pytest.mark.parametrize("shipped", [ONLINE, PRESET])
def test_the_shipped_configs_preview_without_a_client(shipped: Path, tmp_path: Path) -> None:
    raw = {**yaml.safe_load(shipped.read_text(encoding="utf-8")), "output": str(tmp_path / "out")}
    path = tmp_path / shipped.name
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert cli.main(["--config", str(path)], backend_factory=_refuse, detector_factory=_refuse) == 0


@pytest.mark.parametrize("flags", [["--execute"], ["--confirm-paid-api"]])
def test_one_paid_flag_alone_is_a_usage_error(tmp_path: Path, flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(config(tmp_path, scenario(tmp_path))), *flags])

    assert exc.value.code == 2


def test_a_resubmitted_default_name_paid_run_is_refused_before_the_backend_is_built(tmp_path: Path) -> None:
    path, built = config(tmp_path, scenario(tmp_path)), []

    def factory(model: object) -> FakeBackend:
        built.append(model)
        return FakeBackend()

    codes = [
        cli.main(["--config", str(path), "--execute", "--confirm-paid-api"], backend_factory=factory) for _ in range(2)
    ]

    assert codes == [0, 1] and len(built) == 1
    assert [p.name for p in (tmp_path / "out").iterdir()] == [f"config-{load_config(path).sha256[:8]}"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: {**raw, "extra": 1}, "extra"),
        (
            lambda raw: {**raw, "prompt": {**raw["prompt"], "user": raw["prompt"]["user"] + "{oops}"}},
            "unknown placeholders",
        ),
        (lambda raw: {**raw, "inputs": {**raw["inputs"], "alarms": None, "recent_messages": {}}}, "unused inputs"),
        (lambda raw: {**raw, "inputs": {**raw["inputs"], "nope": {}}}, "unknown inputs"),
        (lambda raw: {**raw, "inputs": {**raw["inputs"], "trajectory": {"past_steps": 0}}}, "past_steps"),
        (lambda raw: {**raw, "change": clef(socket="/tmp/$USER/s.sock")}, "placeholder"),
        (lambda raw: {**raw, "change": clef(socket=None, base_url="http://127.0.0.1:8000")}, "api_key_env"),
        (lambda raw: {**raw, "change": clef(base_url="http://127.0.0.1:8000")}, "exactly one"),
        (lambda raw: {**raw, "change": clef(api_key_env="sk-pasted-key")}, "api_key_env"),
    ],
)
def test_a_bad_run_config_fails_at_load_naming_the_key(tmp_path: Path, change, message: str) -> None:
    path = config(tmp_path, scenario(tmp_path))
    path.write_text(yaml.safe_dump(change(yaml.safe_load(path.read_text()))), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("messages", "targets", "message"),
    [
        ([{**QUESTION, "to": ["team:A"]}], ["aircraft:A0100"], "which targets does not list"),
        ([QUESTION, {**QUESTION, "id": "q2"}], ["aircraft:A0100"], "one question per target per step"),
        ([], ["plane:A0100"], "pattern"),
    ],
)
def test_a_bad_scenario_fails_at_load(tmp_path: Path, messages: list, targets: list, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(config(tmp_path, scenario(tmp_path, messages, targets)))


def test_a_duplicate_yaml_key_fails_at_load(tmp_path: Path) -> None:
    path = config(tmp_path, scenario(tmp_path))
    path.write_text(path.read_text() + "output: elsewhere\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)


def test_a_detector_budget_the_states_cannot_fit_fails_before_any_directory(tmp_path: Path) -> None:
    path = config(tmp_path, scenario(tmp_path), clef(input_token_budget=2000))

    assert run(path, paid=False) == 1 and not (tmp_path / "out").exists()


# --- the OpenAI backend ---------------------------------------------------------------------------


class _SDKErrors:
    """Stand-ins for the openai SDK exception classes (only their names and module matter)."""

    APIError = type("APIError", (Exception,), {"__module__": "openai._exceptions"})
    APIConnectionError = type("APIConnectionError", (APIError,), {"__module__": "openai._exceptions"})
    APITimeoutError = type("APITimeoutError", (APIConnectionError,), {"__module__": "openai._exceptions"})
    APIStatusError = type("APIStatusError", (APIError,), {"__module__": "openai._exceptions", "status_code": 503})


class _SDK:
    def __init__(self, outcome: object) -> None:
        self.responses, self.outcome = self, outcome

    def create(self, **body: object) -> object:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _response(
    status: str = "completed", text: str = '{"intent": "x", "evidence": ["e"]}', part: str = "output_text"
) -> dict:
    key = "text" if part == "output_text" else "refusal"
    return {
        "id": "r1",
        "model": "m",
        "status": status,
        "output": [{"type": "message", "content": [{"type": part, key: text}]}],
    }


def test_the_backend_body_is_strict_unstored_and_hashable() -> None:
    model = load_config(PRESET).config.model
    body = OpenAIBackend.request_body(
        model, Prompt("sys", (input_text("user"),), "intent_translation", {"type": "object"})
    )

    assert body["store"] is False and body["text"]["format"]["strict"] is True and "temperature" not in body
    assert (body["model"], body["reasoning"], body["max_output_tokens"]) == ("gpt-6.1-sol", {"effort": "high"}, 25000)
    assert len(request_sha256(body)) == 64


@pytest.mark.parametrize(
    ("outcome", "uncertain"),
    [
        (_SDKErrors.APITimeoutError(), True),
        (_SDKErrors.APIConnectionError(), False),
        (_SDKErrors.APIStatusError(), False),
        (_SDKErrors.APIError(), True),
        (_response(status="incomplete"), False),
        (_response(part="refusal", text="no"), False),
        (_response(status="queued"), True),
        ({"output": "broken"}, True),
    ],
)
def test_backend_failures_are_failed_or_uncertain_and_never_retried(outcome: object, uncertain: bool) -> None:
    sdk = _SDK(outcome)

    with pytest.raises(BackendError) as exc:
        OpenAIBackend(None, client=sdk).send({"model": "m"})

    assert exc.value.uncertain is uncertain


def test_a_completed_response_parses_into_a_reply() -> None:
    reply = parse_reply({**_response(), "usage": {"input_tokens": 3, "output_tokens": 2}})

    assert (reply.text, reply.response_id, reply.usage["input_tokens"]) == (
        '{"intent": "x", "evidence": ["e"]}',
        "r1",
        3,
    )


def test_the_client_refuses_sdk_logging_and_needs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_LOG", "debug")
    with pytest.raises(BackendError, match="OPENAI_LOG"):
        make_client(timeout_s=1)
    monkeypatch.delenv("OPENAI_LOG")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        make_client(timeout_s=1)


# --- the detector client --------------------------------------------------------------------------


def _clef(tmp_path: Path, **changes: object) -> object:
    return load_config(config(tmp_path, scenario(tmp_path), clef(**changes))).config.change


def test_the_detector_client_sends_one_choice_question_over_loopback_with_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "k3y")
    change = _clef(tmp_path, socket=None, base_url="http://127.0.0.1:8000", api_key_env="VLLM_API_KEY")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": change.model}]})
        answer = {"type": "choice", "choice": "change", "probabilities": {"none": 0.2, "change": 0.8}}
        return httpx.Response(
            200,
            json={"model": "m", "answers": {"point_kind": answer}, "usage": {"input_tokens": 5, "output_tokens": 1}},
        )

    client = SystemOneClient(change, transport=httpx.MockTransport(handler))
    client.health()
    label, probabilities, _ = parse_decision(client.decide(request_body(change, "state")), change)

    assert (label, probabilities["change"]) == ("change", 0.8)
    assert (
        seen[1].headers["authorization"] == "Bearer k3y"
        and json.loads(seen[1].content)["questions"]["point_kind"]["type"] == "choice"
    )


def test_the_detector_refuses_remote_hosts_open_socket_directories_and_truncated_states(tmp_path: Path) -> None:
    for url in ("http://10.0.0.1:8000", "http://user:pw@127.0.0.1:8000", "http://127.0.0.1:8000/v1"):
        with pytest.raises(DetectorError):
            check_tcp(url)
    (tmp_path / "sock").mkdir(mode=0o755)
    os.chmod(tmp_path / "sock", 0o755)
    with pytest.raises(DetectorError, match="0700"):
        SystemOneClient(_clef(tmp_path / "c", socket=str(tmp_path / "sock" / "s.sock")))
    over = {"answers": {"point_kind": {"type": "choice", "choice": "none", "probabilities": {"none": 1, "change": 0}}}}
    with pytest.raises(DetectorError, match="truncated"):
        parse_decision({**over, "usage": {"input_tokens": 16384, "output_tokens": 1}}, _clef(tmp_path / "d"))


def test_detector_calls_are_journaled_like_translations(tmp_path: Path) -> None:
    from .fakes import FakeDetector

    run(config(tmp_path, scenario(tmp_path), clef()), detector=FakeDetector())

    requests = lines(tmp_path / "out" / "run" / "requests.jsonl")
    assert {r["layer"] for r in requests} == {"translation", "detector"}
    assert all(r["request_sha256"] == request_sha256(r["body"]) for r in requests)
