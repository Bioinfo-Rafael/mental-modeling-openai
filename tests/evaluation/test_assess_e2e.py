"""E2E of the 3-points x 4-aspects evaluation on a real Translator run, event schema 3 (offline).

The Translator CLI runs the example scenario on the real 13_LAG episode (cut at step 30) with the
production adapters over fakes: ``SystemOneClient`` over ``httpx.MockTransport`` and the OpenAI
client over a scripted SDK stand-in. The detector answers ``change`` for A0100 at tick 15 (after the
alarm at 12) and ``none`` elsewhere, so the run holds the forced ``start`` events at the episode start
(開始点), one 要求点 (step 10, with its question) and one 変化点. The evaluation then runs
``check -> judge preview -> judge --execute -> import-ratings -> report`` with the default judge
config and the committed example annotation and ratings. No socket is opened anywhere.
"""

from __future__ import annotations

import json
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from evaluation import assess_cli, assess_data, trigger_kinds
from evaluation.assess_data import canonical_json, load_config, sha256_text
from tests.llm.llm_fakes import (
    DumpableResponse,
    ScriptedResponsesClient,
    SystemOneServer,
    json_response,
    message_item,
    models_body,
    responses_body,
)
from tests.translate.tr_fakes import (
    CHANGE,
    EXAMPLE_SCENARIO,
    RUN_CONFIG,
    SHORT_LAST_STEP,
    over_tcp,
    systemone_responder,
)
from translate import cli as translator_cli
from translate import events as translator_schema

from .assess_fakes import find, read_jsonl, verdict_text

EXAMPLES = Path(__file__).resolve().parents[2] / "src" / "evaluation" / "examples"
POINTS, RATINGS = EXAMPLES / "assess-points.example.yaml", EXAMPLES / "human-ratings.example.jsonl"
SCENARIO_ID = "placeholder-episode-01"
DETECTOR_ANSWERS = {("aircraft A0100", 15): CHANGE}
DETECTOR_CALLS, TRANSLATION_CALLS = 12, 4  # ticks 5..30 x {A0100, team A}; start x 2, request, change
E_START, E_REQUEST, E_CHANGE = (f"{SCENARIO_ID}:aircraft:A0100:{step:06d}" for step in (0, 10, 15))
E_TEAM_START = f"{SCENARIO_ID}:team:A:000000"


class SocketGuard:
    """Records (and refuses) every attempt to resolve a name or open a connection."""

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def install(self, patch: pytest.MonkeyPatch) -> None:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "getaddrinfo")):
            patch.setattr(owner, name, self._refuse(name))
        patch.setattr(socket, "create_connection", self._refuse("create_connection"))

    def _refuse(self, name: str) -> Any:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            self.attempts.append(name)
            raise OSError(f"network disabled in this test ({name})")

        return refuse


@pytest.fixture(scope="module")
def current_run(
    tmp_path_factory: pytest.TempPathFactory, episode_frame: pd.DataFrame
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """A completed Translator run (event schema 3) written by the real Translator CLI (read-only)."""
    frame = episode_frame[episode_frame["step"] <= SHORT_LAST_STEP]
    root, guard = tmp_path_factory.mktemp("assess-e2e"), SocketGuard()
    responder = systemone_responder(DETECTOR_ANSWERS)
    server = SystemOneServer([json_response(200, models_body("clef-flash")), *[responder] * DETECTOR_CALLS])
    answers = [json.dumps({"intent": f"[FAKE] intent {i}", "evidence": [f"[FAKE] evidence {i}"]}) for i in range(5)]
    client = ScriptedResponsesClient([DumpableResponse(responses_body(output=[message_item(a)])) for a in answers])
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch)
        patch.setattr(translator_cli, "load_episode", lambda path: frame)
        patch.setenv("VLLM_API_KEY", "local-test-key")
        argv = [
            "--config",
            str(RUN_CONFIG),
            "--scenario",
            str(EXAMPLE_SCENARIO),
            "--output-root",
            str(root / "translator"),
            "--run-name",
            "run1",
        ]
        code = translator_cli.main(
            [*argv, "--execute", "--confirm-paid-api"],
            factories=translator_cli.Factories(
                detector_client=lambda config: translator_cli.build_system_one_client(
                    over_tcp(config), transport=server.transport
                ),
                openai_client=lambda *, timeout_s: client,
            ),
        )
    assert code == 0 and guard.attempts == []
    yield root / "translator" / "run1", frame


class AutoClient:
    """SDK stand-in for the judge: every asked criterion gets "4"."""

    def __init__(self) -> None:
        self.responses, self.bodies = self, []

    def create(self, **body: Any) -> object:
        self.bodies.append(body)
        schema = body["text"]["format"]["schema"]
        ids = schema["properties"]["criteria"]["items"]["properties"]["criterion_id"]["enum"]
        return DumpableResponse(responses_body(output=[message_item(verdict_text(ids, "4"))]))


class Factory:
    def __init__(self, client: object = None) -> None:
        self.client, self.calls = client, 0

    def __call__(self, **kwargs: Any) -> object:
        self.calls += 1
        if self.client is None:
            raise AssertionError("no client may be built here")
        return self.client


def _assess(
    capsys: pytest.CaptureFixture[str], frame: pd.DataFrame, argv: list[str], factory: Factory | None = None
) -> tuple[int, dict[str, Any]]:
    capsys.readouterr()
    code = assess_cli.main(argv, openai_client_factory=factory or Factory(), episode_loader=lambda path: frame)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


# --- (a) the evaluation's own contract constants equal the Translator's ---------------------------------


def test_contract_constants_match_the_translator() -> None:
    assert trigger_kinds.CURRENT_EVENT_SCHEMA_VERSION == translator_schema.EVENT_SCHEMA_VERSION == 3
    assert trigger_kinds.TRIGGER_KINDS[3] == translator_schema.TRIGGER_PRECEDENCE == ("start", "change", "request")
    assert trigger_kinds.TRIGGER_KINDS[2] == ("initial", "start", "change", "request")  # runs before 2026-11-08
    assert assess_data.RECORD_TYPES == translator_schema.RECORD_TYPES
    assert assess_data.TARGET_KINDS == translator_schema.TARGET_KINDS
    assert assess_data.STEP_DIGITS == translator_schema.STEP_DIGITS
    assert assess_data.SAFE_ID_PATTERN.pattern == translator_schema.SAFE_ID_PATTERN.pattern
    default = load_config()
    assert sorted(default.triggers) == [3]  # schema 2 runs: assess-legacy.yaml only (2026-11-09)
    assert set(default.triggers[3].point_kinds) == set(trigger_kinds.TRIGGER_KINDS[3])


def test_the_legacy_config_refuses_a_schema_3_run_before_writing(
    tmp_path: Path, current_run: tuple[Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, frame = current_run
    legacy = assess_data.DEFAULT_CONFIG.parent / "assess-legacy.yaml"
    argv = ["check", "--translator-run", str(run_dir), "--output-root", str(tmp_path / "out")]

    code = assess_cli.main([*argv, "--config", str(legacy)], episode_loader=lambda path: frame)

    assert code == 1 and not (tmp_path / "out").exists()
    assert "rates event schema versions [2] only, not 3" in capsys.readouterr().err


# --- (b) check -> judge preview -> judge execute -> import-ratings -> report -----------------------------


def test_start_change_and_request_run_end_to_end(
    tmp_path: Path,
    current_run: tuple[Path, pd.DataFrame],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, frame = current_run
    guard = SocketGuard()
    guard.install(monkeypatch)
    base = ["--translator-run", str(run_dir), "--output-root", str(tmp_path / "out")]
    directory, config = tmp_path / "out" / "run1", load_config()

    # check: every call of both layers is ok; timing thresholds come from assess.yaml
    code, checked = _assess(capsys, frame, ["check", *base, "--points", str(POINTS)])
    assert code == 0
    assert checked["leakage"] == {"calls": 16, "ok": 16, "leak": 0, "error": 0, "invalid_targets": []}
    assert {line["layer"] for line in read_jsonl(directory / "leakage.jsonl")} == {"translation", "detector"}
    machine = read_jsonl(directory / "machine_results.jsonl")
    (latency,) = find(machine, criterion_id="timing.request_latency")
    assert config.timing.request_deadline_s == 3.0 and config.timing.min_question_interval_s == 10.0
    assert latency["subject"]["event_id"] == E_REQUEST and latency["measurement"]["request_deadline_s"] == 3.0
    assert latency["measurement"]["processing_s"] >= 0 and latency["measurement"]["passed"] is True
    intervals = find(machine, criterion_id="timing.question_interval")
    assert {(r["reason"], r["measurement"]["requests"]) for r in intervals} == {
        ("fewer_than_two_requests", 1),  # A0100: one request, so the 10.0 s check has nothing to compare
        ("fewer_than_two_requests", 0),  # team A
    }
    points = {r["subject"]["reference_point_id"]: r for r in find(machine, criterion_id="timing.point_matched")}
    assert points["p-start-000"]["reason"] == "forced_trigger"  # the episode start
    assert (
        points["p-change-001"]["measurement"]["delay_steps"] == 1
        and points["p-change-001"]["subject"]["event_id"] == E_CHANGE
    )

    # judge preview: packets, no client, no call; each point kind gets its own criteria
    preview_factory = Factory()
    code, preview = _assess(capsys, frame, ["judge", *base, "--points", str(POINTS)], preview_factory)
    assert code == 0 and preview_factory.calls == 0 and preview["plan"]["to_send"] == TRANSLATION_CALLS
    assert not (directory / "judge_requests.jsonl").exists()
    (packets_file,) = directory.glob("packets-*.jsonl")
    packets = {packet["event_id"]: packet for packet in read_jsonl(packets_file)}
    asked = {event_id: set(packet["asked_criteria"]) for event_id, packet in packets.items()}
    assert "correctness.start_future" in asked[E_START] - asked[E_CHANGE] - asked[E_REQUEST]
    assert "correctness.change_cause" in asked[E_CHANGE] - asked[E_START] - asked[E_REQUEST]
    assert "correctness.unknown_not_asserted" in asked[E_REQUEST] - asked[E_START] - asked[E_CHANGE]
    assert packets[E_START]["excluded_criteria"] == {}  # no start timing: the episode start is fixed
    assert packets[E_REQUEST]["future_end_step"] is None and "## FUTURE" not in packets[E_REQUEST]["text"]
    assert packets[E_START]["future_end_step"] == 12  # the alarm at 12 ends the start's FUTURE block
    assert "## TRAJECTORY (human-visible telemetry up to t)" in packets[E_START]["text"]
    assert packets[E_CHANGE]["future_end_step"] == 20  # team A's new instruction reaches A0100 at 20
    assert "it ends before step 20 (next_instruction)" in packets[E_CHANGE]["text"]
    assert "mhi_v1.3" not in "".join(packet["text"] for packet in packets.values())

    # judge execute with a scripted Responses client
    client = AutoClient()
    flags = ["--execute", "--confirm-paid-api"]
    code, judged = _assess(capsys, frame, ["judge", *base, "--points", str(POINTS), *flags], Factory(client))
    assert code == 0 and judged["counts"]["judged"] == TRANSLATION_CALLS == len(client.bodies)
    assert all(body["model"] == "gpt-6.1-sol" and body["store"] is False for body in client.bodies)
    judge = read_jsonl(directory / "judge_results.jsonl")
    assert {r["evaluator"]["kind"] for r in judge} == {"judge"}
    for event_id, criteria in asked.items():
        assert {r["criterion_id"] for r in find(judge, event_id=event_id)} == criteria | {"overall"}
    judge_bytes = (directory / "judge_results.jsonl").read_bytes()

    # import-ratings (cannot_judge, not_applicable and a superseding re-rating)
    code, imported = _assess(capsys, frame, ["import-ratings", *base, "--ratings", str(RATINGS)])
    assert code == 0 and imported["ratings"] == 4
    assert (directory / "judge_results.jsonl").read_bytes() == judge_bytes  # nothing overwrote the judge store
    human = read_jsonl(directory / "human_results.jsonl")
    assert {r["evaluator"]["kind"] for r in human} == {"human"}
    (cannot,) = find(human, criterion_id="correctness.unknown_not_asserted")
    assert (cannot["status"], cannot["score"], cannot["reason"]) == ("not_evaluated", None, "cannot_judge")

    # report: not_evaluated is never 0; human and judge side by side, never merged
    code, reported = _assess(capsys, frame, ["report", *base])
    assert code == 0
    report = json.loads(Path(reported["report"]).read_text(encoding="utf-8"))
    section = report["sections"]["online"]
    assert report["sections"]["retrospective"] is None and section["leakage"]["ok"] == 16
    unknown = section["cells"]["request"]["correctness"]["correctness.unknown_not_asserted"]
    assert unknown["reference"] == "observable_records"
    assert (unknown["human"]["n_evaluated"], unknown["human"]["mean"]) == (0, None)
    assert unknown["human"]["not_evaluated"] == {"cannot_judge": 1}
    assert (unknown["judge"]["n_evaluated"], unknown["judge"]["mean"]) == (1, 4.0)
    resolved = section["cells"]["request"]["contribution"]["contribution.question_resolved"]
    assert (resolved["human"]["mean"], resolved["judge"]["mean"]) == (3.0, 4.0)  # r-0004 supersedes r-0001
    assert "timing.start_in_time" not in json.dumps(section["cells"]["start"])
    assert section["cells"]["change"]["correctness"]["correctness.change_cause"]["human"]["not_evaluated"] == {
        "no_human_rating": 1
    }
    assert report["overall"]["status"] == "not_evaluated" and report["overall"]["reason"] == "aggregation_unset"
    assert guard.attempts == []


def test_a_changed_deadline_in_the_config_changes_the_machine_verdict_input(
    tmp_path: Path, current_run: tuple[Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir, frame = current_run
    config_path = tmp_path / "assess.yaml"
    text = assess_data.DEFAULT_CONFIG.read_text(encoding="utf-8")
    config_path.write_text(text.replace("request_deadline_s: 3.0", "request_deadline_s: 2.0", 1), encoding="utf-8")

    code, _ = _assess(
        capsys,
        frame,
        [
            "check",
            "--translator-run",
            str(run_dir),
            "--output-root",
            str(tmp_path / "out"),
            "--config",
            str(config_path),
        ],
    )

    (latency,) = find(
        read_jsonl(tmp_path / "out" / "run1" / "machine_results.jsonl"), criterion_id="timing.request_latency"
    )
    assert code == 0 and latency["measurement"]["request_deadline_s"] == 2.0


# --- (c) a payload that shows a step after t ---------------------------------------------------------------


def test_a_future_step_in_one_payload_is_a_leak_that_invalidates_its_target(
    tmp_path: Path, current_run: tuple[Path, pd.DataFrame], capsys: pytest.CaptureFixture[str]
) -> None:
    source, frame = current_run
    run_dir = Path(shutil.copytree(source, tmp_path / "run1"))
    events, requests = read_jsonl(run_dir / "events.jsonl"), read_jsonl(run_dir / "requests.jsonl")
    (event,) = [e for e in events if e.get("event_id") == E_REQUEST]
    last = json.loads(event["input"]["trajectory"]["text"].split("\n")[-1])
    future = {**last, "step": last["step"] + 1, "time_s": round((last["step"] + 1) * 0.2, 1), "relative_step": 1}
    future_line = json.dumps(future, ensure_ascii=False, separators=(",", ":"))
    (request,) = [r for r in requests if r["call_id"] == E_REQUEST]
    content = request["payload"]["input"][0]["content"][0]
    leaked_text = content["text"].replace("\nEND WINDOW DATA", f"\n{future_line}\nEND WINDOW DATA", 1)
    assert leaked_text != content["text"]
    payload = {
        **request["payload"],
        "input": [{**request["payload"]["input"][0], "content": [{**content, "text": leaked_text}]}],
    }
    digest = sha256_text(canonical_json(payload))  # a consistent log: the Translator recorded what it sent
    _write_jsonl(run_dir / "requests.jsonl", [{**r, "payload": payload} if r is request else r for r in requests])
    _write_jsonl(run_dir / "events.jsonl", [{**e, "request_sha256": digest} if e is event else e for e in events])

    code, checked = _assess(
        capsys, frame, ["check", "--translator-run", str(run_dir), "--output-root", str(tmp_path / "out")]
    )

    assert code == 1 and checked["leakage"]["leak"] == 1 and checked["leakage"]["invalid_targets"] == ["aircraft:A0100"]
    (line,) = [line for line in read_jsonl(tmp_path / "out" / "run1" / "leakage.jsonl") if line["call_id"] == E_REQUEST]
    assert line["status"] == "leak" and [p["check"] for p in line["problems"]] == ["trajectory_future"]
    machine = [r for r in read_jsonl(tmp_path / "out" / "run1" / "machine_results.jsonl") if r["aspect"] != "overall"]
    a0100 = [r for r in machine if r["subject"]["target"] == {"kind": "aircraft", "id": "A0100"}]
    team = [r for r in machine if r["subject"]["target"] == {"kind": "team", "id": "A"}]
    assert a0100 and {(r["status"], r["reason"]) for r in a0100} == {("not_evaluated", "generation_input_invalid")}
    assert any(r["status"] == "evaluated" for r in team)  # the other target keeps its results
    code, preview = _assess(
        capsys, frame, ["judge", "--translator-run", str(run_dir), "--output-root", str(tmp_path / "out")]
    )
    assert (
        code == 0 and preview["plan"]["packets"] == 1 and preview["plan"]["generation_input_invalid"] == 3
    )  # A0100: start, request, change


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
