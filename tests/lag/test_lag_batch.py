"""Contracts of ``mtd_rl_xavier.lag.execution`` / ``batch`` with local fakes only (no API, no network).

The workspace 14-17 suites still run these modules with their real prompts and evaluation; these
tests pin the transport itself inside ``poe test``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mtd_rl_xavier.lag import batch
from mtd_rl_xavier.lag import execution as ex

MODEL_ALIASES = ("sol", "terra", "luna")
QUERY_COUNT = 10


def obj(data: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**data, model_dump=lambda **kwargs: deepcopy(data))


def make_plan() -> dict[str, Any]:
    sample = {
        "messages": [{"role": "user", "content": "Describe the window"}],
        "episode_id": "episode",
        "target_id": "A0100",
        "task": "commentary",
        "H": 61,
        "start": 15,
        "target_step": 45,
        "observation_scope": "all_aircraft_missiles",
        "sample_number": 0,
    }
    plan: dict[str, Any] = {
        "samples": {"sample": sample},
        "queries": [],
        "request_options": {"reasoning_effort": "medium"},
        "source_sha256": {},
    }
    for index in range(QUERY_COUNT):
        alias = MODEL_ALIASES[index % len(MODEL_ALIASES)]
        query = {
            "condition_id": alias,
            "sample_id": "sample",
            "model": f"gpt-5.6-{alias}",
            "model_alias": alias,
            "repeat_id": index,
        }
        query["request_sha256"] = ex.digest(ex.request_for(plan, query))
        query["query_id"] = ex.digest(query)
        plan["queries"].append(query)
    plan["plan_id"] = ex.digest(plan)
    return plan


class Recorder:
    """Stands in for an experiment's ``evaluation.evaluate_saved``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, plan: dict[str, Any], directory: Path) -> None:
        self.calls.append((plan["plan_id"], directory))


class FakeClient:
    def __init__(self) -> None:
        self.inputs: dict[str, list[dict[str, Any]]] = {}
        self.outputs: dict[str, str] = {}
        self.remote: dict[str, dict[str, Any]] = {}
        self.created: list[str] = []
        self.fail_create = False
        self.closed = False
        self.files = SimpleNamespace(create=self.upload, content=lambda id: SimpleNamespace(text=self.outputs[id]))
        self.batches = SimpleNamespace(
            create=self.create,
            retrieve=lambda id: obj(self.remote[id]),
            list=lambda **kwargs: [obj(x) for x in self.remote.values()],
        )

    def close(self) -> None:
        self.closed = True

    def upload(self, file: Any, purpose: str) -> SimpleNamespace:
        assert purpose == "batch"
        file_id = f"file_{len(self.inputs)}"
        self.inputs[file_id] = [json.loads(line) for line in file.read().decode().splitlines()]
        return SimpleNamespace(id=file_id)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        batch_id = f"batch_{len(self.remote)}"
        record = {"id": batch_id, "status": "in_progress", **kwargs}
        self.remote[batch_id] = record
        self.created.append(batch_id)
        if self.fail_create:
            raise TimeoutError("Connection lost AFTER remote acceptance")
        return obj(record)

    def finish(self, *, fail_first: bool = False) -> None:
        for batch_id, remote in self.remote.items():
            outputs, errors = [], []
            for index, row in enumerate(self.inputs[remote["input_file_id"]]):
                body = {"choices": [{"message": {"content": "answer"}}]}
                result = {"custom_id": row["custom_id"], "error": None, "response": {"status_code": 200, "body": body}}
                if fail_first and index == 0:
                    errors.append({**result, "response": None, "error": {"code": "batch_expired"}})
                else:
                    outputs.append(result)
            self.outputs[batch_id + "_out"] = "\n".join(map(json.dumps, outputs))
            self.outputs[batch_id + "_err"] = "\n".join(map(json.dumps, errors))
            remote.update(
                status="expired" if fail_first else "completed",
                output_file_id=batch_id + "_out",
                error_file_id=batch_id + "_err",
            )


@pytest.fixture
def run(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    plan = make_plan()
    ex.write_json(tmp_path / "manifest.json", plan)
    return plan, tmp_path


def test_digest_is_canonical_sorted_compact_json_sha256() -> None:
    assert ex.canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert ex.digest({"b": 1, "a": "é"}) == ex.digest({"a": "é", "b": 1})


def test_verify_plan_with_edited_message_raises_value_error() -> None:
    plan = make_plan()
    plan["samples"]["sample"]["messages"][0]["content"] = "edited"

    with pytest.raises(ValueError, match="Plan digest mismatch"):
        ex.verify_plan(plan)


def test_batch_execute_submits_once_and_collect_is_idempotent(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    client, evaluate = FakeClient(), Recorder()

    batch.execute(plan, directory, repository_root=directory, evaluate=evaluate, client_factory=lambda: client)
    client.finish()
    result = batch.execute(
        plan, directory, repository_root=directory, evaluate=evaluate, submit_new=False, client_factory=lambda: client
    )
    before = (directory / "api/responses.jsonl").read_bytes()
    batch.execute(plan, directory, repository_root=directory, evaluate=evaluate, client_factory=lambda: client)

    assert len(client.created) == len(MODEL_ALIASES)
    assert result["responses_saved"] == QUERY_COUNT and result["status"] == "complete"
    assert (directory / "api/responses.jsonl").read_bytes() == before
    assert evaluate.calls and all(call == (plan["plan_id"], directory) for call in evaluate.calls)


def test_batch_lost_submission_is_recovered_without_resend(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    client = FakeClient()
    client.fail_create = True

    with pytest.raises(TimeoutError):
        batch.execute(plan, directory, repository_root=directory, evaluate=Recorder(), client_factory=lambda: client)
    client.fail_create = False
    batch.execute(plan, directory, repository_root=directory, evaluate=Recorder(), client_factory=lambda: client)

    assert len(client.created) == len(MODEL_ALIASES)


def test_batch_partial_failure_keeps_answers_and_records_errors(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    client = FakeClient()
    batch.execute(plan, directory, repository_root=directory, evaluate=Recorder(), client_factory=lambda: client)
    client.finish(fail_first=True)

    result = batch.execute(
        plan, directory, repository_root=directory, evaluate=Recorder(), submit_new=False, client_factory=lambda: client
    )

    assert result["responses_saved"] == QUERY_COUNT - len(MODEL_ALIASES)
    assert result["unresolved"] == len(MODEL_ALIASES)


def test_batch_state_with_tampered_input_file_raises(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    state = batch.prepare(plan, directory)
    first = directory / "api/batch" / state["jobs"][0]["input_file"]
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash/path mismatch"):
        batch.load_state(plan, directory)


def test_batch_execute_with_changed_source_refuses_before_any_client(run: tuple[dict[str, Any], Path]) -> None:
    _, directory = run
    source = directory / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    plan = make_plan()
    plan = {**{k: v for k, v in plan.items() if k != "plan_id"}, "source_sha256": {"source.py": "0" * 64}}
    plan["plan_id"] = ex.digest(plan)
    ex.write_json(directory / "manifest.json", plan)

    with pytest.raises(ValueError, match="Source changed since planning: source.py"):
        batch.execute(
            plan,
            directory,
            repository_root=directory,
            evaluate=Recorder(),
            client_factory=lambda: pytest.fail("client"),
        )


def test_sync_execute_refuses_a_batch_run(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    batch.prepare(plan, directory)

    with pytest.raises(ValueError, match="Batch run cannot be executed by the synchronous runner"):
        ex.execute(plan, directory, repository_root=directory, client_factory=lambda: pytest.fail("client"))


def test_sync_execute_journals_every_request_before_its_response(run: tuple[dict[str, Any], Path]) -> None:
    plan, directory = run
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **request: obj({"echo": request["model"]}))),
        close=lambda: None,
    )

    summary = ex.execute(plan, directory, repository_root=directory, client_factory=lambda: client)

    requests = ex.read_journal(directory / "api/requests.jsonl")
    responses = ex.read_journal(directory / "api/responses.jsonl")
    assert summary["status"] == "complete" and summary["responses_saved"] == QUERY_COUNT
    assert [r["query_id"] for r in requests] == [r["query_id"] for r in responses]
    assert ex.read_json(directory / "api/identity.json") == {
        "plan_id": plan["plan_id"],
        "sdk_max_retries": 0,
        "timeout_seconds": 180,
    }
