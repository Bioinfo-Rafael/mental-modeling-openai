"""Offline resume safety tests. No API client/network needed."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import common


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    directory = tmp_path / "results"
    directory.mkdir()
    q = {"query_id": "q", "condition_id": "c", "model": "gpt-3.5-turbo",
         "system_prompt": "system", "user_prompt": "user"}
    plan = {"experiment": "03_gpt35_history_n30", "queries": [q],
            "provenance": {"semantics_sha256": "same"}, "upstream_retries": 0,
            "conditions": [{"condition_id": "c", "logical_queries": 1}]}
    request = {"query_id": "q", "attempt_id": "q:1", "api_request_made": True,
               "kwargs": {"model": q["model"], "temperature": 0, "messages": [
                   {"role": "system", "content": "system"},
                   {"role": "user", "content": "user"}]}}
    raw = {"id": "offline", "object": "chat.completion", "created": 0,
           "model": q["model"], "choices": [{"index": 0, "finish_reason": "stop",
           "message": {"role": "assistant", "content": "2"}}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
    response = {"query_id": "q", "attempt_id": "q:1", "api_request_made": True,
                "raw_response": raw, "request_elapsed_seconds": 2.5}
    def save(name, value):
        (directory / name).write_text(json.dumps(value) + "\n")
    save("manifest.json", plan)
    save("requests.jsonl", request)
    save("responses.jsonl", response)
    (directory / "records.jsonl").write_text("")
    return directory, plan, request, response, save


def test_recovers_response_before_scoring_and_counts(journal):
    directory, plan, _, _, _ = journal
    cached, uncertain, destination = common.prepare_resume(plan, directory)
    assert set(cached) == {"q"} and not uncertain
    assert cached["q"]["query_time_recovered_from_request"]
    assert plan["planned_api_requests"] == 0
    assert destination == directory / "resumes/0001"


def test_timeout_requires_explicit_acknowledgment(journal):
    directory, plan, _, response, save = journal
    response["raw_response"] = None
    save("responses.jsonl", response)
    cached, uncertain, _ = common.prepare_resume(plan, directory)
    assert not cached and uncertain == {"q"}
    assert plan["planned_api_requests"] == 1
    assert not plan["retry_uncertain_confirmed"]


def test_prompt_change_refuses_resume(journal):
    directory, plan, _, _, _ = journal
    changed = copy.deepcopy(plan)
    changed["queries"][0]["user_prompt"] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        common.prepare_resume(changed, directory)


def test_request_parameter_change_refuses_resume(journal):
    directory, plan, request, _, save = journal
    request["kwargs"]["temperature"] = 1
    save("requests.jsonl", request)
    with pytest.raises(ValueError, match="parameters"):
        common.prepare_resume(plan, directory)


def test_scored_response_missing_refuses_resend(journal):
    directory, plan, _, _, save = journal
    (directory / "responses.jsonl").write_text("")
    save("records.jsonl", {"query_id": "q", "status": "match"})
    with pytest.raises(ValueError, match="missing saved responses"):
        common.prepare_resume(plan, directory)


def test_paid_resume_without_uncertain_ack_never_executes(journal, monkeypatch):
    directory, plan, _, response, save = journal
    response["raw_response"] = None
    save("responses.jsonl", response)
    spec = common.Experiment(plan["experiment"], ("3.5",), (), (), (), 1)
    monkeypatch.setattr(common, "EXPERIMENTS", directory.parent)
    # _run computes its source path; retain fixture journal via a bounded wrapper.
    prepare = common.prepare_resume
    monkeypatch.setattr(common, "prepare_resume", lambda p, d, a: prepare(p, directory, a))
    monkeypatch.setattr(common, "make_plan", lambda *a: copy.deepcopy(plan))
    monkeypatch.setattr(common, "check_inputs", lambda *a: {})
    execute = Mock(side_effect=AssertionError("must not send"))
    monkeypatch.setattr(common, "execute_plan", execute)
    assert common._run(spec, ["--resume", "--execute", "--confirm-paid-api"]) == 2
    execute.assert_not_called()


def test_resumed_failure_retains_earlier_cache(journal):
    directory, plan, _, _, _ = journal
    later = directory / "resumes/0001"
    later.mkdir(parents=True)
    (later / "manifest.json").write_text(json.dumps(plan))
    for name in ("requests.jsonl", "responses.jsonl", "records.jsonl"):
        (later / name).write_text("")
    cached, uncertain, destination = common.prepare_resume(plan, directory)
    assert set(cached) == {"q"} and not uncertain
    assert destination.name == "0002"
    assert common.latest_results(directory) == later


def test_cached_prefix_is_not_sent_and_original_latency_is_preserved(journal, monkeypatch):
    directory, plan, request, response, _ = journal
    row = plan["queries"][0]
    row["reuse_required"] = False
    second = {**row, "query_id": "q2", "user_prompt": "second"}
    cached, _, _ = common.prepare_resume({**plan, "queries": [{k: v for k, v in row.items() if k != "reuse_required"}]}, directory)
    network = Mock(return_value=__import__("openai").types.chat.ChatCompletion.model_validate(response["raw_response"]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=network)),
                             base_url="https://api.openai.com/v1", with_options=lambda **kw: client)
    def initialize(self, **kwargs):
        self._client, self.model, self.retries = client, kwargs["model"], 0
    monkeypatch.setattr(common.OpenAIChatBackend, "__init__", initialize)
    monkeypatch.setattr(common, "verify_prompt", lambda *a: None)
    monkeypatch.setattr(common, "append_jsonl", lambda *a: None)
    session = common.RecordingSession([row, second], directory, plan["experiment"], cached)
    backend = session.backend_class()(model=row["model"])
    assert backend.complete(system_prompt="system", user_prompt="user") == "2"
    network.assert_not_called()
    assert session.completed[0]["query_elapsed_seconds"] == 2.5
    assert backend.complete(system_prompt="system", user_prompt="second") == "2"
    assert network.call_count == 1 and session.api_attempts == 1
