import json
import getpass
import os
import re
import shutil
import sys
import warnings

import httpx
from openai import OpenAI
import pytest

from decision_experiment import planning
from decision_experiment import prompts
from decision_experiment.dataset import build_blocks, lag, make_samples
from decision_experiment.evaluation import compare_runs, export_run
from decision_experiment.execution import execute, read_api_key
from decision_experiment.metrics import match_events
from decision_experiment.outputs import estimate_cost, parse_output
from decision_experiment.prompts import build_request, load_prompt
from decision_experiment.storage import HERE, append_jsonl, read_json, read_jsonl, write_json


@pytest.fixture(scope="module")
def blocks():
    annotations = read_json(HERE / "data/annotations.json")
    return build_blocks(lag.load_episode(lag.PROCESSED / (annotations["episode_id"] + ".csv")))


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(planning, "RESULTS", tmp_path / "results")
    return planning.create_plan("develop", "v001", "test")


@pytest.mark.parametrize("history", [30, 40])
def test_exact_positive_groups_and_weak_negatives(blocks, history):
    annotations = read_json(HERE / "data/annotations.json")
    samples = make_samples("develop", {"history": history}, annotations, blocks)
    assert len(samples) == 10
    positives = [s for s in samples.values() if s["expected_decision"] == "YES"]
    negatives = [s for s in samples.values() if s["expected_decision"] == "NO"]
    assert [[e["step"] for e in s["expected_events"]] for s in positives] == [[46, 49], [83, 95], [143], [243], [311]]
    assert len(negatives) == 5
    for sample in samples.values():
        w = sample["window"]
        assert [b["step"] for b in w["blocks"]] == list(range(w["start_step"], w["end_step"] + 1))
        assert len(w["blocks"]) == history
        assert w["elapsed_s"] == round((history - 1) * .2, 1)
    assert all(s["label_basis"] == "annotation_absent_not_reviewed" for s in negatives)


@pytest.mark.parametrize("history,count", [(40, 62), (30, 64)])
def test_sliding_causal_windows_complete_episode(blocks, history, count):
    samples = make_samples("evaluate", {"history": history, "stride": 5}, read_json(HERE / "data/annotations.json"), blocks)
    assert len(samples) == count
    assert list(samples.values())[0]["window"]["start_step"] == 0
    assert list(samples.values())[-1]["window"]["end_step"] == 340
    for sample in samples.values():
        assert max(b["step"] for b in sample["window"]["blocks"]) == sample["window"]["end_step"]


def test_masking_and_unobserved_fields(blocks):
    assert 341 not in blocks
    for block in blocks.values():
        for entity in block["entities"]:
            assert "target_id" not in entity and "firing_action" not in entity
            if entity["state_valid"] != 1:
                assert all(v is None for v in entity["state"])
            if entity["entity_type"] == "aircraft" and entity["action_status"] != "recorded":
                assert entity["action"] == [None, None, None]


def test_snapshot_requests_have_no_ground_truth_and_are_stable(plan):
    manifest = planning.verify_plan(plan)
    assert len(manifest["queries"]) == 10
    q = manifest["queries"][0]
    request = planning.request_for(plan, manifest, q)
    rendered = json.dumps(request, ensure_ascii=False)
    for hidden in ("expected_decision", "expected_events", "ヒットアンドアウェー", "positive_end", "scene01", "mhi_v1.3_ic100601_win"):
        assert hidden not in rendered
    assert request["model"] == "gpt-6-luna"
    assert request["text"]["format"]["strict"]
    with pytest.raises(FileExistsError):
        planning.create_plan("develop", "v001", "test")
    (plan / "prompt/system.txt").write_text("tampered")
    with pytest.raises(ValueError, match="Frozen run input changed"):
        planning.verify_plan(plan)


def test_same_semantic_task_and_choices_across_api_adapters(plan):
    manifest = planning.verify_plan(plan)
    prompt = load_prompt(plan / "prompt")
    text = (plan / "inputs/case01.txt").read_text()
    bodies = {c: build_request(c, prompt, text, manifest["config"]) for c in ("responses_choice", "responses_diagnostic", "decisions")}
    assert bodies["decisions"]["input"][0] == bodies["responses_choice"]["input"][0]
    assert bodies["responses_choice"]["input"][1]["content"].startswith(bodies["decisions"]["input"][1]["content"])
    for choice in bodies["decisions"]["questions"][0]["choices"]:
        assert choice["description"] in bodies["responses_choice"]["input"][1]["content"]
    assert "reasoning" not in bodies["decisions"]
    assert set(bodies["responses_choice"]["text"]["format"]["schema"]["properties"]) == {"decision"}


def responses_raw(payload, status="completed"):
    return {"id": "resp_mock", "object": "response", "created_at": 0, "model": "gpt-6-luna",
            "status": status, "output": [{"type": "message", "id": "msg_mock", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 20, "total_tokens": 1020,
                      "input_tokens_details": {"cached_tokens": 200}, "output_tokens_details": {"reasoning_tokens": 0}}}


def event(step=46, aircraft="A0100"):
    return {"aircraft_id": aircraft, "step": step, "evidence_steps": [step], "reason": "Observed a change in the recorded action."}


@pytest.mark.parametrize("payload", [
    {"decision": "YES", "events": [], "reason": "Unsupported"},
    {"decision": "NO", "events": [event()], "reason": "Contradiction"},
    {"decision": "YES", "events": [event(56)], "reason": "Future"},
    {"decision": "YES", "events": [event(True)], "reason": "Boolean is not a step"},
    {"decision": "YES", "events": [event(), event()], "reason": "Duplicate"},
    {"decision": "YES", "events": [event(46, "B0100")], "reason": "Wrong aircraft"},
])
def test_invalid_diagnostics_are_not_no(payload):
    with pytest.raises(ValueError):
        parse_output(responses_raw(payload), "responses_diagnostic", {"start_step": 16, "end_step": 55})


def test_truncation_and_refusal_are_errors():
    with pytest.raises(ValueError, match="not completed"):
        parse_output(responses_raw({"decision": "NO"}, "incomplete"), "responses_choice", {})
    raw = responses_raw({"decision": "NO"})
    raw["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    with pytest.raises(ValueError, match="refusal"):
        parse_output(raw, "responses_choice", {})
    with pytest.raises(ValueError, match="refusal"):
        parse_output({"answers": [{"type": "refusal", "name": "explanation_needed"}]}, "decisions", {})


def test_one_to_one_matching_does_not_credit_two_events_to_one_prediction():
    truth = [{"id": "a", "step": 46, "aircraft_id": "A0100"}, {"id": "b", "step": 49, "aircraft_id": "A0100"}]
    assert len(match_events([event(47)], truth, 2)) == 1
    assert {m["event_id"] for m in match_events([event(47), event(46)], truth, 2)} == {"a", "b"}
    assert match_events([event(46, "A0200")], truth, 2) == []
    assert match_events([event(47)], truth, 0) == []


def test_cost_subsets_unknown_and_long_context():
    p = read_json(HERE / "configs/pricing.json")
    u = {"input_tokens": 1000, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 100}}
    assert estimate_cost(u, "responses_choice", p) == pytest.approx((700*.1 + 200*.01 + 100*.125 + 20*.5)/1e6)
    assert estimate_cost(u, "decisions", p) == pytest.approx((700*.1 + 200*.01 + 100*.125)/1e6)
    assert estimate_cost(None, "decisions", p) is None
    assert estimate_cost({"input_tokens": 1, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 2}}, "decisions", p) is None
    assert estimate_cost({"input_tokens": 300000, "output_tokens": 20}, "responses_choice", p) == pytest.approx((300000*.1*2+20*.5*1.5)/1e6)


def test_real_sdk_mock_execution_resume_and_export(plan):
    calls = []
    def handler(request):
        calls.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        return httpx.Response(200, json=responses_raw({"decision": "NO", "events": [], "reason": "No observed transition."}))
    with OpenAI(api_key="offline-test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0) as client:
        execute(plan, client=client, limit=2)
        with pytest.raises(ValueError, match="resume"):
            execute(plan, client=client)
        execute(plan, client=client, resume=True)
    assert len(calls) == 10
    summary = export_run(plan)["conditions"]["responses_diagnostic"]
    assert summary["TN"] == summary["FN"] == 5
    assert summary["valid"] == 10 and summary["recall_valid"] == 0
    assert summary["event_localization"]["expected_event_window_occurrences"] == 7
    assert summary["cost_complete"]
    assert (plan / "report/review.html").exists()


@pytest.mark.parametrize("continue_on_timeout", [False, True])
def test_timeout_continuation_preserves_failure_and_never_retries(plan, continue_on_timeout):
    calls = []
    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("test timeout", request=request)
        return httpx.Response(200, json=responses_raw({"decision": "NO", "events": [], "reason": "No transition."}))
    with OpenAI(api_key="offline-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        result = execute(plan, client=client, limit=2, continue_on_timeout=continue_on_timeout)
        assert result["attempted_now"] == (2 if continue_on_timeout else 1)
        first = planning.verify_plan(plan)["queries"][0]["id"]
        path = plan / "responses" / f"{first}.json"
        saved = path.read_bytes()
        assert read_json(path)["error_type"] == "APITimeoutError"
        execute(plan, client=client, resume=True, continue_on_timeout=continue_on_timeout)
    assert len(calls) == 10
    assert path.read_bytes() == saved
    assert export_run(plan)["conditions"]["responses_diagnostic"]["errors_or_uncertain"] == 1


@pytest.mark.parametrize("continue_on_timeout", [False, True])
def test_started_without_result_is_not_resent(plan, continue_on_timeout):
    qid = planning.verify_plan(plan)["queries"][0]["id"]
    append_jsonl(plan / "journal.jsonl", {"event": "started", "query_id": qid})
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "test auth error", "type": "invalid_request_error"}})
    with OpenAI(api_key="offline-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        result = execute(plan, client=client, resume=True, continue_on_timeout=continue_on_timeout)
    assert len(calls) == 1  # fail fast on HTTP errors
    assert result["attempted_now"] == 1
    summary = export_run(plan)["conditions"]["responses_diagnostic"]
    assert summary["errors_or_uncertain"] == 2 and summary["valid"] == 0
    assert summary["cost_unknown_attempts"] == 2
    assert read_json(plan / "report/predictions.json")[0]["status"] == "uncertain"


def test_decisions_and_sliding_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(planning, "RESULTS", tmp_path / "results")
    directory = planning.create_plan("evaluate", "v003", "sliding", {"repeats": 1})
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path == "/v1/decisions":
            assert body["questions"][0]["name"] == "changed_aircraft"
            assert {c["value"] for c in body["questions"][0]["choices"]} == {"NONE", "A1", "A2", "BOTH"}
            return httpx.Response(200, json={"model": "gpt-6-luna", "answers": [{"type": "choice", "name": "changed_aircraft", "choice": "BOTH", "confidence": .7,
                "probabilities": [{"value": "NONE", "probability": .1}, {"value": "A1", "probability": .1}, {"value": "A2", "probability": .1}, {"value": "BOTH", "probability": .7}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0}}})
        diagnostic = "events" in body["text"]["format"]["schema"]["properties"]
        end = int(re.search(r"Current step: (\d+)", body["input"][1]["content"]).group(1))
        truth = [e for e in read_json(directory / "annotations.json")["events"] if end - 39 <= e["step"] <= end]
        payload = {"decision": "YES" if truth else "NO", "events": [event(e["step"], e["aircraft_id"]) for e in truth],
                   "reason": "Synthetic oracle for testing only."} if diagnostic else {"decision": "NO"}
        return httpx.Response(200, json=responses_raw(payload))
    with OpenAI(api_key="offline-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        execute(directory, client=client)
    assert len(calls) == 124 and calls.count("/v1/decisions") == 62
    summary = export_run(directory)
    assert summary["conditions"]["decisions"]["recall_valid"] == 1
    assert summary["conditions"]["decisions"]["FP"] == 21
    assert summary["conditions"]["responses_diagnostic"]["event_localization"]["recall"] == 1
    assert summary["conditions"]["responses_diagnostic"]["aircraft_selection"]["exact_set_accuracy_valid"] == 1
    assert summary["conditions"]["decisions"]["aircraft_selection"]["exact_set_accuracy_valid"] < 1
    assert all(v["recall"] == 1 for v in summary["conditions"]["decisions"]["aircraft_selection"]["per_aircraft"].values())
    assert all(p["valid_pairs"] == 62 for p in summary["paired_comparisons"])
    streams = summary["streams"]["decisions"][0]
    assert all(e["binary_coverage_proxy"] for e in streams["events"])
    assert all(e["localized"] is None for e in streams["events"])
    assert not streams["localization_available"]
    localized = summary["streams"]["responses_diagnostic"][0]["events"]
    assert localized[0]["localized_first_delay_steps"] == 3
    assert localized[0]["localized_excess_over_grid_steps"] == 0
    assert localized[0]["localized_duplicate_windows"] == 7
    assert localized[1]["localized_first_delay_steps"] == 0
    assert streams["cooldown_simulations"][0]["trigger_count"] > streams["cooldown_simulations"][-1]["trigger_count"]
    assert "Same samples and matching tolerance: True" in compare_runs([directory])


def test_missing_key_leaves_plan_unsent(plan, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        execute(plan)
    assert not read_jsonl(plan / "journal.jsonl")


def test_env_key_does_not_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-env-key")
    monkeypatch.setattr(getpass, "getpass", lambda _: pytest.fail("Should use the environment"))
    assert read_api_key() == "offline-env-key"


@pytest.mark.parametrize("outcome", ["", EOFError(), KeyboardInterrupt(), getpass.GetPassWarning("Cannot hide input")])
def test_cancelled_or_unsafe_key_input_sends_nothing(plan, monkeypatch, outcome):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    def hidden_input(prompt):
        if isinstance(outcome, getpass.GetPassWarning):
            warnings.warn(outcome)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(getpass, "getpass", hidden_input)
    with pytest.raises(ValueError):
        execute(plan)
    assert not read_jsonl(plan / "journal.jsonl")


def test_entered_key_used_without_persistence(plan, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    key = "offline-entered-key"
    prompts_seen = []
    def hidden_input(prompt):
        prompts_seen.append(prompt)
        return "  " + key + "  "
    monkeypatch.setattr(getpass, "getpass", hidden_input)
    requests_seen = []
    def handler(request):
        requests_seen.append(request)
        assert request.headers["authorization"] == f"Bearer {key}"
        # An upstream error that echoes a key must not persist it in the journal/result.
        return httpx.Response(401, json={"error": {"message": f"Invalid API key: {key}", "type": "invalid_request_error"}})
    def client_factory(**kwargs):
        assert kwargs["api_key"] == key
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("openai.OpenAI", client_factory)
    execute(plan, limit=1)
    assert len(prompts_seen) == len(requests_seen) == 1
    assert "OPENAI_API_KEY" not in os.environ
    for path in (plan / "journal.jsonl", *list((plan / "responses").glob("*.json"))):
        assert key not in path.read_text()


def test_manifest_tampering_is_detected(plan):
    manifest = read_json(plan / "manifest.json")
    manifest["config"]["history"] = 30
    write_json(plan / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Manifest changed"):
        planning.verify_plan(plan)


def test_prompt_iteration_keeps_parent_and_refuses_overwrite(tmp_path, monkeypatch):
    shutil.copytree(HERE / "prompts/v001", tmp_path / "prompts/v001")
    monkeypatch.setattr(prompts, "HERE", tmp_path)
    target = prompts.clone_prompt("v001", "v002", "Test a clearer onset criterion")
    assert read_json(target / "metadata.json")["parent"] == "v001"
    before = (tmp_path / "prompts/v001/system.txt").read_text()
    (target / "system.txt").write_text(before + "\nAn edited criterion.\n")
    assert (tmp_path / "prompts/v001/system.txt").read_text() == before
    with pytest.raises(FileExistsError):
        prompts.clone_prompt("v001", "v002", "Do not replace")


def test_offline_viewer_preserves_script_like_model_text():
    from build_results_viewer import render_html
    payload = {"answer": '</script><script>alert("model text")</script>&\u2028'}
    html = render_html(payload)
    embedded = re.search(r'<script id="dataset" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(embedded) == payload
    assert '</script><script>alert' not in html
    assert '<script src=' not in html


def test_offline_viewer_uses_frozen_requests_and_leaves_run_unchanged(plan, tmp_path, monkeypatch):
    import build_results_viewer as viewer
    from decision_experiment.storage import file_hash
    monkeypatch.setattr(viewer, "RESULTS", plan.parents[2])
    before = {str(p.relative_to(plan)): file_hash(p) for p in plan.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match="No attempted runs"):
        viewer.build_viewer([plan], tmp_path / 'viewer.html')
    output, bundle = viewer.build_viewer([plan], tmp_path / 'viewer.html', include_planned=True)
    run = bundle['runs'][0]
    assert len(run['rows']) == 10
    assert run['responses'] == {}
    assert all(row['status'] == 'pending' for row in run['rows'])
    assert run['requests'][run['rows'][0]['request_sha256']]['input'][0]['content'] == (plan / 'prompt/system.txt').read_text()
    assert output.exists()
    assert before == {str(p.relative_to(plan)): file_hash(p) for p in plan.rglob('*') if p.is_file()}
