"""Offline contracts for Exp.05/06: no real API client is needed."""
import copy
import csv
from importlib import import_module
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from experiments import common
from experiments.joint_questions import JOINT_QUESTIONS
from experiments.common_analysis import joint_data as data
from experiments.common_analysis.joint_aggregate import aggregate_all
from experiments.common_analysis.runner import write_csv


@pytest.fixture(scope="module")
def plans():
    specs = [import_module(f"experiments.{name}.run").EXPERIMENT for name in
             ("05_new_models_single", "06_new_models_n10")]
    return specs, [common.make_plan(spec, 0) for spec in specs]


def test_joint_questions_and_exact_nonoverlap_budget(plans):
    specs, (five, six) = plans
    assert specs[0].questions is specs[1].questions is JOINT_QUESTIONS
    assert import_module("experiments.03_1_gpt35_history_n30_joint.run").JOINT_QUESTIONS is JOINT_QUESTIONS
    assert five["planned_api_requests"] == 120
    assert six["planned_api_requests"] == 840 and six["planned_reused_queries"] == 120
    assert six["planned_logical_queries"] == 960
    for alias in common.NEW_MODELS:
        first = {q["query_id"] for q in five["queries"] if q["model_alias"] == alias}
        reused = {q["query_id"] for q in six["queries"] if q["model_alias"] == alias and q["reuse_required"]}
        new = {q["query_id"] for q in six["queries"] if q["model_alias"] == alias and not q["reuse_required"]}
        assert first == reused and len(first) == 40 and len(new) == 280
        assert first.isdisjoint(new) and len(first | new) == 320
    for q in [*five["queries"], *six["queries"]]:
        assert q["actual_history_steps"] == q["H"]
        assert q["question_name"] == JOINT_QUESTIONS[q["task"]][q["metric"]]
        if q["metric"].endswith("state"):
            assert "Final state values" in q["user_prompt"] and "Final state deltas" in q["user_prompt"]
        elif q["task"] == "Pendulum-v1":
            assert "Final action bins" in q["user_prompt"]


@pytest.fixture
def split_world(plans, tmp_path, monkeypatch):
    """Translate old offline responses into synthetic new-model batches in tmp_path."""
    import shutil
    from experiments import split_execution as split
    specs, planned = plans
    original = common.read_jsonl(common.ROOT / "experiments/03_1_gpt35_history_n30_joint/results/records.jsonl")
    by_key = {(r["task"], r["metric"], r["H"], r["ordinal"]): r for r in original}
    for relative in {r["episode_path"] for r in original}:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(common.ROOT / relative, target)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(common, "make_plan", lambda spec, retries: copy.deepcopy(planned[0 if spec.name == specs[0].name else 1]))

    def save(source, plan, status="complete"):
        source.mkdir(parents=True, exist_ok=True)
        rows, runs = [], {}
        for q in plan["queries"]:
            r = copy.deepcopy(by_key[q["task"], q["metric"], q["H"], q["ordinal"]])
            r.update(q)
            r.update(reused=False, api_request_made=True, api_attempts=1, source_experiment=plan["experiment"])
            r["raw_response"]["model"] = q["model"]
            r["request"] = common.expected_api_request(q["model"], q["system_prompt"], q["user_prompt"])
            r["upstream_output"] = str((source / "runs" / q["condition_id"] / "episode_000").relative_to(tmp_path))
            rows.append(r)
            runs.setdefault(r["upstream_output"], []).append(r["score"])
        for relative, scores in runs.items():
            path = tmp_path / relative
            path.mkdir(parents=True)
            (path / "predictions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scores))
        (source / "manifest.json").write_text(json.dumps(plan))
        (source / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (source / "summary.json").write_text(json.dumps({"status": status, "successful_queries": len(rows),
            "failed_queries": 0, "reused_queries": 0, "api_attempts": len(rows), "experiment_elapsed_seconds": 100.0}))
        return rows

    save(common.EXPERIMENTS / specs[0].name / "results", planned[0])
    root = split.result_root(specs[1])
    return split, specs[1], planned[1], root, save


def test_split_selection_exact_120_and_overlap_rules(plans):
    from experiments import split_execution as split
    full = plans[1][1]
    first = split.select_plan(full, {}, ["Pendulum-v1"], [5])
    assert len(first["queries"]) == first["planned_api_requests"] == 120
    assert first["planned_reused_queries"] == 0
    assert {q["H"] for q in first["queries"]} == {5}
    assert {q["task"] for q in first["queries"]} == {"Pendulum-v1"}
    assert len(first["conditions"]) == 12
    claims = {c["condition_id"]: {"status": "complete", "source": "test"} for c in first["conditions"]}
    for tasks, histories in [(["Pendulum-v1"], [5]), (["Pendulum-v1"], [5, 10]), (None, None)]:
        with pytest.raises(ValueError, match="実行済み"):
            split.select_plan(full, claims, tasks, histories)
    with pytest.raises(ValueError, match="05"):
        split.select_plan(full, {}, ["Pendulum-v1"], [20])
    remaining = split.select_plan(full, claims, remaining=True)
    assert len(remaining["queries"]) == 720
    assert {q["query_id"] for q in first["queries"]}.isdisjoint(q["query_id"] for q in remaining["queries"])
    assert len(split.select_plan(full, {})["queries"]) == 840


def test_model_order_changes_only_order_and_accepts_old_batches(plans):
    from experiments import split_execution as split
    full = plans[1][1]
    default = split.select_plan(full, {})
    assert default["model_order"] == ["terra", "luna", "sol"]
    assert [q["model_alias"] for q in default["queries"]] == ["terra"] * 280 + ["luna"] * 280 + ["sol"] * 280
    assert default["continue_on_api_error"] is True
    custom = split.select_plan(full, {}, model_order=["luna", "sol", "terra"])
    assert [q["model_alias"] for q in custom["queries"]] == ["luna"] * 280 + ["sol"] * 280 + ["terra"] * 280
    for alias in common.NEW_MODELS:
        assert [q for q in default["queries"] if q["model_alias"] == alias] == [
            q for q in full["queries"] if q["model_alias"] == alias and not q["reuse_required"]]
    old = split.select_plan(full, {}, model_order=common.NEW_MODELS)
    del old["model_order"], old["continue_on_api_error"]
    for plan in (old, default, custom):
        assert len(split.validate_plan(plan, full)) == 84
    with pytest.raises(ValueError, match="exactly once"):
        split.select_plan(full, {}, model_order=["sol", "sol", "terra"])


@pytest.fixture
def api_error_batch(split_world, monkeypatch):
    """Real CLI/prompt/scorer + fake SDK transport: success, API error, then success."""
    from types import SimpleNamespace
    import httpx
    from openai import InternalServerError
    from openai.types.chat import ChatCompletion

    split, spec, full, root, _ = split_world
    plan = split.select_plan(full, {})
    plan["conditions"] = plan["conditions"][:1]
    cid = plan["conditions"][0]["condition_id"]
    plan["queries"] = [q for q in plan["queries"] if q["condition_id"] == cid]
    plan.update(planned_api_requests=10, planned_logical_queries=10, maximum_api_attempts=10)
    source = root / "batches/api-error-test"
    source.mkdir(parents=True)
    common.write_json(source / "manifest.json", plan)
    response = httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
                              headers={"x-request-id": "req_offline_fixture"})
    error = InternalServerError("offline simulated error", response=response,
                                body={"message": "offline simulated error", "type": "server_error"})
    calls = []

    def send(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if len(calls) == 2:
            raise error
        return ChatCompletion.model_validate({"id": "offline", "object": "chat.completion", "created": 0,
            "model": kwargs["model"], "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "2"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})

    def initialize(backend, **kwargs):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)),
                                 base_url="https://api.openai.com/v1", close=Mock())
        client.with_options = Mock(return_value=client)
        backend.model, backend.retries, backend._client = kwargs["model"], kwargs["retries"], client

    monkeypatch.setattr(common.OpenAIChatBackend, "__init__", initialize)
    monkeypatch.setattr(httpx.Client, "send", Mock(side_effect=AssertionError("real network forbidden")))
    return spec, plan, source, calls


def test_continue_after_api_error_keeps_exact_next_query_and_durable_details(api_error_batch, split_world):
    spec, plan, source, calls = api_error_batch
    split, _, full, root, save = split_world
    common.execute_plan(spec, plan, source, {}, 0)
    assert calls == [common.expected_api_request(q["model"], q["system_prompt"], q["user_prompt"])
                     for q in plan["queries"]]
    summary = common.read_json(source / "summary.json")
    assert summary["status"] == "complete_with_errors"
    assert (summary["logical_queries"], summary["successful_queries"], summary["failed_queries"],
            summary["api_attempts"], summary["unstarted_queries"], summary["unscored_queries"]) == (10, 9, 1, 10, 0, 0)
    rows = split.completed_records(source, plan)
    assert [r["query_index"] for r in rows] == list(range(5, 15))
    failed = rows[1]
    assert failed == common.read_jsonl(source / "failed_queries.jsonl")[0]
    detail = failed["attempts"][0]["exception"]
    assert detail["status_code"] == 500 and detail["request_id"] == "req_offline_fixture"
    assert detail["type"] == "InternalServerError" and detail["body"]["type"] == "server_error"
    assert failed["request"] == calls[1] and failed["user_prompt"] == plan["queries"][1]["user_prompt"]
    assert failed["usage"] is None and failed["input_tokens"] is None and "score" not in failed
    assert failed["query_id"] in (source / "run.log").read_text()
    assert len(common.read_jsonl(source / "requests.jsonl")) == len(common.read_jsonl(source / "responses.jsonl")) == 10

    claims, _ = split.scan_batches(root, full)
    with pytest.raises(ValueError, match="実行済み"):
        split.select_plan(full, claims)
    rest = split.select_plan(full, claims, remaining=True)
    assert len(rest["queries"]) == 830 and not {r["query_id"] for r in rows} & {q["query_id"] for q in rest["queries"]}
    save(root / "batches/rest", rest)
    protected = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = split.merge_results(spec)
    merged = common.read_jsonl(output / "records.jsonl")
    assert len(merged) == 960 and sum(r["status"] == "failed" for r in merged) == 1
    assert common.read_json(output / "summary.json")["successful_queries"] == 959
    merged_failed = common.read_jsonl(output / "failed_queries.jsonl")[0]
    assert merged_failed["source_record"] and merged_failed["source_requests"] and merged_failed["source_responses"]
    assert all(p.read_bytes() == old for p, old in protected.items())
    with pytest.raises(ValueError, match="API failures remain"):
        data.load_and_parse(output, spec)
    _, _, parsed_with_failure, _ = data.load_and_parse(output, spec, allow_api_failures=True)
    assert len(parsed_with_failure) == 960
    missing = [r for r in parsed_with_failure if r["api_error"]]
    assert len(missing) == 1 and missing[0]["components"]["action"]["error"] == "api_error"
    assert missing[0]["source_run"] is None and missing[0]["response_sha256"] is None
    assert missing[0]["input_tokens"] is None and missing[0]["query_elapsed_seconds"] is None
    assert missing[0]["failed_attempt_elapsed_seconds"] >= 0
    from experiments.common_analysis.joint_aggregate import accuracy_rows, diagnostic_rows
    selected = [r for r in parsed_with_failure if r["condition_id"] == missing[0]["condition_id"]]
    accuracy = accuracy_rows(selected)[0][0]
    # The successful mock response is bare "2", not the required Joint marker.
    # Keep API success distinct from component parse success.
    assert accuracy["selected_n"] == 10 and accuracy["valid_n"] == 0
    timing = next(r for r in diagnostic_rows(selected) if r["metric_name"] == "query_elapsed_seconds")
    assert timing["valid_n"] == 9
    # An altered failure log must prevent further duplicate checks/merges.
    common.write_text(source / "responses.jsonl", "")
    with pytest.raises(ValueError, match="response journal mismatch"):
        split.scan_batches(root, full)


@pytest.mark.parametrize("filename", ["requests.jsonl", "responses.jsonl", "records.jsonl", "failed_queries.jsonl"])
def test_continue_mode_still_stops_on_journal_failure(api_error_batch, monkeypatch, filename):
    spec, plan, source, calls = api_error_batch
    original = common.append_jsonl
    def append(path, row):
        if path.name == filename and len(calls) >= 2:
            raise common.SafetyStop("simulated disk failure")
        original(path, row)
    monkeypatch.setattr(common, "append_jsonl", append)
    with pytest.raises(common.SafetyStop, match="disk failure"):
        common.execute_plan(spec, plan, source, {}, 0)
    assert len(calls) == 2
    assert common.read_json(source / "summary.json")["status"] == "incomplete"


def test_default_failfast_behavior_is_unchanged(api_error_batch):
    spec, plan, source, calls = api_error_batch
    plan.pop("continue_on_api_error")
    with pytest.raises(RuntimeError, match="Upstream CLI exited"):
        common.execute_plan(spec, plan, source, {}, 0)
    assert len(calls) == 2
    summary = common.read_json(source / "summary.json")
    assert summary["status"] == "incomplete" and summary["successful_queries"] == 1


def test_real_cli_finishes_each_model_before_starting_next(api_error_batch, split_world):
    spec, plan, source, calls = api_error_batch
    split, _, full, _, _ = split_world
    ordered = split.select_plan(full, {}, ["MountainCar-v0"], [5])
    plan["conditions"] = [c for c in ordered["conditions"] if c["metric"] == "next-action"]
    plan["queries"] = [q for q in ordered["queries"] if q["metric"] == "next-action"]
    common.execute_plan(spec, plan, source, {}, 0)
    assert [call["model"] for call in calls] == [common.MODELS[a] for a in ("terra", "luna", "sol") for _ in range(10)]
    assert [r["query_id"] for r in common.read_jsonl(source / "records.jsonl")] == [q["query_id"] for q in plan["queries"]]


@pytest.mark.parametrize("failure", ["connection", "local", "interrupt", "empty"])
def test_only_sdk_api_errors_are_skipped(api_error_batch, monkeypatch, failure):
    import httpx
    from openai import APIConnectionError
    from openai.types.chat import ChatCompletion
    spec, plan, source, calls = api_error_batch
    initialize = common.OpenAIChatBackend.__init__
    def init(backend, **kwargs):
        initialize(backend, **kwargs)
        send = backend._client.chat.completions.create
        def wrapped(**request):
            if len(calls) == 1:
                calls.append(copy.deepcopy(request))
                if failure == "connection":
                    raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
                if failure == "local":
                    raise RuntimeError("local bug")
                if failure == "interrupt":
                    raise KeyboardInterrupt()
                return ChatCompletion.model_validate({"id": "offline", "object": "chat.completion", "created": 0,
                    "model": request["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": ""}}]})
            return send(**request)
        backend._client.chat.completions.create = wrapped
    monkeypatch.setattr(common.OpenAIChatBackend, "__init__", init)
    if failure == "connection":
        common.execute_plan(spec, plan, source, {}, 0)
        assert len(calls) == 10
        assert common.read_json(source / "summary.json")["status"] == "complete_with_errors"
    else:
        with pytest.raises((RuntimeError, KeyboardInterrupt)):
            common.execute_plan(spec, plan, source, {}, 0)
        assert len(calls) == 2
        assert common.read_json(source / "summary.json")["status"] == "incomplete"


def test_split_cli_mock_send_duplicate_error_and_failed_reservation(split_world, monkeypatch):
    split, spec, full, root, save = split_world
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    calls = []
    def execute(spec, plan, directory, cached, retries):
        assert cached == {} and retries == 0
        calls.append(directory)
        save(directory, plan)
    monkeypatch.setattr(common, "execute_plan", execute)
    args = ["--task", "Pendulum-v1", "--history", "5"]
    assert split.run(spec, args) == 0
    assert not (root / "batches").exists()  # Dry-run never reserves.
    paid = [*args, "--execute", "--confirm-paid-api"]
    assert split.run(spec, paid) == 0 and len(calls) == 1
    assert split.run(spec, paid) == 2 and len(calls) == 1
    assert split.run(spec, ["--task", "Pendulum-v1", "--history", "5", "10", "--execute", "--confirm-paid-api"]) == 2
    assert len(calls) == 1  # Whole overlapping request rejected, not partially executed.
    summary = calls[0] / "summary.json"
    summary.write_text('{"status": "incomplete"}')
    assert split.run(spec, paid) == 2
    assert split.run(spec, ["--remaining", "--execute", "--confirm-paid-api"]) == 2
    assert len(calls) == 1
    # A disjoint selection can still be planned, but the failed batch cannot resume.
    assert split.run(spec, ["--task", "Pendulum-v1", "--history", "10"]) == 0
    with pytest.raises(SystemExit):
        split.run(spec, ["--resume"])


def test_split_missing_prerequisite_and_lock_prevent_send(split_world, monkeypatch):
    split, spec, _, root, _ = split_world
    execute = Mock(side_effect=AssertionError("no send"))
    monkeypatch.setattr(common, "execute_plan", execute)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    args = ["--task", "Pendulum-v1", "--history", "5", "--execute", "--confirm-paid-api"]
    with split.execution_lock(root):
        assert split.run(spec, args) == 2
    (common.EXPERIMENTS / spec.reuse_from / "results/summary.json").write_text('{"status":"incomplete"}')
    assert split.run(spec, args) == 2
    execute.assert_not_called()
    assert not (root / "batches").exists()


def test_split_cli_signals_errors_after_all_attempts(split_world, monkeypatch, capsys):
    split, spec, _, _, _ = split_world
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    def execute(spec, plan, directory, cached, retries):
        assert plan["continue_on_api_error"] and plan["model_order"] == ["luna", "terra", "sol"]
        common.write_json(directory / "summary.json", {"status": "complete_with_errors"})
    monkeypatch.setattr(common, "execute_plan", execute)
    assert split.run(spec, ["--task", "Pendulum-v1", "--history", "5", "--model-order", "luna", "terra", "sol",
                            "--execute", "--confirm-paid-api"]) == 2
    assert "All queries attempted" in capsys.readouterr().out


def test_split_merge_full_grid_immutable_and_compatible_with_analysis(split_world):
    split, spec, full, root, save = split_world
    first = split.select_plan(full, {}, ["Pendulum-v1"], [5])
    save(root / "batches/first", first)
    with pytest.raises(ValueError, match="missing"):
        split.merge_results(spec)
    assert not (root / "merged").exists()
    claims, _ = split.scan_batches(root, full)
    rest = split.select_plan(full, claims, remaining=True)
    save(root / "batches/rest", rest)
    protected = {p: p.read_bytes() for source in [root / "batches", common.EXPERIMENTS / spec.reuse_from / "results"]
                 for p in source.rglob("*") if p.is_file()}
    output = split.merge_results(spec)
    _, manifest, parsed, _ = data.load_and_parse(output, spec)
    assert len(parsed) == 960 and len(manifest["queries"]) == 960
    records = common.read_jsonl(output / "records.jsonl")
    assert sum(r["reused"] for r in records) == 120
    assert sum(r["api_request_made"] for r in records) == 840
    for r in records:
        assert r["source_record"] and r["source_upstream_output"]
        if r["reused"]:
            assert r["api_attempts"] == 0 and r["query_elapsed_seconds"] > 0
    assert all(p.read_bytes() == old for p, old in protected.items())
    assert len(common.check_inputs(spec, manifest)) == 120
    with pytest.raises(ValueError, match="実行済み"):
        claims, _ = split.scan_batches(root, full)
        split.select_plan(full, claims, remaining=True)
    second = split.merge_results(spec)
    assert second != output and (output / "summary.json").is_file()


def test_split_analysis_runner_automatically_merges(split_world, monkeypatch):
    from experiments.common_analysis import runner
    split, spec, full, root, save = split_world
    save(root / "batches/all", split.select_plan(full, {}))
    monkeypatch.setattr(runner, "ROOT", common.ROOT)
    exported = Mock()
    monkeypatch.setattr(runner, "export_model", exported)
    runner.main(spec.name, sizes=(10,), argv=[])
    assert exported.call_count == 3
    for call in exported.call_args_list:
        records, source = call.args[:2]
        assert len(records) == 320 and source.parent == root / "merged"
        assert len({r["model"] for r in records}) == 1
    assert len(list((root / "merged").iterdir())) == 1


@pytest.mark.parametrize("damage", ["duplicate_batch", "response", "request", "prompt", "semantics", "incomplete", "missing_manifest"])
def test_split_corrupt_batches_are_never_merged(split_world, damage):
    split, spec, full, root, save = split_world
    plan = split.select_plan(full, {})
    source = root / "batches/all"
    rows = save(source, plan)
    if damage == "duplicate_batch":
        save(root / "batches/duplicate", plan)
    elif damage == "incomplete":
        (source / "summary.json").write_text('{"status":"incomplete"}')
    elif damage == "missing_manifest":
        (source / "manifest.json").unlink()
    elif damage in {"prompt", "semantics"}:
        changed = copy.deepcopy(plan)
        if damage == "prompt":
            changed["queries"][0]["user_prompt"] = "changed"
        else:
            changed["provenance"]["semantics_sha256"] = "changed"
        (source / "manifest.json").write_text(json.dumps(changed))
    else:
        if damage == "response":
            rows[0]["assistant_text"] = "changed"
        else:
            rows[0]["request"]["temperature"] = 0
        (source / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises((ValueError, KeyError)):
        split.merge_results(spec)
    assert not (root / "merged").exists()


@pytest.fixture
def prerequisite(plans, tmp_path, monkeypatch):
    specs, original = plans
    five, six = copy.deepcopy(original)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "EXPERIMENTS", tmp_path / "experiments")
    source = common.EXPERIMENTS / specs[0].name / "results"
    source.mkdir(parents=True)
    records = []
    for q in five["queries"]:
        text = "predictions = [0.6]\n>>Final action bins: [6]"
        raw = {"id": "offline-test", "object": "chat.completion", "created": 0, "model": q["model"],
               "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        request = {"model": q["model"], "reasoning_effort": "medium", "messages": [
            {"role": "system", "content": q["system_prompt"]}, {"role": "user", "content": q["user_prompt"]}]}
        records.append({**q, "status": "ignored", "assistant_text": text, "raw_response": raw,
                        "request": request, "source_experiment": specs[0].name, "api_request_made": True,
                        "reused": False, "query_elapsed_seconds": 3.5, "request_elapsed_seconds": 3.0,
                        "score": {"status": "ignored", "prompt": q["user_prompt"], "index": q["query_index"]}})
    def save_records():
        (source / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (source / "manifest.json").write_text(json.dumps(five))
    (source / "summary.json").write_text(json.dumps({"status": "complete", "successful_queries": 120}))
    save_records()
    return specs[1], six, source, records, save_records


def test_all_120_ignored_or_scored_responses_can_be_reused(prerequisite):
    spec, plan, _, _, _ = prerequisite
    assert len(common.check_inputs(spec, plan)) == 120


@pytest.mark.parametrize("damage", ["missing", "duplicate", "changed_response", "old_n1", "incomplete", "changed_question", "temperature", "reasoning_effort"])
def test_invalid_prerequisite_stops_before_any_send(prerequisite, monkeypatch, damage):
    spec, plan, source, records, save = prerequisite
    if damage == "missing":
        records.pop()
    elif damage == "duplicate":
        records[-1] = copy.deepcopy(records[0])
    elif damage == "changed_response":
        records[0]["assistant_text"] = "changed"
    elif damage == "temperature":
        records[0]["request"]["temperature"] = 0
    elif damage == "reasoning_effort":
        records[0]["request"]["reasoning_effort"] = "high"
    elif damage in {"old_n1", "changed_question"}:
        old = json.loads((source / "manifest.json").read_text())
        if damage == "old_n1":
            old["queries"] = [q for q in old["queries"] if q["ordinal"] == 0]
        else:
            old["queries"][0]["question_name"] = "old_question"
        (source / "manifest.json").write_text(json.dumps(old))
    else:
        (source / "summary.json").write_text(json.dumps({"status": "incomplete", "successful_queries": 119}))
    save()
    monkeypatch.setattr(common, "make_plan", lambda *args: copy.deepcopy(plan))
    execute = Mock(side_effect=AssertionError("must never execute"))
    monkeypatch.setattr(common, "execute_plan", execute)
    assert common._run(spec, ["--execute", "--confirm-paid-api"]) == 2
    execute.assert_not_called()


def test_cached_entire_condition_uses_no_client_and_preserves_latency(prerequisite, monkeypatch):
    spec, plan, source, _, _ = prerequisite
    cache = common.check_inputs(spec, plan)
    rows = [q for q in plan["queries"] if q["reuse_required"]][:10]
    constructor = Mock(side_effect=AssertionError("must not construct live client"))
    monkeypatch.setattr(common.OpenAIChatBackend, "__init__", constructor)
    session = common.RecordingSession(rows, source, spec.name, cache, replay_only=True)
    backend = session.backend_class()(model=rows[0]["model"])
    for q in rows:
        backend.complete(system_prompt=q["system_prompt"], user_prompt=q["user_prompt"])
    assert len(session.completed) == 10 and session.api_attempts == 0
    constructor.assert_not_called()
    for r in session.completed:
        assert r["reused"] and not r["api_request_made"]
        assert r["query_elapsed_seconds"] == 3.5 and r["request_elapsed_seconds"] == 3
        assert r["replay_elapsed_seconds"] >= 0 and r["source_experiment"] == "05_new_models_single"


@pytest.mark.parametrize("alias", ["sol", "terra", "luna", "3.5"])
def test_actual_send_and_journal_use_exact_generation_options(prerequisite, monkeypatch, alias):
    from types import SimpleNamespace
    from openai.types.chat import ChatCompletion

    _, plan, source, records, _ = prerequisite
    q = copy.deepcopy(plan["queries"][0])
    q.update(model=common.MODELS[alias], reuse_required=False)
    q["query_id"] = common.query_id(q)
    raw = copy.deepcopy(records[0]["raw_response"])
    raw["model"] = q["model"]
    send = Mock(return_value=ChatCompletion.model_validate(raw))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=send)),
                             base_url="https://api.openai.com/v1")
    client.with_options = Mock(return_value=client)

    def initialize(backend, **kwargs):
        backend.model, backend.retries, backend._client = kwargs["model"], 0, client

    monkeypatch.setattr(common.OpenAIChatBackend, "__init__", initialize)
    session = common.RecordingSession([q], source, "offline-test", {})
    backend = session.backend_class()(model=q["model"])
    backend.complete(system_prompt=q["system_prompt"], user_prompt=q["user_prompt"])
    options = {"temperature": 0} if alias == "3.5" else {"reasoning_effort": "medium"}
    expected = {"model": q["model"], **options, "messages": [
        {"role": "system", "content": q["system_prompt"]},
        {"role": "user", "content": q["user_prompt"]}]}
    send.assert_called_once_with(**expected)
    assert common.read_jsonl(source / "requests.jsonl")[0]["kwargs"] == expected
    assert session.completed[0]["request"] == expected


def test_required_cache_never_falls_back_to_send(prerequisite):
    spec, plan, source, _, _ = prerequisite
    rows = [q for q in plan["queries"] if q["reuse_required"]][:1]
    session = common.RecordingSession(rows, source, spec.name, {}, replay_only=True)
    backend = session.backend_class()(model=rows[0]["model"])
    with pytest.raises(common.SafetyStop, match="no API fallback"):
        backend.complete(system_prompt=rows[0]["system_prompt"], user_prompt=rows[0]["user_prompt"])


def test_refactor_reproduces_all_previous_numeric_tables(tmp_path):
    source, _, records, _ = data.load_and_parse()
    subsets = data.select_subsets(records, 42)
    saved_dir = source.parent / "analysis"
    assert subsets == json.loads((saved_dir / "query_subsets.json").read_text())
    for name, rows in aggregate_all(records, subsets).items():
        target = tmp_path / (name + ".csv")
        write_csv(target, rows)
        with target.open() as stream:
            actual = list(csv.DictReader(stream))
        with (saved_dir / "tables" / target.name).open() as stream:
            expected = list(csv.DictReader(stream))
        assert len(actual) == len(expected), name
        for a, e in zip(actual, expected):
            assert {k: a[k] for k in e} == e, name


def test_n10_only_and_models_never_pool():
    from experiments.common_analysis.runner import export_model
    _, _, records, _ = data.load_and_parse()
    prefixes = [r for r in records if r["ordinal"] < 10]
    subsets = data.select_subsets(prefixes, 42, sizes=(10,), expected_n=10)
    assert all(set(group) == {"10"} for group in subsets["conditions"].values())
    tables = aggregate_all(prefixes, subsets, sizes=(10,))
    assert {r["subset_n"] for r in tables["all_metrics_long"]} == {10}
    with pytest.raises(ValueError, match="one model"):
        export_model([records[0], {**records[0], "model": "different"}], None, None, None, None, (10,))


def test_n10_plot_export(tmp_path):
    from experiments.common_analysis.joint_plots import make_figures
    from experiments.common_analysis.runner import validate_saved_outputs
    _, _, records, _ = data.load_and_parse()
    prefixes = [r for r in records if r["ordinal"] < 10]
    subsets = data.select_subsets(prefixes, 42, sizes=(10,), expected_n=10)
    tables = aggregate_all(prefixes, subsets, sizes=(10,))
    figures = make_figures(tables, tmp_path, sizes=(10,), model_label="GPT-3.5 offline fixture")
    assert len(figures) == 44
    assert {Path(f).parts[0] for f in figures} == {"cross_task_10", "pendulum_10", "diagnostics_10"}
    validate_saved_outputs({}, figures, tmp_path)
    assert "GPT-3.5 offline fixture" in (tmp_path / "cross_task_10/accuracy.svg").read_text()


def test_modern_load_and_export_end_to_end_with_offline_fixtures(plans, tmp_path, monkeypatch):
    """Exercise the real 3-model loader/runner in temp storage, never fake real run results."""
    import shutil
    from experiments.common_analysis import runner
    specs, planned = plans
    real_root = common.ROOT
    original = common.read_jsonl(real_root / "experiments/03_1_gpt35_history_n30_joint/results/records.jsonl")
    by_key = {(r["task"], r["metric"], r["H"], r["ordinal"]): r for r in original}
    for relative in {r["episode_path"] for r in original} | {"upstream/LLM-Xavier/llm_x/task.py"}:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real_root / relative, destination)
    for spec, plan in zip(specs, planned):
        source = tmp_path / "experiments" / spec.name / "results"
        source.mkdir(parents=True)
        records, run_rows = [], {}
        for q in plan["queries"]:
            r = copy.deepcopy(by_key[q["task"], q["metric"], q["H"], q["ordinal"]])
            r.update(q)
            r["raw_response"]["model"] = q["model"]
            r["request"]["model"] = q["model"]
            r["request"].pop("temperature", None)
            r["request"]["reasoning_effort"] = "medium"
            r["reused"] = q["reuse_required"]
            r["api_request_made"] = not r["reused"]
            r["api_attempts"] = int(not r["reused"])
            r["source_experiment"] = specs[0].name if r["reused"] else spec.name
            r["upstream_output"] = f"experiments/{spec.name}/results/runs/{q['condition_id']}/episode_000"
            records.append(r)
            run_rows.setdefault(r["upstream_output"], []).append(r["score"])
        for relative, scores in run_rows.items():
            directory = tmp_path / relative
            directory.mkdir(parents=True)
            (directory / "predictions.jsonl").write_text("".join(json.dumps(s)+"\n" for s in scores))
        (source / "manifest.json").write_text(json.dumps(plan))
        (source / "summary.json").write_text(json.dumps({"status": "complete", "successful_queries": len(records)}))
        (source / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    plots = Mock(return_value=[])  # Actual rendering is covered by test_n10_plot_export.
    monkeypatch.setattr(runner, "make_figures", plots)
    runner.main(specs[0].name, sizes=(10,), tables_only=True, argv=[])
    plots.assert_not_called()
    runner.main(specs[1].name, sizes=(10,), argv=[])
    assert plots.call_count == 3
    for call in plots.call_args_list:
        assert call.kwargs["sizes"] == (10,)
    for spec in specs:
        for alias in common.NEW_MODELS:
            output = tmp_path / "experiments" / spec.name / "analysis" / alias
            parsed = list(csv.DictReader((output / "tables/parsed_records.csv").open()))
            assert len(parsed) == (40 if spec is specs[0] else 320)
            assert {r["model_alias"] for r in parsed} == {alias}
            if spec is specs[1]:
                assert sum(r["reused"] == "True" for r in parsed) == 40
            assert json.loads((output / "analysis_metadata.json").read_text())["subset_sizes"] == [10]
