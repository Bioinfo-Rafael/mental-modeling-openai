"""Offline contracts: temporal leakage, sampling, scoring, and resumable execution."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(HERE.parent / "13_LAG"), str(ROOT / "upstream/LLM-Xavier")]

from lag_data.dataset import PROCESSED, index_records, load_episode
from lag_data.definitions import ACTIONS, STATE_FEATURES
from lag_experiment.config import Experiment, TASKS
from lag_experiment import evaluation, execution, prompts, runner
from llm_x.evaluation import _history_range


@pytest.fixture(scope="module")
def full_plan():
    return runner.make_plan(Experiment())


@pytest.fixture(scope="module")
def episodes():
    return {stem: index_records(load_episode(PROCESSED / f"{stem}.csv")) for stem in Experiment().episodes}


def test_full_grid_and_temporal_sampling(full_plan):
    assert full_plan["counts"] == dict(samples=1200, requests=3600, conditions=720, n=5)
    assert set(Counter(q["condition_id"] for q in full_plan["queries"]).values()) == {5}
    assert set(Counter(q["sample_id"] for q in full_plan["queries"]).values()) == {3}
    assert all(q["repeat_id"] == 1 for q in full_plan["queries"])
    assert runner.select_uniform_starts(list(range(400)), 5) == [0, 80, 160, 240, 320]
    for coverage in full_plan["coverage"]:
        m = coverage["valid_start_count"]
        starts = coverage["selected_starts"]
        assert starts == [coverage["valid_start_min"] + k * m // 5 for k in range(5)]
        assert len(set(starts)) == 5
    with pytest.raises(ValueError, match="temporal gaps"):
        runner.select_uniform_starts([0, 1, 3], 2)
    with pytest.raises(ValueError, match="at least"):
        runner.select_uniform_starts([0, 1], 10)


def test_all_samples_have_only_allowed_input_and_no_target_leak(full_plan, episodes):
    contexts = {name: runner.index_context(records) for name, records in episodes.items()}
    for sample in full_plan["samples"].values():
        task, target = sample["task"], sample["target_id"]
        visible, truth, q = runner.build_query(episodes[sample["episode_id"]], task=task,
                                              history_length=sample["H"], start=sample["start"], target_id=target,
                                              observation_scope=sample["observation_scope"],
                                              context_records=contexts[sample["episode_id"]])
        start, end = _history_range(task, q, sample["H"] - 1)
        assert sample["history_steps"] == list(range(start, end))
        assert len(visible["history"]) == sample["H"]
        assert truth == sample["ground_truth"]
        assert sample["messages"] == prompts.build_messages(visible)
        blocks = visible["history"] + visible["query_context"]
        for block in blocks:
            for state in block.get("states", []):
                assert len(state["values"]) == 10
                if sample["observation_scope"] == "self":
                    assert state["object_id"] == target
                else:
                    assert state["entity_type"] in ("aircraft", "missile")
                    row = episodes[sample["episode_id"]][block["step"], state["object_id"]]
                    expected_values = [None if not state["valid"] or pd.isna(row[f]) else row[f]
                                       for f in STATE_FEATURES]
                    assert state["values"] == expected_values
            for action in block.get("actions", []):
                assert len(action["values"]) == 4
                if sample["observation_scope"] == "self":
                    assert action["object_id"] == target
                else:
                    row = episodes[sample["episode_id"]][block["step"], action["object_id"]]
                    expected_values = [int(row[f"{f}_id"]) for f in ACTIONS] if row["action_status"] == "recorded" else [None] * 4
                    assert action["values"] == expected_values
        hidden = "actions" if task.endswith("action") else "states"
        assert not any(x["object_id"] == target for b in blocks if b["step"] == sample["target_step"]
                       for x in b.get(hidden, []))
        expected = {
            "next-action": [(q, {"states"})],
            "last-action": [(q, {"states"}), (q + 1, {"states"})],
            "next-state": [(q, {"states", "actions"})],
            "last-state": [(q, {"actions"}), (q + 1, {"states"})],
        }[task]
        assert [(b["step"], {kind for kind in ("states", "actions")
                             if any(x["object_id"] == target for x in b.get(kind, []))})
                for b in visible["query_context"]] == expected
        allowed_steps = set(range(start, end)) | {s for s, _ in expected}
        assert {b["step"] for b in blocks} == allowed_steps
        assert len(truth) == (3 if task.endswith("action") else 10)
        text = sample["messages"][1]["content"]
        assert "p_shoot" not in text and "missile_launched" not in text
        assert sample["episode_id"] not in text  # No win/loss label or policy filename.


def test_scopes_pair_the_same_questions_and_preserve_all_recorded_objects(full_plan, episodes):
    pairs = {}
    seen_missile = seen_removed = seen_inactive = False
    for sample in full_plan["samples"].values():
        key = (sample["problem_id"], sample["H"])
        pairs.setdefault(key, {})[sample["observation_scope"]] = sample
        if sample["observation_scope"] == "self":
            continue
        records = episodes[sample["episode_id"]]
        query, _, _ = runner.build_query(records, task=sample["task"], history_length=sample["H"],
                                         start=sample["start"], target_id=sample["target_id"],
                                         observation_scope="all_aircraft_missiles")
        for block in query["history"] + query["query_context"]:
            ids = {uid for (t, uid), r in records.items() if t == block["step"]
                   and r["entity_type"] in ("aircraft", "missile")}
            if sample["task"] == "last-state" and block["step"] == sample["target_step"]:
                ids.remove(sample["target_id"])
            assert {s["object_id"] for s in block["states"]} == ids
            for state in block["states"]:
                if state["entity_type"] == "missile":
                    seen_missile = True
                    assert state["values"][6:] == [None] * 4
                if state["event"] == "remove":
                    seen_removed = True
                    assert state["values"] == [None] * 10
                if state["alive"] == 0:
                    seen_inactive = True
                    assert state["values"] == [None] * 10
    assert seen_missile and seen_removed and seen_inactive
    for pair in pairs.values():
        assert set(pair) == {"self", "all_aircraft_missiles"}
        left, right = pair["self"], pair["all_aircraft_missiles"]
        for field in ("ground_truth", "start", "history_steps", "target_step", "sample_number"):
            assert left[field] == right[field]
        assert execution.prompt_filename(left) != execution.prompt_filename(right)


def test_invalid_and_partial_terminal_states_excluded(episodes):
    records = episodes["ours_seed100100_ic100601_loss"]
    with pytest.raises(runner.InvalidWindow, match="invalid_state"):
        runner.build_query(records, task="next-state", history_length=2, start=208, target_id="A0200")
    records = episodes["mhi_v1.3_ic100601_win"]
    with pytest.raises(runner.InvalidWindow, match="incomplete_state"):
        runner.build_query(records, task="next-state", history_length=2, start=339, target_id="A0100")


def test_unmodified_llmx_wrappers_are_used(monkeypatch, episodes):
    query, _, _ = runner.build_query(next(iter(episodes.values())), task="last-state",
                                     history_length=2, start=0, target_id="A0100")
    original_system, original_user = prompts.llmx.system_prompt, prompts.llmx.user_prompt
    calls = []
    def system(description):
        calls.append("system")
        return original_system(description)
    def user(history, **kwargs):
        calls.append("user")
        assert kwargs["history_end"] - kwargs["history_start"] == 2
        return original_user(history, **kwargs)
    monkeypatch.setattr(prompts.llmx, "system_prompt", system)
    monkeypatch.setattr(prompts.llmx, "user_prompt", user)
    prompts.build_messages(query)
    assert calls == ["system", "user"]


@pytest.mark.parametrize("answer", ("[1,2,0,1]", "[3,2,0]", "[1.0,2,0]", "[true,2,0]", "[1,5,0]"))
def test_invalid_action_output_rejected(answer):
    assert evaluation.parse_prediction(prompts.ACTION_MARKER + answer, "next-action")[0] is None


def test_scores_and_denominators():
    action = evaluation.score("last-action", ACTIONS[:3], [1, 2, 0], prompts.ACTION_MARKER + " [1,2,1]")
    assert not action["exact_match"] and action["component_matches"]["heading_action"]
    expected, predicted = [0] * 10, [0] * 10
    expected[5], predicted[5] = 359, 1
    state = evaluation.score("next-state", STATE_FEATURES, expected, prompts.STATE_MARKER + json.dumps(predicted))
    assert state["absolute_errors"]["yaw_deg"] == 358
    assert state["circular_angle_errors_deg"]["yaw_deg"] == 2
    for answer in ("[0,1]", "[NaN,0,0,0,0,0,0,0,0,0]", "[1e309,0,0,0,0,0,0,0,0,0]"):
        assert evaluation.parse_prediction(prompts.STATE_MARKER + answer, "next-state")[0] is None
    assert evaluation.parse_prediction(prompts.ACTION_MARKER + " [1,2,0]\n" + prompts.ACTION_MARKER + " [1,2,0]", "next-action")[0] is None
    rows = [dict(condition_id="one", **action), dict(condition_id="one", parse_success=False, api_error="Timeout")]
    summary = evaluation.summarize(rows, [dict(condition_id="one")] * 10)[0]
    assert summary["planned"] == 10 and summary["parsed"] == 1
    assert summary["api_errors"] == 1 and summary["unscored"] == 8 and not summary["complete"]


def small_plan(full_plan):
    plan = deepcopy(full_plan)
    plan["queries"] = plan["queries"][:2]
    sids = {q["sample_id"] for q in plan["queries"]}
    plan["samples"] = {sid: s for sid, s in plan["samples"].items() if sid in sids}
    plan.pop("plan_id")
    plan["plan_id"] = execution.digest(plan)
    return plan


class FakeClient:
    def __init__(self, *, fail=False):
        self.calls, self.closed, self.fail = [], False, fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls.append(request)
        if self.fail:
            raise TimeoutError("mock timeout")
        return SimpleNamespace(model_dump=lambda **kw: {"choices": [{"message": {"content": prompts.ACTION_MARKER + " [1,2,0]"}}]})

    def close(self):
        self.closed = True


def test_execute_resume_and_offline_rescore(full_plan, tmp_path):
    plan, client = small_plan(full_plan), FakeClient()
    execution.save_plan(plan, tmp_path)
    execution.execute(plan, tmp_path, repository_root=ROOT, client_factory=lambda: client)
    assert len(client.calls) == 2 and client.closed
    before = (tmp_path / "api/responses.jsonl").read_bytes()
    execution.execute(plan, tmp_path, repository_root=ROOT, resume=True,
                      client_factory=lambda: pytest.fail("No client should be made"))
    summary = evaluation.evaluate_saved(plan, tmp_path)
    assert sum(s["responses"] for s in summary) == 2
    assert (tmp_path / "api/responses.jsonl").read_bytes() == before
    assert (tmp_path / "analysis" / evaluation.SCORING_VERSION / "conditions.csv").exists()
    with pytest.raises(FileExistsError, match="API run exists"):
        execution.execute(plan, tmp_path, repository_root=ROOT, client_factory=lambda: client)


def test_timeout_is_journaled_and_never_resent(full_plan, tmp_path):
    plan, client = small_plan(full_plan), FakeClient(fail=True)
    execution.save_plan(plan, tmp_path)
    with pytest.raises(TimeoutError):
        execution.execute(plan, tmp_path, repository_root=ROOT, client_factory=lambda: client)
    assert len(execution.read_journal(tmp_path / "api/requests.jsonl")) == 1
    client2 = FakeClient()
    result = execution.execute(plan, tmp_path, repository_root=ROOT, resume=True, client_factory=lambda: client2)
    assert len(client2.calls) == 1 and client2.calls[0] != client.calls[0]
    assert result["unresolved"] == 1 and result["status"] == "incomplete"
    assert len(execution.read_journal(tmp_path / "api/requests.jsonl")) == 2


def test_plan_preview_and_source_integrity(full_plan, tmp_path):
    plan = small_plan(full_plan)
    execution.save_plan(plan, tmp_path)
    changed = deepcopy(plan)
    changed["queries"][0]["model"] = "something-else"
    with pytest.raises(ValueError, match="digest"):
        execution.verify_plan(changed)
    preview = next((tmp_path / "prompts").glob("*.txt"))
    preview.write_text("edited", encoding="utf-8")
    with pytest.raises(ValueError, match="Preview was edited"):
        execution.save_plan(plan, tmp_path)
    other = tmp_path / "other"
    plan["source_sha256"] = {"pyproject.toml": "wrong"}
    plan.pop("plan_id")
    plan["plan_id"] = execution.digest(plan)
    execution.save_plan(plan, other)
    with pytest.raises(ValueError, match="Source changed"):
        execution.execute(plan, other, repository_root=ROOT, client_factory=lambda: pytest.fail("API must not start"))


def test_cli_default_dry_run_never_executes(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "HERE", tmp_path)
    plan = dict(plan_id="test", counts={})
    monkeypatch.setattr(runner, "make_plan", lambda experiment: plan)
    monkeypatch.setattr(execution, "save_plan", lambda *args: None)
    monkeypatch.setattr(execution, "execute", lambda *args, **kwargs: pytest.fail("Dry run executed"))
    assert runner.run(replace(Experiment(), n=1), ["--output", "dry"]) == 0
    with pytest.raises(SystemExit):
        runner.run(Experiment(), ["--resume"])
