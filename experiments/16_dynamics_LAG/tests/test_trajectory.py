"""Offline contracts for joint forecasts, complete windows, scoring and execution."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(HERE), str(HERE.parent / "13_LAG"), str(ROOT / "upstream/LLM-Xavier")]

from lag_data.dataset import PROCESSED, index_records, load_episode
from lag_data.definitions import ACTIONS, STATE_FEATURES
from dynamics_experiment.config import Experiment
from dynamics_experiment import batch, evaluation, execution, prompts, runner


@pytest.fixture(scope="module")
def full_plan():
    return runner.make_plan(Experiment())


@pytest.fixture(scope="module")
def records():
    return index_records(load_episode(PROCESSED / f"{Experiment().episodes[0]}.csv"))


def answer(rows):
    return prompts.TRAJECTORY_MARKER + " " + json.dumps(rows)


def test_default_grid_and_pairing(full_plan):
    assert full_plan["counts"] == dict(samples=20, requests=60, conditions=12, n=5)
    assert set(Counter(q["condition_id"] for q in full_plan["queries"]).values()) == {5}
    assert set(Counter(q["sample_id"] for q in full_plan["queries"]).values()) == {3}
    assert all(q["repeat_id"] == 1 for q in full_plan["queries"])
    pairs = {}
    for sample in full_plan["samples"].values():
        assert sample["H"] == 20 and sample["horizon"] == 10
        pairs.setdefault(sample["problem_id"], {})[sample["observation_scope"]] = sample
    assert len(pairs) == 10
    for pair in pairs.values():
        left, right = pair["self"], pair["all_aircraft_missiles"]
        for field in ("ground_truth", "start", "history_steps", "target_steps", "sample_number"):
            assert left[field] == right[field]
        assert execution.prompt_filename(left) != execution.prompt_filename(right)
    for coverage in full_plan["coverage"]:
        assert coverage["selected_starts"] == [
            coverage["valid_start_min"] + k * coverage["valid_start_count"] // 5 for k in range(5)
        ]
    with pytest.raises(ValueError, match="temporal gaps"):
        runner.select_uniform_starts([0, 1, 3], 2)
    with pytest.raises(ValueError, match="at least"):
        runner.select_uniform_starts([0, 1], 5)


def test_exact_future_truth_and_no_future_input(full_plan, records):
    for sample in full_plan["samples"].values():
        start, target = sample["start"], sample["target_id"]
        visible, truth, q = runner.build_query(
            records, task="trajectory", history_length=20, start=start, target_id=target,
            observation_scope=sample["observation_scope"],
        )
        assert q == start + 20
        assert visible["query_context"] == []
        assert [b["step"] for b in visible["history"]] == list(range(start, q))
        assert visible["target_steps"] == list(range(q, q + 10))
        assert truth == sample["ground_truth"]
        assert prompts.build_messages(visible) == sample["messages"]
        for expected in truth:
            row = records[expected["step"], target]
            assert expected["state"] == [row[f] for f in STATE_FEATURES]
            assert expected["action"] == [row[f"{f}_id"] for f in ACTIONS[:3]]
        for block in visible["history"]:
            assert block["step"] < q
            ids = {state["object_id"] for state in block["states"]}
            if sample["observation_scope"] == "self":
                assert ids == {target}
            else:
                assert ids == {uid for (t, uid), row in records.items()
                               if t == block["step"] and row["entity_type"] in ("aircraft", "missile")}
                for state in block["states"]:
                    row = records[block["step"], state["object_id"]]
                    assert state["values"] == [None if not state["valid"] or pd.isna(row[f]) else row[f]
                                               for f in STATE_FEATURES]
            assert all(len(a["values"]) == 4 for a in block["actions"])
        text = sample["messages"][1]["content"]
        assert sample["episode_id"] not in text
        assert "missile_launched" not in text and "p_shoot" not in text
        assert "BOTH" in text and "SAME step" in text
        assert prompts.TRAJECTORY_MARKER in text


def test_future_changes_cannot_leak_into_prompt(records):
    kwargs = dict(task="trajectory", history_length=20, start=0, target_id="A0100",
                  observation_scope="all_aircraft_missiles")
    before, truth_before, _ = runner.build_query(records, **kwargs)
    changed = deepcopy(records)
    for (step, _), row in changed.items():
        if step >= 20:
            row["alt_m"] = 9999.0
            if row["action_status"] == "recorded":
                row["altitude_action_id"] = (int(row["altitude_action_id"]) + 1) % 3
    after, truth_after, _ = runner.build_query(changed, **kwargs)
    assert truth_before != truth_after
    assert prompts.build_messages(before) == prompts.build_messages(after)


@pytest.mark.parametrize("kind", ["missing", "state", "action", "nonfinite"])
def test_entire_future_window_must_be_complete(records, kind):
    changed = deepcopy(records)
    row = changed[29, "A0100"]  # Last forecast step must be checked too.
    if kind == "missing":
        del changed[29, "A0100"]
    elif kind == "state":
        row["state_valid"] = 0
    elif kind == "action":
        row["action_status"] = "not_recorded"
    else:
        row["alt_m"] = float("inf")
    with pytest.raises(runner.InvalidWindow):
        runner.build_query(changed, task="trajectory", history_length=20, start=0, target_id="A0100")


def test_terminal_window_and_sampling(full_plan, records):
    for coverage in full_plan["coverage"]:
        kwargs = dict(task="trajectory", history_length=20, target_id=coverage["target_id"])
        runner.build_query(records, start=coverage["valid_start_max"], **kwargs)
        with pytest.raises(runner.InvalidWindow):
            runner.build_query(records, start=coverage["valid_start_max"] + 1, **kwargs)
    with pytest.raises(ValueError, match="ten"):
        replace(Experiment(), horizon=9).validate()


@pytest.mark.parametrize("corruption", ["short", "long", "reorder", "duplicate", "wrong_step", "bool_step",
                                         "state_short", "nan", "inf", "bool_state", "shoot", "float_action",
                                         "bool_action", "out_of_range", "missing_key", "extra_key"])
def test_strict_trajectory_contract(full_plan, corruption):
    sample = next(iter(full_plan["samples"].values()))
    rows = deepcopy(sample["ground_truth"])
    if corruption == "short": rows.pop()
    elif corruption == "long": rows.append(deepcopy(rows[-1]))
    elif corruption == "reorder": rows.reverse()
    elif corruption == "duplicate": rows[1] = deepcopy(rows[0])
    elif corruption == "wrong_step": rows[0]["step"] += 1
    elif corruption == "bool_step": rows[0]["step"] = True
    elif corruption == "state_short": rows[0]["state"].pop()
    elif corruption == "nan": rows[0]["state"][0] = float("nan")
    elif corruption == "inf": rows[0]["state"][0] = float("inf")
    elif corruption == "bool_state": rows[0]["state"][0] = True
    elif corruption == "shoot": rows[0]["action"].append(0)
    elif corruption == "float_action": rows[0]["action"][0] = 1.0
    elif corruption == "bool_action": rows[0]["action"][0] = True
    elif corruption == "out_of_range": rows[0]["action"][1] = 5
    elif corruption == "missing_key": del rows[0]["action"]
    elif corruption == "extra_key": rows[0]["extra"] = 1
    assert evaluation.parse_prediction(answer(rows), sample["target_steps"])[0] is None


def test_markers_and_json(full_plan):
    sample = next(iter(full_plan["samples"].values()))
    text = answer(sample["ground_truth"])
    assert evaluation.parse_prediction(text, sample["target_steps"]) == (sample["ground_truth"], None)
    for bad in ("", text + "\n" + text, text + " trailing junk", text.replace('"step":', '"step": 20, "step":', 1),
                prompts.TRAJECTORY_MARKER + "\n" + json.dumps(sample["ground_truth"])):
        assert evaluation.parse_prediction(bad, sample["target_steps"])[0] is None


def test_joint_scores_horizons_and_failure_denominators(full_plan):
    sample = next(iter(full_plan["samples"].values()))
    truth = deepcopy(sample["ground_truth"])
    prediction = deepcopy(truth)
    truth[0]["state"][5], prediction[0]["state"][5] = 359, 1
    prediction[9]["state"][2] += 10
    prediction[9]["action"][0] = (truth[9]["action"][0] + 1) % 3
    score = evaluation.score(sample["prediction_fields"], truth, answer(prediction))
    assert score["parse_success"]
    assert score["action_step_accuracy"] == 0.9
    assert not score["action_trajectory_exact_match"]
    assert score["action_component_accuracy"]["altitude_action"] == 0.9
    assert score["state_mae_by_field"]["alt_m"] == 1
    assert score["state_mae_by_field"]["yaw_deg"] == 35.8
    assert score["circular_angle_mae_deg"]["yaw_deg"] == 0.2
    rows = [dict(condition_id="one", **score),
            dict(condition_id="one", parse_success=False, api_error="Timeout"),
            dict(condition_id="one", parse_success=False, api_error=None)]
    summary = evaluation.summarize(rows, [dict(condition_id="one")] * 5)[0]
    assert summary["parsed"] == 1 and summary["unscored"] == 2 and not summary["complete"]
    assert summary["api_errors"] == summary["parse_failures"] == 1
    assert summary["scored_steps"] == 10 and summary["planned_steps"] == 50
    assert summary["by_horizon"][0]["action_step_accuracy"] == 1
    assert summary["by_horizon"][9]["action_step_accuracy"] == 0
    assert summary["by_horizon"][9]["state_mae_by_field"]["alt_m"] == 10


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
        return SimpleNamespace(model_dump=lambda **kw: {"choices": [{"message": {"content": answer([dict(step=t, state=[0.0] * 10, action=[1, 2, 0]) for t in range(20, 30)])}}]})

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
    monkeypatch.setattr(batch, "execute", lambda *args, **kwargs: pytest.fail("Dry run sent a Batch"))
    monkeypatch.setattr(batch, "prepare", lambda *args: dict(jobs=[]))
    assert runner.run(replace(Experiment(), n=1), ["--output", "dry"]) == 0
    with pytest.raises(SystemExit):
        runner.run(Experiment(), ["--resume"])
