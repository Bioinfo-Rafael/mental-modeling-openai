"""Plan temporal queries, then optionally execute and score the frozen plan.

Read build_query for the joint forecast, select_uniform_starts for sampling, and
make_plan for the experiment grid. Dataset parsing lives in Experiment 13.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from lag_data.dataset import PROCESSED, index_records, load_episode
from lag_data.definitions import ACTIONS, DT, STATE_FEATURES
from .config import EPISODES, MODELS, OBSERVATION_SCOPES, REQUEST_OPTIONS, TARGET_ACTION_FIELDS, TASKS
from . import batch, evaluation, execution, prompts

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


class InvalidWindow(ValueError):
    """Expected missing/inactive data: exclude this start, with a recorded reason."""


def index_context(records):
    """Recorded aircraft and missile rows by step; effects are not telemetry."""
    by_step = {}
    for (step, object_id), row in records.items():
        if row["entity_type"] in ("aircraft", "missile"):
            by_step.setdefault(step, []).append((object_id, row))
    for rows in by_step.values():
        rows.sort(key=lambda item: (item[1]["entity_type"], item[0]))
    return by_step


def build_query(records, *, task, history_length, start, target_id,
                observation_scope="self", context_records=None, horizon=10):
    """Observe (s_t, a_t) for b..b+H-1; jointly forecast b+H..b+H+9.

    The future state and movement action at each step share that step index.
    No future telemetry or actions, including other entities, are visible.
    """
    if task not in TASKS or history_length < 1 or start < 0:
        raise ValueError("Invalid task or history interval")
    if observation_scope not in OBSERVATION_SCOPES:
        raise ValueError("Invalid observation scope")
    if type(horizon) is not int or horizon != 10:
        raise ValueError("Expected a ten-step forecast")
    if observation_scope == "all_aircraft_missiles" and context_records is None:
        context_records = index_context(records)

    def nullable(value):
        return None if pd.isna(value) else value

    def context_state(object_id, row):
        valid = nullable(row["state_valid"]) == 1
        data = [nullable(row[field]) if valid else None for field in STATE_FEATURES]
        return dict(object_id=object_id, values=data, valid=int(valid),
                    entity_type=row["entity_type"], team=nullable(row["team"]),
                    alive=nullable(row["alive"]), launcher_id=nullable(row["launcher_id"]),
                    event=row["acmi_event"])

    def values(step, kind):
        row = records.get((step, target_id))
        if row is None:
            raise InvalidWindow("missing_record")
        if kind == "state":
            if pd.isna(row["state_valid"]) or row["state_valid"] != 1:
                raise InvalidWindow("invalid_state")
            fields = STATE_FEATURES
        else:
            if pd.isna(row["action_status"]) or row["action_status"] != "recorded":
                raise InvalidWindow("unrecorded_action")
            fields = tuple(f"{field}_id" for field in ACTIONS)
        data = [row[f] for f in fields]
        if any(pd.isna(x) for x in data):
            raise InvalidWindow("incomplete_" + kind)
        if any(not math.isfinite(float(x)) for x in data):
            raise InvalidWindow("nonfinite_" + kind)
        return [float(x) for x in data] if kind == "state" else [int(x) for x in data]

    def block(step, *, state=False, action=False):
        result = {"step": step, "time_s": float(DT * step)}
        if state:
            result["states"] = [{"object_id": target_id, "values": values(step, "state"), "valid": 1}]
        if action:
            result["actions"] = [{"object_id": target_id, "values": values(step, "action"), "status": "recorded"}]
        if observation_scope == "all_aircraft_missiles":
            # Preserve the target's task-specific mask. Other objects add only
            # telemetry at this already-visible step; no future rows are added.
            result["states"] = [
                context_state(object_id, row)
                for object_id, row in context_records.get(step, [])
                if object_id != target_id or state
            ]
            if action:
                result["actions"] = [
                    dict(object_id=object_id,
                         values=[nullable(row[f"{field}_id"]) for field in ACTIONS]
                         if row["action_status"] == "recorded" else [None] * len(ACTIONS),
                         status=row["action_status"])
                    for object_id, row in context_records.get(step, [])
                    if row["entity_type"] == "aircraft"
                ]
        return result

    history = [block(t, state=True, action=True) for t in range(start, start + history_length)]
    q = start + history_length
    target_steps = list(range(q, q + horizon))
    truth = [dict(step=t, state=values(t, "state"), action=values(t, "action")[:3])
             for t in target_steps]
    visible = dict(task=task, target_id=target_id, target_step=q,
                   target_steps=target_steps, horizon=horizon,
                   observation_scope=observation_scope,
                   state_fields=list(STATE_FEATURES), history_action_fields=list(ACTIONS),
                   history=history, query_context=[])
    return visible, truth, q


def select_uniform_starts(eligible, n):
    """M consecutive valid starts -> min + floor(k*M/n), k=0,...,n-1.

    End-exclusive temporal bins reproduce 0,40,...,360 for M=400,n=10.
    Reject holes so 'uniform time' never silently becomes 'uniform rank'.
    """
    if n < 1 or len(eligible) < n:
        raise ValueError(f"Need at least {n} valid starts, found {len(eligible)}")
    if eligible != list(range(eligible[0], eligible[-1] + 1)):
        raise ValueError("Valid starts contain temporal gaps; define a sampling policy for these data")
    return [eligible[0] + k * len(eligible) // n for k in range(n)]


def make_plan(experiment):
    experiment.validate()
    plan = dict(schema_version="trajectory-1.0", experiment=asdict(experiment),
                prompt_version=prompts.PROMPT_VERSION, request_options=dict(REQUEST_OPTIONS),
                samples={}, queries=[], coverage=[], source_sha256={})
    sources = [PROCESSED / "manifest.json", HERE / "run.py", ROOT / "upstream/LLM-Xavier/llm_x/prompting.py"]
    sources += sorted((HERE / "trajectory_experiment").glob("*.py"))
    sources += sorted((HERE.parent / "13_LAG/lag_data").glob("*.py"))
    for episode in experiment.episodes:
        path = PROCESSED / f"{episode}.csv"
        sources.append(path)
        frame = load_episode(path)
        records = index_records(frame)
        context_records = index_context(records)
        source_hash = frame.attrs["csv_sha256"]
        for target in experiment.target_ids:
            last_step = int(frame.loc[frame.object_id.eq(target), "step"].max())
            for h in experiment.histories:
                for task in experiment.tasks:
                    eligible, excluded = [], Counter()
                    for start in range(last_step + 1):
                        try:
                            build_query(records, task=task, history_length=h, start=start, target_id=target,
                                        horizon=experiment.horizon)
                        except InvalidWindow as exc:
                            excluded[str(exc)] += 1
                        else:
                            eligible.append(start)
                    selected = select_uniform_starts(eligible, experiment.n)
                    for scope in experiment.observation_scopes:
                        coverage = dict(episode_id=episode, target_id=target, task=task, H=h, observation_scope=scope,
                                        valid_start_min=eligible[0], valid_start_max=eligible[-1],
                                        valid_start_count=len(eligible), selected_starts=selected,
                                        excluded_start_counts=dict(excluded), selected_target_steps=[],
                                        horizon=experiment.horizon, selected_target_ranges=[])
                        plan["coverage"].append(coverage)
                        for sample_number, start in enumerate(selected):
                            visible, truth, q = build_query(records, task=task, history_length=h, start=start, target_id=target,
                                                            observation_scope=scope, context_records=context_records,
                                                            horizon=experiment.horizon)
                            target_step = visible["target_step"]
                            coverage["selected_target_steps"].append(target_step)
                            coverage["selected_target_ranges"].append(visible["target_steps"])
                            fields = dict(action=list(TARGET_ACTION_FIELDS), state=list(STATE_FEATURES))
                            sample = dict(episode_id=episode, csv_sha256=source_hash, target_id=target, task=task,
                                          H=h, start=start, query_step=q, target_step=target_step, observation_scope=scope,
                                          horizon=experiment.horizon, target_steps=visible["target_steps"],
                                          sample_number=sample_number, history_steps=[b["step"] for b in visible["history"]],
                                          prediction_fields=fields, ground_truth=truth,
                                          messages=prompts.build_messages(visible),
                                          problem_id=execution.digest([source_hash, target, task, visible["target_steps"], fields]))
                            sid = execution.digest(sample)[:24]
                            if sid in plan["samples"]:
                                raise ValueError("Duplicate sample identity")
                            plan["samples"][sid] = sample
                            for alias in experiment.models:
                                condition = f"{alias}/{episode}/{target}/H{h}/{task}/{scope}"
                                query = dict(condition_id=condition, sample_id=sid, model_alias=alias,
                                             model=MODELS[alias], repeat_id=1)
                                query["request_sha256"] = execution.digest(execution.request_for(plan, query))
                                query["query_id"] = execution.digest(query)
                                plan["queries"].append(query)
    plan["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    plan["counts"] = dict(samples=len(plan["samples"]), requests=len(plan["queries"]),
                          conditions=len({q["condition_id"] for q in plan["queries"]}), n=experiment.n)
    # Round-trip tuples to JSON lists before hashing/comparing with the saved plan.
    plan = json.loads(execution.canonical(plan))
    plan["plan_id"] = execution.digest(plan)
    execution.verify_plan(plan)
    return plan


def run(experiment, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Save prompts/plan only (default)")
    mode.add_argument("--execute", action="store_true", help="Submit/resume model-specific Batch API jobs")
    mode.add_argument("--collect", action="store_true", help="Retrieve Batch results and score; never submit new jobs")
    mode.add_argument("--status", action="store_true", help="Show saved local Batch status (no API call)")
    mode.add_argument("--adopt", action="store_true", help="Recover a submission with an unknown Batch ID")
    mode.add_argument("--rescore", action="store_true", help="Score saved responses offline, without rebuilding prompts")
    parser.add_argument("--resume", action="store_true", help="With --execute: compatibility flag; Batch resume is automatic")
    parser.add_argument("--wait", action="store_true", help="With --execute/--collect: poll until active batches finish")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--job", help="With --adopt: local job name, e.g. sol_000")
    parser.add_argument("--batch-id", help="With --adopt: remote Batch ID")
    parser.add_argument("--output", default=experiment.name, help="Run directory name under this experiment's results/")
    parser.add_argument("--models", nargs="+", choices=MODELS)
    parser.add_argument("--histories", nargs="+", type=int)
    parser.add_argument("--episodes", nargs="+", choices=EPISODES)
    parser.add_argument("--targets", nargs="+", choices=("A0100", "A0200"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS)
    parser.add_argument("--scopes", nargs="+", choices=OBSERVATION_SCOPES)
    parser.add_argument("--n", type=int)
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.wait and not (args.execute or args.collect):
        parser.error("--wait requires --execute or --collect")
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    if args.adopt and not (args.job and args.batch_id):
        parser.error("--adopt requires --job and --batch-id")
    if (args.job or args.batch_id) and not args.adopt:
        parser.error("--job and --batch-id require --adopt")
    changes = {field: tuple(getattr(args, option)) for field, option in (
        ("models", "models"), ("histories", "histories"), ("episodes", "episodes"),
        ("target_ids", "targets"), ("tasks", "tasks"), ("observation_scopes", "scopes")) if getattr(args, option) is not None}
    if args.n is not None:
        changes["n"] = args.n
    saved_mode = args.rescore or args.collect or args.status or args.adopt
    if saved_mode and changes:
        parser.error("This mode uses the saved plan; do not specify new conditions")
    directory = (HERE / "results" / args.output).resolve()
    if not directory.is_relative_to((HERE / "results").resolve()) or directory == (HERE / "results").resolve():
        parser.error("--output must identify a directory inside results/")
    with execution.run_lock(directory):
        if saved_mode:
            plan = execution.read_json(directory / "manifest.json")
            execution.verify_plan(plan)
            if args.rescore:
                evaluation.evaluate_saved(plan, directory)
            elif args.status:
                print(json.dumps(batch.local_status(batch.load_state(plan, directory)), indent=2))
            else:
                batch.execute(plan, directory, repository_root=ROOT, submit_new=False,
                              wait=args.wait, poll_seconds=args.poll_seconds,
                              adopt_job=args.job, batch_id=args.batch_id)
        else:
            plan = make_plan(replace(experiment, **changes))
            execution.save_plan(plan, directory)
            print(f"Plan: {plan['counts']}\nSaved: {directory}", flush=True)
            if args.execute:
                batch.execute(plan, directory, repository_root=ROOT,
                              wait=args.wait, poll_seconds=args.poll_seconds)
            else:
                state = batch.prepare(plan, directory)
                print(json.dumps(batch.local_status(state), indent=2))
                print("Dry run complete. No API calls made. Use --execute to submit Batch jobs; --collect --wait to retrieve results.")
    return 0
