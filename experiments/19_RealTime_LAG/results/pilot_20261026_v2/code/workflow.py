"""Frozen requests, bounded parallel calls, durable recording and replay export."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Any
import csv
import json
import math
import random
import shutil
import threading
import time

from .replay import build_replay

storage = import_module("experiments.18_desicion_LAG.decision_experiment.storage")
dataset = import_module("experiments.18_desicion_LAG.decision_experiment.dataset")
legacy_execution = import_module("experiments.18_desicion_LAG.decision_experiment.execution")
metrics = import_module("experiments.18_desicion_LAG.decision_experiment.metrics")
legacy_outputs = import_module("experiments.18_desicion_LAG.decision_experiment.outputs")
read_json, write_json, digest = storage.read_json, storage.write_json, storage.digest
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
VIEWER = ROOT / "experiments/13_LAG/data/viz/viewer/commentary"
UIDS = ("A0100", "A0200")


def resolve_run(value):
    path = Path(value)
    path = (path if path.exists() else RESULTS / path).resolve()
    if not path.is_relative_to(RESULTS.resolve()) or path == RESULTS.resolve():
        raise ValueError("Run must be inside experiments/19_RealTime_LAG/results")
    return path


def request_for(directory, query):
    directory = Path(directory)
    window = read_json(directory / "inputs" / f'{query["sample_id"]}.json')
    data = storage.canonical(window["blocks"])
    prefix = (f'Episode: episode_01\nCurrent step: {window["end_step"]}\n'
              f'Window: steps {window["start_step"]} through {window["end_step"]}, inclusive\n'
              f'Time points: {window["time_points"]}\n')
    if query["stage"] == "gate":
        # Exactly the v003 input template and selection contract from experiment 18.
        user = (directory / "prompts/gate/user.txt").read_text().format(
            **window, window_data=data)
        system = (directory / "prompts/gate/system.txt").read_text()
        question = (directory / "prompts/gate/aircraft_question.txt").read_text().strip()
        choices = read_json(directory / "prompts/gate/aircraft_choices.json")
        name = "changed_aircraft"
    else:
        user = prefix + f'Target aircraft: {query["aircraft_id"]}\nBEGIN WINDOW DATA\n{data}\nEND WINDOW DATA'
        system = (directory / "prompts/goal/system.txt").read_text()
        question = (directory / "prompts/goal/question.txt").read_text().format(
            aircraft_id=query["aircraft_id"], end_step=window["end_step"])
        choices = read_json(directory / "choices" / f'{query["choice_count"]}.json')
        name = "primary_goal"
    body = {"model": query["model"], "input": [{"role": "developer", "content": system},
            {"role": "user", "content": user}], "questions": [
            {"type": "choice", "name": name, "instructions": question, "choices": choices}]}
    if query.get("request_sha256") and digest(body) != query["request_sha256"]:
        raise ValueError(f'Request changed: {query["id"]}')
    return body


def create_plan(name, repeats=1, max_windows=None):
    storage.safe_name(name)
    if repeats < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError("repeats and max-windows must be positive")
    config = read_json(HERE / "configs/experiment.json")
    source = ROOT / config["input_source"]
    frame = dataset.lag.load_episode(source)
    blocks = dataset.build_blocks(frame)
    h, stride = config["history_points"], config["stride_steps"]
    ends = list(range(min(blocks) + h - 1, max(blocks) + 1, stride))
    if ends[-1] != max(blocks):
        ends.append(max(blocks))
    if max_windows is not None:
        ends = ends[:max_windows]
    directory = RESULTS / name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "inputs").mkdir()
    shutil.copytree(HERE / "choices", directory / "choices")
    shutil.copy2(ROOT / 'experiments/18_desicion_LAG/configs/pricing.json', directory / 'pricing.json')
    gate_source = ROOT / config["gate"]["prompt_source"]
    (directory / "prompts/gate").mkdir(parents=True)
    for filename in ("system.txt", "user.txt", "aircraft_question.txt", "aircraft_choices.json"):
        shutil.copy2(gate_source / filename, directory / "prompts/gate" / filename)
    shutil.copytree(HERE / "prompts/goal_v001", directory / "prompts/goal")
    shutil.copytree(HERE / "realtime_experiment", directory / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    samples = []
    for end in ends:
        window = dataset.window(blocks, end, h)
        sid = f"end_{end:04d}"
        write_json(directory / "inputs" / f"{sid}.json", window)
        entities = {e["object_id"]: e for e in window["blocks"][-1]["entities"]}
        samples.append({"id": sid, "end_step": end, "start_step": window["start_step"],
                        "eligible": {uid: entities[uid]["alive"] != 0 and entities[uid]["state_valid"] == 1 for uid in UIDS}})
    rng = random.Random(config["goal"]["condition_order_seed"])
    base_order = [4, 8, 32]
    rng.shuffle(base_order)
    queries = []
    for repeat in range(1, repeats + 1):
        for index, sample in enumerate(samples):
            stem = f'r{repeat:02d}_{sample["id"]}'
            common = {"sample_id": sample["id"], "repeat": repeat,
                      "source_end_step": sample["end_step"], "source_start_step": sample["start_step"]}
            gate = {**common, "id": stem + "_gate", "stage": "gate", "model": config["gate"]["model"],
                    "gate_query_id": None, "aircraft_id": None, "choice_count": None}
            queries.append(gate)
            shift = (index + repeat - 1) % 3
            order = base_order[shift:] + base_order[:shift]
            for k in order:
                aircraft = list(UIDS) if (index + repeat + k) % 2 else list(reversed(UIDS))
                for uid in aircraft:
                    queries.append({**common, "id": f"{stem}_k{k:02d}_{uid}", "stage": "goal",
                                    "model": config["goal"]["model"], "gate_query_id": gate["id"],
                                    "aircraft_id": uid, "choice_count": k, "eligible": sample["eligible"][uid]})
    for query in queries:
        query["request_sha256"] = digest(request_for(directory, query))
    files = {str(p.relative_to(directory)): storage.file_hash(p)
             for p in directory.rglob("*") if p.is_file()}
    manifest = {"schema_version": 1, "run_id": name, "created_at": storage.utc_now(), "config": config,
                "repeats": repeats, "samples": samples, "queries": queries, "files": files,
                "source_csv_sha256": storage.file_hash(source), "maximum_api_calls": len(queries),
                "recording_last_step": int(frame.step.max()), "source_code": {
                    "dataset": storage.file_hash(Path(dataset.__file__)),
                    "storage": storage.file_hash(Path(storage.__file__))}}
    write_json(directory / "manifest.json", manifest)
    (directory / "manifest.sha256").write_text(storage.file_hash(directory / "manifest.json") + "\n")
    return directory


def verify(directory):
    directory = Path(directory)
    if storage.file_hash(directory / "manifest.json") != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed; make a new run")
    manifest = read_json(directory / "manifest.json")
    for filename, sha in manifest["files"].items():
        if storage.file_hash(directory / filename) != sha:
            raise ValueError(f"Frozen file changed: {filename}")
    return manifest


def parse_answer(raw, body):
    question = body["questions"][0]
    answers = raw.get("answers", [])
    if len(answers) != 1 or answers[0].get("name") != question["name"]:
        raise ValueError("Expected exactly one named answer")
    answer = answers[0]
    if answer.get("type") == "refusal":
        return {"api_status": "refusal"}
    allowed = {c["value"] for c in question["choices"]}
    if answer.get("type") != "choice" or answer.get("choice") not in allowed:
        raise ValueError("Invalid choice")
    probabilities = answer.get("probabilities", [])
    if len(probabilities) != len(allowed) or {p.get("value") for p in probabilities} != allowed:
        raise ValueError("Incomplete probability vector")
    values = [p.get("probability") for p in probabilities]
    confidence = answer.get("confidence")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values + [confidence]):
        raise ValueError("Invalid probabilities/confidence")
    if not math.isclose(sum(values), 1, abs_tol=0.001):
        raise ValueError("Probabilities do not sum to one")
    return {"api_status": "ok", "choice": answer["choice"], "probabilities": probabilities, "confidence": confidence}


def gate_eligible(record, deadline=3.0):
    return bool(record and record.get("api_status") == "ok" and
                record.get("latency_s") is not None and record["latency_s"] <= deadline and
                record.get("choice") in ("A1", "A2", "BOTH"))


def records_for(directory):
    path = Path(directory) / "responses"
    return {p.stem: read_json(p) for p in path.glob("*.json")} if path.exists() else {}


def execute(directory, *, resume=False, limit=None, client=None):
    directory = Path(directory)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    with storage.run_lock(directory):
        manifest = verify(directory)
        journal = storage.read_jsonl(directory / "requests.jsonl")
        started = {r["query_id"] for r in journal}
        if started and not resume:
            raise ValueError("This run has started; use --resume (started queries are never resent)")
        records = records_for(directory)
        if all(q["id"] in records for q in manifest["queries"]):
            return
        owns_client = client is None
        if owns_client:
            from openai import OpenAI
            client = OpenAI(api_key=legacy_execution.read_api_key(), max_retries=0,
                            timeout=manifest["config"]["deadline"]["transport_timeout_seconds"])
        deadline = manifest["config"]["deadline"]["limit_seconds"]
        lock = threading.Lock()
        attempted = 0

        def skip(query, reason):
            record = {"query_id": query["id"], "api_status": reason, "latency_s": None,
                      "choice": None, "finished_at": storage.utc_now()}
            write_json(directory / "responses" / f'{query["id"]}.json', record)
            records[query["id"]] = record

        def call(query, batch_start=None):
            body = request_for(directory, query)
            with lock:
                storage.append_jsonl(directory / "requests.jsonl", {
                    "query_id": query["id"], "started_at": storage.utc_now(),
                    "request_sha256": query["request_sha256"]})
            record = {"query_id": query["id"], "started_at": storage.utc_now(), "api_status": "error", "choice": None}
            start = time.perf_counter()
            record["dispatch_offset_s"] = start - batch_start if batch_start is not None else 0.0
            try:
                raw = client.post("/decisions", cast_to=dict[str, Any], body=body)
                record["latency_s"] = time.perf_counter() - start
                record["raw"] = raw
                try:
                    record.update(parse_answer(raw, body))
                except (ValueError, TypeError, KeyError) as exc:
                    record.update(api_status="invalid_output", error=str(exc))
            except Exception as exc:
                record["latency_s"] = time.perf_counter() - start
                error = str(exc)
                key = getattr(client, "api_key", None)
                if key:
                    error = error.replace(key, "[REDACTED]")
                record.update(api_status="transport_timeout" if type(exc).__name__ == "APITimeoutError" else "error",
                              error_type=type(exc).__name__, error=error,
                              http_status=getattr(exc, "status_code", None))
            record.update(finished_at=storage.utc_now(), deadline_status="within" if record["latency_s"] <= deadline else "exceeded")
            write_json(directory / "responses" / f'{query["id"]}.json', record)
            with lock:
                records[query["id"]] = record
                storage.append_jsonl(directory / "responses.jsonl", record)
                print(f'{query["id"]}: {record["api_status"]}, {record["choice"]}, '
                      f'{record["latency_s"]:.3f}s ({record["deadline_status"]})', flush=True)
            return record

        def usable(query):
            if query["id"] in records:
                return False
            if query["id"] in started:
                skip(query, "uncertain")
                return False
            return True

        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                queries = manifest["queries"]
                index = 0
                while index < len(queries):
                    query = queries[index]
                    if query["stage"] == "gate":
                        if usable(query):
                            if limit is not None and attempted >= limit:
                                break
                            record = call(query)
                            attempted += 1
                            if record["api_status"] == "error":
                                break
                        index += 1
                    else:
                        batch = queries[index:index + 2]
                        assert all(q["stage"] == "goal" and q["gate_query_id"] == query["gate_query_id"] and
                                   q["choice_count"] == query["choice_count"] for q in batch)
                        pending = []
                        for q in batch:
                            if not usable(q):
                                continue
                            gate = records.get(q["gate_query_id"])
                            if not gate_eligible(gate, deadline):
                                skip(q, "skipped_gate")
                            elif not q["eligible"]:
                                skip(q, "skipped_inactive")
                            else:
                                pending.append(q)
                        if limit is not None and attempted + len(pending) > limit:
                            break  # preserve parallel pairs; a one-call remainder stays unused
                        batch_start = time.perf_counter()
                        results = [f.result() for f in as_completed([pool.submit(call, q, batch_start) for q in pending])]
                        attempted += len(pending)
                        if any(r["api_status"] == "error" for r in results):
                            break
                        index += 2
                    # Publish progress after each gate / pair; no API work in export.
                    export(directory, manifest=manifest)
        finally:
            if owns_client:
                client.close()
            export(directory, manifest=manifest)


def export(directory, *, manifest=None):
    directory = Path(directory)
    manifest = manifest or verify(directory)
    records = records_for(directory)
    labels = read_json(directory / "choices/catalog.json")
    pricing = read_json(directory / "pricing.json")
    replay = build_replay(manifest, records, labels)
    report = {"run_id": manifest["run_id"], "created_at": storage.utc_now(),
              "planned_queries": len(manifest["queries"]), "recorded_queries": len(records),
              "complete": len(records) == len(manifest["queries"]), "groups": {}}
    rows = []
    for q in manifest["queries"]:
        r = records.get(q["id"], {})
        rows.append({**{key: q.get(key) for key in ("id", "repeat", "stage", "choice_count", "aircraft_id", "source_end_step")},
                     "api_status": r.get("api_status", "pending"), "choice": r.get("choice"),
                     "latency_s": r.get("latency_s"), "deadline_status": r.get("deadline_status"),
                     "started_at": r.get("started_at"), "finished_at": r.get("finished_at")})
    for group in ("gate", "goal_4", "goal_8", "goal_32"):
        selected = [r for r in rows if (r["stage"] if r["stage"] == "gate" else f'goal_{r["choice_count"]}') == group]
        measured = [r for r in selected if r["latency_s"] is not None]
        ok = [r for r in measured if r["api_status"] == "ok"]
        times = [r["latency_s"] for r in ok]
        report["groups"][group] = {"planned": len(selected), "attempted": len(measured), "ok": len(ok),
            "within_deadline": sum(r["deadline_status"] == "within" for r in ok),
            "late_ok": sum(r["deadline_status"] == "exceeded" for r in ok),
            "errors": len(measured) - len(ok), "p50_s": metrics.quantile(times, .5), "p95_s": metrics.quantile(times, .95)}
    report["replay_status_counts"] = dict(__import__("collections").Counter(e["status"] for e in replay["events"]))
    costs = [legacy_outputs.estimate_cost(r.get('raw', {}).get('usage'), 'decisions', pricing)
             for r in records.values() if r.get('latency_s') is not None]
    report['estimated_cost'] = {'known_usd': sum(c for c in costs if c is not None),
                              'unknown_attempts': sum(c is None for c in costs),
                              'pricing_as_of': pricing['as_of'], 'invoice_verified': False}
    review_path = directory / 'report/review_template.csv'
    if not review_path.exists():
        review_path.parent.mkdir(parents=True, exist_ok=True)
        with review_path.open('w', encoding='utf-8-sig', newline='') as handle:
            fields = ['sample_id','source_end_step','aircraft_id','review_status','goal_4','goal_8','goal_32','acceptable_alternatives','evidence']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in manifest['samples']:
                for uid in UIDS:
                    writer.writerow({'sample_id':sample['id'],'source_end_step':sample['end_step'],'aircraft_id':uid})
    write_json(directory / "report/summary.json", report)
    with (directory / "report/latency.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(directory / "replay.json", replay)
    for k in (4, 8, 32):
        write_json(directory / f"replay/k{k:02d}.json", {**replay, "events": [e for e in replay["events"] if e["choice_count"] in (None, k)]})
    write_json(VIEWER / f'{manifest["run_id"]}.json', replay)
    with storage.run_lock(VIEWER):
        index_path = VIEWER / "index.json"
        index = read_json(index_path) if index_path.exists() else []
        index = [r for r in index if r["id"] != manifest["run_id"]]
        index.append({"id": manifest["run_id"], "episode_id": manifest["config"]["episode_id"],
                      "created_at": manifest["created_at"], "complete": report["complete"],
                      "recorded_queries": len(records), "repeats": manifest["repeats"],
                      "url": f'/data/viz/viewer/commentary/{manifest["run_id"]}.json'})
        write_json(index_path, sorted(index, key=lambda r: r["created_at"], reverse=True))
    return report
