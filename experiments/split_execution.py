"""Exp.06 condition batches and offline merging; no failed-run resume.

Each batch reserves whole conditions before sending. Overlap is an error, not a
silent skip. Only --remaining explicitly skips previously completed conditions.
The existing common.execute_plan remains the sole paid execution implementation.
"""
from contextlib import contextmanager
import copy
import fcntl
import os
from pathlib import Path
import shutil
import sys
import uuid
import argparse

from experiments import common

NAME = "06_new_models_n10"
DEFAULT_MODEL_ORDER = ("terra", "luna", "sol")
TERMINAL_STATUSES = {"complete", "complete_with_errors"}


def order_models(rows, aliases):
    """Stable model-only ordering; preserve task/metric/H/query order within a model."""
    require(len(aliases) == len(common.NEW_MODELS) and set(aliases) == set(common.NEW_MODELS),
            "--model-order must contain terra, luna and sol exactly once")
    ranks = {common.MODELS[alias]: i for i, alias in enumerate(aliases)}
    return sorted(rows, key=lambda row: ranks[row["model"]])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def result_root(spec):
    require(spec.name == NAME, "Split execution is supported only for Exp.06")
    return common.output_path(common.EXPERIMENTS / spec.name / "results")


@contextmanager
def execution_lock(root):
    """Same lock as common.run: serialize validation/reservation/send/merge."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".execution.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("06 is already running or merging; no action taken") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_plan(saved, full):
    """Reject changed prompts, selection, semantics or generation settings."""
    require(saved["experiment"] == NAME, "Unexpected batch experiment")
    require(saved["provenance"]["semantics_sha256"] == full["provenance"]["semantics_sha256"],
            "Batch upstream/preprocessing semantics changed")
    require(saved.get("api_request_options_by_model") == full["api_request_options_by_model"],
            "Batch API generation settings changed")
    expected = {q["query_id"]: q for q in full["queries"] if not q["reuse_required"]}
    queries = saved["queries"]
    require(bool(queries) and len({q["query_id"] for q in queries}) == len(queries), "Empty/duplicate batch query IDs")
    for q in queries:
        require(q == expected.get(q["query_id"]), "Batch query differs from current full plan")
    selected = {q["condition_id"] for q in queries}
    # Older batches have the original sol/terra/luna order and no policy fields.
    aliases = saved.get("model_order", common.NEW_MODELS)
    require(queries == order_models([q for q in full["queries"] if q["condition_id"] in selected], aliases),
            "Batch must contain complete conditions in plan order")
    expected_conditions = order_models(copy.deepcopy([c for c in full["conditions"] if c["condition_id"] in selected]), aliases)
    for c in expected_conditions:
        c["maximum_api_attempts"] = c["planned_api_requests"] * (saved["upstream_retries"] + 1)
    require(saved["conditions"] == expected_conditions,
            "Batch condition definitions differ")
    return selected


def completed_records(source, plan):
    """Validate all attempted queries, retaining API failures without inventing scores."""
    summary = common.read_json(source / "summary.json")
    rows = common.read_jsonl(source / "records.jsonl")
    expected = {q["query_id"]: q for q in plan["queries"]}
    failed_count = sum(r["status"] == "failed" for r in rows)
    require(summary["status"] == ("complete_with_errors" if failed_count else "complete")
            and summary["successful_queries"] == len(expected) - failed_count
            and summary["failed_queries"] == failed_count and summary["reused_queries"] == 0,
            f"Incomplete batch: {source}")
    if failed_count:
        require(plan.get("continue_on_api_error") is True and summary.get("unstarted_queries") == 0
                and summary.get("unscored_queries") == 0, "Failure batch did not finish all attempts")
        requests = common.read_jsonl(source / "requests.jsonl")
        responses = common.read_jsonl(source / "responses.jsonl")
        failures = common.read_jsonl(source / "failed_queries.jsonl")
        require(failures == [r for r in rows if r["status"] == "failed"], "Failure journal/records mismatch")
    require(len(rows) == len(expected) == len({r["query_id"] for r in rows}), "Missing/duplicate batch records")
    for r in rows:
        q = expected.get(r["query_id"])
        require(q is not None and all(r.get(k) == v for k, v in q.items()), "Record/manifest mismatch")
        require(r["status"] in common.SCORED_STATUSES | {"failed"} and r["api_request_made"] and not r["reused"],
                "Batch contains failed/reused records")
        require(r["source_experiment"] == NAME, "Unexpected batch record source")
        common.verify_prompt(r, q["system_prompt"], q["user_prompt"])
        require(r["request"] == common.expected_api_request(q["model"], q["system_prompt"], q["user_prompt"]),
                "Recorded request parameters changed")
        if r["status"] == "failed":
            attempts = r["attempts"]
            require(r.get("api_error") is True and bool(attempts)
                    and not r.get("raw_response") and not r.get("score")
                    and r.get("usage") is None, "Invalid failed API record")
            require(r["api_attempts"] == len(attempts) <= plan["upstream_retries"] + 1,
                    "Failure attempt count mismatch")
            require([a for a in responses if a["query_id"] == r["query_id"]] == attempts,
                    "Failure response journal mismatch")
            sent = [a for a in requests if a["query_id"] == r["query_id"]]
            require(len(sent) == len(attempts), "Failure request journal missing")
            for index, (request, response) in enumerate(zip(sent, attempts), 1):
                require(request["kwargs"] == r["request"] and response.get("api_error") is True
                        and bool(response.get("exception")) and response.get("raw_response") is None
                        and request["attempt_id"] == response["attempt_id"] == f"{r['query_id']}:{index}",
                        "Failure request/response provenance mismatch")
            continue
        require(r["assistant_text"] == r["raw_response"]["choices"][0]["message"]["content"], "Response text changed")
        require(r["score"]["status"] == r["status"] and r["score"]["prompt"] == q["user_prompt"]
                and r["score"]["index"] == q["query_index"], "Score provenance mismatch")
        run_dir = common.ROOT / r["upstream_output"]
        require(run_dir.resolve() == run_dir and run_dir.is_relative_to(source / "runs"), "Unsafe source run path")
        scored = [s for s in common.read_jsonl(run_dir / "predictions.jsonl") if s["index"] == q["query_index"]]
        require(len(scored) == 1 and scored[0] == r["score"], "Source prediction differs from saved score")
    require({r["query_id"] for r in rows} == set(expected), "Unexpected batch query IDs")
    return rows


def scan_batches(root, full):
    """Persistent batch manifests are reservations, including failed/unstarted rows."""
    require(not (root / "manifest.json").exists() and not (root / ".started.json").exists(),
            "Legacy unsplit 06 results exist; keep them intact and inspect before splitting")
    claims, batches = {}, []
    batch_root = common.output_path(root / "batches")
    for source in sorted(batch_root.iterdir()) if batch_root.exists() else []:
        require(source.is_dir() and source.resolve() == source, f"Unexpected batch path: {source}")
        require((source / "manifest.json").exists(), f"Incomplete batch reservation needs manual inspection: {source}")
        plan = common.read_json(source / "manifest.json")
        conditions = validate_plan(plan, full)
        status = common.read_json(source / "summary.json").get("status") if (source / "summary.json").exists() else "reserved"
        rows = completed_records(source, plan) if status in TERMINAL_STATUSES else []
        for cid in conditions:
            require(cid not in claims, f"Duplicate stored condition: {cid}")
            claims[cid] = {"status": status, "source": str(source)}
        batches.append({"source": source, "plan": plan, "status": status, "records": rows})
    return claims, batches


def select_plan(full, claims, tasks=None, histories=None, remaining=False, model_order=None):
    """Select and order whole conditions without changing query identity/windows."""
    selected = [c for c in full["conditions"] if (tasks is None or c["task"] in tasks)
                and (histories is None or c["H"] in histories)]
    explicit = tasks is not None or histories is not None
    chosen = []
    for c in selected:
        cid = c["condition_id"]
        if c["planned_reused_queries"]:
            if explicit and not remaining:
                raise ValueError(f"実行済みです（05で取得済み）: {cid}; select another condition or use --remaining")
            continue  # The unfiltered full 06 always imports these from 05 at merge.
        if cid in claims:
            old = claims[cid]
            if old["status"] in TERMINAL_STATUSES:
                if remaining:
                    continue
                raise ValueError(f"実行済みです: {cid} (status={old['status']}; {old['source']})")
            raise ValueError(f"予約済み/未完了の条件です（再開非対応・自動再送禁止）: {cid} ({old['source']})")
        chosen.append(cid)
    require(bool(chosen), "No unexecuted conditions remain in the selected range (実行済みです)")
    plan = copy.deepcopy(full)
    aliases = list(DEFAULT_MODEL_ORDER if model_order is None else model_order)
    plan["queries"] = order_models([q for q in plan["queries"] if q["condition_id"] in chosen], aliases)
    plan["conditions"] = order_models([c for c in plan["conditions"] if c["condition_id"] in chosen], aliases)
    plan.update(model_order=aliases, continue_on_api_error=True)
    plan.update(split_execution=True, selection_options={"tasks": tasks, "histories": histories, "remaining": remaining},
                planned_logical_queries=len(plan["queries"]), full_grid_logical_queries=len(full["queries"]))
    for key in ("planned_api_requests", "planned_reused_queries", "maximum_api_attempts"):
        plan[key] = sum(c[key] for c in plan["conditions"])
    return plan


def source_hashes(source):
    return {str(p.relative_to(common.ROOT)): common.file_sha256(p)
            for name in ("manifest.json", "summary.json", "records.jsonl", "requests.jsonl",
                         "responses.jsonl", "failed_queries.jsonl", "run.log")
            if (p := source / name).is_file()}


def merge_results(spec):
    """Copy a validated complete 05+06 grid to a NEW derived results snapshot."""
    root = result_root(spec)
    with execution_lock(root):
        full = common.make_plan(spec, 0)
        cached = common.check_inputs(spec, full)
        claims, batches = scan_batches(root, full)
        needed = {c["condition_id"] for c in full["conditions"] if not c["planned_reused_queries"]}
        require(set(claims) == needed, f"Cannot merge: {len(needed - set(claims))} conditions are still missing")
        require(all(b["status"] in TERMINAL_STATUSES for b in batches), "Cannot merge: incomplete batch exists (no resume supported)")
        source05 = common.EXPERIMENTS / spec.reuse_from / "results"
        sources = [source05, *(b["source"] for b in batches)]
        hashes = {key: value for source in sources for key, value in source_hashes(source).items()}
        by_id = {r["query_id"]: (r, source05) for r in cached.values()}
        for batch in batches:
            for r in batch["records"]:
                require(r["query_id"] not in by_id, "Duplicate query during merge")
                by_id[r["query_id"]] = (r, batch["source"])
        require(set(by_id) == {q["query_id"] for q in full["queries"]}, "Merged grid is not exactly 960 unique queries")
        destination = common.output_path(root / "merged" / uuid.uuid4().hex)
        destination.mkdir(parents=True, exist_ok=False)
        common.write_json(destination / "summary.json", {"status": "merging"}, exclusive=True)
        copied, merged = {}, []
        for q in full["queries"]:
            record, source = by_id[q["query_id"]]
            r = {**copy.deepcopy(record), **q}
            r["source_record"] = f"{source.relative_to(common.ROOT)}/records.jsonl#{q['query_id']}"
            if r["status"] == "failed":
                # There is no model response or upstream score/run to copy.
                r["source_requests"] = str((source / "requests.jsonl").relative_to(common.ROOT))
                r["source_responses"] = str((source / "responses.jsonl").relative_to(common.ROOT))
                merged.append(r)
                continue
            original = common.ROOT / record["upstream_output"]
            require(original.resolve() == original and original.is_relative_to(source / "runs"), "Unsafe original run path")
            if original not in copied:
                require(not any(p.is_symlink() for p in original.rglob("*")), "Symlink in source run")
                target = destination / "runs" / f"source_{sources.index(source):03d}" / original.relative_to(source / "runs")
                shutil.copytree(original, target)
                for p in original.rglob("*"):
                    if p.is_file():
                        require(common.file_sha256(p) == common.file_sha256(target / p.relative_to(original)), "Run copy mismatch")
                copied[original] = target
            r.update(source_record=f"{source.relative_to(common.ROOT)}/records.jsonl#{q['query_id']}",
                     source_upstream_output=record["upstream_output"],
                     upstream_output=str(copied[original].relative_to(common.ROOT)))
            if q["reuse_required"]:
                r.update(reused=True, api_request_made=False, api_attempts=0, retry_attempts=0,
                         source_api_attempts=record["api_attempts"], attempts=[], source_experiment=spec.reuse_from)
            merged.append(r)
        common.write_text(destination / "records.jsonl", "".join(common.json_text(r) + "\n" for r in merged), exclusive=True)
        failed = [r for r in merged if r["status"] == "failed"]
        common.write_text(destination / "failed_queries.jsonl", "".join(common.json_text(r) + "\n" for r in failed), exclusive=True)
        full.update(mode="offline_merge", merge_sources=[str(p.relative_to(common.ROOT)) for p in sources],
                    merge_source_sha256=hashes, merge_api_calls=0)
        common.save_manifest(destination, full)
        for path, expected in hashes.items():
            require(common.file_sha256(common.ROOT / path) == expected, "Source changed while merging")
        summary = {"experiment": spec.name, "status": "complete_with_errors" if failed else "complete", "mode": "offline_merge",
                   "logical_queries": len(merged), "successful_queries": len(merged) - len(failed), "failed_queries": len(failed),
                   "reused_queries": len(cached), "merge_api_calls": 0,
                   "source_api_attempts": sum(common.read_json(b["source"] / "summary.json")["api_attempts"] for b in batches),
                   "source_06_experiment_seconds": sum(common.read_json(b["source"] / "summary.json")["experiment_elapsed_seconds"] for b in batches),
                   "note": "Derived dataset, not an API run. Original source timings retained in records; 05 is reused."}
        common.write_json(destination / "summary.json", summary)
        print(f"Merged 960 queries (05:120 + 06:840; API failures: {len(failed)}); API calls: 0\nSaved: {destination}")
        return destination


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description="Exp.06: disjoint condition batches; no resume")
    parser.add_argument("--task", nargs="+", choices=spec.tasks)
    parser.add_argument("--history", nargs="+", type=int, choices=spec.histories)
    parser.add_argument("--model-order", nargs=3, choices=common.NEW_MODELS,
                        help="Execution order; default: terra luna sol (each exactly once)")
    parser.add_argument("--remaining", action="store_true", help="Skip fully attempted conditions, including complete_with_errors; never skip interrupted reservations")
    parser.add_argument("--merge", action="store_true", help="Offline merge of a complete 05+06 grid")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("Paid execution requires BOTH --execute and --confirm-paid-api")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.merge and (args.execute or args.task or args.history or args.remaining or args.retries or args.model_order):
        parser.error("--merge is offline and cannot be combined with selection/paid/retry flags")
    try:
        if args.merge:
            merge_results(spec)
            return 0
        root = result_root(spec)
        with execution_lock(root):
            full = common.make_plan(spec, args.retries)
            # Always validate the COMPLETE 05 prerequisite, even for a non-H20 batch.
            common.check_inputs(spec, full)
            claims, _ = scan_batches(root, full)
            plan = select_plan(full, claims, args.task, args.history, args.remaining, args.model_order)
            print("Model order: " + " -> ".join(plan["model_order"]))
            print("API errors: save failure and continue; local/safety errors: stop")
            if not args.execute:
                plan["mode"] = "dry_run"
                common.save_manifest(root / "dry_run", plan)
                print("DRY RUN: no API calls; no conditions reserved")
                return 0
            require(bool(os.environ.get("OPENAI_API_KEY")), "OPENAI_API_KEY must be set")
            destination = common.output_path(root / "batches" / uuid.uuid4().hex)
            destination.mkdir(parents=True, exist_ok=False)
            common.write_json(destination / ".started.json", {"started_at_utc": common.utc_now()}, exclusive=True)
            plan["mode"] = "execute"
            common.save_manifest(destination, plan)  # Reservation persists even on interruption.
            print(f"Batch results: {destination}", flush=True)
            common.execute_plan(spec, plan, destination, {}, args.retries)
            if common.read_json(destination / "summary.json")["status"] == "complete_with_errors":
                print("All queries attempted; API failures remain. See failed_queries.jsonl (exit code 2).")
                return 2
            return 0
    except (Exception, common.SafetyStop, KeyboardInterrupt) as exc:
        print("Stopped:", common.redact(str(exc)), file=sys.stderr)
        return 2
