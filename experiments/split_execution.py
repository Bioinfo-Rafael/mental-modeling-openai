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
    require(queries == [q for q in full["queries"] if q["condition_id"] in selected],
            "Batch must contain complete conditions in plan order")
    expected_conditions = copy.deepcopy([c for c in full["conditions"] if c["condition_id"] in selected])
    for c in expected_conditions:
        c["maximum_api_attempts"] = c["planned_api_requests"] * (saved["upstream_retries"] + 1)
    require(saved["conditions"] == expected_conditions,
            "Batch condition definitions differ")
    return selected


def completed_records(source, plan):
    """Validate complete saved responses before duplicate checks or integration."""
    summary = common.read_json(source / "summary.json")
    rows = common.read_jsonl(source / "records.jsonl")
    expected = {q["query_id"]: q for q in plan["queries"]}
    require(summary["status"] == "complete" and summary["successful_queries"] == len(expected)
            and summary["failed_queries"] == 0 and summary["reused_queries"] == 0,
            f"Incomplete batch: {source}")
    require(len(rows) == len(expected) == len({r["query_id"] for r in rows}), "Missing/duplicate batch records")
    for r in rows:
        q = expected.get(r["query_id"])
        require(q is not None and all(r.get(k) == v for k, v in q.items()), "Record/manifest mismatch")
        require(r["status"] in common.SCORED_STATUSES and r["api_request_made"] and not r["reused"],
                "Batch contains failed/reused records")
        require(r["source_experiment"] == NAME, "Unexpected batch record source")
        common.verify_prompt(r, q["system_prompt"], q["user_prompt"])
        require(r["request"] == common.expected_api_request(q["model"], q["system_prompt"], q["user_prompt"]),
                "Recorded request parameters changed")
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
        rows = completed_records(source, plan) if status == "complete" else []
        for cid in conditions:
            require(cid not in claims, f"Duplicate stored condition: {cid}")
            claims[cid] = {"status": status, "source": str(source)}
        batches.append({"source": source, "plan": plan, "status": status, "records": rows})
    return claims, batches


def select_plan(full, claims, tasks=None, histories=None, remaining=False):
    """Only select conditions; query IDs/windows/order stay exactly as in full plan."""
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
            if old["status"] == "complete":
                if remaining:
                    continue
                raise ValueError(f"実行済みです: {cid} ({old['source']})")
            raise ValueError(f"予約済み/未完了の条件です（再開非対応・自動再送禁止）: {cid} ({old['source']})")
        chosen.append(cid)
    require(bool(chosen), "No unexecuted conditions remain in the selected range (実行済みです)")
    plan = copy.deepcopy(full)
    plan["queries"] = [q for q in plan["queries"] if q["condition_id"] in chosen]
    plan["conditions"] = [c for c in plan["conditions"] if c["condition_id"] in chosen]
    plan.update(split_execution=True, selection_options={"tasks": tasks, "histories": histories, "remaining": remaining},
                planned_logical_queries=len(plan["queries"]), full_grid_logical_queries=len(full["queries"]))
    for key in ("planned_api_requests", "planned_reused_queries", "maximum_api_attempts"):
        plan[key] = sum(c[key] for c in plan["conditions"])
    return plan


def source_hashes(source):
    return {str(p.relative_to(common.ROOT)): common.file_sha256(p)
            for p in (source / "manifest.json", source / "summary.json", source / "records.jsonl")}


def merge_results(spec):
    """Copy a validated complete 05+06 grid to a NEW derived results snapshot."""
    root = result_root(spec)
    with execution_lock(root):
        full = common.make_plan(spec, 0)
        cached = common.check_inputs(spec, full)
        claims, batches = scan_batches(root, full)
        needed = {c["condition_id"] for c in full["conditions"] if not c["planned_reused_queries"]}
        require(set(claims) == needed, f"Cannot merge: {len(needed - set(claims))} conditions are still missing")
        require(all(b["status"] == "complete" for b in batches), "Cannot merge: incomplete batch exists (no resume supported)")
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
        full.update(mode="offline_merge", merge_sources=[str(p.relative_to(common.ROOT)) for p in sources],
                    merge_source_sha256=hashes, merge_api_calls=0)
        common.save_manifest(destination, full)
        for path, expected in hashes.items():
            require(common.file_sha256(common.ROOT / path) == expected, "Source changed while merging")
        summary = {"experiment": spec.name, "status": "complete", "mode": "offline_merge",
                   "logical_queries": len(merged), "successful_queries": len(merged), "failed_queries": 0,
                   "reused_queries": len(cached), "merge_api_calls": 0,
                   "source_api_attempts": sum(common.read_json(b["source"] / "summary.json")["api_attempts"] for b in batches),
                   "source_06_experiment_seconds": sum(common.read_json(b["source"] / "summary.json")["experiment_elapsed_seconds"] for b in batches),
                   "note": "Derived dataset, not an API run. Original source timings retained in records; 05 is reused."}
        common.write_json(destination / "summary.json", summary)
        print(f"Merged 960 queries (05:120 + 06:840); API calls: 0\nSaved: {destination}")
        return destination


def run(spec, argv=None):
    parser = argparse.ArgumentParser(description="Exp.06: disjoint condition batches; no resume")
    parser.add_argument("--task", nargs="+", choices=spec.tasks)
    parser.add_argument("--history", nargs="+", type=int, choices=spec.histories)
    parser.add_argument("--remaining", action="store_true", help="Explicitly skip complete conditions (never skip failed reservations)")
    parser.add_argument("--merge", action="store_true", help="Offline merge of a complete 05+06 grid")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error("Paid execution requires BOTH --execute and --confirm-paid-api")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.merge and (args.execute or args.task or args.history or args.remaining or args.retries):
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
            plan = select_plan(full, claims, args.task, args.history, args.remaining)
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
            return 0
    except (Exception, common.SafetyStop, KeyboardInterrupt) as exc:
        print("Stopped:", common.redact(str(exc)), file=sys.stderr)
        return 2
