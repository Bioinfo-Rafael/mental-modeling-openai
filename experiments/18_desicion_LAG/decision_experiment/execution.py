"""Serial API execution with durable intent/result journaling and explicit resume."""
from importlib.metadata import version
import getpass
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

from .outputs import parse_output
from .planning import request_for, verify_plan
from .storage import append_jsonl, read_json, read_jsonl, run_lock, utc_now, write_json


def read_api_key():
    """Use the environment or hidden terminal input; never persist the entered key."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise ValueError("ターミナルで実行してAPIキーを入力するか、OPENAI_API_KEYを設定してください")
    try:
        # Refuse getpass's echoing fallback if this terminal cannot hide input.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("OpenAI APIキー（入力は表示されません）: ").strip()
    except getpass.GetPassWarning:
        raise ValueError("入力を非表示にできません。別のターミナルかOPENAI_API_KEYを使用してください") from None
    except (EOFError, KeyboardInterrupt):
        raise ValueError("APIキーの入力を中止しました。APIには送信していません") from None
    if not key:
        raise ValueError("APIキーが空です。再実行して入力してください")
    return key


def call_api(client, condition, body):
    if condition == "decisions":
        # The installed SDK need not expose client.decisions yet.
        return client.post("/decisions", cast_to=dict[str, Any], body=body)
    return client.responses.create(**body).model_dump(mode="json")


def execute(directory, *, resume=False, limit=None, client=None, continue_on_timeout=False):
    directory = Path(directory)
    with run_lock(directory):
        manifest = verify_plan(directory)
        journal = read_jsonl(directory / "journal.jsonl")
        started = {row["query_id"] for row in journal if row["event"] == "started"}
        if started and not resume:
            raise ValueError("This run has attempted queries. Use --resume to send only never-started queries.")
        pending = [q for q in manifest["queries"] if q["id"] not in started]
        if limit is not None:
            if limit < 1:
                raise ValueError("--limit must be positive")
            pending = pending[:limit]
        if not pending:
            return {"attempted_now": 0, "remaining_never_started": 0}
        # Verify every request before any network request; changes cannot fail halfway through.
        for query in manifest["queries"]:
            request_for(directory, manifest, query)
        owns_client = client is None
        if owns_client:
            from openai import OpenAI
            client = OpenAI(api_key=read_api_key(), max_retries=0,
                            timeout=manifest["config"]["timeout_seconds"])
        samples = read_json(directory / "samples.json")
        session = {"event": "session", "time": utc_now(), "openai_version": version("openai"),
                   "base_url": str(client.base_url), "automatic_retries": 0, "concurrency": 1,
                   "continue_on_timeout": continue_on_timeout}
        append_jsonl(directory / "journal.jsonl", session)
        attempted = 0
        try:
            for query in pending:
                qid = query["id"]
                body = request_for(directory, manifest, query)
                append_jsonl(directory / "journal.jsonl", {"event": "started", "query_id": qid,
                             "time": utc_now(), "request_sha256": query["request_sha256"]})
                record = {"query_id": qid, "started_at": utc_now(), "status": "api_error"}
                start = time.perf_counter()
                try:
                    raw = call_api(client, query["condition"], body)
                    record["latency_s"] = time.perf_counter() - start
                    record["raw"] = raw
                    try:
                        record["parsed"] = parse_output(raw, query["condition"], samples[query["sample_id"]]["window"],
                                                       decisions_task=manifest["config"].get("decisions_task", "binary"))
                        record["status"] = "ok"
                    except (ValueError, KeyError, TypeError) as exc:
                        record.update(status="invalid_output", error=str(exc))
                except Exception as exc:
                    message = str(exc)
                    key = client.api_key
                    if isinstance(key, str) and key:
                        message = message.replace(key, "[REDACTED]")
                    record.update(latency_s=time.perf_counter() - start,
                                  error_type=type(exc).__name__, error=message,
                                  http_status=getattr(exc, "status_code", None),
                                  request_id=getattr(exc, "request_id", None))
                record["finished_at"] = utc_now()
                write_json(directory / "responses" / f"{qid}.json", record)
                append_jsonl(directory / "journal.jsonl", {"event": "finished", "query_id": qid,
                             "time": utc_now(), "status": record["status"]})
                attempted += 1
                print(f'{qid}: {record["status"]} ({record["latency_s"]:.3f}s)', flush=True)
                # Optional timeout continuation sends the next query, never retries this one.
                # Authentication/rate limit/server errors still stop the run.
                if record["status"] == "api_error":
                    print(f'  {record["error_type"]}: {record["error"]}', flush=True)
                    if continue_on_timeout and record["error_type"] == "APITimeoutError":
                        continue
                    print('Stopped. Use --resume to continue with never-started queries.', flush=True)
                    break
        finally:
            if owns_client:
                client.close()
        return {"attempted_now": attempted,
                "remaining_never_started": len(manifest["queries"]) - len(started) - attempted}
