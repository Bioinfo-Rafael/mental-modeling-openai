"""Frozen LAG request plans: identity digests, journals, and synchronous execution.

Promoted from the byte-identical ``execution.py`` copies in ``workspace/15_trajectory_LAG``,
``16_dynamics_LAG`` and ``17_commentary_LAG`` (``14_mental-modeling_LAG`` held the same file
without the batch-transport guard in ``execute``). The behaviour is unchanged:
``canonical``/``digest`` define every saved ``plan_id``, ``request_sha256`` and ``query_id``,
so they must stay byte-for-byte stable or committed plans stop verifying.

The write-before-send and no-resend resume behavior follows Experiment 09. There is no NPZ
loader, upstream evaluation loop, or scorer in this module. Nothing here imports ``workspace/``.
"""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Plan = dict[str, Any]
SDK_TIMEOUT_SECONDS = 180.0


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_journal(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    # A torn line is a hard error. Do not silently discard a possibly sent request.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def append_journal(path: str | Path, value: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def run_lock(directory: str | Path) -> Iterator[None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def request_for(plan: Plan, query: dict[str, Any]) -> dict[str, Any]:
    sample = plan["samples"][query["sample_id"]]
    return {"model": query["model"], **plan["request_options"], "messages": sample["messages"]}


def verify_plan(plan: Plan) -> None:
    identity = {k: v for k, v in plan.items() if k != "plan_id"}
    if plan.get("plan_id") != digest(identity):
        raise ValueError("Plan digest mismatch")
    ids = set()
    for query in plan["queries"]:
        if query["query_id"] in ids:
            raise ValueError("Duplicate query_id")
        ids.add(query["query_id"])
        if digest(request_for(plan, query)) != query["request_sha256"]:
            raise ValueError("Request differs from its saved hash")


def prompt_filename(sample: dict[str, Any]) -> str:
    """Readable condition and window identity; prompts are shared by models."""
    return (
        f"{sample['episode_id']}__{sample['target_id']}__{sample['task']}"
        f"__{sample['observation_scope']}"
        f"__H{sample['H']:02d}__sample{sample['sample_number'] + 1:02d}"
        f"__start{sample['start']:04d}__step{sample['target_step']:04d}.txt"
    )


def save_plan(plan: Plan, directory: str | Path) -> None:
    """A changed plan needs another directory; never overwrite an existing run."""
    verify_plan(plan)
    directory = Path(directory)
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != plan:
            raise FileExistsError("This directory contains a different plan; select a new --output name")
    else:
        write_json(path, plan)
    previews = directory / "prompts"
    previews.mkdir(exist_ok=True)
    for sample in plan["samples"].values():
        # One file per distinct input, shared by models and repeats.
        text = "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in sample["messages"])
        preview = previews / prompt_filename(sample)
        if preview.exists() and preview.read_text(encoding="utf-8") != text + "\n":
            raise ValueError(f"Preview was edited; change prompts.py and create a new plan: {preview}")
        preview.write_text(text + "\n", encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_error(exc: BaseException, api_key: str | None) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", message)


def _check_sources(plan: Plan, repository_root: str | Path) -> None:
    for relative, expected in plan["source_sha256"].items():
        actual = hashlib.sha256((Path(repository_root) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source changed since planning: {relative}")


def execute(
    plan: Plan,
    directory: str | Path,
    *,
    repository_root: str | Path,
    resume: bool = False,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Send only unattempted requests. A supplied client_factory supports offline tests."""
    verify_plan(plan)
    directory = Path(directory)
    if read_json(directory / "manifest.json") != plan:
        raise ValueError("Saved preview and execution plan differ")
    _check_sources(plan, repository_root)
    api = directory / "api"
    if (api / "identity.json").exists() and read_json(api / "identity.json").get("transport") == "batch":
        raise ValueError("Batch run cannot be executed by the synchronous runner")
    if api.exists() and not resume:
        raise FileExistsError("API run exists; use --resume for unattempted requests only")
    if resume and not api.exists():
        raise ValueError("No API run to resume")
    if api.exists() and read_json(api / "identity.json")["plan_id"] != plan["plan_id"]:
        raise ValueError("Resume identity differs")
    requests = read_journal(api / "requests.jsonl")
    responses = read_journal(api / "responses.jsonl")
    lookup = {q["query_id"]: q for q in plan["queries"]}
    attempted = {r["query_id"] for r in requests}
    responded = {r["query_id"] for r in responses}
    if len(attempted) != len(requests) or not attempted <= lookup.keys():
        raise ValueError("Invalid request journal")
    if len(responded) != len(responses) or not responded <= attempted:
        raise ValueError("Invalid response journal")
    for record in requests:
        if record["request"] != request_for(plan, lookup[record["query_id"]]):
            raise ValueError("Journaled request differs from plan")
    pending = [q for q in plan["queries"] if q["query_id"] not in attempted]
    api_key, client = None, None
    if pending:
        if client_factory is not None:
            client = client_factory()
        else:
            from openai import OpenAI

            api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: ").strip()
            if not api_key:
                raise ValueError("Empty API key; no requests sent")
            client = OpenAI(api_key=api_key, max_retries=0, timeout=SDK_TIMEOUT_SECONDS)
    if not api.exists():
        api.mkdir()
        write_json(
            api / "identity.json",
            {"plan_id": plan["plan_id"], "sdk_max_retries": 0, "timeout_seconds": int(SDK_TIMEOUT_SECONDS)},
        )
    try:
        for query in pending:
            qid = query["query_id"]
            request = request_for(plan, query)
            started, clock = _now(), time.perf_counter()
            append_journal(api / "requests.jsonl", {"query_id": qid, "started_at": started, "request": request})
            attempted.add(qid)
            print(
                f"[{len(attempted)}/{len(lookup)}] {query['condition_id']} "
                f"sample={query['sample_id']} repeat={query['repeat_id']}",
                flush=True,
            )
            try:
                result = client.chat.completions.create(**request)
                raw = result.model_dump(mode="json")
            except BaseException as exc:
                append_journal(
                    api / "responses.jsonl",
                    {
                        "query_id": qid,
                        "raw_response": None,
                        "started_at": started,
                        "elapsed_seconds": time.perf_counter() - clock,
                        "error_type": type(exc).__name__,
                        "error": _redact_error(exc, api_key),
                    },
                )
                raise
            append_journal(
                api / "responses.jsonl",
                {
                    "query_id": qid,
                    "raw_response": raw,
                    "started_at": started,
                    "finished_at": _now(),
                    "elapsed_seconds": time.perf_counter() - clock,
                    "error_type": None,
                    "error": None,
                },
            )
    finally:
        if client is not None:
            client.close()
        saved = read_journal(api / "responses.jsonl")
        successful = {r["query_id"] for r in saved if r["raw_response"] is not None}
        write_json(
            api / "summary.json",
            {
                "planned": len(lookup),
                "attempted": len(attempted),
                "responses_saved": len(successful),
                "unattempted": len(lookup) - len(attempted),
                "unresolved": len(attempted - successful),
                "status": "complete" if len(successful) == len(lookup) else "incomplete",
            },
        )
    return read_json(api / "summary.json")
