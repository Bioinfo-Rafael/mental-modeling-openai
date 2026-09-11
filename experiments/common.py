"""Planning, guarded upstream CLI execution, and lossless (secret-redacted) logging.

No client is constructed by planning/preview. Scoring lives exclusively in llm_x.
Run entry points are intentionally small; offline statistics live in analysis.py.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

# Do not create __pycache__ in the read-only vendored source tree.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"

from preprocessing.llmx_original import (  # noqa: E402
    DEFAULT_DATA_ROOT, build_prompt_queries, discover_episodes, file_sha256,
)
from tools.output_safety import derived_path  # noqa: E402
# @Rafa: 元論文の実装の再利用
from llm_x import cli as llmx_cli  # noqa: E402
from llm_x.backends import OpenAIChatBackend  # noqa: E402
from llm_x.evaluation import EvaluationConfig, _summarize  # noqa: E402

PILOT_H = 5
TASKS = ("MountainCar-v0", "Pendulum-v1")
METRICS = ("next-action", "last-action", "next-state", "last-state")
H_VALUES = (5, 10, 20, 30)
NEW_MODELS = ("sol", "terra", "luna")
MODELS = {
    "3.5": "gpt-3.5-turbo",
    "luna": "gpt-5.6-luna",
    "terra": "gpt-5.6-terra",
    "sol": "gpt-5.6-sol",
}
QUESTIONS = {
    "MountainCar-v0": {
        "next-action": "next_action_prediction",
        "last-action": "last_action_prediction",
        "next-state": "next_state_prediction",
        "last-state": "last_state_prediction",
    },
    "Pendulum-v1": {
        "next-action": "next_action_prediction_continuous_bins",
        "last-action": "last_action_prediction_continuous_bins",
        "next-state": "next_state_prediction",
        "last-state": "last_state_prediction",
    },
}
IDENTITY_FIELDS = (
    "model", "task", "metric", "question_name", "H", "episode_path",
    "episode_sha256", "query_index", "system_prompt_sha256", "user_prompt_sha256",
)
SCORED_STATUSES = {"match", "mismatch", "ignored"}


def api_request_options(model):
    """Explicit generation options; omitted options use the API defaults."""
    if model in {MODELS[alias] for alias in NEW_MODELS}:
        return {"reasoning_effort": "medium"}
    return {"temperature": 0}  # Preserve the existing GPT-3.5 experiments.


def expected_api_request(model, system_prompt, user_prompt):
    """Single request contract shared by sending, reuse, and resume validation."""
    return {"model": model, **api_request_options(model), "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]}


@dataclass(frozen=True)
class Experiment:
    name: str
    models: tuple[str, ...]
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    histories: tuple[int, ...]
    n: int
    preview: bool = False
    preview_from: str | None = None
    reuse_from: str | None = None
    questions: dict[str, dict[str, str]] | None = None
    reuse_n: int = 1  # Pendulum H20 prefix required from reuse_from; old specs retain N1 behavior.


class SafetyStop(BaseException):
    """Bypass upstream's broad Exception retry on validation/logging failures."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def query_id(row):
    return digest(json_text({key: row[key] for key in IDENTITY_FIELDS}))

# @Rafa: なぜか論文の実装は実際に渡すtimestepをHで指定せずにそれを-1した値で指定するので、その変換
def history_size(H):
    """The ONLY conversion from actual history length to upstream's parameter."""
    if type(H) is not int or H < 1:
        raise ValueError("H must be a positive integer")
    return H - 1


def output_path(path):
    """All writes stay inside an actual experiment's results, not via symlinks."""
    path = Path(path).absolute()
    relative = path.relative_to(EXPERIMENTS)
    if len(relative.parts) < 2 or relative.parts[1] != "results":
        raise ValueError("Experiment output must be experiments/<name>/results/...")
    safe = derived_path(path)
    if safe.resolve() != path:
        raise ValueError("Symlinked experiment output paths are not supported")
    return safe


def redact(value):
    """Never serialize clients, environment, HTTP headers, or credential values."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if re.search(
                r"authorization|api.?key|password|secret|credential|cookie|access.?token|refresh.?token",
                str(key), re.I,
            ) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        secret = os.environ.get("OPENAI_API_KEY")
        if secret:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", value)
        return value
    return value


def write_text(path, text, *, exclusive=False):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(redact(text))
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value, *, exclusive=False):
    write_text(path, json.dumps(redact(value), ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n", exclusive=exclusive)


def write_csv(path, rows):
    buffer = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json_text(value) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    write_text(path, buffer.getvalue())


def append_jsonl(path, row):
    # Logging failures must NOT be interpreted by upstream as retryable API errors.
    try:
        with output_path(path).open("a", encoding="utf-8") as handle:
            handle.write(json_text(redact(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        raise SafetyStop(f"Cannot persist {Path(path).name}: {type(exc).__name__}") from None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def provenance():
    upstream = ROOT / "upstream/LLM-Xavier"
    files = sorted((upstream / "llm_x").glob("*.py"))
    files += [ROOT / "preprocessing/llmx_original.py", ROOT / "dataset_adapters/llmx.py"]
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in files}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    return {
        "code_revision": revision,
        "upstream_revision": read_json(ROOT / "configs/sources.json")["llm_xavier"]["commit"],
        "semantics_files_sha256": hashes,
        "semantics_sha256": digest(json_text(hashes)),
        "experiment_files_sha256": {
            str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(EXPERIMENTS.rglob("*.py"))
            if "results" not in p.relative_to(EXPERIMENTS).parts
        },
    }


def make_plan(spec, retries):
    """Use existing discovery + prompt wrapper, taking each episode's valid prefix."""
    if spec.reuse_from and not 1 <= spec.reuse_n <= spec.n:
        raise ValueError("reuse_n must be between 1 and the condition's N")
    sources = sorted(discover_episodes(DEFAULT_DATA_ROOT), key=lambda s: str(s.path))
    questions = spec.questions or QUESTIONS
    conditions, queries = [], []
    for alias in spec.models: # @Rafa: モデルの選択、aliasがモデルの略称
        model = MODELS[alias] # @Rafa: モデルの略称から正式名称を取得
        # @Rafa: task(Acrobatとか), metric(s_t,s_t+1->a_tの行動予測とか), H(timestepの長さ)の組み合わせを一つ選ぶ
        for task in spec.tasks:
            for metric in spec.metrics:
                for H in spec.histories:
                    question = questions[task][metric]
                    upstream_h = history_size(H)
                    config = EvaluationConfig(task, metric, question, upstream_h) # @Rafa:型定義
                    cid = f"{alias}__{task}__{metric}__H{H}"
                    selected = []
                    for source in (s for s in sources if s.task == task): # @Rafa: npzファイルの選択
                        for q in build_prompt_queries(source, metric=metric,
                                                      question_name=question, history_size=upstream_h):
                            actual = q.history_end - q.history_start
                            if actual != H or len(re.findall(r"^Step \d+:$", q.history_text, re.M)) != H:
                                raise ValueError("Upstream history length no longer matches requested H")
                            assert actual == H
                            row = {
                                "condition_id": cid, "ordinal": len(selected), "model_alias": alias,
                                "model": model, "task": task, "metric": metric, "question_name": question,
                                "H": H, "requested_H": H, "upstream_history_size": upstream_h,
                                "actual_history_steps": actual, "history_start": q.history_start,
                                "history_end": q.history_end, "history_end_is_exclusive": True,
                                "dataset": q.dataset, "episode": q.episode,
                                "episode_path": str(q.episode_path.relative_to(ROOT)),
                                "episode_sha256": q.dataset_sha256, "query_index": q.query_index,
                                "system_prompt_sha256": digest(q.system_prompt),
                                "user_prompt_sha256": digest(q.user_prompt),
                                "system_prompt": q.system_prompt, "user_prompt": q.user_prompt,
                            }
                            row["query_id"] = query_id(row)
                            # These are required reuse slots, not optional cache hits.
                            row["reuse_required"] = bool(spec.reuse_from and task == "Pendulum-v1"
                                                         and H == 20 and row["ordinal"] < spec.reuse_n)
                            selected.append(row)
                            if len(selected) == spec.n:
                                break
                        if len(selected) == spec.n:
                            break
                    if len(selected) != spec.n:
                        raise ValueError(f"{cid}: need {spec.n} valid queries, found {len(selected)}")
                    reused = sum(row["reuse_required"] for row in selected)
                    new = 0 if spec.preview else len(selected) - reused
                    conditions.append({
                        "condition_id": cid, "model": model, "task": task, "metric": metric, "H": H,
                        "question_name": question, "evaluation_config": asdict(config),
                        "logical_queries": len(selected), "planned_reused_queries": reused,
                        "planned_api_requests": new, "maximum_api_attempts": new * (retries + 1),
                    })
                    queries.extend(selected)
    if len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate planned query identity")
    return {
        "schema_version": 1, "experiment": spec.name, "created_at_utc": utc_now(),
        "provenance": provenance(), "selection": "path-sorted episodes; valid query prefixes",
        "sdk_max_retries": 0, "upstream_retries": retries,
        "api_request_options_by_model": {MODELS[a]: api_request_options(MODELS[a]) for a in spec.models},
        "conditions": conditions, "queries": queries,
        "planned_logical_queries": len(queries),
        "planned_api_requests": sum(c["planned_api_requests"] for c in conditions),
        "planned_reused_queries": sum(c["planned_reused_queries"] for c in conditions),
        "maximum_api_attempts": sum(c["maximum_api_attempts"] for c in conditions),
        "api_requests_made": 0,
    }


def verify_prompt(row, system, user):
    if (digest(system) != row["system_prompt_sha256"]
            or digest(user) != row["user_prompt_sha256"] or query_id(row) != row["query_id"]):
        raise SafetyStop(f"Prompt/identity mismatch: {row['query_id']}")


def check_inputs(spec, plan):
    """Validate ALL prerequisites before constructing any client. Never fallback to resend."""
    cached = {}
    source_name = spec.preview_from or spec.reuse_from
    if not source_name:
        return cached
    source = EXPERIMENTS / source_name / "results"
    previous = read_json(source / "manifest.json")
    if previous["experiment"] != source_name:
        raise ValueError("Unexpected source experiment")
    if previous["provenance"]["semantics_sha256"] != plan["provenance"]["semantics_sha256"]:
        raise ValueError("Upstream/preprocessing source changed since prerequisite experiment")
    expected = {q["query_id"]: q for q in previous["queries"]}
    if len(expected) != len(previous["queries"]):
        raise ValueError("Prerequisite manifest has duplicate identities")
    required = plan["queries"] if spec.preview_from else [q for q in plan["queries"] if q["reuse_required"]]
    if set(expected) != {q["query_id"] for q in required}:
        raise ValueError("Prerequisite query identities do not match the entire required set")
    for row in required:
        old = expected[row["query_id"]]
        verify_prompt(old, row["system_prompt"], row["user_prompt"])
        verify_prompt(row, old["system_prompt"], old["user_prompt"])
        for key in (*IDENTITY_FIELDS, "history_start", "history_end", "actual_history_steps",
                    "upstream_history_size", "ordinal"):
            if row[key] != old[key]:
                raise ValueError(f"Prerequisite mismatch in {key}")
    if spec.reuse_from:
        summary = read_json(source / "summary.json")
        if summary["status"] != "complete" or summary["successful_queries"] != len(required):
            raise ValueError("Exp.5 must be complete before Exp.6 can send anything")
        records = read_jsonl(source / "records.jsonl")
        if len(records) != len(required) or len({r["query_id"] for r in records}) != len(records):
            raise ValueError("Exp.5 records are missing or duplicated")
        for record in records:
            qid = record["query_id"]
            if qid not in expected or query_id(record) != qid or record["status"] not in SCORED_STATUSES:
                raise ValueError("Invalid Exp.5 scored record")
            raw = record["raw_response"]
            if not raw["choices"][0]["message"]["content"]:
                raise ValueError("Exp.5 response has no assistant content")
            if (raw["choices"][0]["message"]["content"] != record["assistant_text"]
                    or record["score"]["status"] != record["status"]
                    or record["score"]["prompt"] != expected[qid]["user_prompt"]
                    or record["score"]["index"] != expected[qid]["query_index"]
                    or record["source_experiment"] != spec.reuse_from
                    or not record["api_request_made"] or record["reused"]):
                raise ValueError("Exp.5 raw response/score/provenance mismatch")
            verify_prompt(record, record["request"]["messages"][0]["content"],
                          record["request"]["messages"][1]["content"])
            if record["request"]["model"] != record["model"]:
                raise ValueError("Exp.5 request model mismatch")
            # Check the actual experiment request shape before ANY Exp.6 call.
            expected_request = expected_api_request(
                record["model"], expected[qid]["system_prompt"], expected[qid]["user_prompt"])
            if record["request"] != expected_request:
                raise ValueError("Exp.5 API parameters differ from the current experiment settings")
            from openai.types.chat import ChatCompletion
            ChatCompletion.model_validate(raw)  # Local schema validation; no client/network.
            cached[qid] = {**record, "source_record": f"{source.relative_to(ROOT)}/records.jsonl#{qid}"}
        if set(cached) != set(expected):
            raise ValueError("Exp.5 reuse records do not match manifest")
    return cached


def save_manifest(directory, plan):
    write_json(directory / "manifest.json", plan)
    write_csv(directory / "manifest.csv", [
        {k: v for k, v in q.items() if k not in {"system_prompt", "user_prompt"}}
        for q in plan["queries"]
    ])
    print("model | task | metric | H | logical queries | planned API requests | reused")
    for c in plan["conditions"]:
        print(" | ".join(str(c[k]) for k in ("model", "task", "metric", "H", "logical_queries",
                                             "planned_api_requests", "planned_reused_queries")))
    print(f"planned logical queries: {plan['planned_logical_queries']}")
    print(f"planned API requests (no retries): {plan['planned_api_requests']}")
    print(f"maximum API attempts (including retries): {plan['maximum_api_attempts']}")
    print("API requests made: 0 (planning)")


def exception_info(exc):
    return redact({"type": type(exc).__name__, "message": str(exc)})


class RecordingSession:
    """One episode CLI call; queue joins upstream complete() calls to query identities."""

    def __init__(self, rows, directory, experiment, cached, *, replay_only=False):
        self.rows, self.directory, self.experiment = rows, directory, experiment
        self.cached, self.replay_only = cached, replay_only
        self.completed = []
        self.failed = []
        self.api_attempts = 0
        self.retry_attempts = 0
        self.clients = []

    # @Rafa: 元論文の実装のOpenAIChatBackendをRecordingOpenAIBackendに置き換えるための関数
    def backend_class(self):
        session = self

        class RecordingOpenAIBackend(OpenAIChatBackend):
            def __init__(self, **kwargs):
                self.position = 0
                if session.replay_only:
                    # Offline replay still passes through super().complete() and the CLI scorer.
                    self.model, self.retries = kwargs["model"], 0
                    self._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
                else:
                    super().__init__(**kwargs)
                    self._client = self._client.with_options(max_retries=0)
                    session.clients.append(self._client)
                    if str(self._client.base_url).rstrip("/") != "https://api.openai.com/v1":
                        raise SafetyStop("Only the explicit OpenAI API endpoint is allowed")

            # @Rafa: evaluation.py:evaluate_episode()の中で、backend.complete()が呼ばれるので、ここでRecordingOpenAIBackend.complete()が呼ばれる
            def complete(self, *, system_prompt, user_prompt):
                if self.position >= len(session.rows):
                    raise SafetyStop("Upstream emitted more queries than planned")
                row = session.rows[self.position]
                self.position += 1
                # @Rafa: rowが自分の実装で作ったプロンプト。system_prompt, user_promptが元論文の実装で作成されたプロンプト。verify_promptで両者が一致するかを確認する
                verify_prompt(row, system_prompt, user_prompt)
                qid = row["query_id"]
                cached = session.cached.get(qid)
                if (row["reuse_required"] or session.replay_only) and cached is None:
                    raise SafetyStop("Required replay is missing; no API fallback is allowed")
                start, clock = utc_now(), time.perf_counter()
                attempts = []
                latest = {}
                resource = self._client.chat.completions
                original_create = None if session.replay_only else resource.create

                def recording_create(**kwargs):
                    verify_prompt(row, kwargs["messages"][0]["content"], kwargs["messages"][1]["content"])
                    if kwargs["model"] != row["model"]:
                        raise SafetyStop("Actual API model differs from plan")
                    # Upstream supplies temperature=0. Override only modern-model
                    # generation options here, before journaling, replay, or sending.
                    options = api_request_options(row["model"])
                    if "reasoning_effort" in options:
                        kwargs.pop("temperature", None)
                        kwargs.update(options)
                    if kwargs != expected_api_request(row["model"], row["system_prompt"], row["user_prompt"]):
                        raise SafetyStop("Actual API parameters differ from experiment settings")
                    if cached:
                        if kwargs != cached["request"]:
                            raise SafetyStop("Replay API parameters differ from the saved request")
                        from openai.types.chat import ChatCompletion
                        latest.update(request=kwargs, raw_response=cached["raw_response"])
                        return ChatCompletion.model_validate(cached["raw_response"])
                    attempt = len(attempts) + 1
                    attempt_id = f"{qid}:{attempt}"
                    request_start, request_clock = utc_now(), time.perf_counter()
                    request = {"query_id": qid, "attempt_id": attempt_id, "attempt": attempt,
                               "request_started_at_utc": request_start, "kwargs": kwargs,
                               "api_request_made": True, "reused": False}
                    append_jsonl(session.directory / "requests.jsonl", request)
                    request_clock = time.perf_counter()  # Exclude the pre-send journal fsync from API latency.
                    session.api_attempts += 1
                    session.retry_attempts += int(attempt > 1)
                    result = {"query_id": qid, "attempt_id": attempt_id, "attempt": attempt,
                              "request_started_at_utc": request_start, "response_received_at_utc": None,
                              "api_request_made": True, "reused": False}
                    try:
                        response = original_create(**kwargs)
                    except BaseException as exc:
                        result.update(exception=exception_info(exc), raw_response=None,
                                      request_elapsed_seconds=time.perf_counter() - request_clock)
                        raise
                    else:
                        result.update(response_received_at_utc=utc_now(),
                                      request_elapsed_seconds=time.perf_counter() - request_clock)
                        # Persist the SDK object in full, then return the SAME object to upstream.
                        try:
                            raw = response.model_dump(mode="json")
                            result.update(raw_response=raw, exception=None,
                                          assistant_text=(raw.get("choices") or [{}])[0].get("message", {}).get("content"),
                                          usage=raw.get("usage"))
                            latest.update(request=kwargs, raw_response=raw)
                        except Exception as exc:
                            raise SafetyStop(f"Cannot serialize response: {type(exc).__name__}") from None
                        return response
                    finally:
                        result["attempt_finished_at_utc"] = utc_now()
                        attempts.append(result)
                        append_jsonl(session.directory / "responses.jsonl", result)

                try:
                    with patch.object(resource, "create", recording_create, create=session.replay_only):
                        text = super().complete(system_prompt=system_prompt, user_prompt=user_prompt)
                except BaseException as exc:
                    failed = {**row, "status": "failed", "exception": exception_info(exc),
                              "query_started_at_utc": start, "query_finished_at_utc": utc_now(),
                              "query_elapsed_seconds": time.perf_counter() - clock,
                              "api_attempts": len(attempts), "retry_attempts": max(0, len(attempts) - 1),
                              "api_request_made": bool(attempts), "reused": False, "attempts": attempts}
                    session.failed.append(failed)
                    append_jsonl(session.directory / "records.jsonl", failed)
                    raise
                finished = {**row, **latest, "assistant_text": text,
                            "query_started_at_utc": start, "query_finished_at_utc": utc_now(),
                            "query_elapsed_seconds": time.perf_counter() - clock,
                            "request_elapsed_seconds": sum(a["request_elapsed_seconds"] for a in attempts),
                            "api_attempts": len(attempts), "retry_attempts": max(0, len(attempts) - 1),
                            "api_request_made": bool(attempts), "reused": bool(cached),
                            "source_experiment": session.experiment, "attempts": attempts}
                if cached:
                    finished.update(source_experiment=cached["source_experiment"],
                                    source_record=cached.get("source_record"),
                                    source_query_elapsed_seconds=cached["query_elapsed_seconds"])
                    # Cross-experiment reuse must also retain inference latency for analysis.
                    # New communication remains zero via api_request_made/api_attempts.
                    finished.update(replay_elapsed_seconds=finished["query_elapsed_seconds"],
                                    query_elapsed_seconds=cached["query_elapsed_seconds"],
                                    request_elapsed_seconds=cached.get("request_elapsed_seconds"),
                                    query_time_recovered_from_request=cached.get("query_time_recovered_from_request", False))
                usage = latest["raw_response"].get("usage") or {}
                finished.update(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                                total_tokens=usage.get("total_tokens"), usage=usage)
                session.completed.append(finished)
                if cached and not session.replay_only:
                    append_jsonl(session.directory / "requests.jsonl", {
                        "query_id": qid, "attempt_id": f"{qid}:reuse", "attempt": 0,
                        "api_request_made": False, "reused": True, "source_record": finished["source_record"],
                        "kwargs": latest["request"],
                    })
                    append_jsonl(session.directory / "responses.jsonl", {
                        "query_id": qid, "attempt_id": f"{qid}:reuse", "attempt": 0,
                        "api_request_made": False, "reused": True, "source_record": finished["source_record"],
                        "raw_response": latest["raw_response"], "usage": usage, "assistant_text": text,
                    })
                return text

        return RecordingOpenAIBackend


@contextmanager
def quiet_sdk_logging():
    # Debug HTTP logs are outside our redaction boundary; suppress and restore them.
    loggers = [logging.getLogger(name) for name in ("openai", "httpx", "httpcore")]
    saved = [(logger, logger.disabled, logger.level) for logger in loggers]
    old_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        for logger in loggers:
            logger.disabled = True
        yield
    finally:
        logging.disable(old_disable)
        for logger, disabled, level in saved:
            logger.disabled, logger.level = disabled, level

# @Rafa: 元論文の実装(../upstream/LLM-Xavier)の実行を行う部分。
def invoke_cli(session, output, retries):
    """Only the CLI backend symbol is patched; evaluate_episode and scorer are untouched."""
    row = session.rows[0]
    output = output_path(output)
    if output.exists():
        raise SafetyStop("Upstream output already exists; refusing requests before overwrite checks")
    if file_sha256(ROOT / row["episode_path"]) != row["episode_sha256"]:
        raise SafetyStop("Episode changed after planning")
    argv = [
        "evaluate", "--data-path", str(ROOT / row["episode_path"]), "--task-name", row["task"],
        "--metric", row["metric"], "--question-name", row["question_name"],
        "--history-size", str(row["upstream_history_size"]), "--max-queries", str(len(session.rows)),
        "--backend", "openai", "--model", row["model"], "--api-key-env", "OPENAI_API_KEY",
        "--endpoint", "https://api.openai.com/v1", "--retries", str(retries),
        "--indexed-history", "--include-prompts", "--output-dir", str(output),
    ]
    captured = io.StringIO()
    try:
        # @Rafa: 元論文の実装のOpenAIChatBackendをRecordingOpenAIBackendに置き換えて、llmx_cli.main(argv)を実行する.
        with patch.object(llmx_cli, "OpenAIChatBackend", session.backend_class()), quiet_sdk_logging(), \
                redirect_stdout(captured), redirect_stderr(captured):
            return llmx_cli.main(argv)
    finally:
        for client in session.clients:
            try:
                client.close()
            except Exception as exc:
                captured.write("Client close error: " + json_text(exception_info(exc)) + "\n")
        with output_path(session.directory / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(redact(captured.getvalue()))


def collect_scores(session, output):
    scores = read_jsonl(output / "predictions.jsonl")
    if len(scores) != len(session.completed):
        raise SafetyStop("Upstream scores and recorded responses have different counts")
    combined = []
    for record, score in zip(session.completed, scores):
        if (score["index"] != record["query_index"] or score["prompt"] != record["user_prompt"]
                or score["history_start"] != record["history_start"]
                or score["history_end_exclusive"] != record["history_end"]):
            raise SafetyStop("Upstream score/query join mismatch")
        merged = {**record, "score": score, "status": score["status"],
                  "prediction": score.get("prediction"), "ground_truth": score.get("ground_truth"),
                  "element_accuracy": score.get("element_accuracy"),
                  "upstream_output": str(output.relative_to(ROOT))}
        append_jsonl(session.directory / "records.jsonl", merged)
        combined.append(merged)
    return combined


def summarize_scored(records, metric):
    scores = [r["score"] for r in records if r["status"] in SCORED_STATUSES]
    return _summarize(scores, metric) if scores else {}

# @Rafa: ここから元論文のコマンド実行を行う
def execute_plan(spec, plan, directory, cached, retries):
    """Fail fast. Recover completed prefix scores by OFFLINE replay after an API failure."""
    started, clock = utc_now(), time.perf_counter()
    summary = {"experiment": spec.name, "status": "running", "started_at_utc": started,
               "planned_logical_queries": len(plan["queries"]), "logical_queries": 0,
               "api_attempts": 0, "retry_attempts": 0, "successful_queries": 0,
               "failed_queries": 0, "reused_queries": 0, "conditions": []}
    for name in ("requests.jsonl", "responses.jsonl", "records.jsonl", "run.log"):
        write_text(directory / name, "", exclusive=True)
    write_json(directory / "summary.json", summary)
    try:
        for condition in plan["conditions"]:
            condition_start, condition_clock = utc_now(), time.perf_counter()
            rows = [q for q in plan["queries"] if q["condition_id"] == condition["condition_id"]]
            episode_paths = list(dict.fromkeys(q["episode_path"] for q in rows))
            condition_records = []
            csummary = {**condition, "started_at_utc": condition_start, "status": "running"}
            counters_before = {k: summary[k] for k in ("logical_queries", "api_attempts", "retry_attempts", "failed_queries")}
            summary["conditions"].append(csummary)
            try:
                for number, path in enumerate(episode_paths):
                    batch = [q for q in rows if q["episode_path"] == path]
                    session = RecordingSession(batch, directory, spec.name, cached,
                                               replay_only=all(q["query_id"] in cached for q in batch))
                    output = directory / "runs" / condition["condition_id"] / f"episode_{number:03d}"
                    error = None
                    try:
                        # @Rafa: これが元論文のコマンド実行
                        rc = invoke_cli(session, output, retries)
                        if rc != 0:
                            error = RuntimeError(f"Upstream CLI exited {rc}; see run.log")
                    except BaseException as exc:
                        error = exc
                    finally:
                        summary["api_attempts"] += session.api_attempts
                        summary["retry_attempts"] += session.retry_attempts
                        summary["logical_queries"] += len(session.completed) + len(session.failed)
                        summary["failed_queries"] += len(session.failed)
                    if error:
                        # No further paid calls. Preserve scores for responses returned before failure.
                        if session.completed and not isinstance(error, (SafetyStop, KeyboardInterrupt)):
                            replay_rows = batch[:len(session.completed)]
                            replay = RecordingSession(replay_rows, directory, spec.name,
                                                      {r["query_id"]: r for r in session.completed}, replay_only=True)
                            partial = output.parent / (output.name + "_partial")
                            if invoke_cli(replay, partial, 0) == 0:
                                condition_records.extend(collect_scores(session, partial))
                        raise error
                    if len(session.completed) != len(batch):
                        raise SafetyStop("CLI returned without completing its planned prefix")
                    condition_records.extend(collect_scores(session, output))
                csummary["status"] = "complete"
            finally:
                csummary.update(finished_at_utc=utc_now(),
                                condition_elapsed_seconds=time.perf_counter() - condition_clock,
                                scored_queries=len(condition_records),
                                metrics=summarize_scored(condition_records, condition["metric"]))
                csummary["Accuracy"] = csummary["metrics"].get("legacy_compatible_match_rate")
                csummary.update({"actual_" + k: summary[k] - value for k, value in counters_before.items()})
                csummary["successful_queries"] = len(condition_records)
                csummary["reused_queries"] = sum(r["reused"] for r in condition_records)
                if csummary["status"] != "complete":
                    csummary["status"] = "incomplete"
                summary["successful_queries"] += len(condition_records)
                summary["reused_queries"] += sum(r["reused"] for r in condition_records)
                write_json(directory / "summary.json", summary)
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(status="incomplete", exception=exception_info(exc))
        raise
    finally:
        summary.update(finished_at_utc=utc_now(), experiment_elapsed_seconds=time.perf_counter() - clock,
                       api_requests_made=summary["api_attempts"],
                       unstarted_queries=len(plan["queries"]) - summary["logical_queries"],
                       unscored_queries=summary["logical_queries"] - summary["successful_queries"] - summary["failed_queries"])
        write_json(directory / "summary.json", summary)
        write_csv(directory / "summary.csv", [
            {**{k: v for k, v in c.items() if k not in {"metrics", "evaluation_config"}}, **c.get("metrics", {})}
            for c in summary["conditions"]
        ])
        print(f"{summary['status']}: logical_queries={summary['logical_queries']}, "
              f"api_attempts={summary['api_attempts']}, retry_attempts={summary['retry_attempts']}")


def latest_results(directory):
    """Original run and immutable, ordered resume generations."""
    generations = sorted((directory / "resumes").glob("[0-9][0-9][0-9][0-9]"))
    return generations[-1] if generations else directory


def prepare_resume(plan, directory, retry_uncertain=False):
    """Recover saved responses, including responses not yet scored. Never send here."""
    sources = [directory, *sorted((directory / "resumes").glob("[0-9][0-9][0-9][0-9]"))]
    expected = {q["query_id"]: q for q in plan["queries"]}
    cached, attempted = {}, set()
    from openai.types.chat import ChatCompletion
    for source in sources:
        previous = read_json(source / "manifest.json")
        if (previous["experiment"] != plan["experiment"]
                or previous["queries"] != plan["queries"]
                or previous["provenance"]["semantics_sha256"] != plan["provenance"]["semantics_sha256"]):
            raise ValueError("Resume manifest/prompt/data/semantics mismatch")
        requests = {}
        for request in read_jsonl(source / "requests.jsonl"):
            if not request["api_request_made"]:
                continue
            qid, aid = request["query_id"], request["attempt_id"]
            if qid not in expected or aid in requests:
                raise ValueError("Invalid resume request journal")
            q = expected[qid]
            wanted = expected_api_request(q["model"], q["system_prompt"], q["user_prompt"])
            if request["kwargs"] != wanted:
                raise ValueError("Resume request parameters changed")
            requests[aid] = request
            attempted.add(qid)
        records = {r["query_id"]: r for r in read_jsonl(source / "records.jsonl")
                   if r["status"] in SCORED_STATUSES}
        seen = set()
        for response in read_jsonl(source / "responses.jsonl"):
            if not response["api_request_made"]:
                continue
            aid, qid = response["attempt_id"], response["query_id"]
            if aid not in requests or aid in seen or requests[aid]["query_id"] != qid:
                raise ValueError("Invalid resume response journal")
            seen.add(aid)
            raw = response.get("raw_response")
            if raw is None:
                continue
            parsed = ChatCompletion.model_validate(raw)
            if not parsed.choices or not parsed.choices[0].message.content:
                raise ValueError("Saved response cannot be replayed; refusing automatic resend")
            record = records.get(qid, {})
            cached[qid] = {
                "request": requests[aid]["kwargs"], "raw_response": raw,
                "source_experiment": plan["experiment"],
                "source_record": f"{source.relative_to(ROOT)}/responses.jsonl#{aid}",
                "query_elapsed_seconds": record.get("query_elapsed_seconds", response["request_elapsed_seconds"]),
                "request_elapsed_seconds": record.get("request_elapsed_seconds", response["request_elapsed_seconds"]),
                "query_time_recovered_from_request": not bool(record),
            }
        if set(records) - cached.keys():
            raise ValueError("Scored queries are missing saved responses; refusing resend")
    uncertain = attempted - cached.keys()
    plan["resume_cached_queries"] = len(cached)
    plan["resume_uncertain_queries"] = sorted(uncertain)
    for c in plan["conditions"]:
        reused = sum(q["query_id"] in cached for q in plan["queries"] if q["condition_id"] == c["condition_id"])
        c["planned_reused_queries"] = reused
        c["planned_api_requests"] = c["logical_queries"] - reused
        c["maximum_api_attempts"] = c["planned_api_requests"] * (plan["upstream_retries"] + 1)
    for key in ("planned_reused_queries", "planned_api_requests", "maximum_api_attempts"):
        plan[key] = sum(c[key] for c in plan["conditions"])
    plan["resume_sources"] = [str(s.relative_to(ROOT)) for s in sources]
    plan["retry_uncertain_confirmed"] = retry_uncertain
    return cached, uncertain, directory / "resumes" / f"{len(sources):04d}"


# @Rafa: run.pyで呼び出す、smoke testかどうかを判定して、さもなくばmkdirする
def run(spec, argv=None):
    """Serialize paid executions, including resume, to prevent duplicate sends."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    # --executeがなければAPI実行用ロックは取らず、dry-runまたはpreviewへ進む。
    if "--execute" not in arguments:
        return _run(spec, arguments)
    directory = output_path(EXPERIMENTS / spec.name / "results")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Stopped: another execution holds the results lock", file=sys.stderr)
            return 2
        return _run(spec, arguments)

# @Rafa: これが本体
def _run(spec, argv=None):
    parser = argparse.ArgumentParser(description=spec.name)
    parser.add_argument("--execute", action="store_true", help="Allow paid execution (also requires confirmation)")
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Exp.3: replay saved responses and send only missing queries")
    parser.add_argument("--retry-uncertain", action="store_true", help="Acknowledge possible duplicate billing for timed-out/interrupted requests")
    parser.add_argument("--retries", type=int, default=0,
                        help="Explicit upstream retries per query; default 0 avoids uncertain duplicate billing")
    args = parser.parse_args(argv)
    # --resume は、再開処理を実装している Exp.3 だけで許可する。
    if args.resume and spec.name not in {
        "03_gpt35_history_n30", "03_1_gpt35_history_n30_joint",
    }:
        parser.error("--resume is currently supported for the Exp.3 N30 runs only")
    # 結果不明requestの再送指定は、resumeと組み合わせた場合だけ許可する。
    if args.retry_uncertain and not args.resume:
        parser.error("--retry-uncertain requires --resume")
    # query数Nは1以上、追加retry回数は0以上でなければ停止する。
    if spec.n < 1 or args.retries < 0:
        parser.error("N must be positive and retries non-negative")
    # 有料API実行は、--executeと--confirm-paid-apiが両方ある場合だけ許可する。
    if args.execute != args.confirm_paid_api:
        parser.error("Paid execution requires BOTH --execute and --confirm-paid-api")
    # prompt確認専用のpreview実験では、有料API実行を禁止する。
    if spec.preview and args.execute:
        parser.error("Exp.1 is local preview only; paid flags are not accepted")
    directory = EXPERIMENTS / spec.name / "results"
    try:
        plan = make_plan(spec, args.retries)
        cached = {}
        resume_directory = None
        uncertain = set()
        # resume指定時は、保存済み応答・結果不明request・新しい保存先を復元する。
        if args.resume:
            cached, uncertain, resume_directory = prepare_resume(plan, directory, args.retry_uncertain)
        try:
            cached.update(check_inputs(spec, plan))
            plan["prerequisites"] = "ready"
        except (OSError, ValueError, KeyError, SafetyStop) as exc:
            plan["prerequisites"] = exception_info(exc)
            # API実行前提なら失敗理由をdry-run manifestへ残し、送信せず停止する。
            if args.execute:
                # Show the full plan and reason even though no API call is allowed.
                save_manifest(directory / "dry_run", plan)
                raise ValueError("Prerequisite check failed; see results/dry_run/manifest.json") from None

        # 通常実験で--executeがなければ、計画だけ保存するdry-runとして終了する。
        if not spec.preview and not args.execute:
            plan["mode"] = "dry_run"
            print("DRY RUN")
            save_manifest(directory / "dry_run", plan)
            print("prerequisites:", plan["prerequisites"])
            return 0

        # 結果不明requestの再送は、明示確認がなければ禁止する。
        if uncertain and not args.retry_uncertain:
            raise ValueError(f"{len(uncertain)} requests have unknown outcomes; inspect --resume dry-run, then explicitly add --retry-uncertain to resend")
        # 全queryを回収済みで最新runも完了済みなら、再送せず正常終了する。
        if args.resume and len(cached) == len(plan["queries"]) and read_json(latest_results(directory) / "summary.json")["status"] == "complete":
            print("Already complete. API requests made: 0")
            return 0
        # resume時は、元結果を上書きせず新しいresumes/XXXXへ保存する。
        if resume_directory is not None:
            directory = resume_directory
        # API実行時に認証キーがなければ、client生成前に停止する。
        if args.execute and not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY must be set in the environment")
        # Exclusive, persistent marker: reruns must not overwrite results or resend paid queries.
        output_path(directory).mkdir(parents=True, exist_ok=True)
        # 許可された補助物以外の既存結果があれば、上書き防止のため停止する。
        if any(p.name not in {".gitkeep", "dry_run", ".execution.lock"} for p in directory.iterdir()):
            raise FileExistsError("Existing results: archive them manually before a deliberate new run")

        write_json(directory / ".started.json", {"started_at_utc": utc_now()}, exclusive=True)
        plan["mode"] = "preview" if spec.preview else "execute"
        save_manifest(directory, plan)
        # preview実験ではpromptをtxtへ保存し、APIを呼ばずに終了する。
        if spec.preview:
            for row in plan["queries"]:
                write_text(directory / f"{row['condition_id']}.txt",
                           "SYSTEM MESSAGE\n================\n" + row["system_prompt"]
                           + "\n\nUSER MESSAGE\n==============\n" + row["user_prompt"] + "\n", exclusive=True)
            print("Preview complete. API requests made: 0")
            return 0

        execute_plan(spec, plan, directory, cached, args.retries)
        return 0
    except (Exception, SafetyStop, KeyboardInterrupt) as exc:
        print("Stopped:", redact(str(exc)), file=sys.stderr)
        return 2


def analysis_main(argv=None):
    # Separate domain, imported only for Exp.4; it never constructs a backend/client.
    from experiments.analysis import main
    return main(argv)
