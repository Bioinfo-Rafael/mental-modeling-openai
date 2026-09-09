#!/usr/bin/env python3
"""Plan a five-query pilot, or execute it only behind an explicit paid-API flag."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import tiktoken

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from tools.count_input_tokens import resolve_encoding
from tools.llmx_adapter import (
    DEFAULT_DATA_ROOT,
    WORKSPACE_ROOT,
    PromptQuery,
    build_prompt_queries,
    default_metric_question,
    stable_sample,
    task_sources,
)


DEFAULT_OUTPUT = WORKSPACE_ROOT / "outputs" / "pilot"


def select_queries(
    grouped_sources: dict[str, list[Any]],
    *,
    sample_size: int,
    seed: int,
    history_size: int,
    metric: str | None,
    question_name: str | None,
) -> tuple[list[PromptQuery], list[dict[str, Any]]]:
    selected: list[PromptQuery] = []
    skipped: list[dict[str, Any]] = []
    from llm_x.data import Episode

    for task, sources in sorted(grouped_sources.items()):
        first = Episode.load(sources[0].path)
        chosen_metric, chosen_question = (
            (metric, question_name)
            if metric is not None and question_name is not None
            else default_metric_question(task, first)
        )
        candidates = []
        try:
            for source in sources:
                candidates.extend(
                    build_prompt_queries(
                        source,
                        metric=chosen_metric,
                        question_name=chosen_question,
                        history_size=history_size,
                        seed=seed,
                    )
                )
            selected.extend(stable_sample(candidates, sample_size, seed))
        except ValueError as exc:
            skipped.append({"task": task, "episode": "*", "reason": str(exc)})
    return selected, skipped


def query_metadata(query: PromptQuery) -> dict[str, Any]:
    return {
        "dataset": query.dataset,
        "task": query.task,
        "episode": query.episode,
        "episode_path": str(query.episode_path),
        "dataset_sha256": query.dataset_sha256,
        "metric": query.metric,
        "question_name": query.question_name,
        "history_size": query.history_size,
        "query_index": query.query_index,
        "history_start": query.history_start,
        "history_end": query.history_end,
    }


def usage_record(
    query: PromptQuery,
    response: Any,
    encoding: Any,
    *,
    model: str,
    timestamp: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    usage = response.usage
    output_details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = getattr(output_details, "reasoning_tokens", None)
    output_text = response.output_text or ""
    return {
        **query_metadata(query),
        "model": model,
        "timestamp_utc": timestamp,
        "parameters": parameters,
        "prediction": output_text,
        "actual_input_tokens": int(usage.input_tokens),
        "output_tokens": int(usage.output_tokens),
        "visible_output_tokens": len(encoding.encode(output_text)),
        "visible_output_tokenizer_encoding": encoding.name,
        "reasoning_tokens": int(reasoning_tokens) if reasoning_tokens is not None else None,
        "total_tokens": int(usage.total_tokens),
        "raw_api_response_stored": False,
    }


def execute_queries(
    queries: list[PromptQuery],
    *,
    client: Any,
    model: str,
    encoding: Any,
    max_output_tokens: int,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> list[dict[str, Any]]:
    records = []
    parameters = {"max_output_tokens": max_output_tokens}
    for query in queries:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": query.system_prompt},
                {"role": "user", "content": query.user_prompt},
            ],
            max_output_tokens=max_output_tokens,
        )
        records.append(
            usage_record(
                query,
                response,
                encoding,
                model=model,
                timestamp=now().isoformat(),
                parameters=parameters,
            )
        )
    return records


def metric_stats(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record[field] for record in records if record.get(field) is not None]
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.pstdev(values) if values else None,
    }


def pilot_summary(
    records: list[dict[str, Any]],
    *,
    model: str,
    seed: int,
    sample_size: int,
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)

    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "number_of_queries": len(group),
            "input_tokens": metric_stats(group, "actual_input_tokens"),
            "output_tokens": metric_stats(group, "output_tokens"),
            "visible_output_tokens": metric_stats(group, "visible_output_tokens"),
            "reasoning_tokens": metric_stats(group, "reasoning_tokens"),
            "total_tokens": metric_stats(group, "total_tokens"),
        }

    return {
        "schema_version": 1,
        "model": model,
        "sample_size_per_task": sample_size,
        "seed": seed,
        "standard_deviation": "population standard deviation (ddof=0)",
        "raw_api_responses_stored": False,
        "overall": stats(records),
        "by_task": {task: stats(group) for task, group in sorted(grouped.items())},
        "skipped": skipped,
    }


def _atomic_text(path: Path, content: str) -> None:
    from tools.output_safety import derived_path
    path = derived_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _model_directory(output_root: Path, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("._") or "model"
    return output_root / safe_model


def write_dry_run(
    output_root: Path,
    model: str,
    queries: list[PromptQuery],
    skipped: list[dict[str, Any]],
    *,
    seed: int,
    sample_size: int,
) -> Path:
    path = _model_directory(output_root, model) / "dry_run_plan.json"
    body = {
        "schema_version": 1,
        "dry_run": True,
        "api_requests_made": 0,
        "model": model,
        "seed": seed,
        "sample_size_per_task": sample_size,
        "would_send": len(queries),
        "queries": [query_metadata(query) for query in queries],
        "skipped": skipped,
    }
    _atomic_text(path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def write_paid_results(
    output_root: Path,
    model: str,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    overwrite: bool,
) -> Path:
    directory = _model_directory(output_root, model)
    queries_path = directory / "queries.jsonl"
    summary_path = directory / "token_summary.json"
    if not overwrite and (queries_path.exists() or summary_path.exists()):
        raise FileExistsError(f"pilot output exists; pass --overwrite to replace it: {directory}")
    _atomic_text(queries_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    _atomic_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--task")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--metric")
    parser.add_argument("--question-name")
    parser.add_argument("--history-size", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--fallback-encoding", default="cl100k_base")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--execute-paid-api",
        action="store_true",
        help="actually send the selected prompts to the paid OpenAI Responses API",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.history_size < 0:
        raise SystemExit("error: --history-size must be non-negative")
    if (args.metric is None) != (args.question_name is None):
        raise SystemExit("error: --metric and --question-name must be supplied together")
    grouped, skipped = task_sources(
        data_root=args.data_root,
        task=args.task,
        data_path=args.data_path,
    )
    queries, selection_skips = select_queries(
        grouped,
        sample_size=args.sample_size,
        seed=args.seed,
        history_size=args.history_size,
        metric=args.metric,
        question_name=args.question_name,
    )
    skipped.extend(selection_skips)
    if not args.execute_paid_api:
        path = write_dry_run(
            args.output_root,
            args.model,
            queries,
            skipped,
            seed=args.seed,
            sample_size=args.sample_size,
        )
        print("DRY RUN")
        print(f"Would send {len(queries)} queries.")
        print("No API request was made.")
        print(f"Wrote {path}")
        return 0

    # The environment is intentionally consulted only after the explicit paid flag.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("error: OPENAI_API_KEY is required with --execute-paid-api")
    from openai import OpenAI

    encoding, fallback_used = resolve_encoding(args.model, args.fallback_encoding)
    if fallback_used:
        print(f"WARNING: visible output token count uses fallback encoding {encoding.name!r}.")
    records = execute_queries(
        queries,
        client=OpenAI(api_key=api_key),
        model=args.model,
        encoding=encoding,
        max_output_tokens=args.max_output_tokens,
    )
    summary = pilot_summary(
        records,
        model=args.model,
        seed=args.seed,
        sample_size=args.sample_size,
        skipped=skipped,
    )
    directory = write_paid_results(
        args.output_root,
        args.model,
        records,
        summary,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(records)} pilot records to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
