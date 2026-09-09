#!/usr/bin/env python3
"""Count LLM-Xavier prompt tokens locally; this module never calls an API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import tiktoken

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from tools.llmx_adapter import (
    DEFAULT_DATA_ROOT,
    WORKSPACE_ROOT,
    PromptQuery,
    build_prompt_queries,
    default_metric_question,
    task_sources,
)


DEFAULT_OUTPUT = WORKSPACE_ROOT / "outputs" / "token_counts"


def resolve_encoding(model: str, fallback: str = "cl100k_base") -> tuple[Any, bool]:
    try:
        return tiktoken.encoding_for_model(model), False
    except KeyError:
        return tiktoken.get_encoding(fallback), True


def count_query(
    query: PromptQuery,
    encoding: Any,
    *,
    model: str,
    encoding_name: str,
    fallback_used: bool,
    tokens_per_message: int = 3,
    priming_tokens: int = 3,
) -> dict[str, Any]:
    system_tokens = len(encoding.encode(query.system_prompt))
    user_tokens = len(encoding.encode(query.user_prompt))
    content_tokens = system_tokens + user_tokens
    estimated = content_tokens + 2 * tokens_per_message + priming_tokens
    return {
        "dataset": query.dataset,
        "episode": query.episode,
        "episode_path": str(query.episode_path),
        "task": query.task,
        "metric": query.metric,
        "question_name": query.question_name,
        "history_size": query.history_size,
        "query_index": query.query_index,
        "history_start": query.history_start,
        "history_end": query.history_end,
        "system_tokens": system_tokens,
        "user_tokens": user_tokens,
        "content_only_tokens": content_tokens,
        "estimated_total_input_tokens": estimated,
        "model": model,
        "tokenizer_encoding": encoding_name,
        "fallback_encoding_used": fallback_used,
        "dataset_sha256": query.dataset_sha256,
    }


def token_statistics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [record["estimated_total_input_tokens"] for record in records]
    if not values:
        return {
            "number_of_queries": 0, "mean": None, "std": None, "min": None,
            "max": None, "median": None, "p95": None, "total_input_tokens": 0,
        }
    ordered = sorted(values)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "number_of_queries": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "p95": ordered[rank],
        "total_input_tokens": sum(values),
    }


def summarize(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return [{key: name, **token_statistics(group)} for name, group in sorted(groups.items())]


def _atomic_text(path: Path, content: str) -> None:
    from tools.output_safety import derived_path
    path = derived_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import io

    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_text(path, buffer.getvalue())


def write_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    model: str,
    encoding_name: str,
    fallback_used: bool,
) -> None:
    by_dataset = summarize(records, "dataset")
    by_task = summarize(records, "task")
    _write_csv(output_dir / "queries.csv", records)
    _write_csv(output_dir / "summary_by_dataset.csv", by_dataset)
    _write_csv(output_dir / "summary_by_task.csv", by_task)
    summary = {
        "schema_version": 1,
        "model": model,
        "tokenizer_encoding": encoding_name,
        "fallback_encoding_used": fallback_used,
        "chat_framing_note": (
            "Estimated as content tokens + two message headers + assistant priming. "
            "Actual server counts can differ and should be compared with pilot usage."
        ),
        "overall": token_statistics(records),
        "by_dataset": by_dataset,
        "by_task": by_task,
        "skipped": skipped,
    }
    _atomic_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")


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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fallback-encoding", default="cl100k_base")
    parser.add_argument("--tokens-per-message", type=int, default=3)
    parser.add_argument("--priming-tokens", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def legacy_main(argv: Sequence[str] | None = None) -> int:
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
    encoding, fallback_used = resolve_encoding(args.model, args.fallback_encoding)
    if fallback_used:
        print(
            f"WARNING: token count uses fallback encoding {encoding.name!r}; "
            f"model {args.model!r} is unknown to tiktoken."
        )
    records: list[dict[str, Any]] = []
    for task, sources in sorted(grouped.items()):
        first_episode = __import__("llm_x.data", fromlist=["Episode"]).Episode.load(sources[0].path)
        metric, question = (
            (args.metric, args.question_name)
            if args.metric is not None
            else default_metric_question(task, first_episode)
        )
        try:
            for source in sources:
                for query in build_prompt_queries(
                    source,
                    metric=metric,
                    question_name=question,
                    history_size=args.history_size,
                    seed=args.seed,
                ):
                    records.append(
                        count_query(
                            query,
                            encoding,
                            model=args.model,
                            encoding_name=encoding.name,
                            fallback_used=fallback_used,
                            tokens_per_message=args.tokens_per_message,
                            priming_tokens=args.priming_tokens,
                        )
                    )
        except ValueError as exc:
            if not args.all:
                raise
            skipped.append({"task": task, "episode": "*", "reason": str(exc)})
    write_outputs(
        args.output_dir,
        records,
        skipped,
        model=args.model,
        encoding_name=encoding.name,
        fallback_used=fallback_used,
    )
    print(f"Counted {len(records)} prompts locally. No API request was made.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from tools.token_inventory import main as inventory_main
    return inventory_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
