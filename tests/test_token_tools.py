from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.count_input_tokens import count_query, token_statistics
from tools.llmx_adapter import EpisodeSource, PromptQuery, build_prompt_queries, stable_sample
from tools.pilot_output_tokens import (
    execute_queries,
    main as pilot_main,
    pilot_summary,
    write_paid_results,
)
from tools.summarize_token_costs import estimate_costs


class CharacterEncoding:
    name = "fixture-character"

    def encode(self, text: str) -> list[str]:
        return list(text)


def make_query(index: int) -> PromptQuery:
    return PromptQuery(
        dataset="physics_data",
        task="MountainCar-v0",
        episode="episode.npz",
        episode_path=Path("episode.npz"),
        dataset_sha256="a" * 64,
        metric="next-action",
        question_name="next_action_prediction",
        history_size=1,
        query_index=index,
        history_start=max(0, index - 2),
        history_end=index,
        system_prompt="system",
        user_prompt="user",
    )


def test_input_token_statistics_are_population_statistics() -> None:
    encoding = CharacterEncoding()
    records = [
        count_query(
            make_query(i), encoding, model="fixture", encoding_name=encoding.name,
            fallback_used=False, tokens_per_message=0, priming_tokens=value - 10,
        )
        for i, value in enumerate([10, 20, 30])
    ]
    stats = token_statistics(records)
    assert stats["mean"] == 20
    assert stats["std"] == pytest.approx(8.1649658093)
    assert stats["median"] == 20
    assert stats["p95"] == 30
    assert stats["total_input_tokens"] == 60


def test_prompt_generation_reuses_upstream(episode_path: Path) -> None:
    source = EpisodeSource(episode_path, "physics_data", "MountainCar-v0", episode_path.name)
    query = next(
        build_prompt_queries(
            source,
            metric="next-action",
            question_name="next_action_prediction",
            history_size=1,
        )
    )
    assert query.query_index == 2
    assert query.history_start == 0
    assert query.history_end == 2
    assert "Step 0:" in query.user_prompt
    assert "MountainCar-v0" in query.system_prompt


def test_sampling_seed_is_repeatable() -> None:
    queries = [make_query(index) for index in range(20)]
    first = [q.query_index for q in stable_sample(queries, 5, 0)]
    second = [q.query_index for q in stable_sample(queries, 5, 0)]
    assert first == second
    assert first != list(range(5))


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=101,
            output_tokens=23,
            total_tokens=124,
            output_tokens_details=SimpleNamespace(reasoning_tokens=7),
        )
        return SimpleNamespace(output_text="ok", usage=usage)


def test_mock_pilot_usage_and_output_format(tmp_path: Path) -> None:
    responses = FakeResponses()
    records = execute_queries(
        [make_query(4), make_query(9)],
        client=SimpleNamespace(responses=responses),
        model="fixture-model",
        encoding=CharacterEncoding(),
        max_output_tokens=64,
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert len(responses.calls) == 2
    assert records[0]["actual_input_tokens"] == 101
    assert records[0]["reasoning_tokens"] == 7
    assert records[0]["visible_output_tokens"] == 2
    summary = pilot_summary(records, model="fixture-model", seed=0, sample_size=2, skipped=[])
    assert summary["overall"]["output_tokens"]["mean"] == 23
    directory = write_paid_results(tmp_path, "fixture-model", records, summary, overwrite=False)
    lines = (directory / "queries.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["raw_api_response_stored"] is False
    assert (directory / "token_summary.json").is_file()


def test_default_pilot_is_dry_run(episode_path: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = pilot_main(
        [
            "--task", "MountainCar-v0", "--data-path", str(episode_path),
            "--history-size", "0", "--model", "gpt-4o", "--sample-size", "5",
            "--output-root", str(tmp_path / "pilot"),
        ]
    )
    assert result == 0
    plan = json.loads((tmp_path / "pilot" / "gpt-4o" / "dry_run_plan.json").read_text())
    assert plan["api_requests_made"] == 0
    assert plan["would_send"] == 5


def test_cost_estimate_uses_explicit_prices() -> None:
    input_summary = {"overall": {"number_of_queries": 100, "total_input_tokens": 2_000_000}}
    pilot = {"overall": {"output_tokens": {"mean": 50}}}
    result = estimate_costs(
        input_summary,
        pilot,
        input_price_per_million=3.0,
        output_price_per_million=12.0,
    )
    assert result["estimated_input_cost_usd"] == 6.0
    assert result["projected_output_tokens"] == 5000
    assert result["estimated_total_cost_usd"] == pytest.approx(6.06)
