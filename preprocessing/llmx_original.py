"""Original LLM-X preprocessing: calls upstream prompt semantics unchanged."""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = WORKSPACE_ROOT / "upstream" / "LLM-Xavier"
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "llmx_data"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from llm_x.data import Episode  # noqa: E402
from llm_x.evaluation import (  # noqa: E402
    EvaluationConfig,
    _argue_inputs,
    _history_range,
    _query_indices,
)
from llm_x.prompting import system_prompt, user_prompt  # noqa: E402
from llm_x.questions import (  # noqa: E402
    available_questions,
    render_question,
    resolve_question,
    resolve_task,
)
from llm_x.task import ALL_CLS  # noqa: E402


from dataset_adapters.llmx import EpisodeSource, discover_episodes, source_for_path


@dataclass(frozen=True)
class PromptQuery:
    dataset: str
    task: str
    episode: str
    episode_path: Path
    dataset_sha256: str
    metric: str
    question_name: str
    history_size: int
    query_index: int
    history_start: int
    history_end: int
    system_prompt: str
    user_prompt: str
    history_text: str = ""
    question_text: str = ""


REGISTERED_TASKS = frozenset(cls.task_name for cls in ALL_CLS)


def is_discrete_episode(episode: Episode) -> bool:
    actions = episode.arrays["actions"]
    if actions.shape[0] == 0:
        return False
    values = actions.reshape(actions.shape[0], -1)
    return values.shape[1] == 1 and bool((values == values.astype(int)).all())


def default_metric_question(task: str, episode: Episode) -> tuple[str, str]:
    """Choose one conservative action-prediction variant already in upstream."""

    if is_discrete_episode(episode):
        return "next-action", "next_action_prediction"
    if "Fetch" in task:
        return "next-action", "next_action_prediction_continuous_bins_fetch"
    if task in {"InvertedPendulum-v4", "InvertedDoublePendulum-v4"}:
        return "next-action", "next_action_prediction_continuous_bins_mjpen"
    return "next-action", "next_action_prediction_continuous_bins"


def validate_combination(metric: str, question_name: str, episode: Episode) -> None:
    resolve_question(metric, question_name)
    discrete = is_discrete_episode(episode)
    continuous_variant = "continuous" in question_name
    if metric in {"next-action", "last-action"}:
        if discrete and continuous_variant:
            raise ValueError(f"{question_name} is a continuous-action question for discrete data")
        if not discrete and not continuous_variant:
            raise ValueError(f"{question_name} is a discrete-action question for continuous data")
    if metric == "argue-action" and not discrete:
        raise ValueError("argue-action requires scalar discrete actions")


def build_prompt_queries(
    source: EpisodeSource,
    *,
    metric: str,
    question_name: str,
    history_size: int,
    seed: int = 0,
) -> Iterator[PromptQuery]:
    """Yield prompts using the exact upstream Episode/query/prompt functions."""

    episode = Episode.load(source.path)
    task = resolve_task(source.task)
    question = resolve_question(metric, question_name)
    validate_combination(metric, question_name, episode)
    scene = system_prompt(task)
    drop_last = "Fetch" in source.task
    indices = _query_indices(metric, history_size, episode.length)
    argue = _argue_inputs(episode, indices, seed) if metric == "argue-action" else {}
    for index in indices:
        presented_action = argue.get(index, (None, None))[0]
        question_text = render_question(
            metric,
            question,
            episode,
            index=index,
            drop_last_feature=drop_last,
            presented_action=presented_action,
        )
        history_start, history_end = _history_range(metric, index, history_size)
        prompt = user_prompt(
            episode,
            history_start=history_start,
            history_end=history_end,
            indexed_history=True,
            drop_last_state_feature=drop_last,
            question=question_text,
        )
        yield PromptQuery(
            dataset=source.dataset,
            task=source.task,
            episode=source.episode,
            episode_path=source.path,
            dataset_sha256=episode.sha256,
            metric=metric,
            question_name=question_name,
            history_size=history_size,
            query_index=index,
            history_start=history_start,
            history_end=history_end,
            system_prompt=scene,
            user_prompt=prompt,
            history_text=episode.history_text(history_start, history_end, indexed=True, drop_last_state_feature=drop_last),
            question_text=question_text,
        )


def stable_sample(items: Iterable[PromptQuery], size: int, seed: int) -> list[PromptQuery]:
    values = list(items)
    if size < 1:
        raise ValueError("sample size must be positive")
    if len(values) < size:
        raise ValueError(f"requested {size} queries, but only {len(values)} are valid")
    # The sorted input plus Python's documented Random instance makes selection repeatable.
    rng = random.Random(seed)
    return [values[index] for index in sorted(rng.sample(range(len(values)), size))]


def task_sources(
    *,
    data_root: Path,
    task: str | None,
    data_path: Path | None,
) -> tuple[dict[str, list[EpisodeSource]], list[dict[str, Any]]]:
    if data_path is not None:
        source = source_for_path(data_path, data_root)
        source = EpisodeSource(source.path, source.dataset, task or source.task, source.episode)
        return {source.task: [source]}, []

    grouped: dict[str, list[EpisodeSource]] = {}
    skipped_counts: dict[str, int] = {}
    for source in discover_episodes(data_root):
        if task and source.task != task:
            continue
        if source.task not in REGISTERED_TASKS:
            skipped_counts[source.task] = skipped_counts.get(source.task, 0) + 1
            continue
        grouped.setdefault(source.task, []).append(source)
    if task and not grouped:
        if task not in REGISTERED_TASKS:
            raise ValueError(f"task {task!r} is not registered by upstream LLM-Xavier")
        raise FileNotFoundError(f"no episodes found for task {task!r} under {data_root}")
    skipped = [
        {
            "task": task_name,
            "episode_count": count,
            "reason": "task is absent from the upstream LLM-Xavier task registry",
        }
        for task_name, count in sorted(skipped_counts.items())
    ]
    return grouped, skipped


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def all_question_names() -> dict[str, list[str]]:
    return available_questions()
