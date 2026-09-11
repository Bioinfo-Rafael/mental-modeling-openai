"""Generate and validate the eight Exp.03_1 prompts without calling an API."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from experiments import common  # noqa: E402
from run import EXPERIMENT, JOINT_QUESTIONS  # noqa: E402
from llm_x.data import Episode  # noqa: E402
from llm_x.evaluation import EvaluationConfig, evaluate_episode  # noqa: E402


SMOKE_H = common.H_VALUES[0]
FILENAMES = {
    ("MountainCar-v0", "next-action"): "mountaincar_next_action.txt",
    ("MountainCar-v0", "last-action"): "mountaincar_last_action.txt",
    ("MountainCar-v0", "next-state"): "mountaincar_next_state.txt",
    ("MountainCar-v0", "last-state"): "mountaincar_last_state.txt",
    ("Pendulum-v1", "next-action"): "pendulum_next_action.txt",
    ("Pendulum-v1", "last-action"): "pendulum_last_action.txt",
    ("Pendulum-v1", "next-state"): "pendulum_next_state.txt",
    ("Pendulum-v1", "last-state"): "pendulum_last_state.txt",
}


class PromptCaptureBackend:
    """Capture the messages at backend.complete() without constructing a client."""

    name = "smoke-capture"
    model = "no-model"

    def __init__(self, response):
        self.response = response
        self.messages = []

    def complete(self, *, system_prompt, user_prompt):
        self.messages.append((system_prompt, user_prompt))
        return self.response


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prompts",
    )
    return parser.parse_args(argv)


def select_prompts(plan):
    selected = {}
    for row in plan["queries"]:
        key = (row["task"], row["metric"])
        if row["H"] == SMOKE_H and row["ordinal"] == 0 and key not in selected:
            selected[key] = row
    if set(selected) != set(FILENAMES):
        missing = sorted(set(FILENAMES) - set(selected))
        raise AssertionError(f"Missing smoke prompts: {missing}")
    return selected


def prompt_text(row, system_prompt, user_prompt):
    return (
        "--- SYSTEM MESSAGE ---\n"
        f"{system_prompt}\n\n"
        "--- USER MESSAGE ---\n"
        f"{user_prompt}\n\n"
        "--- METADATA (NOT PART OF API MESSAGES) ---\n"
        f"task_name: {row['task']}\n"
        f"metric: {row['metric']}\n"
        f"question_name: {row['question_name']}\n"
        f"history_size: {row['H']}\n"
        f"query_index: {row['query_index']}\n"
        f"episode_path: {row['episode_path']}\n"
    )


def capture_runtime_messages(row):
    episode = Episode.load(ROOT / row["episode_path"])
    if row["metric"].endswith("action"):
        response = (
            ">>Final action choice: [0]"
            if row["task"] == "MountainCar-v0"
            else "predictions = [0]\n>>Final action bins: [0]"
        )
    else:
        state_dim = episode.state_vector(row["query_index"]).size
        response = "predictions = [" + ", ".join('"INC"' for _ in range(state_dim)) + "]"
        response += "\n>>Final state values: [" + ", ".join("0" for _ in range(state_dim)) + "]"
        response += "\n>>Final state deltas: [" + ", ".join("0" for _ in range(state_dim)) + "]"

    backend = PromptCaptureBackend(response)
    config = EvaluationConfig(
        task_name=row["task"],
        metric=row["metric"],
        question_name=row["question_name"],
        history_size=row["upstream_history_size"],
        max_queries=1,
    )
    evaluate_episode(episode, config, backend)
    assert len(backend.messages) == 1
    system_prompt, user_prompt = backend.messages[0]
    assert system_prompt == row["system_prompt"]
    assert user_prompt == row["user_prompt"]
    return system_prompt, user_prompt


def validate(selected):
    for (task, metric), row in selected.items():
        prompt = row["user_prompt"]
        assert row["question_name"] == JOINT_QUESTIONS[task][metric]
        assert row["system_prompt"] and "Given the following snippet" in prompt

        if task == "MountainCar-v0" and metric.endswith("action"):
            assert ">>Final action choice: []" in prompt
            assert ">>Final action bins:" not in prompt
            assert "real value from the action range" not in prompt

        if task == "Pendulum-v1" and metric.endswith("action"):
            assert "real value from the action range" in prompt
            assert "10 discrete bins ranging from [-2, 2]" in prompt
            assert "predictions = [-1.52, 1.25]" in prompt
            assert ">>Final action bins: [1, 8]" in prompt

        if metric.endswith("state"):
            for token in ('"INC"', '"DEC"', '"UNCH"'):
                assert token in prompt
            assert "numerical value of each element" in prompt
            assert "signed change of each state element" in prompt
            assert 'predictions = ["INC", "DEC", "UNCH"]' in prompt
            assert ">>Final state values:" in prompt
            assert ">>Final state deltas:" in prompt
            index = row["query_index"]
            assert f"delta_state = s{index + 1} - s{index}" in prompt

        if metric == "next-action":
            assert "In next step" in prompt
        elif metric == "last-action":
            assert "Then the agent took an action" in prompt
            assert "next_state" not in prompt  # The rendered value, not an unresolved placeholder.
        elif metric == "next-state":
            assert "predict the next state" in prompt.lower()
        else:
            assert "Deduce the previous state" in prompt


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the production planner used immediately before execute_plan.
    # It resolves feedback classes and renders prompts, but constructs no client.
    plan = common.make_plan(EXPERIMENT, retries=0)
    selected = select_prompts(plan)
    validate(selected)
    for key, filename in FILENAMES.items():
        row = selected[key]
        system_prompt, user_prompt = capture_runtime_messages(row)
        (output_dir / filename).write_text(
            prompt_text(row, system_prompt, user_prompt), encoding="utf-8"
        )

    print(f"Smoke test passed: {len(selected)} prompts; API requests made: 0")
    print(f"Prompts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
