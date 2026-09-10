"""Small hand-checkable examples. Run with pytest; never contacts an API."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("exp03_plot_metrics", Path(__file__).with_name("plot_metrics.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_state_change_three_classes_and_dimension_average():
    # Third dimension changes by less than 1e-4: unchanged, not a continuous value.
    truth = analysis.state_directions([0, 0, 0], [1, -1, 0.00001],
                                      threshold=1e-4, allow_unchanged=True)
    assert truth == [1, 0, 2]
    assert analysis.element_accuracy(truth, [1, 0, 0]) == pytest.approx(2 / 3)
    legacy = analysis.state_directions([0, 0, 0], [1, -1, 0.00001],
                                       threshold=1e-4, allow_unchanged=False)
    assert legacy == [1, 0, 1]


def test_action_matching_and_population_sd():
    rows = [{"accuracy_pct": value} for value in (100, 0, 100, 0)]
    stats = analysis.summarize_values(rows, "accuracy_pct", 4)
    assert stats == {"subset_n": 4, "N": 4, "missing_count": 0,
                     "mean": 50, "std": 50, "std_ddof": 0}


def test_missing_state_is_not_zero_filled():
    stats = analysis.summarize_values([{"accuracy_pct": 100}, {"accuracy_pct": None}], "accuracy_pct", 2)
    assert stats["N"] == 1 and stats["missing_count"] == 1 and stats["mean"] == 100
    empty = analysis.summarize_values([{"accuracy_pct": None}], "accuracy_pct", 1)
    assert empty["N"] == 0 and empty["mean"] is None and empty["std"] is None


def test_nested_subsets_are_reproducible_and_not_the_first_ten():
    rows = [{"condition_id": "condition", "ordinal": i, "query_id": f"q{i}"} for i in range(30)]
    first = analysis.select_subsets(rows, 42)
    assert first == analysis.select_subsets(list(reversed(rows)), 42)
    subsets = first["conditions"]["condition"]["subsets"]
    assert {n: len(s["query_ids"]) for n, s in subsets.items()} == {"30": 30, "20": 20, "10": 10}
    assert set(subsets["10"]["query_ids"]) < set(subsets["20"]["query_ids"]) < set(subsets["30"]["query_ids"])
    assert subsets["10"]["ordinals"] != list(range(10))


def test_official_pendulum_bin_boundaries():
    # Actual official scoring uses right=True, unlike left-closed bins in the prompt.
    from llm_x.metrics import bin_actions
    assert bin_actions([-3, -2, -1.6, -1.59, 0, 2, 3], start=-2, stop=2) == [0, 0, 0, 1, 4, 9, 9]
