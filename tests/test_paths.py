from pathlib import Path

import pytest

from mtd_rl_xavier import paths
from mtd_rl_xavier.experiment import common


@pytest.mark.parametrize(
    ("old", "current"),
    [
        ("upstream/LLM-Xavier/llm_x/task.py", "src/mentalmodeling/llm_x/task.py"),
        ("experiments/06_new_models_n10/results/runs/x", "workspace/06_new_models_n10/results/runs/x"),
        ("experiments/common.py", "src/mtd_rl_xavier/experiment/common.py"),
        ("experiments/common_analysis/runner.py", "src/mtd_rl_xavier/experiment/common_analysis/runner.py"),
        ("experiments/README.md", "workspace/EXPERIMENTS.md"),
        ("notebooks/01_token_estimation.py", "workspace/notebooks/01_token_estimation.py"),
        ("tools/show_data.py", "src/mtd_rl_xavier/tools/show_data.py"),
        ("data/llmx_data/a.npz", "data/llmx_data/a.npz"),
    ],
)
def test_relpath_mapping_round_trips(old: str, current: str) -> None:
    assert paths.current_relpath(old) == current
    assert paths.legacy_relpath(current) == old


def test_resolve_recorded_prefers_existing_path(tmp_path: Path) -> None:
    recorded = tmp_path / "experiments/01_x/results/a.json"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("{}")

    assert paths.resolve_recorded(tmp_path, "experiments/01_x/results/a.json") == recorded


def test_resolve_recorded_maps_old_layout_when_missing(tmp_path: Path) -> None:
    resolved = paths.resolve_recorded(tmp_path, "upstream/LLM-Xavier/llm_x/task.py")

    assert resolved == tmp_path / "src/mentalmodeling/llm_x/task.py"


def test_pre_move_mentalmodeling_paths_map_to_src() -> None:
    assert paths.current_relpath("mentalmodeling/llm_x/task.py") == "src/mentalmodeling/llm_x/task.py"
    assert paths.legacy_relpaths("src/mentalmodeling/llm_x/task.py") == (
        "upstream/LLM-Xavier/llm_x/task.py",
        "mentalmodeling/llm_x/task.py",
    )
    assert paths.legacy_relpaths("data/llmx_data/a.npz") == ()


def test_resolve_recorded_maps_pre_move_path_when_missing(tmp_path: Path) -> None:
    resolved = paths.resolve_recorded(tmp_path, "mentalmodeling/llm_x/task.py")

    assert resolved == tmp_path / "src/mentalmodeling/llm_x/task.py"


def test_same_semantics_accepts_only_the_recorded_layout_moves() -> None:
    current = common.CURRENT_SEMANTICS
    olds = set(common.RESTRUCTURED_SEMANTICS)

    assert len(olds) == 2 and set(common.RESTRUCTURED_SEMANTICS.values()) == {current}
    assert common.same_semantics(current, current)
    for old in olds:
        assert common.same_semantics(old, current)
        assert not common.same_semantics(current, old)
        assert not common.same_semantics(old, "0" * 64)
    first, second = sorted(olds)
    assert not common.same_semantics(first, second)
    assert not common.same_semantics(second, first)
