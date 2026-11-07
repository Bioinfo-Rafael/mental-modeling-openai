from pathlib import Path

import pytest

from mtd_rl_xavier_workspace import paths
from mtd_rl_xavier_workspace.experiment import common


# Current paths are relative to workspace/ (paths.ROOT).
@pytest.mark.parametrize(
    ("old", "current"),
    [
        ("upstream/LLM-Xavier/llm_x/task.py", "src/mentalmodeling/llm_x/task.py"),
        ("experiments/common.py", "src/mtd_rl_xavier_workspace/experiment/common.py"),
        ("experiments/common_analysis/runner.py", "src/mtd_rl_xavier_workspace/experiment/common_analysis/runner.py"),
        ("tools/show_data.py", "src/mtd_rl_xavier_workspace/tools/show_data.py"),
        ("data/llmx_data/a.npz", "data/llmx_data/a.npz"),
    ],
)
def test_relpath_mapping_round_trips(old: str, current: str) -> None:
    assert paths.current_relpath(old) == current
    assert paths.legacy_relpath(current) == old


@pytest.mark.parametrize(
    ("old", "current"),
    [
        # Experiments sit at the workspace root, so their current prefix is empty (not inverted).
        ("experiments/06_new_models_n10/results/runs/x", "06_new_models_n10/results/runs/x"),
        ("workspace/06_new_models_n10/results/runs/x", "06_new_models_n10/results/runs/x"),
        ("experiments/README.md", "EXPERIMENTS.md"),
        ("workspace/EXPERIMENTS.md", "EXPERIMENTS.md"),
        ("notebooks/01_token_estimation.py", "notebooks/01_token_estimation.py"),
        ("workspace/notebooks/01_token_estimation.py", "notebooks/01_token_estimation.py"),
        # The single-project layout before 2026-11-08 (repository-relative).
        ("src/mtd_rl_xavier/experiment/common.py", "src/mtd_rl_xavier_workspace/experiment/common.py"),
        ("src/mtd_rl_xavier/datasets/llmx.py", "src/mtd_rl_xavier_workspace/datasets/llmx.py"),
        ("src/mentalmodeling/llm_x/task.py", "src/mentalmodeling/llm_x/task.py"),
    ],
)
def test_relpath_mapping_reads_every_recorded_layout(old: str, current: str) -> None:
    assert paths.current_relpath(old) == current


def test_experiment_paths_have_no_derived_legacy_key() -> None:
    assert paths.legacy_relpaths("06_new_models_n10/results/manifest.json") == ()


def test_resolve_recorded_maps_pre_split_workspace_path_when_missing(tmp_path: Path) -> None:
    resolved = paths.resolve_recorded(tmp_path, "workspace/06_new_models_n10/results/manifest.json")

    assert resolved == tmp_path / "06_new_models_n10/results/manifest.json"


def test_resolve_recorded_from_the_repository_root_lands_in_the_workspace() -> None:
    # Experiment scripts pass the repository root; data/ and src/mentalmodeling/ now live under workspace/.
    resolved = paths.resolve_recorded(paths.REPOSITORY_ROOT, "upstream/LLM-Xavier/llm_x/task.py")

    assert resolved == paths.ROOT / "src/mentalmodeling/llm_x/task.py"
    assert resolved.is_file()
    assert paths.resolve_recorded(paths.REPOSITORY_ROOT, "data/llmx_data/a.npz") == paths.ROOT / "data/llmx_data/a.npz"


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
    # Only the longest matching current prefix is inverted.
    assert paths.legacy_relpaths("src/mtd_rl_xavier_workspace/datasets/llmx.py") == ("dataset_adapters/llmx.py",)
    assert paths.legacy_relpaths("src/mtd_rl_xavier_workspace/lag/batch.py") == ("src/mtd_rl_xavier/lag/batch.py",)
    assert paths.legacy_relpaths("data/llmx_data/a.npz") == ()


def test_resolve_recorded_maps_pre_move_path_when_missing(tmp_path: Path) -> None:
    resolved = paths.resolve_recorded(tmp_path, "mentalmodeling/llm_x/task.py")

    assert resolved == tmp_path / "src/mentalmodeling/llm_x/task.py"


def test_same_semantics_accepts_only_the_recorded_layout_moves() -> None:
    current = common.CURRENT_SEMANTICS
    olds = set(common.RESTRUCTURED_SEMANTICS)

    assert len(olds) == 3 and set(common.RESTRUCTURED_SEMANTICS.values()) == {current}
    assert common.same_semantics(current, current)
    for old in olds:
        assert common.same_semantics(old, current)
        assert not common.same_semantics(current, old)
        assert not common.same_semantics(old, "0" * 64)
    for first in olds:
        for second in olds - {first}:
            assert not common.same_semantics(first, second)
