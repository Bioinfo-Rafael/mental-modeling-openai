"""Offline layout and model/color contracts for PNG-only comparison figures."""
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pytest

from experiments.common_analysis import comparison_plots as plots
from experiments.common_analysis import joint_data as data
from experiments.common_analysis.joint_aggregate import aggregate_all


def test_model_colors_and_missing_values_break_lines():
    rows = [{"model_alias": alias, "H": h, "value": None if h == 10 else 50,
             "std": 2, "valid_n": 0 if h == 10 else 9, "selected_n": 10}
            for alias in plots.MODEL_ORDER for h in data.HISTORIES]
    fig, ax = plt.subplots()
    plots.draw_models(ax, rows, band=True)
    assert [line.get_color() for line in ax.lines] == [plots.MODEL_COLORS[a] for a in plots.MODEL_ORDER]
    assert all(np.isnan(line.get_ydata()[1]) for line in ax.lines)
    assert sum(text.get_text() == "N/A" for text in ax.texts) == 4
    with pytest.raises(ValueError, match="exactly four H"):
        plots.draw_models(ax, rows + rows[:1])
    plt.close(fig)


def test_all_65_layouts_separate_dimensions_and_preserve_model_colors(tmp_path, monkeypatch):
    _, _, records, _ = data.load_and_parse()
    subsets = data.select_subsets(records, 42, sizes=(10,))
    original = aggregate_all(records, subsets, repetitions=10, sizes=(10,))
    tables = {}
    for name, rows in original.items():
        tables[name] = [{**copy.deepcopy(row), "model_alias": alias, "model": alias}
                        for alias in plots.MODEL_ORDER for row in rows]
    captured = {}
    def save(fig, root, folder, name, inventory, **kwargs):
        relative = f"{folder}/{name}.png"
        captured[relative] = len(fig.axes)
        for ax in fig.axes:
            if ax.lines:
                assert len(ax.lines) == 4  # Series are ONLY models, never dimensions.
                assert [line.get_color() for line in ax.lines] == [plots.MODEL_COLORS[a] for a in plots.MODEL_ORDER]
        if folder == "cm":
            token = name.split("_")[1]
            alias = "3.5" if token == "35" else token
            np.testing.assert_allclose(fig.axes[0].images[0].cmap(1.0)[:3], to_rgb(plots.MODEL_COLORS[alias]))
        inventory.append(relative)
        plt.close(fig)
    monkeypatch.setattr(plots, "save_png", save)
    figures = plots.make_comparison_figures(tables, tmp_path)
    assert len(figures) == 65 and len(set(figures)) == 65
    assert sum(name.startswith("cm/") for name in figures) == 24
    assert sum("state_dimensions/" in name for name in figures) == 30
    assert "cm/01_35_mountaincar_last_action.png" in figures
    assert "cm/01_luna_mountaincar_last_action.png" in figures
    assert captured["cross_task_10/state_dimensions/state_delta_position_nrmse.png"] == 2
    assert captured["cross_task_10/accuracy.png"] == 8
    assert captured["cross_task_10/state_absolute_macro_nrmse.png"] == 4


def test_save_is_png_only_and_does_not_overwrite(tmp_path):
    fig, _ = plt.subplots(figsize=(8, 7))
    inventory = []
    plots.save_png(fig, tmp_path, "figures", "test", inventory)
    assert inventory == ["figures/test.png"]
    assert not list(tmp_path.rglob("*.svg"))
    fig, _ = plt.subplots()
    with pytest.raises(FileExistsError):
        plots.save_png(fig, tmp_path, "figures", "test", inventory)
    plt.close(fig)
