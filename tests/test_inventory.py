from __future__ import annotations

from pathlib import Path

from tools.inventory_llmx_data import create_inventory, inspect_file, write_inventory


def test_inspect_file_reports_real_schema(episode_path: Path) -> None:
    record = inspect_file(episode_path)
    assert record["timesteps"] == 8
    assert record["arrays"]["states"] == {"dtype": "float32", "shape": [8, 1, 2], "ndim": 3, "num_elements": 16, "hasobject": False, "python_type": "numpy.ndarray"}
    assert record["state_dimension"] == 2
    assert record["action_kind"] == "discrete"


def test_inventory_output_formats(tmp_path: Path, episode_path: Path) -> None:
    target = tmp_path / "data" / "offline_data" / "physics_data" / "raw_transitions" / "MountainCar-v0" / "episodes"
    target.mkdir(parents=True)
    moved = target / episode_path.name
    moved.write_bytes(episode_path.read_bytes())
    inventory = create_inventory(tmp_path / "data")
    json_path = tmp_path / "inventory.json"
    csv_path = tmp_path / "inventory.csv"
    write_inventory(inventory, json_path, csv_path)
    assert inventory["task_count"] == 1
    assert inventory["episode_count"] == 1
    assert inventory["total_timesteps"] == 8
    assert '"task_count": 1' in json_path.read_text()
    assert "MountainCar-v0" in csv_path.read_text()
