#!/usr/bin/env python3
"""Inspect every official LLM-Xavier NPZ without loading pickled objects."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from dataset_adapters.llmx import DEFAULT_DATA_ROOT, WORKSPACE_ROOT, discover_episodes


DEFAULT_JSON = WORKSPACE_ROOT / "outputs" / "dataset_inventory.json"
DEFAULT_CSV = WORKSPACE_ROOT / "outputs" / "dataset_inventory.csv"


def array_spec(value: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "python_type": "numpy.ndarray", "ndim": value.ndim,
            "num_elements": int(value.size), "hasobject": bool(value.dtype.hasobject)}


def inspect_file(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as container:
        members = [{"name": v.filename, "compressed_bytes": v.compress_size, "uncompressed_npy_bytes": v.file_size, "compression_method": v.compress_type} for v in container.infolist()]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"states", "actions", "rewards", "episodic_return"}
    missing = sorted(required - arrays.keys())
    length = int(arrays["states"].shape[0]) if "states" in arrays and arrays["states"].ndim else None
    action = arrays.get("actions")
    action_flat = action.reshape(action.shape[0], -1) if action is not None and action.ndim else None
    discrete = bool(
        action_flat is not None
        and action_flat.shape[1] == 1
        and np.all(np.isfinite(action_flat))
        and np.all(action_flat == action_flat.astype(np.int64))
    )
    return {
        "filename": path.name,
        "relative_path": str(path.relative_to(DEFAULT_DATA_ROOT.resolve()))
        if path.is_relative_to(DEFAULT_DATA_ROOT.resolve())
        else str(path),
        "format": ".npz",
        "raw_size_bytes": path.stat().st_size,
        "archive_members": members,
        "ndarray_payload_bytes": sum(int(v.nbytes) for v in arrays.values()),
        "timesteps": length,
        "npz_keys": sorted(arrays),
        "missing_required_keys": missing,
        "arrays": {name: array_spec(value) for name, value in sorted(arrays.items())},
        "state_dimension": int(np.prod(arrays["states"].shape[1:])) if "states" in arrays else None,
        "action_kind": "discrete" if discrete else "continuous",
        "action_dimension": int(np.prod(action.shape[1:])) if action is not None else None,
        "other_keys": sorted(arrays.keys() - required),
        "reward_dimension": int(np.prod(arrays["rewards"].shape[1:])) if "rewards" in arrays else None,
        "key_presence": {key: key in arrays for key in sorted(required | {"achieved_goals", "desired_goals", "agent_dirs", "dir_vectors"})},
        "example_step_0": {key: value[0].tolist() for key, value in arrays.items() if value.ndim and len(value)},
    }


def _unique_specs(files: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    encoded = {
        json.dumps(file["arrays"][key], sort_keys=True)
        for file in files
        if key in file["arrays"]
    }
    return [json.loads(item) for item in sorted(encoded)]


def create_inventory(data_root: Path) -> dict[str, Any]:
    sources = discover_episodes(data_root)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        record = inspect_file(source.path)
        record["relative_path"] = str(source.path.relative_to(data_root.resolve()))
        grouped[(source.dataset, source.task)].append(record)

    tasks = []
    for (dataset, task), files in sorted(grouped.items()):
        lengths = [file["timesteps"] for file in files if file["timesteps"] is not None]
        keys = sorted({key for file in files for key in file["npz_keys"]})
        schema_groups = defaultdict(list)
        for file in files:
            normalized = {key: {"dtype": spec["dtype"], "shape": (["T"] + spec["shape"][1:] if key != "episodic_return" and spec["shape"] else spec["shape"])} for key, spec in file["arrays"].items()}
            fingerprint = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()[:16]
            file["schema_group"] = fingerprint
            schema_groups[fingerprint].append(file["filename"])
        tasks.append(
            {
                "dataset": dataset,
                "task": task,
                "file_format": ".npz",
                "number_of_episodes": len(files),
                "total_timesteps": sum(lengths),
                "timesteps_per_episode": {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": statistics.fmean(lengths),
                },
                "npz_keys": keys,
                "states": {
                    "specs": _unique_specs(files, "states"),
                    "dimensions": sorted({file["state_dimension"] for file in files}),
                },
                "actions": {
                    "specs": _unique_specs(files, "actions"),
                    "kinds": sorted({file["action_kind"] for file in files}),
                    "dimensions": sorted({file["action_dimension"] for file in files}),
                },
                "rewards": {"specs": _unique_specs(files, "rewards")},
                "episodic_return": {"specs": _unique_specs(files, "episodic_return")},
                "other_keys": sorted({key for file in files for key in file["other_keys"]}),
                "raw_episode_files_size_bytes": sum(file["raw_size_bytes"] for file in files),
                "files": files,
                "schema_groups": dict(schema_groups),
                "schema_consistent_except_T": len(schema_groups) == 1,
            }
        )

    all_files = [path for path in data_root.rglob("*") if path.is_file() and ".cache" not in path.parts]
    return {
        "schema_version": 2,
        "all_file_count": len(all_files),
        "all_files": [{"relative_path": str(p.relative_to(data_root)), "format": p.suffix or "no_extension", "size_bytes": p.stat().st_size, "inspection": "numeric NPZ" if p.suffix == ".npz" else "metadata only; policy/checkpoint/log not deserialized"} for p in sorted(all_files)],
        "data_root": str(data_root.resolve()),
        "task_count": len(tasks),
        "episode_count": sum(task["number_of_episodes"] for task in tasks),
        "total_timesteps": sum(task["total_timesteps"] for task in tasks),
        "dataset_disk_size_bytes": sum(path.stat().st_size for path in all_files),
        "npz_container_bytes": sum(f["raw_size_bytes"] for task in tasks for f in task["files"]),
        "ndarray_payload_bytes": sum(f["ndarray_payload_bytes"] for task in tasks for f in task["files"]),
        "unique_schema_group_count": len({k for task in tasks for k in task["schema_groups"]}),
        "tasks": tasks,
    }


def _atomic_text(path: Path, content: str) -> None:
    from tools.output_safety import derived_path
    path = derived_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_inventory(inventory: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    _atomic_text(json_path, json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    fields = [
        "dataset", "task", "number_of_episodes", "total_timesteps", "timesteps_min",
        "timesteps_max", "timesteps_mean", "npz_keys", "state_specs", "state_dimensions",
        "action_specs", "action_kinds", "action_dimensions", "reward_specs",
        "episodic_return_specs", "other_keys", "raw_episode_files_size_bytes",
    ]
    rows = []
    for task in inventory["tasks"]:
        rows.append(
            {
                "dataset": task["dataset"],
                "task": task["task"],
                "number_of_episodes": task["number_of_episodes"],
                "total_timesteps": task["total_timesteps"],
                "timesteps_min": task["timesteps_per_episode"]["min"],
                "timesteps_max": task["timesteps_per_episode"]["max"],
                "timesteps_mean": task["timesteps_per_episode"]["mean"],
                "npz_keys": json.dumps(task["npz_keys"]),
                "state_specs": json.dumps(task["states"]["specs"]),
                "state_dimensions": json.dumps(task["states"]["dimensions"]),
                "action_specs": json.dumps(task["actions"]["specs"]),
                "action_kinds": json.dumps(task["actions"]["kinds"]),
                "action_dimensions": json.dumps(task["actions"]["dimensions"]),
                "reward_specs": json.dumps(task["rewards"]["specs"]),
                "episodic_return_specs": json.dumps(task["episodic_return"]["specs"]),
                "other_keys": json.dumps(task["other_keys"]),
                "raw_episode_files_size_bytes": task["raw_episode_files_size_bytes"],
            }
        )
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(csv_path, buffer.getvalue())


def write_detailed_inventory(inventory: dict[str, Any], output: Path) -> None:
    write_inventory(inventory, output / "llmx_schema.json", output / "llmx_tasks.csv")
    import io
    buffer = io.StringIO()
    fields = ["task", "filename", "relative_path", "format", "size_bytes", "timesteps", "schema_group", "key", "python_type", "dtype", "shape", "ndim", "num_elements", "hasobject"]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for task in inventory["tasks"]:
        for file in task["files"]:
            for key, spec in file["arrays"].items():
                writer.writerow({"task": task["task"], "filename": file["filename"], "relative_path": file["relative_path"], "format": file["format"], "size_bytes": file["raw_size_bytes"], "timesteps": file["timesteps"], "schema_group": file["schema_group"], "key": key, **spec, "shape": json.dumps(spec["shape"])})
    _atomic_text(output / "llmx_files.csv", buffer.getvalue())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=WORKSPACE_ROOT / "outputs/dataset_inventory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = create_inventory(args.data_root.expanduser().resolve())
    write_inventory(inventory, args.json_output, args.csv_output)
    write_detailed_inventory(inventory, args.output_dir)
    print(
        f"Inspected {inventory['task_count']} tasks, {inventory['episode_count']} episodes, "
        f"{inventory['total_timesteps']} timesteps."
    )
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
