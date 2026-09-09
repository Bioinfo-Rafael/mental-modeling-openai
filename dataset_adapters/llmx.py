"""Official NPZ reader; preserves all arrays and shapes, including T=1 episodes."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from .base import ROOT, RawAdapter, Sequence, inventories, sources

WORKSPACE_ROOT = ROOT
DEFAULT_DATA_ROOT = ROOT / "data/llmx_data"
UPSTREAM_ROOT = ROOT / "upstream/LLM-Xavier"


@dataclass(frozen=True)
class EpisodeSource:
    path: Path
    dataset: str
    task: str
    episode: str


def discover_episodes(data_root=DEFAULT_DATA_ROOT):
    base = Path(data_root).expanduser().resolve()
    result = []
    for path in sorted((base / "offline_data").glob("**/episodes/*.npz")):
        parts = path.relative_to(base).parts
        if "raw_transitions" in parts:
            result.append(EpisodeSource(path, parts[1], parts[parts.index("raw_transitions")+1], path.name))
    return result


def source_for_path(path, data_root=DEFAULT_DATA_ROOT):
    path = Path(path).resolve()
    return next((s for s in discover_episodes(data_root) if s.path == path), EpisodeSource(path, "custom", path.parent.parent.name, path.name))


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class LLMXAdapter(RawAdapter):
    def __init__(self, dataset_id, data_root=DEFAULT_DATA_ROOT):
        self.dataset_id = dataset_id
        self.data_root = Path(data_root).resolve()
        self.inventory = next((t for t in inventories()[0]["tasks"] if t["task"] == dataset_id), None)
        self._sources = [s for s in discover_episodes(self.data_root) if s.task == dataset_id]
        self._sequences = []
        for i, s in enumerate(self._sources):
            record = next((f for f in (self.inventory or {}).get("files", []) if f["relative_path"] == str(s.path.relative_to(self.data_root))), None)
            length = record["timesteps"] if record else len(read_arrays(s.path)["states"])
            source = str(s.path.relative_to(ROOT)) if s.path.is_relative_to(ROOT) else str(s.path)
            self._sequences.append(Sequence(i, source, str(s.path.relative_to(self.data_root)), length, ".npz", episode_id=i, sequential=True))

    def sequences(self):
        return self._sequences

    @lru_cache(maxsize=2)
    def arrays(self, sequence_id):
        return read_arrays(self._sources[sequence_id].path)

    def raw_fields(self, sequence, index):
        arrays = self.arrays(sequence.sequence_id)
        fields = {k: (v if k == "episodic_return" else v[index]) for k, v in arrays.items()}
        return fields, {"field_scope": {k: "episode" if k == "episodic_return" else "timestep" for k in fields}}

    def describe(self):
        t = self.inventory or {}
        arrays = t.get("files", [{}])[0].get("arrays", {})
        return {"dataset_id": self.dataset_id, "display_name": self.dataset_id, "category": "official",
                "source": sources()["llmx_data"], "format": ["NPZ"], "number_of_files": len(self._sequences),
                "number_of_sequences": len(self._sequences), "number_of_records": sum(s.length for s in self._sequences),
                "sequential": True, "multi_agent": False, "state_available": True, "action_available": True, "reward_available": True,
                "state_dimension": t.get("states", {}).get("dimensions"), "action_dimension": t.get("actions", {}).get("dimensions"), "reward_dimension": [1],
                "raw_arrays_example": arrays, "mental_modeling_compatibility": "A (raw contract; prompt compatibility separately reported)",
                "schema_doc": "data/README.md", "id_policy": "0-based lexicographic source-path order; episode_id=sequence_id; stable sequence_key=source path"}
