"""Lossless lexical delimited readers; retain all source columns, no NA imputation."""
from functools import lru_cache
from pathlib import Path
import csv
import numpy as np
from .base import ROOT, RawAdapter, inventories, sources


class CandidateAdapter(RawAdapter):
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.inventory = inventories()[1]
        self.files = [r for r in self.inventory["files"] if r["dataset"] == dataset_id]
        self._sequences = []

    def sequences(self):
        return self._sequences

    def path(self, record):
        return "data/candidate_datasets/" + self.dataset_id + "/" + record["relative_path"]

    @lru_cache(maxsize=3)
    def rows(self, source_file, delimiter=",", header=True, names=()):
        rows = []
        source_rows = []
        with (ROOT / source_file).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter) if delimiter != "whitespace" else (line.split() for line in f)
            columns = list(names)
            for raw_index, row in enumerate(reader):
                if not row:
                    continue
                if header and not columns:
                    columns = row
                    continue
                if len(row) != len(columns):
                    raise ValueError(f"row width differs: {source_file} record {raw_index}")
                rows.append(row)
                source_rows.append(len(source_rows))
        return columns, rows

    def raw_fields(self, sequence, index):
        sel = sequence.selector
        columns, rows = self.rows(sequence.source_file, sel.get("delimiter", ","), sel.get("header", True), tuple(sel.get("names", [])))
        source_index = index
        if "aircraft_id" in sel:
            idcol = columns.index("aircraft_id")
            selected = [i for i, row in enumerate(rows) if row[idcol] == sel["aircraft_id"]]
            source_index = selected[index]
        fields = dict(zip(columns, rows[source_index]))
        return fields, {"source_row_index": source_index, "raw_storage_type": "delimited text; values are original lexical strings", "selector": sel}

    def describe(self):
        n = self.dataset_id
        totals = self.inventory["datasets"].get(n, {})
        capabilities = {
            "f16capstone": (True, True, False, True, False, "B", "state/control available; next_state constructable for aligned streams; reward absent"),
            "trajair": (True, False, False, True, True, "B", "trajectory available; no action/control/reward; gap segmentation is a preprocessing choice"),
            "aircombat_wez": (True, False, False, False, True, "C", "static scenario/target values; no flight sequence/action/reward"),
            "calculated_moves": (False, False, False, False, True, "C", "encounter rule weights/scripts/fitness/outcomes available; fitness is not relabelled as timestep reward; no timestep kinematics or flight actions"),
        }
        state, action, reward, sequential, multi, compat, note = capabilities[n]
        return {"dataset_id": n, "display_name": n, "category": "candidate", "source": sources()[n],
                "format": sorted(totals.get("format_counts", {})), "number_of_files": totals.get("file_count"),
                "number_of_sequences": sum(s.sequential for s in self._sequences) if sequential else None,
                "number_of_reader_units": len(self._sequences), "number_of_records": sum(s.length for s in self._sequences),
                "record_scope": "reader-accessible units; duplicate exports/topics are not unique physical trajectories",
                "sequential": sequential, "multi_agent": multi, "state_available": state, "action_available": action, "reward_available": reward,
                "state_dimension": None, "action_dimension": None, "reward_dimension": None,
                "mental_modeling_compatibility": compat, "capabilities": note,
                "schema_doc": f"data/candidate_datasets/{n}/SCHEMA.md", "feature_selection": "not defined by adapter; all raw columns exposed",
                "id_policy": "0-based deterministic adapter order (F16: CSV/MAT/ULog; TrajAir: tracks then raw tables); sequence_key retains path and selector; static row_id is prefix-sum across all reader units, including script/config units"}
