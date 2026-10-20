from functools import lru_cache
from scipy.io import loadmat
from .base import ROOT, Sequence
from .tables import CandidateAdapter


class F16CapstoneAdapter(CandidateAdapter):
    def __init__(self):
        super().__init__("f16capstone")
        self.unreadable_variables = []
        for r in sorted(self.files, key=lambda r: ({".csv": 0, ".mat": 1, ".ulg": 2}.get(r["format"], 3), r["relative_path"])):
            p = self.path(r)
            if r["format"] == ".csv":
                self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"], r["rows"], ".csv", sequential=True))
            elif r["format"] == ".mat":
                for name, spec in r["variables"].items():
                    if spec["opaque"] or spec["hasobject"]:
                        self.unreadable_variables.append({"source_file": p, "variable": name, "schema": spec, "reason": "opaque MATLAB object, internal rows unknown"})
                        continue
                    self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"]+"#"+name, spec["shape"][0], ".mat", {"variable": name}, sequential=True))
            elif r["format"] == ".ulg":
                for t in r["streams"]:
                    sel = {"topic": t["name"], "multi_id": t["multi_id"]}
                    self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"]+f"#{t['name']}/{t['multi_id']}", t["rows"], ".ulg", sel, sequential=True))

    @lru_cache(maxsize=2)
    def mat(self, source_file, variable):
        return loadmat(ROOT / source_file, variable_names=[variable])[variable]

    @lru_cache(maxsize=1)
    def ulog(self, source_file):
        from pyulog import ULog
        return ULog(str(ROOT / source_file))

    def raw_fields(self, sequence, index):
        sel = sequence.selector
        if sequence.format == ".mat":
            array = self.mat(sequence.source_file, sel["variable"])
            return {sel["variable"]: array[index]}, {"variable_shape": list(array.shape), "selector": sel}
        if sequence.format == ".ulg":
            topic = self.ulog(sequence.source_file).get_dataset(sel["topic"], sel["multi_id"])
            return {k: v[index] for k, v in topic.data.items()}, {"selector": sel}
        return super().raw_fields(sequence, index)

    def describe(self):
        return {**super().describe(), "unreadable_variables": self.unreadable_variables,
                "reader_scope": "all 3 CSV, numeric MAT arrays, every ULog topic; opaque MATLAB metadata is listed, XLSX/model/media remain inventory-only"}
