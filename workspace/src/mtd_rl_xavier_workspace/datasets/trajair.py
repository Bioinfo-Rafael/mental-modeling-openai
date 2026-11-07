from functools import lru_cache
from .base import Sequence
from .tables import CandidateAdapter

PROCESSED_COLUMNS = ["frame_number", "aircraft_id", "x_km", "y_km", "z_km", "wind_x_mps", "wind_y_mps"]


class TrajAirAdapter(CandidateAdapter):
    def __init__(self):
        super().__init__("trajair")
        # Explicit track unit for processed; raw/weather tables are accessible after tracks.
        for r in self.files:
            if "track_rows" in r:
                for ident, length in sorted(r["track_rows"].items(), key=lambda x: int(x[0])):
                    p = self.path(r)
                    self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"]+"#aircraft="+ident, length, ".txt",
                        {"delimiter": "whitespace", "header": False, "names": PROCESSED_COLUMNS, "aircraft_id": ident, "scope": "processed_track"}, sequential=True))
        for r in self.files:
            if r["format"] == ".csv":
                self._sequences.append(Sequence(len(self._sequences), self.path(r), r["relative_path"], r["rows"], ".csv", {"scope": "raw_table"}, sequential=False))

    @lru_cache(maxsize=8)
    def track_rows(self, sequence_id):
        seq = self.sequence(sequence_id)
        columns, rows = self.rows(seq.source_file, "whitespace", False, tuple(PROCESSED_COLUMNS))
        selected = [i for i, row in enumerate(rows) if row[1] == seq.selector["aircraft_id"]]
        return columns, rows, selected

    def raw_fields(self, sequence, index):
        if sequence.selector.get("scope") != "processed_track":
            return super().raw_fields(sequence, index)
        columns, rows, selected = self.track_rows(sequence.sequence_id)
        source_index = selected[index]
        return dict(zip(columns, rows[source_index])), {"source_row_index": source_index, "raw_storage_type": "whitespace text; lexical values unchanged", "selector": sequence.selector}
