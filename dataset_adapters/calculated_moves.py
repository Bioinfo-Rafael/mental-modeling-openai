import re
from functools import lru_cache
from .base import ROOT, Sequence
from .tables import CandidateAdapter


class CalculatedMovesAdapter(CandidateAdapter):
    def __init__(self):
        super().__init__("calculated_moves")
        for r in self.files:
            kind = self.inventory["groups"][r["schema_group"]]["schema"]["kind"]
            if kind == "delimited":
                schema = self.inventory["groups"][r["schema_group"]]["schema"]
                length, selector = r["rows"], {"delimiter": schema["delimiter"], "scope": "csv_table"}
            elif kind == "encounter_script":
                length, selector = r["encounter_count"], {"scope": "script_blocks"}
            elif kind == "winner_sequence":
                length, selector = r["logical_records"], {"scope": "winner_list"}
            elif kind == "configuration":
                length, selector = 1, {"scope": "configuration"}
            else:
                continue
            p = self.path(r)
            self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"], length, r["format"], selector))

    @lru_cache(maxsize=4)
    def text(self, path):
        return (ROOT / path).read_text()

    def raw_fields(self, sequence, index):
        scope = sequence.selector["scope"]
        if scope == "csv_table":
            return super().raw_fields(sequence, index)
        text = self.text(sequence.source_file)
        if scope == "configuration":
            fields = {"raw_text": text, **dict(re.findall(r"(\w+)=([^,\]]+)", text))}
        elif scope == "winner_list":
            fields = {"winner": text.split(",")[index].strip()}
        else:
            blocks = re.split(r"(?m)(?=^Encounter \d+)", text)
            fields = {"script_block": [b for b in blocks if b.startswith("Encounter")][index]}
        return fields, {"source_record_index": index, "scope": scope}
