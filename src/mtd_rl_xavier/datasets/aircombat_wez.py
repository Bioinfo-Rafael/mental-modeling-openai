from .base import Sequence
from .tables import CandidateAdapter


class AirCombatWEZAdapter(CandidateAdapter):
    def __init__(self):
        super().__init__("aircombat_wez")
        for r in self.files:
            if r["format"] == ".csv":
                p = self.path(r)
                self._sequences.append(Sequence(len(self._sequences), p, r["relative_path"], r["rows"], ".csv"))
