"""Sliding-window mental-modeling experiments over the shared LAG dataset."""

# evaluation.py keeps its original `from . import execution` and stays byte-identical, because saved
# analysis/*/metadata.json records evaluator_sha256 = sha256(evaluation.py). The package attribute below
# makes that import resolve to the shared library module (the local copy was promoted and removed).
from mtd_rl_xavier_workspace.lag import execution  # noqa: F401
