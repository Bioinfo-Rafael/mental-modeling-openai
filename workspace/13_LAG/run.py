"""Regenerate/check the shared LAG data; the only top-level Python entry point."""
import sys

sys.dont_write_bytecode = True

from lag_data import prepare
from lag_data.dataset import PROCESSED

# The processed CSVs moved to <repo>/data/mhi/processed. prepare.py keeps its bytes because
# manifest.json records its SHA-256 under `generator`, so the destination is set here.
prepare.OUTPUT = PROCESSED
main = prepare.main


if __name__ == "__main__":
    main()
