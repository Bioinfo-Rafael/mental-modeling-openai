"""Regenerate/check the shared LAG data; the only top-level Python entry point."""
import sys

sys.dont_write_bytecode = True

from lag_data.prepare import main


if __name__ == "__main__":
    main()
