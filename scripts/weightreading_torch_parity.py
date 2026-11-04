#!/usr/bin/env python3
"""Torch-side parity check for weightreading's restricted checkpoint reader.

Runs in the mtd-rl-lag environment (Python 3.8 + torch), NOT in mtd-rl-xavier's venv, and imports
nothing from this repository. Loads the checkpoint with ``torch.load(weights_only=True)`` and
prints ``{key: sha256(raw little-endian float32 bytes)}``; with ``--expected`` it compares against
the JSON printed by ``python -m weightreading inspect <checkpoint> --tensor-hashes``.

Exit codes: 0 identical, 1 mismatch, 2 bad arguments.
"""

import argparse
import hashlib
import json
import sys


def tensor_hashes(path):
    import torch  # only available in the mtd-rl-lag environment

    state = torch.load(path, map_location="cpu", weights_only=True)
    hashes = {}
    for key, tensor in state.items():
        array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        hashes[key] = hashlib.sha256(array.tobytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--expected", help="JSON from `python -m weightreading inspect --tensor-hashes`")
    args = parser.parse_args()
    hashes = tensor_hashes(args.checkpoint)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    if args.expected is None:
        return 0
    with open(args.expected, encoding="utf-8") as handle:
        expected = json.load(handle)
    if list(expected) != list(hashes) or expected != hashes:
        print("MISMATCH between torch and weightreading reader (keys, order or bytes)", file=sys.stderr)
        return 1
    print("parity OK: {} tensors identical".format(len(hashes)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
