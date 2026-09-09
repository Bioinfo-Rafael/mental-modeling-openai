"""SHA256 manifest of raw files and legacy data docs, never writes under data/."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    result = {}
    for p in sorted((ROOT / "data").rglob("*")):
        if not p.is_file() or any(x in p.parts for x in [".git", ".cache", "__pycache__"]):
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        result[str(p.relative_to(ROOT))] = {"size": p.stat().st_size, "sha256": h.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = ROOT / "outputs/dataset_inventory/raw_integrity_before.json"
    current = manifest()
    if args.verify:
        old = json.loads(output.read_text())
        changes = [p for p in old.keys() | current.keys() if old.get(p) != current.get(p)]
        report = {"files": len(current), "changed": changes, "raw_unchanged": not changes}
        (output.parent / "raw_integrity_verification.json").write_text(json.dumps(report, indent=2))
        print(report)
        if changes:
            raise SystemExit(1)
    else:
        if output.exists():
            raise SystemExit("baseline exists; never silently replace integrity baseline")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(current, sort_keys=True))
        print(f"Recorded {len(current)} raw file hashes")


if __name__ == "__main__":
    main()
