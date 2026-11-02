"""Small, explicit persistence helpers shared by planning and execution."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RESULTS = HERE / "results"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    with pending.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # Do not discard a torn journal: whether its request reached the API is unknown.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("Use 1–80 letters, digits, underscores or hyphens for names")
    return value


def resolve_run(value):
    path = Path(value)
    if not path.is_absolute():
        path = path if path.exists() else RESULTS / path
    path = path.resolve()
    if not path.is_relative_to(RESULTS.resolve()) or path == RESULTS.resolve():
        raise ValueError("Run directory must be inside experiments/18_desicion_LAG/results/")
    return path


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
