"""The run directory and its journals, and the helpers both packages share.

Every request line is appended and fsynced before its call is sent, so a call that may have been
billed always has a record. A run directory is created once (owner-only); only a preview directory
may be written again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]  # relative config paths resolve against the repository
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # run, judgement and message ids
DIR_MODE = 0o700  # run outputs hold model-visible scenario text
GIT_TIMEOUT_S = 10
LINE_BREAK_ESCAPES = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_sha256(body: dict[str, Any]) -> str:
    """The hash every journal records: SHA-256 of the canonical JSON of the exact request body."""
    return sha256_text(canonical_json(body))


def quoted(text: str) -> str:
    """Free text as one JSON string literal on one line (the prompt-injection envelope)."""
    literal = json.dumps(text, ensure_ascii=False)
    for char, escape in LINE_BREAK_ESCAPES.items():
        literal = literal.replace(char, escape)
    return literal


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("a JSON object repeats a key")
    return dict(pairs)


def loads_unique(text: str) -> Any:
    """``json.loads`` that rejects a repeated key (a reader could otherwise show one value and keep another)."""
    return json.loads(text, object_pairs_hook=_unique_keys)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite(value: Any) -> Any:
    """Standard JSON only: a non-finite float from a lax server becomes the string "NaN" or "Infinity"."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def append_jsonl(path: Path, *records: dict[str, Any]) -> None:
    """Append records with one fsync; earlier bytes are never rewritten. A lone surrogate is escaped."""
    lines = [json.dumps(finite(record), ensure_ascii=False, allow_nan=False) + "\n" for record in records]
    with Path(path).open("a", encoding="utf-8", errors="backslashreplace") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(finite(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def new_run_dir(root: Path, name: str, *, reuse: bool) -> Path:
    """``root/name``, owner-only. An existing directory is refused unless ``reuse`` (previews only)."""
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"run name {name!r} must match {SAFE_NAME.pattern} (a name, never a path)")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / name
    run_dir.mkdir(mode=DIR_MODE, exist_ok=reuse)
    if reuse and (run_dir / "requests.jsonl").exists():
        raise ValueError(f"{run_dir} holds a journal: a preview never writes into a run")
    return run_dir


def git_state() -> dict[str, Any]:
    """The commit and whether tracked files differ from it (None when git is unavailable)."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        command = ["git", "-C", str(REPO_ROOT), *args]
        return subprocess.run(command, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False)

    try:
        commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}
    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


class Journal:
    """``events.jsonl``, ``requests.jsonl`` and ``responses.jsonl`` of one run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.sequence = 0

    def request(self, record: dict[str, Any]) -> None:
        append_jsonl(self.run_dir / "requests.jsonl", {**record, "written_at": utc_now()})

    def response(self, record: dict[str, Any]) -> None:
        append_jsonl(self.run_dir / "responses.jsonl", {**record, "written_at": utc_now()})

    def event(self, record: dict[str, Any]) -> None:
        """One event line; ``sequence`` counts the lines written (1, 2, 3, ... without gaps)."""
        append_jsonl(self.run_dir / "events.jsonl", {"schema_version": 3, "sequence": self.sequence + 1, **record})
        self.sequence += 1
