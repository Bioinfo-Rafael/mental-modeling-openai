"""The paid assess request path: moved files changed only by the rename map, the rest pinned as is.

``tests/fixtures/assess-request-path.sha256`` freezes the current bytes of the files that build the
paid assess request; ``tests/fixtures/frozen-moves.json`` ties each moved, unchanged file to its pin
from before the move (``d8b2d12``): reversing the rename map on it must give exactly the pinned
bytes. A file that later changes on purpose leaves ``files`` in that commit and is listed in
``pinned_without_move`` with the reason; the manifest pins its current bytes, and the request
goldens and ``scripts/check_assess_resume.py`` prove the request did not change. Since the judge
prompts moved into config files, the request text lives in the legacy prompt and config files.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from translate import paths

FIXTURE = paths.ROOT / "tests" / "fixtures" / "frozen-moves.json"
MANIFEST = paths.ROOT / "tests" / "fixtures" / "assess-request-path.sha256"


def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def reverse_renames(text: str, rename_map: list[list[str]]) -> str:
    """Undo the map: inverse replacements, last rule first."""
    for old, new in reversed(rename_map):
        text = text.replace(new, old)
    return text


@pytest.mark.parametrize("path", sorted(fixture()["files"]))
def test_moved_file_equals_its_pin_modulo_renames(path: str) -> None:
    pin = fixture()["files"][path]
    text = (paths.ROOT / path).read_text(encoding="utf-8")

    restored = reverse_renames(text, fixture()["rename_map"])

    assert hashlib.sha256(restored.encode("utf-8")).hexdigest() == pin["pinned_sha256"]


def _manifest() -> set[str]:
    return {line.split()[1] for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()}


def test_manifest_freezes_exactly_the_moved_and_the_pinned_files() -> None:
    moved, pinned = set(fixture()["files"]), set(fixture()["pinned_without_move"])

    assert not moved & pinned
    assert _manifest() == moved | pinned
    assert all(isinstance(reason, str) and reason.strip() for reason in fixture()["pinned_without_move"].values())


def test_manifest_pins_the_legacy_judge_config_and_the_prompt_it_names() -> None:
    legacy = "configs/evaluation/assess-legacy.yaml"
    prompt = yaml.safe_load((paths.ROOT / legacy).read_text(encoding="utf-8"))["prompt"]

    assert {legacy, prompt} <= _manifest()
