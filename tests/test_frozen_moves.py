"""The request-path files moved into ``src/`` changed only by the declared module rename map.

``tests/fixtures/assess-request-path.sha256`` freezes the current bytes of the files that build the
paid assess request; ``tests/fixtures/frozen-moves.json`` ties each of them to its pin from before
the move (``d8b2d12``). Reversing the rename map on the moved file must give exactly the pinned
bytes, so the move changed nothing but module names. A file that later changes on purpose leaves
this fixture in that commit and is guarded by the request goldens instead.
"""

from __future__ import annotations

import hashlib
import json

import pytest

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


def test_manifest_freezes_exactly_the_moved_files() -> None:
    frozen = {line.split()[1] for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()}

    assert frozen == set(fixture()["files"])
