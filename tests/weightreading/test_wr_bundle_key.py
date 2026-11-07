"""Bundle directories are keyed by alias + checkpoint sha prefix (the sandbox self-test uses a bundle)."""

from __future__ import annotations

from pathlib import Path

import pytest

from weightreading.bundle import BundleError, bundle_key, write_bundle
from weightreading.tensors import Tensor

from .wr_helpers import small_arrays

ALIAS = "net-0123abcd"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _tensors(seed: int = 0) -> tuple[Tensor, ...]:
    arrays = small_arrays(seed)
    return tuple(Tensor(k, "float32", v.shape, v.astype("<f4").tobytes()) for k, v in arrays.items())


def test_bundle_key_contains_alias_and_twelve_char_sha_prefix() -> None:
    assert bundle_key(ALIAS, SHA_A) == f"{ALIAS}-{'a' * 12}"


def test_write_bundle_with_two_checkpoints_same_alias_gives_two_dirs(tmp_path: Path) -> None:
    first = write_bundle(_tensors(0), ALIAS, SHA_A, tmp_path)
    second = write_bundle(_tensors(1), ALIAS, SHA_B, tmp_path)

    assert first.path != second.path
    assert sorted(p.name for p in tmp_path.iterdir()) == [bundle_key(ALIAS, SHA_A), bundle_key(ALIAS, SHA_B)]


def test_write_bundle_same_checkpoint_twice_is_idempotent(tmp_path: Path) -> None:
    first = write_bundle(_tensors(), ALIAS, SHA_A, tmp_path)
    second = write_bundle(_tensors(), ALIAS, SHA_A, tmp_path)

    assert first.path == second.path
    assert first.manifest_sha256 == second.manifest_sha256


def test_write_bundle_differing_content_at_same_key_refuses(tmp_path: Path) -> None:
    write_bundle(_tensors(0), ALIAS, SHA_A, tmp_path)

    with pytest.raises(BundleError, match="differs"):
        write_bundle(_tensors(1), ALIAS, SHA_A, tmp_path)


def test_write_bundle_tampered_bundle_at_same_key_refuses(tmp_path: Path) -> None:
    bundle = write_bundle(_tensors(), ALIAS, SHA_A, tmp_path)
    target = next(bundle.weights.glob("*.npy"))
    target.write_bytes(target.read_bytes()[:-4] + b"\0\0\0\0")

    with pytest.raises(BundleError, match="changed"):
        write_bundle(_tensors(), ALIAS, SHA_A, tmp_path)
