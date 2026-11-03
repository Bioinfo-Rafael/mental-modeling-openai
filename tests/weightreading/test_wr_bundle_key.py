"""Bundle/run directories are keyed by alias + checkpoint sha prefix; the prompt stays hash-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from mtd_rl_xavier import paths
from weightreading import cli, commands
from weightreading.bundle import BundleError, bundle_key, write_bundle
from weightreading.config import load_config
from weightreading.prompt import first_message
from weightreading.tensors import Tensor

from .wr_helpers import checkpoint_bytes, small_arrays

ALIAS = "net-0123abcd"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _tensors(seed: int = 0) -> tuple[Tensor, ...]:
    arrays = small_arrays(seed)
    return tuple(Tensor(k, "float32", v.shape, v.astype("<f4").tobytes()) for k, v in arrays.items())


def _second_checkpoint(small_files: tuple[Path, Path], tmp_path: Path) -> Path:
    other = tmp_path / "other.pt"
    other.write_bytes(checkpoint_bytes(small_arrays(seed=1)))
    return other


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


def test_prepare_two_checkpoints_same_spec_gives_two_bundles(small_files, tmp_path: Path) -> None:
    checkpoint, spec = small_files
    other = _second_checkpoint(small_files, tmp_path)
    out = tmp_path / "out"

    first = commands.prepare(checkpoint, spec, out)
    second = commands.prepare(other, spec, out)

    assert first.bundle.path != second.bundle.path
    assert len(list((out / "bundles").iterdir())) == 2


def test_first_message_is_identical_for_two_checkpoints_and_hash_free(small_files, tmp_path: Path) -> None:
    checkpoint, spec = small_files
    other = _second_checkpoint(small_files, tmp_path)
    config = load_config(paths.CONFIGS / "weightreading" / "run.yaml", paths.ROOT)
    first = commands.prepare(checkpoint, spec, tmp_path / "out")
    second = commands.prepare(other, spec, tmp_path / "out")

    messages = [
        first_message(p.spec, p.checkpoint.tensors, "shown", commands.executor_limits(config)) for p in (first, second)
    ]

    assert messages[0] == messages[1]
    for prepared in (first, second):
        assert prepared.checkpoint.sha256[:12] not in messages[0]
        assert str(prepared.bundle.path.name) not in messages[0]


def test_output_name_distinguishes_checkpoints(small_files, tmp_path: Path) -> None:
    checkpoint, spec = small_files
    other = _second_checkpoint(small_files, tmp_path)
    config = load_config(paths.CONFIGS / "weightreading" / "run.yaml", paths.ROOT)
    first = commands.prepare(checkpoint, spec, tmp_path / "out")
    second = commands.prepare(other, spec, tmp_path / "out")

    names = [commands.output_name(p, config) for p in (first, second)]

    assert names[0] != names[1]
    assert f"_{ALIAS}-{first.checkpoint.sha256[:12]}_" in names[0]


def test_run_record_stores_full_checkpoint_sha256(small_files, tmp_path: Path) -> None:
    checkpoint, spec = small_files
    config = load_config(paths.CONFIGS / "weightreading" / "run.yaml", paths.ROOT)
    prepared = commands.prepare(checkpoint, spec, tmp_path / "out")

    record = commands.run_record(prepared, config, "message")

    assert record["inputs"]["checkpoint_sha256"] == prepared.checkpoint.sha256
    assert len(record["inputs"]["checkpoint_sha256"]) == 64


def test_cli_preview_two_checkpoints_same_spec_both_succeed(small_files, tmp_path: Path) -> None:
    checkpoint, spec = small_files
    other = _second_checkpoint(small_files, tmp_path)
    out = tmp_path / "out"

    codes = [
        cli.main(["preview", "--checkpoint", str(c), "--spec", str(spec), "--output-root", str(out)])
        for c in (checkpoint, other)
    ]

    assert codes == [0, 0]
    assert len(list((out / "bundles").iterdir())) == 2
