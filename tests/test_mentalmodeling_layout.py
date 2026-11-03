"""The vendored LLM-Xavier code lives in src/mentalmodeling/ as its own distribution.

Moving it must not change what is imported, what is protected, or whether results
saved under the old layout still pass the provenance checks. Committed results are
only read here, never written.
"""

import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from mtd_rl_xavier import paths
from mtd_rl_xavier.experiment import common
from mtd_rl_xavier.experiment import split_execution as split
from mtd_rl_xavier.tools.check_inspection_notebook import GUARD
from mtd_rl_xavier.tools.output_safety import derived_path

FORBIDDEN_IMPORT = re.compile(r"^\s*\"?(?:from|import)\s+(?:src\.)?mentalmodeling\b", re.MULTILINE)
PRE_MOVE_SEMANTICS = "830a5841e1bde42d45272d461f477ba2ea3280094d100437ca0d9864c9a4345b"
EXP06 = "06_new_models_n10"
EXP06_RESULTS = paths.WORKSPACE / EXP06 / "results"
EXP06_BATCHES = sorted((EXP06_RESULTS / "batches").glob("*/manifest.json"))


@pytest.fixture(scope="module")
def exp06():
    """Exp.06 spec and its full plan as recorded, re-stamped with today's provenance.

    Built from the committed merged manifest instead of `make_plan`, so the check runs
    without the raw episodes under data/.
    """
    spec = import_module(f"workspace.{EXP06}.run").EXPERIMENT
    (merged,) = (EXP06_RESULTS / "merged").glob("*/manifest.json")
    return spec, {**common.read_json(merged), "provenance": common.provenance()}


def with_semantics(plan: dict, semantics: str) -> dict:
    return {**plan, "provenance": {**plan["provenance"], "semantics_sha256": semantics}}


def test_mentalmodeling_constant_points_at_vendored_distribution() -> None:
    assert paths.MENTALMODELING == paths.ROOT / "src" / "mentalmodeling"
    assert 'name = "llm-x-evaluator"' in (paths.MENTALMODELING / "pyproject.toml").read_text()
    assert (paths.MENTALMODELING / "llm_x/evaluation.py").is_file()


def test_llm_x_resolves_to_vendored_source(tmp_path: Path) -> None:
    probe = "import llm_x; print(llm_x.__file__)"

    result = subprocess.run(
        [sys.executable, "-I", "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    assert Path(result.stdout.strip()) == paths.MENTALMODELING / "llm_x/__init__.py"


def test_mentalmodeling_is_not_an_importable_name(tmp_path: Path) -> None:
    """The editable install must expose `llm_x`, not `src/` as a whole: a `mentalmodeling`
    import name would be a second module identity for the vendored code."""
    probe = (
        "import importlib.util as u; "
        "print(u.find_spec('mentalmodeling')); "
        "print(u.find_spec('llm_x') is not None)"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", probe], cwd=tmp_path, capture_output=True, text=True, check=True
    )

    assert result.stdout.split() == ["None", "True"]


def test_no_code_imports_vendored_code_by_directory_name() -> None:
    """Defence in depth next to the test above: nothing may even spell the second name."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.py", "*.ipynb"], cwd=paths.ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    vendored = paths.MENTALMODELING.relative_to(paths.ROOT).as_posix() + "/"
    offenders = [
        relative
        for relative in tracked
        if not relative.startswith(vendored)
        and FORBIDDEN_IMPORT.search((paths.ROOT / relative).read_text(encoding="utf-8", errors="replace"))
    ]

    assert offenders == []


def test_setuptools_package_list_covers_every_package_directory() -> None:
    """The list is explicit (see pyproject.toml), so a new subpackage must be added by hand."""
    tomllib = pytest.importorskip("tomllib")
    declared = set(tomllib.loads((paths.ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["packages"])
    src = paths.ROOT / "src"
    on_disk = {
        ".".join(init.parent.relative_to(src).parts)
        for top in ("mtd_rl_xavier", "weightreading")
        for init in (src / top).rglob("__init__.py")
    }

    assert declared == on_disk


@pytest.mark.parametrize("relative", ["", "derived.json", "llm_x/derived.json", "new_dir/out.csv"])
def test_derived_path_rejects_writes_under_mentalmodeling(relative: str) -> None:
    with pytest.raises(ValueError, match="read-only source"):
        derived_path(paths.MENTALMODELING / relative)


def test_derived_path_rejects_symlink_into_mentalmodeling(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(paths.MENTALMODELING)

    with pytest.raises(ValueError, match="read-only source"):
        derived_path(tmp_path / "link/new.json")


def run_guarded(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", GUARD + code], cwd=cwd, capture_output=True, text=True)


def test_notebook_guard_rejects_write_under_mentalmodeling(tmp_path: Path) -> None:
    # The directory does not exist, so even a failing guard cannot create a file there.
    target = paths.MENTALMODELING / "guard_probe_missing_dir" / "out.json"

    result = run_guarded(f"open({str(target)!r}, 'w')", cwd=tmp_path)

    assert result.returncode != 0
    assert "Protected raw write attempted" in result.stderr
    assert not target.parent.exists()


def test_notebook_guard_allows_derived_write(tmp_path: Path) -> None:
    target = tmp_path / "out.json"

    result = run_guarded(f"open({str(target)!r}, 'w').write('{{}}')", cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert target.read_text() == "{}"


def test_current_semantics_alias_target_is_the_live_hash() -> None:
    assert common.provenance()["semantics_sha256"] == common.CURRENT_SEMANTICS


def test_pre_move_semantics_hash_differs_only_in_path_keys() -> None:
    current = common.provenance()["semantics_files_sha256"]
    prefix = paths.MENTALMODELING.relative_to(paths.ROOT).as_posix() + "/"
    pre_move = {key.replace(prefix, "mentalmodeling/", 1): value for key, value in current.items()}

    assert common.digest(common.json_text(pre_move)) == PRE_MOVE_SEMANTICS
    assert common.same_semantics(PRE_MOVE_SEMANTICS, common.CURRENT_SEMANTICS)


def test_committed_prerequisite_saved_before_move_is_reusable(exp06) -> None:
    spec, plan = exp06
    saved = common.read_json(common.EXPERIMENTS / spec.reuse_from / "results/manifest.json")
    assert saved["provenance"]["semantics_sha256"] != plan["provenance"]["semantics_sha256"]

    assert len(common.check_inputs(spec, plan)) == 120
    with pytest.raises(ValueError, match="changed"):
        common.check_inputs(spec, with_semantics(plan, "0" * 64))


@pytest.mark.parametrize("manifest", EXP06_BATCHES, ids=lambda path: path.parent.name)
def test_committed_batch_saved_before_move_passes_split_validation(exp06, manifest: Path) -> None:
    _, plan = exp06
    saved = common.read_json(manifest)
    assert saved["provenance"]["semantics_sha256"] != plan["provenance"]["semantics_sha256"]

    assert split.validate_plan(saved, plan)
    with pytest.raises(ValueError, match="semantics changed"):
        split.validate_plan(saved, with_semantics(plan, "0" * 64))


def test_committed_batches_exist() -> None:
    assert EXP06_BATCHES
