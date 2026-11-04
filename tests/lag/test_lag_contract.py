"""Contract tests for ``mtd_rl_xavier.lag``: parity with the workspace 13/17 originals.

The workspace modules are imported here only to compute references. The library under
``src/`` never imports them (see ``test_lag_sources_do_not_import_workspace``).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from mtd_rl_xavier.lag import dataset as lag_dataset
from mtd_rl_xavier.lag import rendering
from mtd_rl_xavier.lag.dataset import (
    AIRCRAFT,
    EpisodeLoader,
    LagDataError,
    history_steps,
    index_records,
    last_complete_step,
    load_episode,
    select_records,
)
from mtd_rl_xavier.lag.rendering import (
    DEFAULT_DETECTOR_FIELDS,
    RenderError,
    render_detector_compact,
    render_lag17_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAG_SRC = REPO_ROOT / "src" / "mtd_rl_xavier" / "lag"
DIR_13 = REPO_ROOT / "workspace" / "13_LAG"
DIR_17 = REPO_ROOT / "workspace" / "17_commentary_LAG"
PROCESSED = DIR_13 / "data" / "processed"
WORKSPACE_SOURCE_DIRS = (DIR_13 / "lag_data", DIR_17 / "commentary_experiment")
WORKSPACE_MODULE_PREFIXES = ("lag_data", "commentary_experiment")

LAST_COMPLETE_STEPS = {
    "mhi_v1.3_ic100601_win": 340,
    "ours_seed100100_ic100601_loss": 293,
    "ours_seed100100_ic100606_win": 495,
}
MHI = "mhi_v1.3_ic100601_win"
MHI_CENTERS = (45, 85, 145, 245, 310)
HALF_WINDOW = 30
CONSTANT_NAMES = (
    "DT",
    "AIRCRAFT",
    "TRANSFORM",
    "VELOCITY",
    "ACTION_LABELS",
    "PROBABILITIES",
    "CSV_FIELDS",
    "CSV_RENAMES",
    "COLUMNS",
    "INTEGER_COLUMNS",
    "FLOAT_COLUMNS",
    "ACTIONS",
    "STATE_FEATURES",
    "STATE_UNITS",
)


@contextmanager
def workspace_modules() -> Iterator[SimpleNamespace]:
    """Import the workspace 13/17 modules without leaving bytecode, sys.path or sys.modules traces."""
    previous_flag = sys.dont_write_bytecode
    previous_path = list(sys.path)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(DIR_17), str(DIR_13)]
    try:
        yield SimpleNamespace(
            definitions=importlib.import_module("lag_data.definitions"),
            dataset=importlib.import_module("lag_data.dataset"),
            commentary_dataset=importlib.import_module("commentary_experiment.dataset"),
            prompts=importlib.import_module("commentary_experiment.prompts"),
        )
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_flag
        for name in [m for m in sys.modules if m.split(".")[0] in WORKSPACE_MODULE_PREFIXES]:
            del sys.modules[name]


def snapshot_workspace_sources() -> dict[str, str]:
    digests: dict[str, str] = {}
    for directory in WORKSPACE_SOURCE_DIRS:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            digests[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def episode_path(name: str) -> Path:
    return PROCESSED / f"{name}.csv"


@pytest.fixture(scope="module")
def reference() -> Iterator[SimpleNamespace]:
    with workspace_modules() as modules:
        yield modules


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {name: load_episode(episode_path(name)) for name in LAST_COMPLETE_STEPS}


@pytest.fixture(scope="module")
def reference_frames(reference: SimpleNamespace) -> dict[str, pd.DataFrame]:
    return {name: reference.dataset.load_episode(episode_path(name)) for name in LAST_COMPLETE_STEPS}


@pytest.fixture(scope="module")
def mhi(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return frames[MHI]


def reference_lag17_text(reference: SimpleNamespace, frame: pd.DataFrame, center: int) -> str:
    """Window body exactly as workspace 17 builds it for the model prompt."""
    window = reference.commentary_dataset.build_window(frame, center)
    return "\n".join(
        json.dumps(block, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for block in window["blocks"]
    )


def render_centered(frame: pd.DataFrame, center: int) -> rendering.RenderedTrajectory:
    return render_lag17_jsonl(
        frame,
        start_step=center - HALF_WINDOW,
        end_step=center + HALF_WINDOW,
        anchor_step=center,
        episode_alias="episode_01",
    )


def parsed_lines(trajectory: rendering.RenderedTrajectory) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trajectory.text.split("\n")]


# --- Constants, dtypes, attrs, helpers --------------------------------------------------------


@pytest.mark.parametrize("name", CONSTANT_NAMES)
def test_constant_matches_workspace_definition(reference: SimpleNamespace, name: str) -> None:
    assert getattr(lag_dataset, name) == getattr(reference.definitions, name)


@pytest.mark.parametrize("name", list(LAST_COMPLETE_STEPS))
def test_load_episode_matches_workspace_frame_dtypes_and_attrs(
    frames: dict[str, pd.DataFrame], reference_frames: dict[str, pd.DataFrame], name: str
) -> None:
    ours, expected = frames[name], reference_frames[name]

    pd.testing.assert_frame_equal(ours, expected, check_exact=True)
    assert list(ours.dtypes) == list(expected.dtypes)
    assert ours.attrs == expected.attrs
    assert set(ours.attrs) == {"csv_sha256", "schema_version", "dt_s"}
    assert ours.shape[1] == len(lag_dataset.COLUMNS) == 64


def test_load_episode_with_modified_csv_raises_data_error(tmp_path: Path) -> None:
    shutil.copy(PROCESSED / "manifest.json", tmp_path / "manifest.json")
    csv_copy = tmp_path / f"{MHI}.csv"
    csv_copy.write_bytes(episode_path(MHI).read_bytes().replace(b",Red,", b",Rex,", 1))

    with pytest.raises(LagDataError, match="differs from its manifest"):
        load_episode(csv_copy)

    assert load_episode(csv_copy, verify_hash=False)["color"].iloc[0] == "Rex"


def test_load_episode_with_unsupported_schema_version_raises_data_error(tmp_path: Path) -> None:
    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    (tmp_path / "manifest.json").write_text(json.dumps({**manifest, "schema_version": "9.9"}), encoding="utf-8")
    shutil.copy(episode_path(MHI), tmp_path / f"{MHI}.csv")

    with pytest.raises(LagDataError, match="schema version"):
        load_episode(tmp_path / f"{MHI}.csv")


def test_lag_data_error_is_a_value_error() -> None:
    assert issubclass(LagDataError, ValueError)
    assert issubclass(RenderError, ValueError)


def test_load_episode_satisfies_swappable_loader_protocol() -> None:
    assert isinstance(load_episode, EpisodeLoader)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 20, "stride": 5},
        {"offsets": [-20, -10, -5, -1]},
        {"length": 3},
        {"length": 4, "stride": 2},
    ],
)
def test_history_steps_matches_workspace(reference: SimpleNamespace, kwargs: dict[str, Any]) -> None:
    assert history_steps(100, **kwargs) == reference.dataset.history_steps(100, **kwargs)


@pytest.mark.parametrize(
    ("query_step", "kwargs"),
    [
        (-1, {"length": 1}),
        (100, {"offsets": [-1], "length": 2}),
        (100, {"offsets": [0]}),
        (100, {"offsets": [-1, -1]}),
        (100, {"length": 0}),
        (3, {"length": 5}),
    ],
)
def test_history_steps_with_invalid_arguments_raises_like_workspace(
    reference: SimpleNamespace, query_step: int, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        reference.dataset.history_steps(query_step, **kwargs)
    with pytest.raises(LagDataError):
        history_steps(query_step, **kwargs)


def test_select_records_and_index_records_match_workspace(
    reference: SimpleNamespace, mhi: pd.DataFrame, reference_frames: dict[str, pd.DataFrame]
) -> None:
    arguments = {"steps": [12, 10, 11], "object_ids": ["B0100", "A0100"], "columns": ["alive", "lon_deg"]}
    expected_frame = reference_frames[MHI]

    pd.testing.assert_frame_equal(
        select_records(mhi, **arguments), reference.dataset.select_records(expected_frame, **arguments)
    )
    assert index_records(mhi) == reference.dataset.index_records(expected_frame)


def test_select_records_with_unknown_column_raises_data_error(mhi: pd.DataFrame) -> None:
    with pytest.raises(LagDataError):
        select_records(mhi, steps=[1], object_ids=["A0100"], columns=["not_a_column"])


# --- last_complete_step -----------------------------------------------------------------------


@pytest.mark.parametrize(("name", "expected"), list(LAST_COMPLETE_STEPS.items()))
def test_last_complete_step_returns_last_step_with_all_aircraft_csv_records(
    frames: dict[str, pd.DataFrame], name: str, expected: int
) -> None:
    frame = frames[name]

    assert last_complete_step(frame) == expected
    assert int(frame["step"].max()) == expected + 1  # one trailing ACMI-only step


def test_last_complete_step_without_complete_step_raises_data_error(mhi: pd.DataFrame) -> None:
    only_acmi = mhi[mhi["has_csv_record"].eq(0)]

    with pytest.raises(LagDataError):
        last_complete_step(only_acmi)


# --- 17-format parity -------------------------------------------------------------------------


def parity_cases() -> list[tuple[str, int]]:
    cases = [(MHI, center) for center in MHI_CENTERS]
    for name, last in LAST_COMPLETE_STEPS.items():
        if name != MHI:
            cases += [(name, HALF_WINDOW), (name, last // 2), (name, last - HALF_WINDOW)]
    cases += [(MHI, HALF_WINDOW), (MHI, LAST_COMPLETE_STEPS[MHI] - HALF_WINDOW)]
    return cases


@pytest.mark.parametrize(("name", "center"), parity_cases())
def test_render_lag17_jsonl_bytes_equal_workspace_17_window(
    reference: SimpleNamespace, frames: dict[str, pd.DataFrame], name: str, center: int
) -> None:
    expected = reference_lag17_text(reference, frames[name], center)

    rendered = render_centered(frames[name], center)

    assert rendered.text.encode("utf-8") == expected.encode("utf-8")
    assert rendered.sha256 == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert (rendered.start_step, rendered.end_step, rendered.anchor_step) == (
        center - HALF_WINDOW,
        center + HALF_WINDOW,
        center,
    )
    assert len(rendered.text.split("\n")) == 2 * HALF_WINDOW + 1


def test_render_lag17_jsonl_text_is_embedded_verbatim_in_workspace_prompt(
    reference: SimpleNamespace, mhi: pd.DataFrame
) -> None:
    center = MHI_CENTERS[0]
    window = reference.commentary_dataset.build_window(mhi, center)
    user_message = reference.prompts.build_messages(window)[1]["content"]

    assert render_centered(mhi, center).text in user_message


def test_render_lag17_jsonl_records_metadata(mhi: pd.DataFrame) -> None:
    rendered = render_centered(mhi, 100)

    assert rendered.renderer == "lag17-jsonl"
    assert rendered.renderer_version == "1"
    assert rendered.episode_alias == "episode_01"
    assert set(AIRCRAFT) <= set(rendered.object_ids)
    assert rendered.fields == rendering.LAG17_FIELDS
    assert not rendered.text.endswith("\n")


def test_render_lag17_jsonl_is_deterministic(mhi: pd.DataFrame) -> None:
    assert render_centered(mhi, 100) == render_centered(mhi, 100)


# --- End step, clipping, errors ---------------------------------------------------------------


@pytest.mark.parametrize(("name", "last"), list(LAST_COMPLETE_STEPS.items()))
def test_render_at_acmi_only_step_raises(frames: dict[str, pd.DataFrame], name: str, last: int) -> None:
    frame = frames[name]
    incomplete = last + 1

    with pytest.raises(RenderError, match="Incomplete CSV time point"):
        render_lag17_jsonl(
            frame, start_step=incomplete - 5, end_step=incomplete, anchor_step=incomplete, episode_alias="e"
        )
    with pytest.raises(RenderError, match="Incomplete CSV time point"):
        render_detector_compact(
            frame,
            start_step=incomplete - 5,
            end_step=incomplete,
            anchor_step=incomplete,
            object_ids=("A0100",),
            fields=("alive",),
            float_precision=3,
        )


def test_render_lag17_jsonl_window_past_last_step_clips_at_last_complete_step(mhi: pd.DataFrame) -> None:
    last = LAST_COMPLETE_STEPS[MHI]

    rendered = render_lag17_jsonl(
        mhi, start_step=last - 3, end_step=last + 25, anchor_step=last - 1, episode_alias="episode_01"
    )

    steps = [line["step"] for line in parsed_lines(rendered)]
    assert steps == [last - 3, last - 2, last - 1, last]
    assert rendered.end_step == last
    assert [line["relative_step"] for line in parsed_lines(rendered)] == [-2, -1, 0, 1]


def test_render_lag17_jsonl_window_before_step_zero_clips_at_zero(mhi: pd.DataFrame) -> None:
    rendered = render_lag17_jsonl(mhi, start_step=-8, end_step=4, anchor_step=2, episode_alias="episode_01")

    assert [line["step"] for line in parsed_lines(rendered)] == [0, 1, 2, 3, 4]
    assert rendered.start_step == 0
    assert parsed_lines(rendered)[0]["relative_step"] == -2


@pytest.mark.parametrize(
    "arguments",
    [
        {"start_step": 10, "end_step": 5, "anchor_step": 7},
        {"start_step": 10, "end_step": 20, "anchor_step": 25},
        {"start_step": 10, "end_step": 20, "anchor_step": 5},
        {"start_step": -3, "end_step": 5, "anchor_step": -1},
        {"start_step": 1.0, "end_step": 5, "anchor_step": 2},
        {"start_step": True, "end_step": 5, "anchor_step": 2},
    ],
)
def test_render_lag17_jsonl_with_invalid_window_raises_render_error(
    mhi: pd.DataFrame, arguments: dict[str, Any]
) -> None:
    with pytest.raises(RenderError):
        render_lag17_jsonl(mhi, episode_alias="episode_01", **arguments)


def test_render_lag17_jsonl_with_empty_alias_raises_render_error(mhi: pd.DataFrame) -> None:
    with pytest.raises(RenderError):
        render_lag17_jsonl(mhi, start_step=0, end_step=2, anchor_step=1, episode_alias="")


def test_render_lag17_jsonl_with_missing_aircraft_row_in_window_raises(mhi: pd.DataFrame) -> None:
    broken = mhi[~(mhi["step"].eq(50) & mhi["object_id"].eq("B0200"))]

    with pytest.raises(RenderError, match="Incomplete CSV time point: 50"):
        render_lag17_jsonl(broken, start_step=48, end_step=52, anchor_step=50, episode_alias="episode_01")


# --- Post-death rows --------------------------------------------------------------------------


def first_death_step(frame: pd.DataFrame) -> int:
    downed = frame[frame["entity_type"].eq("aircraft") & frame["alive"].eq(0)]
    return int(downed["step"].min())


@pytest.mark.parametrize("name", list(LAST_COMPLETE_STEPS))
def test_render_lag17_jsonl_window_after_death_renders_alive_zero_rows(
    reference: SimpleNamespace, frames: dict[str, pd.DataFrame], name: str
) -> None:
    frame = frames[name]
    death = first_death_step(frame)
    center = min(death + 5, LAST_COMPLETE_STEPS[name] - HALF_WINDOW)

    rendered = render_centered(frame, center)

    lines = parsed_lines(rendered)
    post_death = [line for line in lines if line["step"] >= death]
    assert post_death, "window must cover steps after the first death"
    downed = [
        entity
        for line in post_death
        for entity in line["entities"]
        if entity["entity_type"] == "aircraft" and entity["alive"] == 0
    ]
    assert downed
    for line in post_death:
        aircraft_ids = {e["object_id"] for e in line["entities"] if e["entity_type"] == "aircraft"}
        assert aircraft_ids == set(AIRCRAFT)
    assert rendered.text == reference_lag17_text(reference, frame, center)


def test_render_detector_compact_window_after_death_renders_alive_zero_rows(mhi: pd.DataFrame) -> None:
    death = first_death_step(mhi)

    rendered = render_detector_compact(
        mhi,
        start_step=death,
        end_step=death + 4,
        anchor_step=death + 4,
        object_ids=AIRCRAFT,
        fields=("alive",),
        float_precision=3,
    )

    alive_values = [obj["alive"] for line in parsed_lines(rendered) for obj in line["objects"]]
    assert 0 in alive_values
    assert all(len(line["objects"]) == len(AIRCRAFT) for line in parsed_lines(rendered))


# --- Compact detector renderer ----------------------------------------------------------------


def compact(frame: pd.DataFrame, **overrides: Any) -> rendering.RenderedTrajectory:
    arguments: dict[str, Any] = {
        "start_step": 91,
        "end_step": 100,
        "anchor_step": 100,
        "object_ids": ("A0100", "B0100"),
        "fields": ("alive", "alt_m", "yaw_deg"),
        "float_precision": 2,
    }
    return render_detector_compact(frame, **{**arguments, **overrides})


def test_default_detector_settings_match_decision_d27() -> None:
    assert DEFAULT_DETECTOR_FIELDS == (
        "alive",
        "state_valid",
        "lon_deg",
        "lat_deg",
        "alt_m",
        "yaw_deg",
        "tas_mps",
        "vn_mps",
        "ve_mps",
        "vu_mps",
    )
    assert rendering.DEFAULT_DETECTOR_WINDOW_STEPS == 10


def test_render_detector_compact_selects_requested_objects_and_fields_in_order(mhi: pd.DataFrame) -> None:
    rendered = compact(mhi)

    lines = parsed_lines(rendered)
    assert [line["step"] for line in lines] == list(range(91, 101))
    assert [line["relative_step"] for line in lines] == list(range(-9, 1))
    for line in lines:
        assert [obj["object_id"] for obj in line["objects"]] == ["A0100", "B0100"]
        for obj in line["objects"]:
            assert list(obj) == ["object_id", "alive", "alt_m", "yaw_deg"]
    assert (rendered.renderer, rendered.renderer_version) == ("lag-detector-compact", "1")
    assert rendered.object_ids == ("A0100", "B0100")
    assert rendered.fields == ("alive", "alt_m", "yaw_deg")
    assert rendered.sha256 == hashlib.sha256(rendered.text.encode("utf-8")).hexdigest()


def test_render_detector_compact_object_order_follows_request(mhi: pd.DataFrame) -> None:
    rendered = compact(mhi, object_ids=("B0200", "A0200"))

    assert [obj["object_id"] for obj in parsed_lines(rendered)[0]["objects"]] == ["B0200", "A0200"]


def test_render_detector_compact_rounds_floats_and_keeps_integers(mhi: pd.DataFrame) -> None:
    row = mhi[mhi["step"].eq(100) & mhi["object_id"].eq("A0100")].iloc[0]

    rendered = compact(mhi, object_ids=("A0100",), float_precision=2)

    obj = parsed_lines(rendered)[-1]["objects"][0]
    assert obj["alt_m"] == round(float(row["alt_m"]), 2)
    assert obj["yaw_deg"] == round(float(row["yaw_deg"]), 2)
    assert obj["alive"] == int(row["alive"])
    assert isinstance(obj["alive"], int)


def test_render_detector_compact_is_shorter_than_lag17_for_same_window(mhi: pd.DataFrame) -> None:
    compact_text = compact(mhi, fields=DEFAULT_DETECTOR_FIELDS).text
    full_text = render_lag17_jsonl(mhi, start_step=91, end_step=100, anchor_step=100, episode_alias="e").text

    assert len(compact_text.encode("utf-8")) < len(full_text.encode("utf-8"))


def test_render_detector_compact_clips_at_step_zero(mhi: pd.DataFrame) -> None:
    rendered = compact(mhi, start_step=-6, end_step=3, anchor_step=3)

    assert [line["step"] for line in parsed_lines(rendered)] == [0, 1, 2, 3]
    assert rendered.start_step == 0
    assert parsed_lines(rendered)[0]["relative_step"] == -3


def test_render_detector_compact_clips_at_last_complete_step(mhi: pd.DataFrame) -> None:
    last = LAST_COMPLETE_STEPS[MHI]

    rendered = compact(mhi, start_step=last - 2, end_step=last + 9, anchor_step=last - 2)

    assert [line["step"] for line in parsed_lines(rendered)] == [last - 2, last - 1, last]
    assert rendered.end_step == last


def test_render_detector_compact_masks_state_features_when_state_is_invalid(mhi: pd.DataFrame) -> None:
    modified = mhi.copy()
    mask = modified["step"].eq(100) & modified["object_id"].eq("A0100")
    modified.loc[mask, "state_valid"] = 0

    rendered = compact(modified, fields=("alive", "state_valid", "alt_m"))

    masked, intact = parsed_lines(rendered)[-1]["objects"]
    assert masked["state_valid"] == 0
    assert masked["alt_m"] is None
    assert masked["alive"] == 1  # non-state fields are never masked
    assert intact["alt_m"] is not None


def test_render_detector_compact_omits_object_with_no_row_at_step(mhi: pd.DataFrame) -> None:
    missiles = mhi[mhi["entity_type"].eq("missile")]
    missile_id = str(missiles["object_id"].iloc[0])
    first_step = int(missiles[missiles["object_id"].eq(missile_id)]["step"].min())

    rendered = compact(
        mhi,
        start_step=first_step - 1,
        end_step=first_step,
        anchor_step=first_step,
        object_ids=("A0100", missile_id),
        fields=("alive",),
    )

    before, at_start = parsed_lines(rendered)
    assert [obj["object_id"] for obj in before["objects"]] == ["A0100"]
    assert [obj["object_id"] for obj in at_start["objects"]] == ["A0100", missile_id]


@pytest.mark.parametrize(
    "overrides",
    [
        {"fields": ("not_a_column",)},
        {"fields": ()},
        {"fields": ("alive", "alive")},
        {"fields": ("object_id",)},
        {"fields": ("step",)},
        {"object_ids": ()},
        {"object_ids": ("Z9999",)},
        {"object_ids": ("A0100", "A0100")},
        {"float_precision": -1},
        {"float_precision": 2.5},
        {"start_step": 101, "end_step": 105, "anchor_step": 100},
    ],
)
def test_render_detector_compact_with_invalid_arguments_raises_render_error(
    mhi: pd.DataFrame, overrides: dict[str, Any]
) -> None:
    with pytest.raises(RenderError):
        compact(mhi, **overrides)


def test_render_detector_compact_is_deterministic(mhi: pd.DataFrame) -> None:
    assert compact(mhi) == compact(mhi)


# --- Isolation from the workspace -------------------------------------------------------------


def imported_module_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("path", sorted(LAG_SRC.glob("*.py")), ids=lambda p: p.name)
def test_lag_sources_do_not_import_workspace(path: Path) -> None:
    forbidden = ("workspace", *WORKSPACE_MODULE_PREFIXES)

    offenders = [n for n in imported_module_names(path) if n.split(".")[0] in forbidden]

    assert offenders == []


def test_workspace_sources_are_unchanged_by_reference_import_and_rendering() -> None:
    before = snapshot_workspace_sources()

    with workspace_modules() as modules:
        frame = modules.dataset.load_episode(episode_path(MHI))
        modules.commentary_dataset.build_window(frame, 45)
    render_centered(load_episode(episode_path(MHI)), 45)

    assert snapshot_workspace_sources() == before
