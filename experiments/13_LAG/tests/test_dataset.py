"""Source coverage, event timing, missing data, and reusable selection checks."""
import csv
from importlib import import_module
import json

import pandas as pd
import pytest

prep = import_module("experiments.13_LAG.lag_data.prepare")
lag = import_module("experiments.13_LAG.lag_data.dataset")
legacy = import_module("experiments.13_LAG.data.viz.acmi_loader")
STEMS = sorted(p.stem for p in prep.RAW.glob("*.acmi"))


@pytest.mark.parametrize("stem", STEMS)
def test_every_source_field_and_acmi_line_survives(stem):
    with (prep.OUTPUT / f"{stem}.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(int(r["step"]), r["object_id"]): r for r in rows}
    assert len(keyed) == len(rows)
    with (prep.RAW / "actions" / f"{stem}_actions.csv").open(newline="") as handle:
        source = list(csv.DictReader(handle))
    for line, original in enumerate(source, 2):
        merged = keyed[int(original["step"]), original["uid"]]
        assert merged["csv_row_number"] == str(line)
        for field, text in original.items():
            assert merged[prep.CSV_RENAMES.get(field, field)] == text
    # The pre-existing visualization parser independently verifies object/event coverage.
    updates, removals, metadata = legacy.load_acmi(prep.RAW / f"{stem}.acmi")
    expected_lines = set(updates.line_number) | set(removals.line_number)
    actual = [r for r in rows if r["has_acmi_record"] == "1"]
    assert {int(r["acmi_line_number"]) for r in actual} == expected_lines
    assert len(actual) == len(expected_lines)
    lines = (prep.RAW / f"{stem}.acmi").read_text(encoding="utf-8-sig").splitlines()
    for r in actual:
        line = lines[int(r["acmi_line_number"]) - 1]
        if r["acmi_event"] == "remove":
            assert line == "-" + r["object_id"]
            assert json.loads(r["acmi_properties_json"]) == {}
            assert r["state_valid"] == "0"
        else:
            uid, *parts = line.split(",")
            props = dict(part.split("=", 1) for part in parts)
            assert uid == r["object_id"]
            assert json.loads(r["acmi_properties_json"]) == props
            if "T" in props:
                assert [r[f"acmi_{x}"] for x in prep.TRANSFORM] == props["T"].split("|")
    manifest = json.loads((prep.OUTPUT / "manifest.json").read_text())
    episode = next(e for e in manifest["episodes"] if e["episode_id"] == stem)
    assert episode["acmi_metadata"] == metadata
    assert set(keyed) == {(int(r["step"]), r["uid"]) for r in source} | {
        (round(r.time_s / 0.2), r.object_id) for r in pd.concat([updates, removals]).itertuples()
    }


@pytest.mark.parametrize("stem", STEMS)
def test_missiles_terminal_state_and_inactive_aircraft(stem):
    frame = lag.load_episode(prep.OUTPUT / f"{stem}.csv")
    aircraft = frame[frame.entity_type.eq("aircraft")]
    assert set(aircraft.object_id) == set(lag.AIRCRAFT)
    assert aircraft.loc[aircraft.team.eq("B"), list(lag.ACTIONS)].isna().all().all()
    dead = aircraft[aircraft.alive.eq(0).fillna(False)]
    assert not dead.empty
    assert dead.state_valid.eq(0).all()
    assert dead.action_status.eq("inactive").all()
    # Stale source values are kept in storage; masking happens in query construction.
    assert dead.lon_deg.notna().all()
    last_csv = int(frame.loc[frame.has_csv_record.eq(1), "step"].max())
    assert frame.step.max() == last_csv + 1
    final_states = aircraft[aircraft.step.eq(last_csv + 1) & aircraft.state_valid.eq(1)]
    assert not final_states.empty
    assert final_states.state_source.eq("acmi").all()
    assert final_states.alive.isna().all()
    assert final_states.vn_mps.isna().all()
    assert final_states.action_status.eq("state_only").all()
    missiles = frame[frame.entity_type.eq("missile")]
    launches = frame[frame.missile_launched.notna()]
    assert set(missiles.object_id) == set(launches.missile_launched)
    for launch in launches.itertuples():
        track = missiles[missiles.object_id.eq(launch.missile_launched)]
        assert track.launcher_id.eq(launch.object_id).all()
        assert track.team.eq(launch.team).all()
        assert int(track.loc[track.acmi_event.eq("update"), "step"].min()) == launch.step + 1
        assert track.acmi_event.eq("remove").sum() == 1
    assert missiles[list(lag.ACTIONS)].isna().all().all()
    assert set(frame.loc[frame.entity_type.eq("effect"), "acmi_type"]) == {"Misc+Explosion"}


def test_saved_outputs_reproducible_and_hash_checked(tmp_path):
    outputs, _ = prep.build_outputs()
    for name, data in outputs.items():
        assert (prep.OUTPUT / name).read_bytes() == data
    path = tmp_path / f"{STEMS[0]}.csv"
    path.write_bytes(outputs[path.name] + b"\n")
    (tmp_path / "manifest.json").write_bytes(outputs["manifest.json"])
    with pytest.raises(ValueError, match="differs"):
        lag.load_episode(path)


def test_exact_temporal_and_object_selection():
    frame = lag.load_episode(prep.OUTPUT / f"{STEMS[0]}.csv")
    assert lag.history_steps(100, length=20, stride=5) == list(range(0, 100, 5))
    assert lag.history_steps(100, offsets=[-1, -20, -5]) == [80, 95, 99]
    with pytest.raises(ValueError, match="before"):
        lag.history_steps(1, length=2)
    selected = lag.select_records(frame, steps=[2, 0], object_ids=["A0200", "A0100"],
                                  columns=["lon_deg", "shoot_action_id"])
    assert list(selected[["step", "object_id"]].itertuples(index=False, name=None)) == [
        (0, "A0200"), (0, "A0100"), (2, "A0200"), (2, "A0100")]
    records = lag.index_records(frame)
    assert records[(0, "A0100")]["lon_deg"] == selected.iloc[1].lon_deg
    with pytest.raises(ValueError, match="Duplicate"):
        lag.select_records(frame, steps=[0, 0], object_ids=["A0100"], columns=[])
