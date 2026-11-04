"""Offline tests of the npz + ACMI -> 13_LAG CSV converter on small synthetic arrays."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from intent_compare_convert import (
    build_manifest,
    convert,
    launch_steps,
    parse_acmi_missiles,
    softmax_heads,
)

from mtd_rl_xavier.lag.dataset import COLUMNS, MANIFEST_NAME, load_episode
from mtd_rl_xavier.lag.rendering import render_lag17_jsonl
from mtd_rl_xavier.translator.pipeline import episode_facts

UIDS = ("A0100", "A0200", "B0100", "B0200")
LEARNERS = ("A0100", "A0200")
HEAD_SIZES = (3, 5, 3, 2)
ROWS = 4
B0200 = 3
DEATH_ROW = 2
ACMI = """﻿FileType=text/acmi/tacview
FileVersion=2.1
0,ReferenceTime=2020-04-01T00:00:00Z
#0.00
A0100,T=120.0|60.0|6000.0|0.0|0.0|10.0,Name=F16,Color=Red,TAS=243.8,Health=100,Missiles=2
#0.20
A0100,T=120.1|60.0|6000.0|0.0|0.0|10.0,Name=F16,Color=Red,TAS=243.8,Health=100,Missiles=1
A01002,T=120.1|60.0|5999.5|0.0|-1.5|370.25,Name=AIM-9L,Color=Red
#0.40
A01002,T=120.2|60.1|5990.125|0.0|-1.0|371.0,Name=AIM-9L,Color=Red
-B0200
B0200X,T=120.3|60.2|5000.0|1.0|2.0|3.0,Type=Misc+Explosion,Name=SHOTDOWN,Color=Blue,Radius=300
#0.60
-A01002
A01002F,T=120.3|60.1|5980.0|0.0|-1.0|371.0,Type=Misc+Explosion,Color=Red,Radius=300
#0.80
A0100,T=120.5|60.0|6000.0|0.0|0.0|10.0,Name=F16,Color=Red,TAS=243.8,Health=100,Missiles=1
"""


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    alive = np.ones((ROWS, len(UIDS)), dtype=bool)
    alive[DEATH_ROW:, B0200] = False
    rpy = np.zeros((ROWS, len(UIDS), 3))
    rpy[:, :, 2] = -0.5  # negative yaw must wrap into [0, 360)
    actions = np.zeros((ROWS, len(LEARNERS), len(HEAD_SIZES)), dtype=np.int64)
    actions[:, :, 1] = 3
    return {
        "step": np.arange(ROWS, dtype=np.int64),
        "alive": alive,
        "geodetic": np.tile(np.array([120.0, 60.0, 6000.0]), (ROWS, len(UIDS), 1)) + rng.normal(0, 1e-3, (ROWS, 4, 3)),
        "rpy": rpy,
        "velocity_neu": np.tile(np.array([3.0, 4.0, 0.0]), (ROWS, len(UIDS), 1)),
        "missiles_left": np.full((ROWS, len(UIDS)), 2, dtype=np.int64),
        "actions": actions,
        "logits": rng.normal(size=(ROWS, len(LEARNERS), sum(HEAD_SIZES))).astype(np.float32),
    }


def _manifest() -> dict[str, Any]:
    return {
        "aircraft_uids": list(UIDS),
        "learner_uids": list(LEARNERS),
        "action_head_sizes": list(HEAD_SIZES),
        "opponent": {"rule_type": "pursue"},
    }


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    npz, acmi = tmp_path / "episode.npz", tmp_path / "replay.txt.acmi"
    np.savez(npz, **_arrays())
    acmi.write_text(ACMI, encoding="utf-8")
    data, info = convert(npz, acmi, _manifest(), "synthetic_seed1")
    csv_path = tmp_path / "seed1.csv"
    csv_path.write_bytes(data)
    manifest = build_manifest(csv_path.name, data, {**info, "episode_id": "synthetic_seed1"}, {"npz": npz})
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path


def test_parse_acmi_missiles_with_launch_and_remove_returns_records() -> None:
    records = parse_acmi_missiles(ACMI, UIDS)

    assert [(r.step, r.object_id, r.event) for r in records] == [
        (1, "A01002", "update"),
        (2, "A01002", "update"),
        (3, "A01002", "remove"),
    ]
    assert records[0].launcher_id == "A0100"
    assert records[0].transform == (120.1, 60.0, 5999.5, 0.0, -1.5, 370.25)
    assert records[2].transform is None


def test_launch_steps_puts_launch_one_step_before_first_frame() -> None:
    assert launch_steps(parse_acmi_missiles(ACMI, UIDS)) == {(0, "A0100"): "A01002"}


def test_softmax_heads_each_head_sums_to_one() -> None:
    heads = softmax_heads(np.arange(13, dtype=np.float32), HEAD_SIZES)

    assert [len(h) for h in heads] == list(HEAD_SIZES)
    assert all(math.isclose(float(h.sum()), 1.0) for h in heads)


def test_convert_output_loads_with_translator_loader(episode: Path) -> None:
    frame = load_episode(episode)

    assert list(frame.columns) == list(COLUMNS)
    facts = episode_facts(frame)
    assert facts.last_step == ROWS - 1
    assert facts.down_steps == {"A0100": None, "A0200": None, "B0100": None, "B0200": DEATH_ROW}


def test_convert_aircraft_rows_follow_13lag_semantics(episode: Path) -> None:
    frame = load_episode(episode).set_index(["step", "object_id"])

    first_dead, later_dead = frame.loc[(DEATH_ROW, "B0200")], frame.loc[(DEATH_ROW + 1, "B0200")]
    assert (first_dead["alive"], first_dead["state_valid"], first_dead["acmi_event"]) == (0, 0, "remove")
    assert first_dead["action_status"] == "inactive"
    assert later_dead["acmi_event"] == "none"
    learner = frame.loc[(0, "A0100")]
    assert (learner["action_status"], learner["heading_action"], learner["heading_action_id"]) == (
        "recorded",
        "right15",
        3,
    )
    assert learner["missile_launched"] == "A01002"
    assert 0.0 <= learner["yaw_deg"] < 360.0
    assert math.isclose(learner["tas_mps"], 5.0)
    assert frame.loc[(0, "B0100")]["action_status"] == "not_recorded"


def test_convert_missile_rows_keep_acmi_values_and_drop_frames_after_last_row(episode: Path) -> None:
    frame = load_episode(episode)
    missile = frame[frame["object_id"].eq("A01002")].set_index("step")

    assert list(missile.index) == [1, 2, 3]
    assert missile.loc[2, "alt_m"] == 5990.125
    assert missile.loc[1, "yaw_deg"] == 370.25  # raw ACMI value, like 13_LAG
    assert (missile.loc[3, "acmi_event"], missile.loc[3, "state_valid"]) == ("remove", 0)
    assert missile["launcher_id"].eq("A0100").all() and missile["has_csv_record"].eq(0).all()
    assert frame[frame["entity_type"].eq("effect")].empty
    assert int(frame["step"].max()) == ROWS - 1  # ACMI frame #0.80 (step 4) is dropped


def test_convert_window_renders_missile_and_masks_dead_state(episode: Path) -> None:
    rendered = render_lag17_jsonl(load_episode(episode), start_step=0, end_step=2, anchor_step=2, episode_alias="e1")
    lines = [json.loads(line) for line in rendered.text.splitlines()]

    ids_at_1 = [entity["object_id"] for entity in lines[1]["entities"]]
    assert ids_at_1 == [*UIDS, "A01002"]
    dead = next(e for e in lines[2]["entities"] if e["object_id"] == "B0200")
    assert dead["state"] == [None] * 10 and dead["action"] == [None, None, None]


def test_convert_with_steps_not_starting_at_zero_raises(tmp_path: Path) -> None:
    npz, acmi = tmp_path / "e.npz", tmp_path / "a.acmi"
    np.savez(npz, **{**_arrays(), "step": np.arange(1, ROWS + 1)})
    acmi.write_text(ACMI, encoding="utf-8")

    with pytest.raises(ValueError, match="steps 0..T-1"):
        convert(npz, acmi, _manifest(), "x")


def test_parse_acmi_missile_with_unknown_launcher_raises() -> None:
    with pytest.raises(ValueError, match="launcher"):
        parse_acmi_missiles("#0.20\nC09001,T=1|2|3|0|0|0,Name=AIM-9L,Color=Red\n", UIDS)
