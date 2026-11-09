"""The episode, View(t) and every input: the renderers, the message rules and the causality poison test."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from translate.config import Message
from translate.episode import Episode, EpisodeError, Translation, View, load_episode, load_frame, render_lag17
from translate.inputs import INPUTS, TrajectoryOptions

from .fakes import EPISODE_CSV, ROOT

PARITY = json.loads((ROOT / "tests" / "fixtures" / "lag17-renderer-parity.json").read_text(encoding="utf-8"))
HALF_WINDOW = 30
A1, TEAM = "aircraft:A0100", "team:A"
POISON_STEP = 120


def write_rollout(
    directory: Path,
    poison_after: int | None = None,
    names: tuple[str, ...] = ("obs", "gru_out", "act_in", "logits", "actions"),
) -> Path:
    """A small synthetic rollout (manifest + npz + head weights); rows after ``poison_after`` differ."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(7)
    arrays = {name: rng.normal(size=(341, 2, 3)).astype(np.float32) for name in names if name != "actions"}
    arrays = {**arrays, **({"actions": rng.integers(0, 3, size=(341, 2, 4))} if "actions" in names else {})}
    if poison_after is not None:
        arrays = {
            name: np.where(np.arange(341)[:, None, None] > poison_after, 999, value).astype(value.dtype)
            for name, value in arrays.items()
        }
    np.savez(directory / "episode_seed1.npz", **arrays)
    np.savez(
        directory / "head_weights.npz", weight=np.ones((13, 3), dtype=np.float32), bias=np.zeros(13, dtype=np.float32)
    )
    manifest = {"learner_uids": ["A0100", "A0200"], "episodes": [{"seed": 1, "file": "episode_seed1.npz"}]}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def option_pairs(tmp_path: Path) -> dict[str, list[tuple]]:
    """(clean, poisoned) options per input; an input not listed runs with its default options on both."""
    policy = INPUTS["policy_trace"].options
    clean, dirty = write_rollout(tmp_path / "clean"), write_rollout(tmp_path / "dirty", POISON_STEP)
    trajectory = [
        TrajectoryOptions(past_steps=60),
        TrajectoryOptions(past_steps=5, objects="target_and_opponents", fields=("alive", "lon_deg")),
    ]
    previous = [INPUTS["previous_translation"].options(), INPUTS["previous_translation"].options(evidence=False)]
    return {
        "trajectory": [(o, o) for o in trajectory],
        "previous_translation": [(o, o) for o in previous],
        "policy_trace": [
            (policy(rollout=clean, seed=1), policy(rollout=dirty, seed=1)),
            (
                policy(rollout=clean, seed=1, past_steps=3, head_weights=False),
                policy(rollout=dirty, seed=1, past_steps=3, head_weights=False),
            ),
        ],
    }


@pytest.fixture(scope="module")
def episode() -> Episode:
    return load_episode(EPISODE_CSV)


def message(id_: str, step: int, kind: str, to: str, text: str) -> Message:
    return Message(id=id_, step=step, kind=kind, to=(to,), text=text)


def view(episode: Episode, step: int, messages: tuple = (), translations: dict | None = None) -> View:
    shown = tuple(m for m in messages if m.step <= step)
    return View(step, episode.rows_until(step), shown, episode.teams, translations or {})


@pytest.mark.parametrize("key", sorted(PARITY["lag17_window_sha256"]))
def test_the_lag17_renderer_matches_the_workspace_17_windows(key: str) -> None:
    name, center = key.split("@")
    frame, _ = load_frame(ROOT / "data" / "mhi" / "processed" / f"{name}.csv")

    text = render_lag17(frame, int(center) - HALF_WINDOW, int(center) + HALF_WINDOW, int(center))

    assert hashlib.sha256(text.encode()).hexdigest() == PARITY["lag17_window_sha256"][key]


def test_an_episode_that_differs_from_its_manifest_is_refused(tmp_path: Path) -> None:
    shutil.copy(EPISODE_CSV.parent / "manifest.json", tmp_path / "manifest.json")
    (tmp_path / EPISODE_CSV.name).write_bytes(EPISODE_CSV.read_bytes() + b"\n")

    with pytest.raises(EpisodeError, match="manifest"):
        load_episode(tmp_path / EPISODE_CSV.name)


def test_a_bare_start_has_no_instruction_in_effect_and_a_later_one_replaces_it(episode: Episode) -> None:
    render = INPUTS["instructions"].render
    target = episode.target(A1)
    messages = (message("i1", 5, "instruction", A1, "first"), message("i2", 9, "instruction", A1, "second"))

    texts = [render(view(episode, step, messages), target, None) for step in (0, 5, 9)]

    assert texts[0] == "(no instruction in effect)"
    assert texts[1] == '- aircraft A0100, issued at step 5: "first" (this target)'
    assert texts[2] == '- aircraft A0100, issued at step 9: "second" (this target)'


def test_an_alarm_reaches_every_target_that_shares_an_aircraft(episode: Episode) -> None:
    alarm = (message("w1", 3, "alarm", A1, "missile"),)

    shown = {
        t: INPUTS["alarms"].render(view(episode, 4, alarm), episode.target(t), None)
        for t in (A1, TEAM, "aircraft:A0200")
    }

    assert shown == {A1: '- step 3: "missile"', TEAM: '- step 3: "missile"', "aircraft:A0200": "(none)"}


def test_a_question_shows_only_at_its_step_and_recent_messages_keep_a_window(episode: Episode) -> None:
    messages = (message("q1", 10, "question", A1, "why?"), message("w1", 2, "alarm", A1, "old alarm"))
    target, recent = episode.target(A1), INPUTS["recent_messages"]

    assert INPUTS["question"].render(view(episode, 10, messages), target, None) == '"why?"'
    assert INPUTS["question"].render(view(episode, 11, messages), target, None) == "(none)"
    assert (
        recent.render(view(episode, 11, messages), target, recent.options(past_steps=5))
        == '- question at step 10: "why?"'
    )


def test_the_previous_translation_with_and_without_evidence(episode: Episode) -> None:
    previous = {A1: Translation("e1", 4, "climb", ("alt rises",))}
    render, options = INPUTS["previous_translation"].render, INPUTS["previous_translation"].options

    full = render(view(episode, 9, (), previous), episode.target(A1), options())
    short = render(view(episode, 9, (), previous), episode.target(A1), options(evidence=False))

    assert (full, short) == ('from step 4: "climb"\n  - "alt rises"', 'from step 4: "climb"')


def test_poison_every_input_at_t_ignores_everything_after_t(episode: Episode, tmp_path: Path) -> None:
    """Two episodes, scenarios and rollouts equal through t give identical inputs through t, for every input."""
    t, pairs = POISON_STEP, option_pairs(tmp_path)
    poisoned_frame = episode.frame.copy()
    later = poisoned_frame["step"] > t
    poisoned_frame.loc[later, ["lon_deg", "lat_deg", "alt_m"]] = 9999.0
    poisoned_frame.loc[later, "alive"] = pd.array([0] * int(later.sum()), dtype=poisoned_frame["alive"].dtype)
    poisoned = Episode.from_frame(poisoned_frame)
    clean_messages = (message("q1", t, "question", A1, "why?"), message("i1", 30, "instruction", TEAM, "go"))
    poison_messages = (
        *clean_messages,
        message("i2", t + 1, "instruction", A1, "FUTURE"),
        message("w9", t + 2, "alarm", TEAM, "FUTURE"),
    )
    previous = {A1: Translation("e1", 60, "climb", ("e",)), TEAM: Translation("e2", 60, "split", ("e",))}

    for name, spec in INPUTS.items():
        for clean_options, dirty_options in pairs[name] if name in pairs else [(spec.options(), spec.options())]:
            for target_id in (A1, TEAM):
                for step in (0, t - 1, t):
                    clean_view, dirty_view = (
                        view(episode, step, clean_messages, previous),
                        view(poisoned, step, poison_messages, previous),
                    )
                    clean = spec.render(clean_view, episode.target(target_id), clean_options)
                    dirty = spec.render(dirty_view, poisoned.target(target_id), dirty_options)
                    assert clean == dirty, (name, target_id, step)


def test_the_policy_trace_shows_each_learner_up_to_t_and_the_head_weights(episode: Episode, tmp_path: Path) -> None:
    options = INPUTS["policy_trace"].options(rollout=write_rollout(tmp_path / "r"), seed=1, past_steps=3)

    text = INPUTS["policy_trace"].render(view(episode, 10, ()), episode.target(TEAM), options)
    blue = INPUTS["policy_trace"].render(view(episode, 10, ()), episode.target("aircraft:B0100"), options)

    rows = [json.loads(line) for line in text.split("\n") if line.startswith('{"step"')]
    assert [row["step"] for row in rows] == [8, 9, 10, 8, 9, 10] and rows[-1]["relative_step"] == 0
    assert set(rows[0]) == {"step", "relative_step", "obs", "gru_out", "act_in", "logits", "actions"}
    assert "Policy trace of A0200" in text and text.count('{"logit"') == 13
    assert blue == "B0100: not flown by the recorded policy" + text[text.index("\n\nAction heads") :]


def test_a_rollout_without_act_in_must_be_re_recorded(episode: Episode, tmp_path: Path) -> None:
    rollout = write_rollout(tmp_path / "old", names=("obs", "gru_out", "logits", "actions"))
    options = INPUTS["policy_trace"].options(rollout=rollout, seed=1)

    with pytest.raises(ValueError, match="re-record"):
        INPUTS["policy_trace"].render(view(episode, 10, ()), episode.target(A1), options)
