"""The episode, View(t) and every input: the renderers, the message rules and the causality poison test."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from translate.config import Message
from translate.episode import Episode, EpisodeError, Translation, View, load_episode, load_frame, render_lag17
from translate.inputs import INPUTS, TrajectoryOptions

from .fakes import EPISODE_CSV, ROOT

PARITY = json.loads((ROOT / "tests" / "fixtures" / "lag17-renderer-parity.json").read_text(encoding="utf-8"))
HALF_WINDOW = 30
A1, TEAM = "aircraft:A0100", "team:A"
OPTIONS = {  # every input with non-default options too, so the poison test covers each code path
    "trajectory": [
        TrajectoryOptions(past_steps=60),
        TrajectoryOptions(past_steps=5, objects="target_and_opponents", fields=("alive", "lon_deg")),
    ],
    "previous_translation": [
        INPUTS["previous_translation"].options(),
        INPUTS["previous_translation"].options(evidence=False),
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


def test_poison_every_input_at_t_ignores_everything_after_t(episode: Episode) -> None:
    """Two episodes and scenarios equal through t give identical inputs through t, for every input."""
    t = 120
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
        for options in OPTIONS.get(name, [spec.options()]):
            for target_id in (A1, TEAM):
                for step in (0, t - 1, t):
                    clean = spec.render(
                        view(episode, step, clean_messages, previous), episode.target(target_id), options
                    )
                    dirty = spec.render(
                        view(poisoned, step, poison_messages, previous), poisoned.target(target_id), options
                    )
                    assert clean == dirty, (name, target_id, step)
