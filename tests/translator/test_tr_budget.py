"""Detector budget fit of the default config on every tick of the three 13_LAG episodes.

The bound is the System One client's conservative one (UTF-8 bytes + fixed reserves) for the
exact state framing plus a worst-case previous intent (4 bytes x ``translation.max_intent_chars``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mtd_rl_xavier.lag.rendering import render_detector_compact
from mtd_rl_xavier.translator.detector import noul_question
from mtd_rl_xavier.translator.pipeline import BudgetFit, check_detector_budget, episode_facts, is_tick
from mtd_rl_xavier.translator.schema import TargetRef

from .tr_fakes import WIN01, make_config

AIRCRAFT_TARGETS = tuple(TargetRef("aircraft", a) for a in ("A0100", "A0200", "B0100", "B0200"))
TEAM_TARGETS = (TargetRef("team", "A"), TargetRef("team", "B"))


@pytest.fixture(scope="module")
def budget_fits(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], BudgetFit]:
    """(episode id, target kind) -> worst fit over every tick 5..last_step, deaths ignored (superset)."""
    config = make_config()
    questions = (noul_question(config.detector.question),)
    fits = {}
    for episode_id, frame in frames.items():
        facts = episode_facts(frame)
        steps = [step for step in range(facts.last_step + 1) if is_tick(step, config.pipeline.tick_interval_steps)]
        for kind, targets in (("aircraft", AIRCRAFT_TARGETS), ("team", TEAM_TARGETS)):
            ticks = tuple((target, step) for target in targets for step in steps)
            fits[(episode_id, kind)] = check_detector_budget(
                frame, ticks=ticks, facts=facts, config=config, questions=questions, renderer=render_detector_compact
            )
    return fits


def test_default_config_fits_every_tick_of_every_episode(budget_fits: dict[tuple[str, str], BudgetFit]) -> None:
    report = {
        f"{episode}/{kind}": (fit.worst_bound, fit.margin, fit.checked_states)
        for (episode, kind), fit in budget_fits.items()
    }
    print(json.dumps(report, indent=1))

    assert all(fit.fits and fit.input_token_budget == 16384 for fit in budget_fits.values()), report
    assert sum(fit.checked_states for fit in budget_fits.values()) == 6 * (68 + 58 + 99)


def test_team_states_are_the_worst_case(budget_fits: dict[tuple[str, str], BudgetFit]) -> None:
    for episode in {episode for episode, _ in budget_fits}:
        assert budget_fits[(episode, "team")].worst_bound > budget_fits[(episode, "aircraft")].worst_bound


def test_precision_five_resolves_per_step_motion_and_three_does_not(frames: dict[str, pd.DataFrame]) -> None:
    def repeated_positions(precision: int) -> int:
        rendered = render_detector_compact(
            frames[WIN01],
            start_step=0,
            end_step=340,
            anchor_step=340,
            object_ids=("A0100",),
            fields=("lon_deg", "lat_deg"),
            float_precision=precision,
        )
        points = [json.loads(line)["objects"][0] for line in rendered.text.split("\n")]
        positions = [(point["lon_deg"], point["lat_deg"]) for point in points]
        return sum(a == b for a, b in zip(positions[:-1], positions[1:], strict=True))

    assert repeated_positions(5) == 0  # A0100 never dies in this episode, so it moves every step
    assert repeated_positions(3) > 0  # ~100 m rounding hides ~50 m steps: a staircase


@pytest.mark.parametrize(
    ("worst_bound", "fits", "margin"), [(16383, True, 1), (16384, False, 0), (20000, False, -3616)]
)
def test_budget_fit_needs_the_worst_bound_strictly_below_the_budget(worst_bound: int, fits: bool, margin: int) -> None:
    fit = BudgetFit(16384, worst_bound, TargetRef("team", "A"), 5, 1, 4000)

    assert (fit.fits, fit.margin) == (fits, margin)
