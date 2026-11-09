"""Detector budget fit of the default config on every tick of the three 13_LAG episodes.

The bound is the System One client's conservative one (UTF-8 bytes + fixed reserves) for the
exact state, instruction, alarm and question texts included, plus a worst-case previous intent
(4 bytes x ``translation.max_intent_chars``). The scenario is dense: every target instructed with a
longest Japanese text, one longest alarm per team in every lookback, one longest question per
aircraft in every lookback.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from translate.detector import choice_question
from translate.events import TargetRef
from translate.lag.rendering import render_detector_compact
from translate.scenario import episode_facts
from translate.triggers import BudgetFit, check_detector_budget, is_tick

from .tr_fakes import WIN01, alarm, instruction, make_config, make_scenario, request

AIRCRAFT_TARGETS = tuple(TargetRef("aircraft", a) for a in ("A0100", "A0200", "B0100", "B0200"))
TEAM_TARGETS = (TargetRef("team", "A"), TargetRef("team", "B"))
JAPANESE = "編隊を維持して北へ向かう。"  # 3 UTF-8 bytes per character


def _longest(chars: int) -> str:
    return (JAPANESE * chars)[:chars]


def dense_scenario(episode_id: str, last_step: int) -> Any:
    """Every target at step 0 with the longest texts; one alarm per team and one question per aircraft
    in every messages lookback (25 steps), so each state carries the largest per-target message set."""
    limits = make_config().scenario_limits
    lookback = make_config().detector.messages_lookback_steps
    targets = [*AIRCRAFT_TARGETS, *TEAM_TARGETS]
    instructions = [
        instruction(f"i-{t.kind}-{t.target_id}", 0, t.kind, t.target_id, _longest(limits.max_instruction_chars))
        for t in targets
    ]
    steps = range(0, last_step + 1, lookback)
    alarms = [
        alarm(f"w-{team.target_id}-{step}", step, (team.kind, team.target_id), text=_longest(limits.max_alarm_chars))
        for team in TEAM_TARGETS
        for step in steps
    ]
    requests = [
        request(f"r-{a.target_id}-{step}", step, (a.kind, a.target_id), question=_longest(limits.max_question_chars))
        for a in AIRCRAFT_TARGETS
        for step in steps
    ]
    return make_scenario(instructions, requests, alarms, episode_id=episode_id)


@pytest.fixture(scope="module")
def budget_fits(frames: dict[str, pd.DataFrame]) -> dict[tuple[str, str], BudgetFit]:
    """(episode id, target kind) -> worst fit over every tick 5..last_step, deaths ignored (superset)."""
    config = make_config()
    questions = (choice_question(config.detector.question),)
    fits = {}
    for episode_id, frame in frames.items():
        facts = episode_facts(frame)
        scenario = dense_scenario(episode_id, facts.last_step)
        steps = [step for step in range(facts.last_step + 1) if is_tick(step, config.triggers.tick_interval_steps)]
        for kind, targets in (("aircraft", AIRCRAFT_TARGETS), ("team", TEAM_TARGETS)):
            ticks = tuple((target, step) for target in targets for step in steps)
            fits[(episode_id, kind)] = check_detector_budget(
                frame,
                ticks=ticks,
                scenario=scenario,
                facts=facts,
                config=config,
                questions=questions,
                renderer=render_detector_compact,
            )
    return fits


def test_default_config_fits_every_tick_of_every_episode(budget_fits: dict[tuple[str, str], BudgetFit]) -> None:
    report = {
        f"{episode}/{kind}": (fit.worst_bound, fit.margin, fit.checked_states)
        for (episode, kind), fit in budget_fits.items()
    }
    print(json.dumps(report, indent=1))

    assert all(fit.fits and fit.input_token_budget == 16384 for fit in budget_fits.values()), report
    assert all(fit.intent_reserve_bytes == 4 * 600 for fit in budget_fits.values())
    assert sum(fit.checked_states for fit in budget_fits.values()) == 6 * (68 + 58 + 99)


def test_the_bound_counts_the_exact_message_texts(frames: dict[str, pd.DataFrame]) -> None:
    config = make_config()
    facts = episode_facts(frames[WIN01])
    target, questions = TargetRef("team", "A"), (choice_question(config.detector.question),)

    def worst(alarm_count: int) -> int:
        instructions = [instruction("ta", 0, "team", "A")]
        alarms = [alarm(f"w{i}", i, ("team", "A"), text="\U0001f600" * 200) for i in range(1, alarm_count + 1)]
        scenario = make_scenario(instructions, alarms=alarms)
        fit = check_detector_budget(
            frames[WIN01],
            ticks=((target, 25),),
            scenario=scenario,
            facts=facts,
            config=config,
            questions=questions,
            renderer=render_detector_compact,
        )
        return fit.worst_bound

    one_more = worst(2) - worst(1)

    assert one_more == len(b"- alarm at step 2: ") + 800 + 1  # its line, 4-byte characters, newline
    assert worst(20) > config.detector.input_token_budget  # all 20 alarms in [1, 25]: none is dropped


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
