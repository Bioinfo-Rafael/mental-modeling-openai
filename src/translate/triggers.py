"""When to translate: three trigger kinds, two of them forced, and the change sources.

- ``start``: step 0, the episode start; every target is translated there, with or without an
  instruction in effect (forced).
- ``request``: a human question to the target, at its step (forced). Instructions and alarms never
  trigger a translation; they are prompt context and detector input (D4, D61, D50).
- ``change``: decided by the change source: ``preset`` (steps listed in the config) or
  ``clef_flash`` (the detector on its tick grid; ``translate.detector``). A change that follows the
  plan is no change point.

Coinciding reasons give one translation that lists all of them. A target that is down gets no
translation: its reasons are logged as ``trigger_suppressed`` (D37).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from translate.episode import Episode, Target, View

KINDS = ("start", "change", "request")  # also the order reasons are listed in


@dataclass(frozen=True)
class Reason:
    kind: str  # start | change | request
    source: str  # episode_start | question | preset | clef_flash
    reference_id: str  # episode-start, the question's id, preset-step-<s>, or the detector call id
    probability: float | None = None  # the detector's p(change)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "reference_id": self.reference_id,
            "probability": self.probability,
        }


def forced(view: View, target: Target) -> tuple[Reason, ...]:
    """``start`` at step 0, and ``request`` for each question to the target at t."""
    start = (Reason("start", "episode_start", "episode-start"),) if view.step == 0 else ()
    asked = tuple(
        Reason("request", "question", m.id)
        for m in view.messages
        if m.kind == "question" and m.step == view.step and target.id in m.to
    )
    return start + asked


def preset_change(view: View, target: Target, change: Any) -> Reason | None:
    """A change at every listed step, for every live target."""
    return Reason("change", "preset", f"preset-step-{view.step}") if view.step in change.steps else None


CHANGE_SOURCES: dict[str, Callable[[View, Target, Any], Reason | None]] = {"preset": preset_change}


def sorted_reasons(reasons: tuple[Reason, ...]) -> tuple[Reason, ...]:
    return tuple(sorted(reasons, key=lambda r: (KINDS.index(r.kind), r.reference_id)))


def kinds_of(reasons: tuple[Reason, ...]) -> list[str]:
    return [kind for kind in KINDS if any(reason.kind == kind for reason in reasons)]


def is_tick(step: int, every_steps: int) -> bool:
    """The detector's fixed grid (D28): questions and instructions never move it."""
    return step > 0 and step % every_steps == 0


@dataclass(frozen=True)
class Schedule:
    """What a run will do, known before any call: detector calls never depend on model outputs."""

    forced: tuple[tuple[int, str], ...]  # (step, target) of every start and request while the target is up
    preset: tuple[tuple[int, str], ...]  # (step, target) of every preset change while up
    ticks: tuple[tuple[int, str], ...]  # (step, target) of every detector call (clef_flash)
    suppressed: tuple[tuple[int, str], ...]  # (step, target) of forced or preset triggers after the target is down

    @property
    def mandatory(self) -> tuple[tuple[int, str], ...]:
        return tuple(sorted(set(self.forced) | set(self.preset)))

    @property
    def translation_bound(self) -> int:
        """Mandatory translations plus one per tick without one."""
        return len(set(self.mandatory) | set(self.ticks))


def schedule(scenario: Any, episode: Episode, change: Any) -> Schedule:
    forced_at, preset, ticks, suppressed = [], [], [], []
    for target_id in scenario.targets:
        down = episode.down_step(episode.target(target_id))
        end = episode.last_step + 1 if down is None else down
        steps = {0} | {m.step for m in scenario.messages if m.kind == "question" and target_id in m.to}
        listed = set(change.steps) if change.source == "preset" else set()
        forced_at += [(s, target_id) for s in sorted(steps) if s < end]
        preset += [(s, target_id) for s in sorted(listed) if s < end]
        suppressed += [(s, target_id) for s in sorted(steps | listed) if end <= s <= episode.last_step]
        if change.source == "clef_flash":
            ticks += [(s, target_id) for s in range(1, end) if is_tick(s, change.every_steps)]
    return Schedule(tuple(forced_at), tuple(preset), tuple(ticks), tuple(suppressed))
