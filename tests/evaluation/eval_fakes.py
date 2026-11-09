"""Offline helper for the assess tests: a Translator run directory of the example scenario with fake layers.

Nothing here touches the network, a GPU, a real key or a paid API.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tests.translate.tr_fakes import (
    CHANGE,
    EXAMPLE_SCENARIO,
    WIN01,
    FakeDetectionModel,
    FakeIntentModel,
    answers_at,
    make_config,
    make_translator,
)
from translate.cli import JsonlRunSink
from translate.lag.dataset import last_complete_step
from translate.lag.rendering import render_detector_compact, render_lag17_jsonl
from translate.pipeline import TranslatorSummary
from translate.scenario import load_scenario

# The example scenario's translation events with fake layers whose detector reports exactly these changes.
EXAMPLE_DETECTOR_ANSWERS = {("aircraft A0100", 15): CHANGE, ("team A", 20): CHANGE}


def write_translator_run(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    detection: FakeDetectionModel | None = None,
    intent: FakeIntentModel | None = None,
) -> TranslatorSummary:
    """Run the example scenario with fake layers and the real renderers into a run directory.

    The default detector reports ``EXAMPLE_DETECTOR_ANSWERS``, so the run has the five ``EVENT_IDS``.
    The files are the ones the Translator CLI writes (``JsonlRunSink``) plus a minimal ``run.json`` that
    carries the resolved config like the CLI's (``config.pipeline.visibility`` = online, 0 future steps).
    """
    run_dir.mkdir(parents=True)
    config = make_config()
    translator = make_translator(
        detection or FakeDetectionModel(answers_at(EXAMPLE_DETECTOR_ANSWERS)),
        intent,
        config=config,
        detector_renderer=render_detector_compact,
        translation_renderer=render_lag17_jsonl,
    )
    scenario = load_scenario(EXAMPLE_SCENARIO, limits=config.scenario_limits)
    with JsonlRunSink(run_dir) as sink:
        summary = translator.run(frame=frame, scenario=scenario, config=config, sink=sink)
    episode = {"episode_id": WIN01, "last_complete_step": last_complete_step(frame)}  # what the video block reads
    resolved = json.loads(json.dumps(asdict(config), default=str))  # as the CLI writes it: visibility online
    run_json = {
        "schema_version": 1,
        "mode": "execute",
        "status": summary.status,
        "inputs": {"episode": episode},
        "config": resolved,
    }
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return summary
