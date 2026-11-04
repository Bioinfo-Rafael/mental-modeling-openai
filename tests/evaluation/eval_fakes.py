"""Offline helpers for the judge tests: Translator run directories, key points, scripted verdicts.

Nothing here touches the network, a GPU, a decoder, a real key or a paid API.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evaluation.configuration import JudgeConfig, VideoResize, parse_config
from evaluation.frames import EncodedFrame, FrameSourceError, VideoInfo
from mtd_rl_xavier import paths
from mtd_rl_xavier.lag.dataset import last_complete_step
from mtd_rl_xavier.lag.rendering import render_detector_compact, render_lag17_jsonl
from mtd_rl_xavier.translator.cli import JsonlRunSink
from mtd_rl_xavier.translator.config import load_scenario
from mtd_rl_xavier.translator.pipeline import TranslatorSummary
from tests.llm.llm_fakes import DumpableResponse, ScriptedResponsesClient, message_item, responses_body
from tests.translator.tr_fakes import (
    EXAMPLE_SCENARIO,
    WIN01,
    FakeDetectionModel,
    FakeIntentModel,
    make_config,
    make_translator,
    with_value,
)

JUDGE_CONFIG = paths.ROOT / "evaluation" / "config" / "judge-text.yaml"
VIDEO_CONFIG = paths.ROOT / "evaluation" / "config" / "judge-text-video.yaml"
EXAMPLE_KEY_POINTS = paths.ROOT / "evaluation" / "examples" / "key-points.example.yaml"
SCENARIO_ID = "placeholder-episode-01"
A1_START = f"{SCENARIO_ID}:aircraft:A0100:000000"
TEAM_START = f"{SCENARIO_ID}:team:A:000000"
A1_REQUEST = f"{SCENARIO_ID}:aircraft:A0100:000010"
A1_CHANGE = f"{SCENARIO_ID}:aircraft:A0100:000015"
TEAM_INSTRUCTION = f"{SCENARIO_ID}:team:A:000020"
# The example scenario's translation events with fake layers that never report a change.
EVENT_IDS = (A1_START, TEAM_START, A1_REQUEST, A1_CHANGE, TEAM_INSTRUCTION)
EXAMPLE_JUDGED = (A1_START, A1_REQUEST)  # the event ids in key-points.example.yaml


def judge_config_raw() -> dict[str, Any]:
    return yaml.safe_load(JUDGE_CONFIG.read_text(encoding="utf-8"))


def make_judge_config(**overrides: Any) -> JudgeConfig:
    """The committed judge-text.yaml with ``a__b=value`` overrides (``__`` separates the levels)."""
    raw = judge_config_raw()
    for dotted, value in overrides.items():
        raw = with_value(raw, dotted.replace("__", "."), value)
    return parse_config(raw)


def video_config_raw() -> dict[str, Any]:
    return yaml.safe_load(VIDEO_CONFIG.read_text(encoding="utf-8"))


def with_video(raw: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """``raw`` with top-level keys of ``inputs.video`` replaced (episode ids contain dots, so no dotted paths)."""
    video = {**raw["inputs"]["video"], **changes}
    return {**raw, "inputs": {**raw["inputs"], "video": video}}


def make_video_config(**video_changes: Any) -> JudgeConfig:
    """The committed judge-text-video.yaml with some ``inputs.video`` keys replaced."""
    return parse_config(with_video(video_config_raw(), **video_changes))


FAKE_FRAME_WIDTH, FAKE_FRAME_HEIGHT = 960, 540


class FakeFrameSource:
    """A ``FrameSource`` with no decoder: ``jpeg`` bytes are ``b"FAKEJPEG-<index>"``; calls are recorded.

    ``videos`` maps a video file name to its frame count and fps; any other path is "not found".
    """

    def __init__(self, videos: dict[str, VideoInfo]) -> None:
        self._videos = videos
        self.info_calls: list[Path] = []
        self.frame_calls: list[tuple[Path, tuple[int, ...], VideoResize]] = []

    @classmethod
    def for_committed_videos(cls, frame_count: int = 400, fps: float = 20) -> FakeFrameSource:
        names = (f"{episode}_with_actions.mp4" for episode in ("mhi_v1.3_ic100601_win",))
        return cls({name: VideoInfo(frame_count, fps) for name in names})

    def info(self, path: Path) -> VideoInfo:
        self.info_calls.append(path)
        if path.name not in self._videos:
            raise FrameSourceError(f"video file not found: {path}")
        return self._videos[path.name]

    def frames(self, path: Path, frame_indices: Sequence[int], resize: VideoResize) -> tuple[EncodedFrame, ...]:
        self.frame_calls.append((path, tuple(frame_indices), resize))
        return tuple(
            EncodedFrame(index, f"FAKEJPEG-{index}".encode(), FAKE_FRAME_WIDTH, FAKE_FRAME_HEIGHT)
            for index in frame_indices
        )

    @property
    def untouched(self) -> bool:
        return not self.info_calls and not self.frame_calls


def write_yaml(path: Path, raw: object) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def key_points_raw(entries: dict[str, list[str]]) -> dict[str, Any]:
    """event id -> key-point texts; ids are kp-001, kp-002, ... per event."""
    events = [
        {
            "event_id": event_id,
            "key_points": [
                {"key_point_id": f"kp-{index:03d}", "text": text} for index, text in enumerate(texts, start=1)
            ],
        }
        for event_id, texts in entries.items()
    ]
    return {"schema_version": 1, "events": events}


def write_key_points(path: Path, entries: dict[str, list[str]]) -> Path:
    return write_yaml(path, key_points_raw(entries))


def write_translator_run(
    run_dir: Path,
    frame: pd.DataFrame,
    *,
    detection: FakeDetectionModel | None = None,
    intent: FakeIntentModel | None = None,
) -> TranslatorSummary:
    """Run the example scenario with fake layers and the real renderers into a run directory.

    The files are the ones the Translator CLI writes (``JsonlRunSink``) plus a minimal ``run.json``.
    """
    run_dir.mkdir(parents=True)
    translator = make_translator(
        detection or FakeDetectionModel(0.0),
        intent,
        detector_renderer=render_detector_compact,
        translation_renderer=render_lag17_jsonl,
    )
    with JsonlRunSink(run_dir) as sink:
        summary = translator.run(frame=frame, scenario=load_scenario(EXAMPLE_SCENARIO), config=make_config(), sink=sink)
    episode = {"episode_id": WIN01, "last_complete_step": last_complete_step(frame)}  # what the video block reads
    run_json = {"schema_version": 1, "mode": "execute", "status": summary.status, "inputs": {"episode": episode}}
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL records, split on "\\n" only (U+2028 and friends are legal inside JSON strings)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in (lines[:-1] if lines[-1] == "" else lines)]


def verdict(key_point_ids: list[str] | tuple[str, ...], *, score: int = 2, overall: int = 4) -> dict[str, Any]:
    items = [{"key_point_id": kp, "rationale": f"[FAKE] reason for {kp}", "score": score} for kp in key_point_ids]
    return {"key_points": items, "overall": {"rationale": "[FAKE] overall reason", "score": overall}}


def judge_response(text: str) -> DumpableResponse:
    """A completed Responses API response whose output text is ``text``."""
    return DumpableResponse(responses_body(output=[message_item(text)]))


def verdict_response(key_point_ids: list[str] | tuple[str, ...], **scores: int) -> DumpableResponse:
    return judge_response(json.dumps(verdict(key_point_ids, **scores)))


class RecordingFactory:
    """Stands in for ``make_openai_client``: counts calls, returns a fixed object or raises."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result, self.error = result, error

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def tree_snapshot(root: Path) -> set[Path]:
    """Every file and directory below ``root``."""
    return set(root.rglob("*"))


class LoggingResponsesClient(ScriptedResponsesClient):
    """``ScriptedResponsesClient`` that also appends ``("send", n)`` to a shared order log."""

    def __init__(self, outcomes: list[object], log: list[tuple[str, str]]) -> None:
        super().__init__(outcomes)
        self._log = log

    def create(self, **kwargs: Any) -> object:
        self._log.append(("send", str(len(self.requests) + 1)))
        return super().create(**kwargs)


class MemoryJudgeSink:
    """Collects the three judge record streams; ``fail`` raises ``OSError`` on a chosen write."""

    def __init__(self, log: list[tuple[str, str]] | None = None, *, fail: tuple[str, int] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.log = log if log is not None else []
        self._fail = fail

    def _append(self, stream: str, records: list[dict[str, Any]], record: dict[str, Any], key: str) -> None:
        if self._fail == (stream, len(records) + 1):
            raise OSError(f"simulated {stream} journal failure")
        records.append(json.loads(json.dumps(record)))  # JSON round trip, as on disk
        self.log.append((stream, record[key]))

    def write_request(self, record: dict[str, Any]) -> None:
        self._append("request", self.requests, record, "call_id")

    def write_response(self, record: dict[str, Any]) -> None:
        self._append("response", self.responses, record, "call_id")

    def write_result(self, record: dict[str, Any]) -> None:
        self._append("result", self.results, record, "event_id")
