"""Offline fakes for the Translator tests: scripted layer models, cheap renderers, a memory sink.

Nothing here touches the network, a GPU, a real key or a paid API.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml

from mtd_rl_xavier import paths
from mtd_rl_xavier.lag.rendering import DETECTOR_RENDERER, LAG17_RENDERER, RENDERER_VERSION, RenderedTrajectory
from mtd_rl_xavier.translator.config import (
    ConnectionConfig,
    DetectorConfig,
    TranslatorConfig,
    parse_config,
    parse_scenario,
)
from mtd_rl_xavier.translator.detector import DetectionModelReply, DetectorLayer
from mtd_rl_xavier.translator.pipeline import Translator
from mtd_rl_xavier.translator.schema import DETECTOR_LABELS, Scenario, sha256_text
from mtd_rl_xavier.translator.translation import IntentModelOutput, TranslationLayer, TranslationPrompt
from tests.llm.llm_fakes import choice_answer, json_response, systemone_body

PROCESSED = paths.ROOT / "data" / "mhi" / "processed"
EPISODE_CSV = {
    episode_id: PROCESSED / f"{episode_id}.csv"
    for episode_id in ("mhi_v1.3_ic100601_win", "ours_seed100100_ic100601_loss", "ours_seed100100_ic100606_win")
}
WIN01 = "mhi_v1.3_ic100601_win"
LOSS01 = "ours_seed100100_ic100601_loss"
RUN_CONFIG = paths.CONFIGS / "translator" / "run.yaml"
EXAMPLE_SCENARIO = paths.CONFIGS / "translator" / "scenario.example.yaml"
SHORT_LAST_STEP = 30
FAKE_DETECTOR_MODEL = "fake-detector"
FAKE_INTENT_MODEL = "fake-intent"
QUESTION_ID = "point_kind"  # the committed run.yaml's detector question
# The state's trajectory framing line names the tick (anchor) step for any renderer.
ANCHOR_PATTERN = re.compile(r"relative_step = step - (\d+)\. Fields: ")
TARGET_PATTERN = re.compile(r"^Target: (\w+ \w+) ")
# The committed run.yaml uses a Unix socket (D39); tests that build the real client over
# httpx.MockTransport use this loopback TCP endpoint instead, whose checks need no directory.
TEST_TCP_CONNECTION = ConnectionConfig(base_url="http://127.0.0.1:8000", uds_path=None, api_key_env="VLLM_API_KEY")


def run_config_raw() -> dict[str, Any]:
    return yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))


def with_value(raw: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """A deep copy of ``raw`` with ``a.b.c`` set to ``value`` (``DELETE`` removes the key)."""
    result = copy.deepcopy(raw)
    *parents, last = dotted.split(".")
    node = result
    for key in parents:
        node = node[key]
    if value is DELETE:
        del node[last]
    else:
        node[last] = value
    return result


DELETE = object()


def make_config(**overrides: Any) -> TranslatorConfig:
    """The committed run.yaml with ``a__b=value`` overrides (``__`` separates the levels)."""
    raw = run_config_raw()
    for dotted, value in overrides.items():
        raw = with_value(raw, dotted.replace("__", "."), value)
    return parse_config(raw)


def target(kind: str, target_id: str) -> dict[str, str]:
    return {"kind": kind, "id": target_id}


def instruction(instruction_id: str, step: int, kind: str, target_id: str, text: str = "") -> dict[str, Any]:
    text = text or f"[PLACEHOLDER] instruction {instruction_id}"
    return {"instruction_id": instruction_id, "issue_step": step, "target": target(kind, target_id), "text": text}


def request(
    request_id: str, step: int, *targets: tuple[str, str], source: str = "human", question: str = ""
) -> dict[str, Any]:
    question = question or f"[PLACEHOLDER] question {request_id}"
    return {
        "request_id": request_id,
        "step": step,
        "targets": [target(*t) for t in targets],
        "source": source,
        "question": question,
    }


def alarm(alarm_id: str, step: int, *targets: tuple[str, str], text: str = "") -> dict[str, Any]:
    text = text or f"[PLACEHOLDER] alarm {alarm_id}"
    return {"alarm_id": alarm_id, "step": step, "targets": [target(*t) for t in targets], "text": text}


def scenario_raw(
    instructions: list[dict[str, Any]],
    requests: list[dict[str, Any]] | None = None,
    alarms: list[dict[str, Any]] | None = None,
    *,
    episode_id: str = WIN01,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scenario_id": "test-scenario",
        "episode": {
            "episode_id": episode_id,
            "alias": "episode_x",
            "csv_path": str(EPISODE_CSV[episode_id]),
            "video_path": None,
        },
        "instructions": instructions,
        "alarms": alarms or [],
        "requests": requests or [],
    }


def make_scenario(*args: Any, **kwargs: Any) -> Scenario:
    """A scenario checked against the committed run.yaml's ``scenario_limits``."""
    return parse_scenario(scenario_raw(*args, **kwargs), base=paths.ROOT, limits=make_config().scenario_limits)


def write_yaml(path: Path, raw: object) -> Path:
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def over_tcp(config: DetectorConfig) -> DetectorConfig:
    """The detector config with ``TEST_TCP_CONNECTION`` (for clients over ``httpx.MockTransport``)."""
    return replace(config, connection=TEST_TCP_CONNECTION)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL records, split on "\\n" only (U+2028 and friends are legal inside JSON strings)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in (lines[:-1] if lines[-1] == "" else lines)]


# --- renderers ------------------------------------------------------------------------------------


def _rendered(
    renderer: str, start: int, end: int, anchor: int, objects: tuple[str, ...], fields: tuple[str, ...], text: str
) -> RenderedTrajectory:
    return RenderedTrajectory(renderer, RENDERER_VERSION, start, end, anchor, objects, fields, text, sha256_text(text))


def fake_detector_renderer(
    frame: pd.DataFrame,
    *,
    start_step: int,
    end_step: int,
    anchor_step: int,
    object_ids: tuple[str, ...],
    fields: tuple[str, ...],
    float_precision: int,
) -> RenderedTrajectory:
    text = f"FAKE-COMPACT anchor={anchor_step} start={start_step} objects={','.join(object_ids)}"
    return _rendered(DETECTOR_RENDERER, start_step, end_step, anchor_step, object_ids, fields, text)


def fake_translation_renderer(
    frame: pd.DataFrame, *, start_step: int, end_step: int, anchor_step: int, episode_alias: str
) -> RenderedTrajectory:
    text = f"FAKE-LAG17 {start_step}..{end_step} anchor={anchor_step}"
    return _rendered(LAG17_RENDERER, start_step, end_step, anchor_step, ("A0100", "A0200", "B0100", "B0200"), (), text)


# --- layer models ---------------------------------------------------------------------------------

Answer = tuple[str, float]  # a scripted detector answer: (label, p(label))
NONE: Answer = ("none", 1.0)
START: Answer = ("start", 0.9)
CHANGE: Answer = ("change", 0.9)


def state_key(state: str) -> tuple[str, int]:
    """(target label, tick step) of a detector state (any renderer)."""
    target_match, anchor_match = TARGET_PATTERN.search(state), ANCHOR_PATTERN.search(state)
    if target_match is None or anchor_match is None:
        raise AssertionError("not a detector state")
    return target_match.group(1), int(anchor_match.group(1))


def answer_probabilities(answer: Answer) -> dict[str, float]:
    """``p(label)`` for the label; the rest is split evenly over the other two options."""
    label, probability = answer
    rest = (1.0 - probability) / (len(DETECTOR_LABELS) - 1)
    return {name: probability if name == label else rest for name in DETECTOR_LABELS}


def answers_at(answers: dict[tuple[str, int], Answer], default: Answer = NONE) -> Callable[[str], Answer]:
    """A state -> answer function: ``answers[(target label, tick step)]``, else ``default``."""
    return lambda state: answers.get(state_key(state), default)


def systemone_responder(
    answers: dict[tuple[str, int], Answer], *, question_id: str = QUESTION_ID
) -> Callable[[httpx.Request], httpx.Response]:
    """A ``SystemOneServer`` outcome that answers a real ``/v1/systemone`` request from its state."""
    answer_of = answers_at(answers)

    def respond(request: httpx.Request) -> httpx.Response:
        answer = answer_of(json.loads(request.content)["state"])
        body = systemone_body({question_id: choice_answer(answer[0], answer_probabilities(answer))})
        return json_response(200, body)

    return respond


class FakeDetectionModel:
    """Scripted choice answers: a constant, or a function of the state. ``fail`` raises on a chosen call."""

    def __init__(
        self,
        answer: Answer | Callable[[str], Answer] = NONE,
        *,
        log: list[tuple[str, str]] | None = None,
        fail: tuple[int, Exception] | None = None,
    ) -> None:
        self._answer = answer
        self._log = log if log is not None else []
        self._fail = fail
        self.states: list[str] = []

    @property
    def question(self) -> dict[str, Any]:
        criteria = {label: f"[FAKE] {label}" for label in DETECTOR_LABELS}
        return {"id": QUESTION_ID, "type": "choice", "instructions": "[FAKE] question", "criteria": criteria}

    def request_payload(self, state: str) -> dict[str, Any]:
        return {"model": FAKE_DETECTOR_MODEL, "state": state, "questions": {QUESTION_ID: self.question}}

    def decide(self, state: str) -> DetectionModelReply:
        self.states.append(state)
        self._log.append(("send", "detector"))
        if self._fail is not None and len(self.states) == self._fail[0]:
            raise self._fail[1]
        answer = self._answer(state) if callable(self._answer) else self._answer
        probabilities = answer_probabilities(answer)
        raw = {
            "model": FAKE_DETECTOR_MODEL,
            "answers": {QUESTION_ID: {"type": "choice", "choice": answer[0], "probabilities": probabilities}},
        }
        usage = {"input_tokens": 1, "output_tokens": 0}
        return DetectionModelReply(FAKE_DETECTOR_MODEL, answer[0], tuple(probabilities.items()), usage, raw)


class FakeIntentModel:
    """Returns ``[FAKE] intent <n>`` (or scripted texts); ``fail`` raises on a chosen call."""

    def __init__(
        self,
        texts: list[str] | None = None,
        *,
        log: list[tuple[str, str]] | None = None,
        fail: tuple[int, Exception] | None = None,
    ) -> None:
        self._texts = list(texts) if texts else None
        self._log = log if log is not None else []
        self._fail = fail
        self.prompts: list[TranslationPrompt] = []

    def request_payload(self, prompt: TranslationPrompt) -> dict[str, Any]:
        return {"model": FAKE_INTENT_MODEL, "instructions": prompt.instructions, "input": prompt.user_text}

    def generate(self, prompt: TranslationPrompt) -> IntentModelOutput:
        self.prompts.append(prompt)
        self._log.append(("send", "translation"))
        number = len(self.prompts)
        if self._fail is not None and number == self._fail[0]:
            raise self._fail[1]
        default = json.dumps({"intent": f"[FAKE] intent {number}", "evidence": [f"[FAKE] evidence {number}"]})
        text = self._texts.pop(0) if self._texts else default
        usage = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "reasoning_tokens": 2}
        return IntentModelOutput(
            FAKE_INTENT_MODEL, f"resp-{number}", text, usage, {"id": f"resp-{number}", "text": text}
        )


class MemorySink:
    """Collects the three record streams; ``log`` interleaves writes with fake model sends."""

    def __init__(self, log: list[tuple[str, str]] | None = None) -> None:
        self.events: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.log = log if log is not None else []

    def write_event(self, record: dict[str, Any]) -> None:
        self.events.append(json.loads(json.dumps(record)))  # JSON round trip, as on disk
        self.log.append(("event", record["record_type"]))

    def write_request(self, record: dict[str, Any]) -> None:
        self.requests.append(json.loads(json.dumps(record)))
        self.log.append(("request", record["layer"]))

    def write_response(self, record: dict[str, Any]) -> None:
        self.responses.append(json.loads(json.dumps(record)))
        self.log.append(("response", record["layer"]))

    def of_type(self, record_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["record_type"] == record_type]


def make_translator(
    detection: FakeDetectionModel | None = None,
    intent: FakeIntentModel | None = None,
    *,
    config: TranslatorConfig | None = None,
    detector_renderer: Callable[..., RenderedTrajectory] = fake_detector_renderer,
    translation_renderer: Callable[..., RenderedTrajectory] = fake_translation_renderer,
) -> Translator:
    config = config or make_config()
    return Translator(
        detector=DetectorLayer(detection or FakeDetectionModel(), threshold=config.detector.threshold),
        translation=TranslationLayer(
            intent or FakeIntentModel(),
            output_language=config.translation.output_language,
            max_intent_chars=config.translation.max_intent_chars,
        ),
        detector_renderer=detector_renderer,
        translation_renderer=translation_renderer,
    )
