"""The causal loop: for every step t and every target, triggers -> inputs -> prompt -> one call -> event.

Per step: targets that went down are closed (``target_closed``); then, per target in canonical order,
the forced reasons (start at 0, request at a question) and the change source's reason are collected.
A closed target only logs them (``trigger_suppressed``). Otherwise, if any reason holds, the target
gets exactly one translation listing every reason. Inputs and change sources see only ``View(t)``.
Every request line is fsynced before its call; a failed call stops the run (``failed``, or
``uncertain`` when it may have been billed) and is never retried; a cap stops it before call cap + 1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, NoReturn

from translate.backends import BackendError, Prompt, input_text
from translate.config import Loaded
from translate.detector import QUESTION_ID, DetectorError, check_budget, parse_decision, request_body, state_text
from translate.episode import Episode, Target, Translation, View, time_s
from translate.inputs import INPUTS, TrajectoryOptions, alarms_for, instructions_in_effect, question_at
from translate.journal import Journal, loads_unique, request_sha256, sha256_text
from translate.prompts import fill, intent_schema
from translate.triggers import CHANGE_SOURCES, Reason, forced, is_tick, kinds_of, sorted_reasons

TIMING_DECIMALS = 3
WITHHELD = "details withheld"


@dataclass(frozen=True)
class Counts:
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    uncertain: int = 0


@dataclass(frozen=True)
class Summary:
    status: str  # completed | failed | uncertain | cap_reached
    counts: dict[str, Counts]
    events: int
    error: dict[str, Any] | None = None
    unsent: dict[str, Any] | None = None


class Stop(Exception):
    def __init__(
        self, status: str, *, error: dict[str, Any] | None = None, unsent: dict[str, Any] | None = None
    ) -> None:
        super().__init__(status)
        self.status, self.error, self.unsent = status, error, unsent


def parse_answer(text: str, max_intent_chars: int) -> tuple[dict[str, Any] | None, str | None]:
    """(answer, None), or (None, why it is not one object with a non-blank intent and evidence)."""
    try:
        answer = loads_unique(text)
    except ValueError:
        return None, "the answer is not one JSON object without repeated keys"
    if not isinstance(answer, dict) or set(answer) != {"intent", "evidence"}:
        return None, "the answer must have exactly intent and evidence"
    intent, evidence = answer["intent"], answer["evidence"]
    if not isinstance(intent, str) or not intent.strip() or len(intent) > max_intent_chars:
        return None, f"intent must be a non-blank string of at most {max_intent_chars} characters"
    if not isinstance(evidence, list) or not evidence or not all(isinstance(e, str) and e.strip() for e in evidence):
        return None, "evidence must be a non-empty list of non-blank strings"
    return answer, None


def _elapsed(since: float) -> float:
    return round(perf_counter() - since, TIMING_DECIMALS)


def prompt_sha256(config: Any) -> str:
    """Identifies the inline prompt in run.json and every event (with ``prompt.name``)."""
    return sha256_text(config.prompt.system + "\n" + config.prompt.user)


def base_values(view: View, target: Target, max_intent_chars: int) -> dict[str, Any]:
    """The values every prompt and detector state may use besides the selected inputs."""
    aircraft = ", ".join(target.aircraft)
    base = {"step": view.step, "time_s": view.time_s, "target": target.label, "target_aircraft": aircraft}
    return {**base, "max_intent_chars": max_intent_chars}


def translation_prompt(config: Any, view: View, target: Target) -> Prompt:
    """The selected inputs at t fill the inline prompt; the answer schema is intent + evidence."""
    values = base_values(view, target, config.answer.max_intent_chars)
    values = {**values, **{name: INPUTS[name].render(view, target, opts) for name, opts in config.inputs.items()}}
    user = (input_text(fill(config.prompt.user, values)),)
    return Prompt(fill(config.prompt.system, values), user, "intent_translation", intent_schema())


class Pipeline:
    def __init__(self, loaded: Loaded, episode: Episode, journal: Journal, backend: Any, detector: Any = None) -> None:
        self.config, self.scenario, self.loaded = loaded.config, loaded.scenario, loaded
        self.episode, self.journal, self.backend, self.detector = episode, journal, backend, detector
        self.targets = [episode.target(target_id) for target_id in sorted(self.scenario.targets)]
        self.prompt_sha256 = prompt_sha256(self.config)
        self.translations: dict[str, Translation] = {}
        self.closed: set[str] = set()
        self.counts = {"translation": Counts(), "detector": Counts()}
        self.in_flight: str | None = None  # a call sent and not yet answered in responses.jsonl
        self.detector_options = loaded.detector_inputs() if self.config.change.source == "clef_flash" else {}

    def run(self) -> Summary:
        try:
            for step in range(self.episode.last_step + 1):
                self._step(step)
        except Stop as stop:
            return self._summary(stop.status, stop.error, stop.unsent)
        except (Exception, KeyboardInterrupt) as exc:  # a journal write failed, or Ctrl-C between calls
            status = "uncertain" if self.in_flight else "failed"
            return self._summary(status, {"call_id": self.in_flight, "type": type(exc).__name__, "message": WITHHELD})
        return self._summary("completed")

    def _summary(self, status: str, error: dict | None = None, unsent: dict | None = None) -> Summary:
        return Summary(status, dict(self.counts), self.journal.sequence, error, unsent)

    def _bump(self, layer: str, field: str) -> None:
        counts = self.counts[layer]
        self.counts = {**self.counts, layer: replace(counts, **{field: getattr(counts, field) + 1})}

    def _step(self, step: int) -> None:
        started = perf_counter()  # processing_s: from the start of step t (a question arrives) to the event
        rows, messages = self.episode.rows_until(step), tuple(m for m in self.scenario.messages if m.step <= step)
        for target in self.targets:
            down = self.episode.down_step(target)
            if target.id not in self.closed and down is not None and step >= down:
                self.closed = self.closed | {target.id}
                last = self.translations.get(target.id)
                body = {"reason": "target_down", "down_step": down, "last_event_id": last.event_id if last else None}
                self._event("target_closed", step, target, body)
        for target in self.targets:
            view = View(step, rows, messages, self.episode.teams, dict(self.translations))
            self._process(view, target, started)

    def _event(self, record_type: str, step: int, target: Target, body: dict[str, Any]) -> None:
        common = {"record_type": record_type, "scenario_id": self.scenario.id, "episode_id": self.episode.episode_id}
        self.journal.event({**common, "step": step, "time_s": time_s(step), "target": target.to_json(), **body})

    def _process(self, view: View, target: Target, started: float) -> None:
        change = self.config.change
        if target.id in self.closed:
            preset = CHANGE_SOURCES[change.source](view, target, change) if change.source in CHANGE_SOURCES else None
            reasons = sorted_reasons(forced(view, target) + ((preset,) if preset else ()))
            if reasons:
                down = self.episode.down_step(target)
                body = {"reason": "target_down", "down_step": down, "kinds": kinds_of(reasons)}
                self._event(
                    "trigger_suppressed", view.step, target, {**body, "trigger_reasons": [r.to_json() for r in reasons]}
                )
            return
        found = self._change(view, target)
        reasons = sorted_reasons(forced(view, target) + ((found,) if found else ()))
        if reasons:
            self._translate(view, target, reasons, started)

    def _change(self, view: View, target: Target) -> Reason | None:
        change = self.config.change
        if change.source in CHANGE_SOURCES:
            return CHANGE_SOURCES[change.source](view, target, change)
        if not is_tick(view.step, change.every_steps) or target.id not in view.translations:
            return None
        return self._detect(view, target)

    def _base(self, view: View, target: Target) -> dict[str, Any]:
        return base_values(view, target, self.config.answer.max_intent_chars)

    def _call(
        self, layer: str, call_id: str, view: View, target: Target, body: dict, send: Callable[[dict], Any]
    ) -> tuple[Any, str]:
        """Cap check, request line (fsynced), one call, response line. A failure stops the run."""
        limits = self.config.limits
        cap = limits.max_translations if layer == "translation" else limits.max_detector_calls
        if self.counts[layer].attempted >= cap:
            raise Stop("cap_reached", unsent={"layer": layer, "target": target.id, "step": view.step, "cap": cap})
        sha256 = request_sha256(body)
        record = {"call_id": call_id, "layer": layer, "step": view.step, "target": target.to_json()}
        self.journal.request({**record, "request_sha256": sha256, "body": body})
        self._bump(layer, "attempted")
        self.in_flight, sent = call_id, perf_counter()
        try:
            result = send(body)
        except (BackendError, DetectorError) as exc:
            self._failed(
                layer, call_id, sent, exc.uncertain, {"type": type(exc).__name__, "message": str(exc)}, exc.raw
            )
        except (Exception, KeyboardInterrupt) as exc:  # the request may have reached the server
            self._failed(
                layer, call_id, sent, True, {"type": f"unexpected:{type(exc).__name__}", "message": WITHHELD}, None
            )
        raw = result.raw if layer == "translation" else result
        self.journal.response(
            {"call_id": call_id, "layer": layer, "status": "ok", "latency_s": _elapsed(sent), "raw": raw, "error": None}
        )
        self.in_flight = None
        self._bump(layer, "completed")
        return result, sha256

    def _failed(
        self, layer: str, call_id: str, sent: float, uncertain: bool, error: dict[str, str], raw: Any
    ) -> NoReturn:
        status = "uncertain" if uncertain else "error"
        self.journal.response(
            {
                "call_id": call_id,
                "layer": layer,
                "status": status,
                "latency_s": _elapsed(sent),
                "raw": raw,
                "error": error,
            }
        )
        self.in_flight = None
        self._bump(layer, "uncertain" if uncertain else "failed")
        raise Stop(
            "uncertain" if uncertain else "failed",
            error={"layer": layer, "call_id": call_id, **error, "uncertain": uncertain},
        )

    def _shown(self, view: View, target: Target) -> dict[str, Any]:
        """The instructions in effect and the alarms that reached the target: what a human sees at t."""
        instructions = [
            {"instruction_id": m.id, "issue_step": m.step, "target": self.episode.target(to).to_json(), "text": m.text}
            for to, m in instructions_in_effect(view)
        ]
        alarms = [
            {
                "alarm_id": m.id,
                "step": m.step,
                "targets": [self.episode.target(to).to_json() for to in m.to],
                "text": m.text,
            }
            for m in alarms_for(view, target)
        ]
        return {"target_aircraft": list(target.aircraft), "instructions_in_effect": instructions, "alarms": alarms}

    def _material(self, view: View, target: Target) -> dict[str, Any]:
        """The question, the previous translation and the trajectory window: judge material, whatever the inputs."""
        asked, previous = question_at(view, target), view.translations.get(target.id)
        question = None if asked is None else {"request_id": asked.id, "step": asked.step, "text": asked.text}
        window = INPUTS["trajectory"].render(view, target, TrajectoryOptions())
        start = max(0, view.step - TrajectoryOptions().past_steps + 1)
        trajectory = {"start_step": start, "end_step": view.step, "text": window, "sha256": sha256_text(window)}
        if previous is None:
            return {"question": question, "previous_translation": None, "trajectory": trajectory}
        shown = {"event_id": previous.event_id, "step": previous.step, "intent": previous.intent}
        shown = {**shown, "evidence": list(previous.evidence)}
        return {"question": question, "previous_translation": shown, "trajectory": trajectory}

    def _detect(self, view: View, target: Target) -> Reason | None:
        change = self.config.change
        body = request_body(change, state_text(change, self.detector_options, view, target, self._base(view, target)))
        call_id = f"{self.scenario.id}:detector:{target.id}:{view.step:06d}"
        try:
            check_budget(change, body)  # before the request line: an over-long state is never sent
        except DetectorError as exc:
            raise Stop(
                "failed", error={"layer": "detector", "call_id": call_id, "type": "DetectorError", "message": str(exc)}
            ) from None
        raw, sha256 = self._call("detector", call_id, view, target, body, self.detector.decide)
        try:
            label, probabilities, usage = parse_decision(raw, change)
        except DetectorError as exc:
            raise Stop(
                "failed", error={"layer": "detector", "call_id": call_id, "type": "DetectorError", "message": str(exc)}
            ) from None
        triggered = label == "change" and probabilities["change"] >= change.threshold
        output = {"label": label, "probabilities": probabilities, "threshold": change.threshold, "triggered": triggered}
        inputs = {**self._shown(view, target), "state_text": body["state"], "question": body["questions"][QUESTION_ID]}
        model = {"model": change.model, "revision": change.revision, "response_model": raw.get("model")}
        body_record = {"call_id": call_id, "request_sha256": sha256, "input": inputs, "output": output}
        self._event(
            "detector_call", view.step, target, {**body_record, "raw_response": raw, "model": model, "usage": usage}
        )
        return Reason("change", "clef_flash", call_id, probabilities["change"]) if triggered else None

    def _translate(self, view: View, target: Target, reasons: tuple[Reason, ...], started: float) -> None:
        config, event_id = self.config, f"{self.scenario.id}:{target.id}:{view.step:06d}"
        body = type(self.backend).request_body(config.model, translation_prompt(config, view, target))
        reply, sha256 = self._call("translation", event_id, view, target, body, self.backend.send)
        answer, problem = parse_answer(reply.text, config.answer.max_intent_chars)
        if answer is None and config.answer.on_invalid == "stop":
            raise Stop(
                "failed",
                error={"layer": "translation", "call_id": event_id, "type": "InvalidAnswer", "message": problem},
            )
        record_input = {
            **self._shown(view, target),
            **self._material(view, target),
            "translator_inputs": list(config.inputs),
        }
        model = {
            "backend": config.model.backend,
            "name": config.model.name,
            "reasoning_effort": config.model.reasoning_effort,
        }
        model = {
            **model,
            "max_output_tokens": config.model.max_output_tokens,
            "response_model": reply.model,
            "response_id": reply.response_id,
        }
        body = {
            "event_id": event_id,
            "kinds": kinds_of(reasons),
            "trigger_reasons": [reason.to_json() for reason in reasons],
            "input": record_input,
            "output": answer or {"intent": "", "evidence": []},
            **({"answer_problem": problem} if problem else {}),
            "prompt": {"name": self.loaded.prompt_name, "sha256": self.prompt_sha256},
            "model": model,
            "usage": reply.usage,
            "request_sha256": sha256,
        }
        self._event("translation_event", view.step, target, {**body, "processing_s": _elapsed(started)})
        if answer is not None:
            translation = Translation(event_id, view.step, answer["intent"], tuple(answer["evidence"]))
            self.translations = {**self.translations, target.id: translation}
