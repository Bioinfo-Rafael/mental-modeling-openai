"""Method A request building on top of the (read-only) Translator: prompts, estimates, provenance.

Only public Translator functions are called: ``load_episode`` (hash-verified CSV),
``episode_facts`` (down steps), ``render_lag17_jsonl`` (the 60-step window), and
``TranslationModelRequest`` / ``build_prompt`` / ``build_openai_request`` / ``build_responses_body``
(the exact prompt and request body). The pipeline, the change detector and its vLLM health check
are bypassed: the request steps are fixed by ``compare.yaml``, so the runner calls the
translation layer directly. Nothing in ``src/mtd_rl_xavier`` is modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mtd_rl_xavier import paths
from mtd_rl_xavier.intent_compare import AnswerLimits, CompareConfig, limits_sentence, payload_problem
from mtd_rl_xavier.lag.dataset import load_episode
from mtd_rl_xavier.lag.rendering import render_lag17_jsonl
from mtd_rl_xavier.llm.openai_responses import API_BASE_URL
from mtd_rl_xavier.llm.retry import DEFAULT_RETRY_POLICY
from mtd_rl_xavier.translator.pipeline import EpisodeFacts, episode_facts
from mtd_rl_xavier.translator.schema import TARGET_AIRCRAFT, Instruction, PreviousTranslation, TargetRef
from mtd_rl_xavier.translator.translation import (
    PROMPT_VERSION,
    TranslationModelRequest,
    TranslationPrompt,
    build_openai_request,
    build_prompt,
)

# The Translator's language mechanism is the ``output_language`` argument of ``build_prompt`` /
# ``TranslationLayer`` (the "{output_language}" slot of its system prompt). Its run.yaml also sets a
# value (now Japanese), but the comparison must not depend on it: ``compare.yaml`` decides.
TRANSLATOR_LANGUAGE_NAMES = {"en": "English", "ja": "Japanese"}
REPO_ROOT = paths.ROOT
TRANSLATOR_RUN_CONFIG = REPO_ROOT / "configs" / "translator" / "run.yaml"
PRICING_CONFIG = REPO_ROOT / "configs" / "intent_compare" / "pricing.yaml"
EPISODE_ALIAS = "episode_01"  # neutral alias; the only episode name the model sees
INSTRUCTION_ID_PREFIX = "compare-neutral"
INSTRUCTIONS_JOINER = "\n\n"  # between the Translator instructions and the appended limits sentence
TOKENIZER = "o200k_base"
CHARS_PER_TOKEN = 4  # the Translator CLI's own approximation
EVIDENCE_ALLOWANCE_CHARS = 1000  # assumed size of the previous translation's evidence list
TOKENS_PER_MILLION = 1_000_000
PROVENANCE_PACKAGES = ("translator", "lag", "llm")
# The same default policy as method B (configs/weightreading/intent_b.yaml has no `retry` override).
RETRY_POLICY = DEFAULT_RETRY_POLICY
OWN_SCRIPTS = (
    "src/mtd_rl_xavier/intent_compare/config.py",
    "src/mtd_rl_xavier/intent_compare/method_a_requests.py",
    "src/mtd_rl_xavier/intent_compare/method_a_cli.py",
    "src/mtd_rl_xavier/intent_compare/convert.py",
)


@dataclass(frozen=True)
class TranslationSettings:
    """A-only settings: language and length limits from compare.yaml, timeout from the Translator run config.

    The Translator's prompt only carries ``max_intent_chars``. When any limit differs from the defaults,
    ``prompt_for`` appends method B's length-limit sentence (``limits_sentence``) to the instructions, and
    ``answer_problem`` applies method B's validation rules to every parsed answer.
    """

    output_language: str
    max_intent_chars: int
    max_output_tokens: int
    timeout_s: float
    max_evidence_items: int | None = None
    max_evidence_chars: int | None = None

    @property
    def answer_limits(self) -> AnswerLimits:
        return AnswerLimits(self.max_intent_chars, self.max_evidence_items, self.max_evidence_chars)

    @property
    def limits_sentence(self) -> str:
        """Sentence appended to A's instructions; ``""`` (nothing appended) when the limits are the defaults."""
        return limits_sentence(self.answer_limits)


@dataclass(frozen=True)
class Slot:
    """One (step, aircraft) of the comparison; ``skipped`` is set when the target is down."""

    step: int
    aircraft: str
    skipped: str | None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_settings(config: CompareConfig, max_output_tokens: int) -> TranslationSettings:
    translation = yaml.safe_load(TRANSLATOR_RUN_CONFIG.read_text(encoding="utf-8"))["translation"]
    return TranslationSettings(
        output_language=TRANSLATOR_LANGUAGE_NAMES[config.output_language],
        max_intent_chars=config.answer_limits.max_intent_chars,
        max_output_tokens=max_output_tokens,
        timeout_s=float(translation["timeout_s"]),
        max_evidence_items=config.answer_limits.max_evidence_items,
        max_evidence_chars=config.answer_limits.max_evidence_chars,
    )


def answer_problem(intent: str, evidence: tuple[str, ...], settings: TranslationSettings) -> str | None:
    """Method B's validation of a parsed answer (characters, evidence count / length); None = valid."""
    return payload_problem({"intent": intent, "evidence": list(evidence)}, settings.answer_limits)


def load_pricing(model: str) -> dict[str, Any] | None:
    """USD per 1M tokens from ``configs/intent_compare/pricing.yaml`` (preview estimate only)."""
    pricing = yaml.safe_load(PRICING_CONFIG.read_text(encoding="utf-8")).get("pricing", {})
    return pricing.get(model)


def load_frame(config: CompareConfig) -> tuple[pd.DataFrame, EpisodeFacts]:
    frame = load_episode(config.episode_csv)  # verifies the CSV against manifest.json
    return frame, episode_facts(frame)


def instructions(config: CompareConfig) -> dict[str, Instruction]:
    """One instruction per target aircraft at step 0, all with the same neutral text."""
    return {
        target: Instruction(
            f"{INSTRUCTION_ID_PREFIX}-{target}", 0, TargetRef(TARGET_AIRCRAFT, target), config.instruction
        )
        for target in config.targets
    }


def plan_slots(config: CompareConfig, facts: EpisodeFacts) -> list[Slot]:
    """Chronological slots; a target with ``alive == 0`` at or before the step is skipped."""
    slots = []
    for step in config.steps:
        if step > facts.last_step:
            raise ValueError(f"step {step} is after the last complete step {facts.last_step}")
        for aircraft in config.targets:
            if aircraft not in facts.down_steps:
                raise ValueError(f"unknown target aircraft {aircraft!r}")
            down = facts.down_steps[aircraft]
            reason = None if down is None or step < down else f"target_down: alive == 0 since step {down}"
            slots.append(Slot(step, aircraft, reason))
    return slots


def build_request(
    config: CompareConfig,
    frame: pd.DataFrame,
    slot: Slot,
    previous: PreviousTranslation | None,
) -> TranslationModelRequest:
    trajectory = render_lag17_jsonl(
        frame,
        start_step=config.window_start(slot.step),
        end_step=slot.step,
        anchor_step=slot.step,
        episode_alias=EPISODE_ALIAS,
    )
    in_effect = instructions(config)
    return TranslationModelRequest(
        target=TargetRef(TARGET_AIRCRAFT, slot.aircraft),
        target_aircraft=(slot.aircraft,),
        step=slot.step,
        episode_alias=EPISODE_ALIAS,
        instructions_in_effect=tuple(in_effect[t] for t in sorted(in_effect)),
        primary_instruction_id=in_effect[slot.aircraft].instruction_id,
        trajectory=trajectory,
        previous=previous,
        alarms=(),  # the comparison issues no alarms: method B sees none either
        question=None,  # one neutral instruction for both methods, no per-step question
    )


def prompt_for(request: TranslationModelRequest, settings: TranslationSettings) -> TranslationPrompt:
    """The Translator prompt, plus B's length-limit sentence only when a limit differs from the defaults."""
    prompt = build_prompt(request, output_language=settings.output_language, max_intent_chars=settings.max_intent_chars)
    sentence = settings.limits_sentence
    if not sentence:
        return prompt
    return replace(prompt, instructions=f"{prompt.instructions}{INSTRUCTIONS_JOINER}{sentence}")


def request_body(prompt: TranslationPrompt, config: CompareConfig, settings: TranslationSettings) -> dict[str, Any]:
    from mtd_rl_xavier.llm.openai_responses import build_responses_body

    request = build_openai_request(
        prompt,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        max_output_tokens=settings.max_output_tokens,
    )
    return build_responses_body(request)


def count_tokens(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding(TOKENIZER).encode(text))


def estimate_call(prompt: TranslationPrompt, settings: TranslationSettings, chained: bool) -> dict[str, Any]:
    """Input-token estimate of one call; a chained call adds an allowance for the previous translation."""
    allowance_chars = settings.max_intent_chars + EVIDENCE_ALLOWANCE_CHARS if chained else 0
    text = prompt.instructions + prompt.user_text
    tokens = count_tokens(text)
    allowance_tokens = -(-allowance_chars // CHARS_PER_TOKEN)
    return {
        "chars": len(text),
        "tokens_chars_div_4": -(-len(text) // CHARS_PER_TOKEN),
        f"tokens_{TOKENIZER}": tokens,
        "previous_translation_allowance_tokens": allowance_tokens,
        "input_tokens_estimate": tokens + allowance_tokens,
    }


def cost_summary(estimates: list[dict[str, Any]], pricing: dict[str, Any] | None, max_out: int) -> dict[str, Any]:
    calls = len(estimates)
    input_tokens = sum(e["input_tokens_estimate"] for e in estimates)
    if pricing is None:
        return {"calls": calls, "input_tokens": input_tokens, "pricing": None, "note": "no price found in repo"}
    input_usd = input_tokens * float(pricing["input"]) / TOKENS_PER_MILLION
    output_cap_usd = calls * max_out * float(pricing["output"]) / TOKENS_PER_MILLION
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "pricing_usd_per_1m": pricing,
        "pricing_source": str(PRICING_CONFIG.relative_to(REPO_ROOT)),
        "input_cost_usd": round(input_usd, 4),
        "output_cost_upper_bound_usd": round(output_cap_usd, 4),
        "total_upper_bound_usd": round(input_usd + output_cap_usd, 4),
        "note": "output billed at actual usage (reasoning + visible); the upper bound assumes every call uses "
        "max_output_tokens. No cached-input discount assumed.",
    }


def provenance(config: CompareConfig, frame: pd.DataFrame, settings: TranslationSettings) -> dict[str, Any]:
    """SHA-256 of every Translator source file used (uncommitted, may change), configs and the episode."""
    package_root = REPO_ROOT / "src" / "mtd_rl_xavier"
    code = {
        f"src/mtd_rl_xavier/{package}/{path.name}": sha256_file(path)
        for package in PROVENANCE_PACKAGES
        for path in sorted((package_root / package).glob("*.py"))
    }
    return {
        "translator_sources_sha256": code,
        "own_scripts_sha256": {name: sha256_file(REPO_ROOT / name) for name in OWN_SCRIPTS},
        "compare_config": {"path": str(config.source_path), "sha256": sha256_file(config.source_path)},
        "translator_run_config": {"path": str(TRANSLATOR_RUN_CONFIG), "sha256": sha256_file(TRANSLATOR_RUN_CONFIG)},
        "episode_csv": {"path": str(config.episode_csv), "sha256": frame.attrs.get("csv_sha256")},
        "prompt_version": PROMPT_VERSION,
        "episode_alias": EPISODE_ALIAS,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "settings": settings.__dict__,
        "answer_limits": asdict(settings.answer_limits),
        "limits_sentence": settings.limits_sentence,
        "limits_sentence_sha256": sha256_text(settings.limits_sentence),
        "call_path": "render_lag17_jsonl -> TranslationModelRequest -> build_prompt -> OpenAIIntentModel "
        "(OpenAIResponsesClient over RetryingClient) -> parse_intent_output; Translator pipeline / detector bypassed",
        "openai_base_url": API_BASE_URL,
        "retry_policy": asdict(RETRY_POLICY),
    }


def json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=False, allow_nan=False)
