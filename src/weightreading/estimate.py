"""Offline token and cost preview (no network: tiktoken would download its encoding on first use).

Token counts are a chars/4 approximation, labelled as such. Two figures are reported:
``worst_case`` (every turn uses ``max_output_tokens`` and a full stdout cap, all replayed, up to
``max_runs``; capped by ``max_total_tokens`` plus one turn of overshoot) and ``typical`` (blog
Exp 3 range: 10 runs, moderate outputs). Real usage is logged per turn in ``transcript.jsonl``.
"""

from __future__ import annotations

import math

from weightreading.config import RunConfig

CHARS_PER_TOKEN = 4.0
TYPICAL_RUNS = 10
TYPICAL_OUTPUT_TOKENS_PER_TURN = 6000  # code + reasoning, assumed
TYPICAL_RESULT_CHARS = 4000  # printed output per run, assumed
MESSAGE_OVERHEAD_CHARS = 400  # run header, fences, stderr label
TOKENS_PER_MILLION = 1_000_000


def approx_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def _session_tokens(first: int, turns: int, output_per_turn: int, result_per_turn: int) -> tuple[int, int]:
    """(billed input, output) when every turn replays the whole history; turn k adds k growth steps."""
    growth = output_per_turn + result_per_turn
    input_tokens = turns * first + growth * turns * (turns - 1) // 2
    return input_tokens, turns * output_per_turn


def _cost(config: RunConfig, input_tokens: int, output_tokens: int) -> float | None:
    price = config.price()
    if price is None:
        return None
    return round((input_tokens * price.input + output_tokens * price.output) / TOKENS_PER_MILLION, 2)


def preview_estimate(message_chars: int, config: RunConfig) -> dict:
    first = approx_tokens(message_chars)
    limits = config.limits
    sandbox = config.sandbox
    worst_result = approx_tokens(sandbox.stdout_cap_chars + sandbox.stderr_tail_chars + MESSAGE_OVERHEAD_CHARS)
    # Each run (and the final answer, and the one budget notice) may be preceded by up to
    # max_protocol_errors - 1 malformed replies; max_total_tokens is what bounds this in practice.
    worst_turns = (limits.max_runs + 2) * limits.max_protocol_errors
    worst_in, worst_out = _session_tokens(first, worst_turns, limits.max_output_tokens, worst_result)
    # The loop stops after the turn that crosses max_total_tokens. That turn's input is the history
    # before it (< max_total_tokens) plus one run result, and its output is at most max_output_tokens.
    overshoot = limits.max_total_tokens + worst_result + limits.max_output_tokens
    cap = limits.max_total_tokens + overshoot
    capped_total = min(worst_in + worst_out, cap)
    worst_ratio = capped_total / (worst_in + worst_out)
    capped_in, capped_out = round(worst_in * worst_ratio), round(worst_out * worst_ratio)
    typical_result = approx_tokens(TYPICAL_RESULT_CHARS + MESSAGE_OVERHEAD_CHARS)
    typical_in, typical_out = _session_tokens(first, TYPICAL_RUNS + 1, TYPICAL_OUTPUT_TOKENS_PER_TURN, typical_result)
    return {
        "token_method": f"approximate: chars / {CHARS_PER_TOKEN:g} (no tokenizer download)",
        "first_message_chars": message_chars,
        "first_message_tokens_approx": first,
        "model": config.model,
        "price_usd_per_1m": None if config.price() is None else vars(config.price()),
        "worst_case": {
            "turns": worst_turns,
            "input_tokens": capped_in,
            "output_tokens": capped_out,
            "cost_usd": _cost(config, capped_in, capped_out),
            "capped_by_max_total_tokens": capped_total < worst_in + worst_out,
        },
        "typical": {
            "assumptions": {
                "runs": TYPICAL_RUNS,
                "output_tokens_per_turn": TYPICAL_OUTPUT_TOKENS_PER_TURN,
                "result_chars_per_run": TYPICAL_RESULT_CHARS,
            },
            "input_tokens": typical_in,
            "output_tokens": typical_out,
            "cost_usd": _cost(config, typical_in, typical_out),
        },
        "note": "Cached-input discounts are ignored (upper bound). Verify prices before a paid run.",
    }
