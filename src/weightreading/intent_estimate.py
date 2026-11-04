"""Offline token and cost bound of one intent-B session (chars / 4 approximation, no tokenizer).

Two figures per session, both under the session's token budget:

* ``hard_cap``: the most a session can bill. A request is only sent while the session total is
  below ``max_total_tokens``, so the requests before the last bill < ``max_total_tokens`` and the
  last one bills at most ``max_total_tokens + one run result`` of input plus ``max_output_tokens``
  (a run result is bounded at one token per character of its stdout/stderr caps).
  All earlier tokens are priced at the dearer of the input/output prices (strict upper bound).
* ``expected``: the blog Exp-3 style turn sizes of ``estimate.py``, replayed until the soft budget
  (``answer_at_total_tokens``) asks for the final answer, plus that final turn.

Prices from the run config; every request stays below the long-context threshold by construction
of ``max_total_tokens`` (checked here and reported).
"""

from __future__ import annotations

from weightreading.config import RunConfig
from weightreading.estimate import (
    CHARS_PER_TOKEN,
    MESSAGE_OVERHEAD_CHARS,
    TOKENS_PER_MILLION,
    TYPICAL_OUTPUT_TOKENS_PER_TURN,
    TYPICAL_RESULT_CHARS,
    approx_tokens,
)

# gpt-6.1-sol bills a request above this many input tokens at a higher rate
# (evaluation/config/judge-text.yaml `long_context_threshold_tokens`).
LONG_CONTEXT_THRESHOLD_TOKENS = 272_000


def _usd(tokens: int, price_per_million: float) -> float:
    return tokens * price_per_million / TOKENS_PER_MILLION


def _expected(first: int, config: RunConfig, answer_at: int) -> dict:
    result = approx_tokens(TYPICAL_RESULT_CHARS + MESSAGE_OVERHEAD_CHARS)
    history, billed_in, billed_out, turns = first, 0, 0, 0
    while turns < config.limits.max_runs and billed_in + billed_out < answer_at:
        billed_in, billed_out, turns = billed_in + history, billed_out + TYPICAL_OUTPUT_TOKENS_PER_TURN, turns + 1
        history += TYPICAL_OUTPUT_TOKENS_PER_TURN + result
    billed_in, billed_out = billed_in + history, billed_out + TYPICAL_OUTPUT_TOKENS_PER_TURN  # final answer
    price = config.price()
    cost = None if price is None else round(_usd(billed_in, price.input) + _usd(billed_out, price.output), 2)
    return {"runs": turns, "input_tokens": billed_in, "output_tokens": billed_out, "cost_usd": cost}


def session_estimate(message_chars: int, config: RunConfig, answer_at: int) -> dict:
    """Token/cost bound for one session whose first message has ``message_chars`` characters."""
    first = approx_tokens(message_chars)
    limits, sandbox = config.limits, config.sandbox
    # Upper bound at one token per character of the stdout/stderr caps (printed ASCII never needs more).
    result = sandbox.stdout_cap_chars + sandbox.stderr_tail_chars + MESSAGE_OVERHEAD_CHARS
    last_input = limits.max_total_tokens + result
    price = config.price()
    hard_cost = None
    if price is not None:
        dearer = max(price.input, price.output)
        hard_cost = round(
            _usd(limits.max_total_tokens, dearer)
            + _usd(last_input, price.input)
            + _usd(limits.max_output_tokens, price.output),
            2,
        )
    return {
        "token_method": f"approximate: chars / {CHARS_PER_TOKEN:g} (no tokenizer download)",
        "first_message_chars": message_chars,
        "first_message_tokens_approx": first,
        "model": config.model,
        "price_usd_per_1m": None if price is None else vars(price),
        "answer_requested_at_total_tokens": answer_at,
        "hard_cap": {
            "billed_tokens": limits.max_total_tokens + last_input + limits.max_output_tokens,
            "max_request_input_tokens": last_input,
            "below_long_context_threshold": last_input < LONG_CONTEXT_THRESHOLD_TOKENS,
            "cost_usd": hard_cost,
        },
        "expected": _expected(first, config, answer_at),
        "note": "Cached-input discounts are ignored. Verify prices before a paid run.",
    }
