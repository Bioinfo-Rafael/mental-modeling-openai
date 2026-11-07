"""Token-estimate constants shared by method B's preview (``intent_estimate.py``).

Token counts are a chars/4 approximation (no network: tiktoken would download its encoding on first
use), labelled as such. Real usage is logged per turn in ``transcript.jsonl``.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4.0
TYPICAL_OUTPUT_TOKENS_PER_TURN = 6000  # code + reasoning, assumed
TYPICAL_RESULT_CHARS = 4000  # printed output per run, assumed
MESSAGE_OVERHEAD_CHARS = 400  # run header, fences, stderr label
TOKENS_PER_MILLION = 1_000_000


def approx_tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)
