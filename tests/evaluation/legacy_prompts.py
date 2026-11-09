"""The legacy assess judge prompt the assess tests assert on (assess-judge-v2)."""

from __future__ import annotations

from evaluation.prompt_files import ASSESS_PROMPT_SPEC, load_judge_prompt
from translate import paths

PROMPTS = paths.ROOT / "configs" / "evaluation" / "prompts"
ASSESS_PROMPT = load_judge_prompt(PROMPTS / "assess-judge-v2.yaml", ASSESS_PROMPT_SPEC)
ASSESS_PROMPT_VERSION = ASSESS_PROMPT.version
