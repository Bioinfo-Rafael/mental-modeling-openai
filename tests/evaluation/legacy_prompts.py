"""The legacy judge prompts the judge tests assert on (judge-text-v3 and assess-judge-v2).

They were the in-code prompts before the prompt texts moved into ``configs/evaluation/prompts/``;
the tests keep checking them through the request. Headings and the key-point rubric are literals, so
finding them in a rendered request also checks the prompt file; versions and the system prompt are
read from the files.
"""

from __future__ import annotations

from evaluation.prompt_files import ASSESS_PROMPT_SPEC, KEY_POINT_PROMPT_SPEC, load_judge_prompt
from translate import paths

PROMPTS = paths.ROOT / "configs" / "evaluation" / "prompts"
KEY_POINT_PROMPT = load_judge_prompt(PROMPTS / "judge-text-v3.yaml", KEY_POINT_PROMPT_SPEC)
ASSESS_PROMPT = load_judge_prompt(PROMPTS / "assess-judge-v2.yaml", ASSESS_PROMPT_SPEC)

JUDGE_PROMPT_VERSION = KEY_POINT_PROMPT.version
JUDGE_SYSTEM_PROMPT = KEY_POINT_PROMPT.texts["system"]
ASSESS_PROMPT_VERSION = ASSESS_PROMPT.version

HEADING_INSTRUCTIONS = "## INSTRUCTIONS IN EFFECT"
HEADING_VIDEO = "## VIDEO FRAMES"
HEADING_TIMESTAMPS = "## TIMESTAMPS"
HEADING_TRAJECTORY = "## TRAJECTORY"
HEADING_TRANSLATED_INTENT = "## TRANSLATED INTENT"
HEADING_KEY_POINTS = "## KEY POINTS"
WINDOW_BEGIN = "BEGIN WINDOW DATA"
WINDOW_END = "END WINDOW DATA"
KEY_POINT_RUBRIC = """\
Reference key points prepared by a human rater for this event. Score each one:
2 = the translated intent reproduces the key point's core purpose and its main action;
1 = it reproduces the key point only partially;
0 = it misses the key point, or it states a different purpose.
Use the id in brackets as key_point_id."""
