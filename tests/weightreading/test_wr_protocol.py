"""Reply protocol: block merging and final-answer validation (pure functions, no I/O)."""

from __future__ import annotations

import pytest

from weightreading.protocol import FINAL_SECTIONS, KIND_CODE, KIND_ERROR, KIND_FINAL, final_answer_problems, parse_reply

SECTIONS = "\n".join(f"## {name}\ntext" for name, _ in FINAL_SECTIONS)
WELL_FORMED = (
    SECTIONS + '\n```json\n{"claims": [{"input": "x0", "head": "head_0", "direction": "+", "confidence": 0.5}]}\n```'
)


def test_parse_reply_two_blocks_merges_in_order_with_separator() -> None:
    parsed = parse_reply("```python\na = 1\n```\n```python\nb = 2\n```\n```python\nc = 3\n```")

    assert parsed.kind == KIND_CODE
    assert parsed.block_count == 3
    assert parsed.code == "a = 1\n\n# --- block 2 ---\nb = 2\n\n# --- block 3 ---\nc = 3\n"


def test_final_answer_problems_well_formed_returns_empty() -> None:
    assert final_answer_problems(WELL_FORMED) == ()


def test_final_answer_problems_lists_each_missing_section() -> None:
    text = '## Summary\nx\n```json\n{"claims": []}\n```'

    problems = final_answer_problems(text)

    assert [p for p in problems if "section heading" in p] == [
        f"section heading '## {name}'" for name, _ in FINAL_SECTIONS if name != "Summary"
    ]


@pytest.mark.parametrize(
    ("json_part", "expected"),
    [
        ("", "found 0"),
        ('```json\n{"claims": []}\n```\n```json\n{"claims": []}\n```', "found 2"),
        ("```json\n{not json}\n```", "not valid JSON"),
        ('```json\n{"other": []}\n```', '"claims" list'),
        ('```json\n{"claims": "none"}\n```', '"claims" list'),
        ("```json\n[1, 2]\n```", '"claims" list'),
    ],
)
def test_final_answer_problems_bad_json_block_is_reported(json_part: str, expected: str) -> None:
    problems = final_answer_problems(SECTIONS + "\n" + json_part)

    assert len(problems) == 1
    assert expected in problems[0]


def test_parse_reply_headings_are_case_insensitive_and_level_agnostic() -> None:
    text = "# observations\n### INPUTS\n## Heads (x)\n## Recurrence\n## Summary\n" + '```json\n{"claims": []}\n```'

    assert parse_reply(text).kind == KIND_FINAL


def test_parse_reply_error_carries_problems() -> None:
    parsed = parse_reply("I'm unable to analyze the network.")

    assert parsed.kind == KIND_ERROR
    assert parsed.problems
