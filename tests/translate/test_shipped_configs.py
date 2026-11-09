"""The shipped run configurations: they load, and their language and limit variants differ only where documented."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from translate import paths
from translate.config import load_config

TRANSLATE = paths.CONFIGS / "translate"
RUN_CONFIGS = sorted(TRANSLATE.glob("*.yaml"))
LIMIT_KEYS = ("max_intent_chars", "max_evidence_items", "max_evidence_chars")


def raw(name: str) -> dict[str, Any]:
    return yaml.safe_load((TRANSLATE / name).read_text(encoding="utf-8"))


def without(config: dict[str, Any], *dotted: str) -> dict[str, Any]:
    """A copy without the given ``section.key`` entries."""
    result = {key: dict(value) if isinstance(value, dict) else value for key, value in config.items()}
    result["prompt"] = {**result["prompt"], "vars": dict(result["prompt"]["vars"])}
    for path in dotted:
        *parents, last = path.split(".")
        node = result
        for key in parents:
            node = node[key]
        node.pop(last)
    return result


def test_shipped_configs_are_found() -> None:
    assert {path.name for path in RUN_CONFIGS} >= {
        "online.example.yaml",
        "seed100601-trajectory.yaml",
        "seed100601-trajectory-ja.yaml",
        "seed100601-trajectory-ja-short.yaml",
        "seed100601-policy.yaml",
        "seed100601-policy-ja.yaml",
        "seed100601-policy-ja-short.yaml",
    }


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda path: path.name)
def test_shipped_config_loads(path: Path) -> None:
    config = load_config(path)

    assert config.prompt.version in {"translator-intent-v2", "intent-b-v1"}
    assert config.scenario_path is not None and config.scenario_path.is_file()


@pytest.mark.parametrize("kind, language_var", [("trajectory", "output_language"), ("policy", "language_instruction")])
def test_ja_variant_differs_only_in_the_language_var(kind: str, language_var: str) -> None:
    english, japanese = raw(f"seed100601-{kind}.yaml"), raw(f"seed100601-{kind}-ja.yaml")

    assert without(english, f"prompt.vars.{language_var}") == without(japanese, f"prompt.vars.{language_var}")
    assert english["prompt"]["vars"][language_var] != japanese["prompt"]["vars"][language_var]


@pytest.mark.parametrize("kind, limits_var", [("trajectory", "limits_paragraph"), ("policy", "limits_line")])
def test_ja_short_variant_differs_only_in_the_limits_and_their_wording(kind: str, limits_var: str) -> None:
    japanese, short = raw(f"seed100601-{kind}-ja.yaml"), raw(f"seed100601-{kind}-ja-short.yaml")
    dropped = (f"prompt.vars.{limits_var}", *(f"answer.{key}" for key in LIMIT_KEYS))

    assert without(japanese, *dropped) == without(short, *dropped)
    assert [short["answer"][key] for key in LIMIT_KEYS] == [200, 4, 80]
    assert (
        "intent at most 200 characters; at most 4 evidence items, each at most 80 characters"
        in short["prompt"]["vars"][limits_var]
    )
    assert japanese["prompt"]["vars"][limits_var] == ""
