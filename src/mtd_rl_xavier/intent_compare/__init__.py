"""The intent-translation comparison: shared contract, method A, the viewer and the converters.

``config`` holds the shared ``compare.yaml`` contract and answer limits (also used by method B in
``weightreading``); its public names are re-exported here so ``from mtd_rl_xavier.intent_compare import
CompareConfig`` keeps working. Entry points (``python -m``): ``method_a_cli`` (method A), ``viewer``,
``convert`` and ``clips`` (episode preparation), ``to_translator_run`` (one method A call -> a
Translator v2 run directory for the judge; kept byte-identical, its SHA-256 is recorded in that run).
"""

from mtd_rl_xavier.intent_compare.config import (
    CLIPS_DIR_NAME,
    CLIPS_MANIFEST,
    DEFAULT_CONFIG,
    DEFAULT_LANGUAGE,
    DEFAULT_LIMITS,
    EPISODE_CSV_TEMPLATE,
    EPISODE_DIR_NAME,
    MAX_INTENT_CHARS,
    METHODS,
    OUTPUT_LANGUAGES,
    REQUIRED_KEYS,
    RUN_ID_PATTERN,
    TIME_DECIMALS,
    AnswerLimits,
    CompareConfig,
    CompareConfigError,
    check_run_id,
    clip_bounds,
    limits_sentence,
    load_compare,
    parse_compare,
    payload_problem,
)

__all__ = [
    "CLIPS_DIR_NAME",
    "CLIPS_MANIFEST",
    "DEFAULT_CONFIG",
    "DEFAULT_LANGUAGE",
    "DEFAULT_LIMITS",
    "EPISODE_CSV_TEMPLATE",
    "EPISODE_DIR_NAME",
    "MAX_INTENT_CHARS",
    "METHODS",
    "OUTPUT_LANGUAGES",
    "REQUIRED_KEYS",
    "RUN_ID_PATTERN",
    "TIME_DECIMALS",
    "AnswerLimits",
    "CompareConfig",
    "CompareConfigError",
    "check_run_id",
    "clip_bounds",
    "limits_sentence",
    "load_compare",
    "parse_compare",
    "payload_problem",
]
