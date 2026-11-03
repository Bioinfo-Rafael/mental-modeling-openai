"""Run configuration (``configs/weightreading/run.yaml``): strict loading, no hidden defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

BACKENDS = ("openai-responses",)
TASK_MODES = ("shown", "blind")
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
TOP_LEVEL_KEYS = {"backend", "model", "reasoning_effort", "task_mode", "limits", "sandbox", "pricing"}
LIMIT_KEYS = {"max_runs", "max_output_tokens", "max_total_tokens", "max_protocol_errors"}
SANDBOX_KEYS = {
    "python",
    "timeout_s",
    "memory_bytes",
    "file_size_bytes",
    "stdout_cap_chars",
    "stderr_tail_chars",
    "tmp_size_mb",
    "threads",
    "work_max_bytes",
}
PRICE_KEYS = {"input", "cached_input", "output"}


class ConfigError(ValueError):
    """The run configuration is malformed."""


@dataclass(frozen=True)
class Limits:
    max_runs: int
    max_output_tokens: int
    max_total_tokens: int
    max_protocol_errors: int


@dataclass(frozen=True)
class SandboxConfig:
    python: Path
    timeout_s: int
    memory_bytes: int
    file_size_bytes: int
    stdout_cap_chars: int
    stderr_tail_chars: int
    tmp_size_mb: int
    threads: int
    work_max_bytes: int


@dataclass(frozen=True)
class Price:
    input: float
    cached_input: float
    output: float


@dataclass(frozen=True)
class RunConfig:
    backend: str
    model: str
    reasoning_effort: str
    task_mode: str
    limits: Limits
    sandbox: SandboxConfig
    pricing: tuple[tuple[str, Price], ...]

    def price(self) -> Price | None:
        return dict(self.pricing).get(self.model)


def _section(raw: dict, key: str, allowed: set[str]) -> dict:
    section = raw.get(key)
    if not isinstance(section, dict):
        raise ConfigError(f"config section {key!r} is missing")
    if set(section) != allowed:
        raise ConfigError(f"config section {key!r} must have exactly {sorted(allowed)}")
    return section


def _positive_ints(section: dict, keys: set[str], name: str) -> dict[str, int]:
    values = {}
    for key in keys:
        value = section[key]
        if type(value) is not int or value <= 0:
            raise ConfigError(f"{name}.{key} must be a positive integer")
        values[key] = value
    return values


def _choice(raw: dict, key: str, choices: tuple[str, ...]) -> str:
    value = raw.get(key)
    if value not in choices:
        raise ConfigError(f"{key} must be one of {list(choices)}")
    return value


def _pricing(raw: object) -> tuple[tuple[str, Price], ...]:
    if not isinstance(raw, dict):
        raise ConfigError("pricing must be a mapping of model id -> prices")
    entries = []
    for model, prices in raw.items():
        if not isinstance(prices, dict) or set(prices) != PRICE_KEYS:
            raise ConfigError(f"pricing.{model} must have exactly {sorted(PRICE_KEYS)}")
        if not all(isinstance(v, int | float) and v >= 0 for v in prices.values()):
            raise ConfigError(f"pricing.{model} values must be non-negative numbers")
        entries.append((str(model), Price(**{k: float(v) for k, v in prices.items()})))
    return tuple(entries)


def parse_config(raw: object, root: Path) -> RunConfig:
    """Validate a decoded YAML mapping. Relative sandbox.python is resolved against ``root``."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    unknown = sorted(set(raw) - TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"unknown config keys: {unknown}")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("model must be set explicitly in the config")
    limits = Limits(**_positive_ints(_section(raw, "limits", LIMIT_KEYS), LIMIT_KEYS, "limits"))
    sandbox_raw = _section(raw, "sandbox", SANDBOX_KEYS)
    python = sandbox_raw["python"]
    if not isinstance(python, str) or not python:
        raise ConfigError("sandbox.python must be a path")
    numbers = _positive_ints(sandbox_raw, SANDBOX_KEYS - {"python"}, "sandbox")
    sandbox = SandboxConfig(python=root / python, **numbers)
    return RunConfig(
        backend=_choice(raw, "backend", BACKENDS),
        model=model.strip(),
        reasoning_effort=_choice(raw, "reasoning_effort", REASONING_EFFORTS),
        task_mode=_choice(raw, "task_mode", TASK_MODES),
        limits=limits,
        sandbox=sandbox,
        pricing=_pricing(raw.get("pricing", {})),
    )


def load_config(path: Path, root: Path) -> RunConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config is not valid YAML: {exc}") from exc
    return parse_config(raw, root)
