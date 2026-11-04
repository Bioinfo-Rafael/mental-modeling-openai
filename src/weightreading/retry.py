"""Narrow, logged retry for transient OpenAI server errors.

A request is retried only when it almost certainly was not processed (so it was not billed):
an HTTP 5xx answer, or a connection failure that is not a timeout. A timeout may have been
processed and billed, so it is never retried; neither are 4xx answers (including 429, which also
covers insufficient quota), bad requests, authentication errors or any non-``openai`` exception.
The SDK's own retries stay off (``SDK_MAX_RETRIES = 0``) so that every attempt is visible here.
The SDK is imported lazily, only after an exception has been raised.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TypeVar

DEFAULT_MAX_RETRIES = 2  # 3 attempts in total
DEFAULT_INITIAL_BACKOFF_S = 10.0
DEFAULT_BACKOFF_FACTOR = 3.0  # delays 10 s, then 30 s
SERVER_ERROR_MIN_STATUS = 500
SERVER_ERROR_MAX_STATUS = 599
MAX_ALLOWED_RETRIES = 2  # hard ceiling for the config override: a retried paid request may be billed

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff_s: float = DEFAULT_INITIAL_BACKOFF_S
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR

    def delay_s(self, attempt: int) -> float:
        """Seconds to wait after the failed ``attempt`` (1-based) before the next one."""
        return self.initial_backoff_s * self.backoff_factor ** (attempt - 1)


@dataclass(frozen=True)
class RetryEvent:
    """One failed attempt. Carries no request body, no message text and no credentials."""

    attempt: int
    max_attempts: int
    error_type: str
    status_code: int | None
    request_id: str | None
    delay_s: float | None  # None: retries exhausted, the error is re-raised

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_RETRY_POLICY = RetryPolicy()


def is_transient(exc: BaseException) -> bool:
    try:
        import openai  # lazy: previews and tests never import the SDK
    except ImportError:
        return False
    if isinstance(exc, openai.APITimeoutError):
        return False
    if isinstance(exc, openai.APIConnectionError):
        return True
    if not isinstance(exc, openai.APIStatusError):
        return False
    return SERVER_ERROR_MIN_STATUS <= exc.status_code <= SERVER_ERROR_MAX_STATUS


def _describe(exc: BaseException, attempt: int, max_attempts: int, delay_s: float | None) -> RetryEvent:
    return RetryEvent(
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=type(exc).__name__,
        status_code=getattr(exc, "status_code", None),
        request_id=getattr(exc, "request_id", None),
        delay_s=delay_s,
    )


def log_to_stderr(event: RetryEvent) -> None:
    outcome = "giving up" if event.delay_s is None else f"retrying in {event.delay_s:g} s"
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(
        f"[{stamp}] openai request attempt {event.attempt}/{event.max_attempts} failed: "
        f"{event.error_type} status={event.status_code} request_id={event.request_id}; {outcome}",
        file=sys.stderr,
        flush=True,
    )


def call_with_retry(
    call: Callable[[], T],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[RetryEvent], None] = log_to_stderr,
) -> tuple[T, tuple[RetryEvent, ...]]:
    """Run ``call``; return its result and the failed attempts that preceded it.

    Exhausted retries re-raise the last error unchanged.
    """
    max_attempts = policy.max_retries + 1
    failed: tuple[RetryEvent, ...] = ()
    attempt = 1
    while True:
        try:
            return call(), failed
        except Exception as exc:
            if not is_transient(exc):
                raise
            has_retry_left = attempt < max_attempts
            delay_s = policy.delay_s(attempt) if has_retry_left else None
            event = _describe(exc, attempt, max_attempts, delay_s)
            log(event)
            if delay_s is None:
                raise
            failed = failed + (event,)
            sleep(delay_s)
            attempt += 1
