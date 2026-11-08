"""``python -m evaluation.cli``: judge a Translator run's translation events with ``gpt-6.1-sol``.

Preview by default: no client is built and no credential is read. The judge always calls a paid
API, so a real run needs BOTH ``--execute`` and ``--confirm-paid-api``; either one alone is a
usage error. This module is the only composition root: it alone builds the OpenAI SDK client.
Exit codes: 0 preview written or run completed; 1 deterministic failure; 2 usage or gate misuse;
3 a call has an uncertain outcome (may have been billed; never resent); 4 ``cap_reached``.
Run it from the repository root only (a stray ``evaluation.py`` on ``sys.path`` would shadow it).
With ``inputs.video.include`` the frames of every judged event are decoded while the inputs are
loaded (preview and run alike): a missing video fails before any client exists, and no file written
here contains base64.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from translate import paths
from translate.events import SAFE_ID_PATTERN, redacted_error, sha256_text
from translate.llm.openai_responses import (
    API_BASE_URL,
    OpenAIClientError,
    OpenAIResponsesClient,
    build_responses_body,
    make_openai_client,
    set_provenance_env_names,
)
from translate.output_safety import derived_path

from .configuration import JudgeConfig, JudgeConfigError, JudgePrice, load_config
from .frames import CachingFrameSource, FrameSource, PyAVFrameSource
from .judge import (
    INTERRUPTIONS,
    STATUS_CAP_REACHED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    JsonlJudgeSink,
    JudgeCallError,
    JudgeSummary,
    OpenAIJudgeModel,
    PlannedEvent,
    run_judge,
    utc_now,
)
from .rendering import (
    CHARS_PER_TOKEN,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    KEY_POINT_RUBRIC,
    VideoContext,
    approx_input_tokens,
    approx_text_tokens,
    build_judge_request,
    redact_images,
)
from .schemas import (
    EventKeyPoints,
    TranslatorRun,
    load_key_points,
    load_translator_run,
    match_key_points,
    run_episode,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "judge-text.yaml"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_UNCERTAIN = 3
EXIT_CAP_REACHED = 4
EXIT_BY_STATUS = {
    STATUS_COMPLETED: EXIT_OK,
    STATUS_FAILED: EXIT_FAILED,
    STATUS_UNCERTAIN: EXIT_UNCERTAIN,
    STATUS_CAP_REACHED: EXIT_CAP_REACHED,
}
RUN_SCHEMA_VERSION = 1
MODE_PREVIEW = "preview"
MODE_EXECUTE = "execute"
STATUS_PREVIEW = "preview"
STATUS_RUNNING = "running"
TRANSLATOR_COMPLETED = "completed"
TOKENS_PER_MILLION = 1_000_000
COST_DECIMALS = 4
GIT_TIMEOUT_S = 10
RUN_DIR_MODE = 0o700  # outputs hold model-visible scenario text and key points: owner-only
GIT_TRACKED_PATHS = ("evaluation",)
GATE_MESSAGE = "the judge always calls a paid API: use BOTH --execute and --confirm-paid-api"

OpenAIClientFactory = Callable[..., Any]


@dataclass(frozen=True)
class Inputs:
    config_path: Path
    key_points_path: Path
    config: JudgeConfig
    run: TranslatorRun
    entries: tuple[EventKeyPoints, ...]
    planned: tuple[PlannedEvent, ...]

    @property
    def judged(self) -> int:
        return sum(1 for item in self.planned if item.request is not None)


def run_name(value: str) -> str:
    """argparse type: a run directory name, never a path. A bad one is a usage error (exit 2)."""
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {SAFE_ID_PATTERN.pattern} (a name, not a path)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.cli",
        description="LLM-as-a-judge (gpt-6.1-sol) for Translator events. Writes a preview unless "
        "--execute --confirm-paid-api are both given.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="judge config YAML")
    parser.add_argument("--translator-run", type=Path, required=True, help="Translator run directory (events.jsonl)")
    parser.add_argument(
        "--key-points", type=Path, required=True, help="key-points YAML (real ones: private/evaluation/)"
    )
    parser.add_argument("--output-root", type=Path, help="override output.root of the config")
    parser.add_argument("--run-name", type=run_name, help="run directory name (default: <UTC stamp>_judge_<mode>)")
    parser.add_argument("--execute", action="store_true", help="send the judge calls (needs --confirm-paid-api)")
    parser.add_argument("--confirm-paid-api", action="store_true", help="confirm paid calls (needs --execute)")
    return parser


# --- inputs and preflight -------------------------------------------------------------------------


def _video_context(config: JudgeConfig, run: TranslatorRun, frame_source: FrameSource | None) -> VideoContext | None:
    """The video block's frame source and episode, or ``None`` when the block is excluded."""
    if not config.inputs.video.include:
        return None
    episode = run_episode(run)
    source = frame_source if frame_source is not None else CachingFrameSource(PyAVFrameSource())
    return VideoContext(frame_source=source, episode_id=episode.episode_id, last_step=episode.last_complete_step)


def load_inputs(
    config_path: Path, run_dir: Path, key_points_path: Path, *, frame_source: FrameSource | None = None
) -> Inputs:
    """Config, event log, key points, matching, cap and every request. No client, no credential.

    With the video block included this also decodes the frames of every judged event, so a missing
    or unreadable video fails here, before any client exists.
    """
    config = load_config(config_path)
    run = load_translator_run(run_dir)
    entries = load_key_points(key_points_path)
    matched = match_key_points(run.event_ids(), entries)
    if len(matched) > config.max_judged_events:
        raise JudgeConfigError(
            f"{len(matched)} events have key points; limits.max_judged_events is {config.max_judged_events}"
        )
    events = run.translation_events
    video = _video_context(config, run, frame_source) if matched else None
    planned = tuple(
        PlannedEvent(
            event=event,
            request=build_judge_request(
                event=event, event_log=events, key_points=matched[event["event_id"]], config=config, video=video
            )
            if event["event_id"] in matched
            else None,
        )
        for event in events
    )
    return Inputs(config_path, key_points_path, config, run, entries, planned)


def _priced(tokens: int, rate: float, multiplier: float, long_context: bool) -> float:
    return tokens * rate * (multiplier if long_context else 1.0) / TOKENS_PER_MILLION


def _estimate(inputs: Inputs) -> dict[str, Any]:
    price: JudgePrice = inputs.config.price
    max_out = inputs.config.max_output_tokens
    tokens = [approx_input_tokens(item.request) for item in inputs.planned if item.request is not None]
    long_flags = [count > price.long_context_threshold_tokens for count in tokens]
    input_cost = sum(
        _priced(t, price.input, price.long_input_multiplier, long) for t, long in zip(tokens, long_flags, strict=True)
    )
    output_cost = sum(_priced(max_out, price.output, price.long_output_multiplier, long) for long in long_flags)
    requests = [item.request for item in inputs.planned if item.request is not None]
    video = inputs.config.inputs.video
    return {
        "approximate": f"text tokens ~= ceil(chars / {CHARS_PER_TOKEN}) of instructions, user text and schema; "
        "no cache hits assumed; output priced at max_output_tokens per call (upper bound)",
        "pricing_note": "list price, verify before paid runs",
        "judge_calls": len(tokens),
        "images_total": sum(len(request.frames) for request in requests),
        "image_tokens_total": sum(request.image_tokens for request in requests),
        "image_tokens_per_image": video.image_tokens.per_image if video.include else 0,
        "image_tokens_status": video.image_tokens.status if video.include else None,
        "text_tokens_total": sum(approx_text_tokens(request) for request in requests),
        "input_tokens_total": sum(tokens),
        "input_tokens_max_per_call": max(tokens, default=0),
        "long_context_calls": sum(long_flags),
        "cost_usd_input": round(input_cost, COST_DECIMALS),
        "cost_usd_output_upper_bound": round(output_cost, COST_DECIMALS),
        "cost_usd_upper_bound": round(input_cost + output_cost, COST_DECIMALS),
    }


def _usage_cost(usage: dict[str, int], price: JudgePrice) -> float:
    """Approximate USD of the reported usage at list price (long-context surcharge not applied)."""
    uncached = usage["input_tokens"] - usage["cached_input_tokens"]
    cost = uncached * price.input + usage["cached_input_tokens"] * price.cached_input
    return round((cost + usage["output_tokens"] * price.output) / TOKENS_PER_MILLION, COST_DECIMALS)


def _warnings(inputs: Inputs, estimate: dict[str, Any]) -> list[str]:
    warnings = []
    status = inputs.run.run_json.get("status")
    if status != TRANSLATOR_COMPLETED:
        warnings.append(f"translator run status is {status!r}, not 'completed': its event log may be partial")
    if inputs.judged == 0:
        warnings.append("no translation event has key points: nothing will be judged")
    if not inputs.config.inputs.translated_intent.include:
        warnings.append("inputs.translated_intent is excluded: the judge cannot see the output it scores")
    if not inputs.config.inputs.key_points.include:
        warnings.append("inputs.key_points is excluded: only the overall score is requested")
    video = inputs.config.inputs.video
    if video.include:
        warnings.append(
            "inputs.video is included: the judge sees A-team actions and outcomes after the judged step that the "
            "Translator never saw (evaluation input only)"
        )
        if video.image_tokens.status != "verified":
            warnings.append(
                f"image token estimate ({video.image_tokens.per_image} per image) is {video.image_tokens.status}: "
                "check it against the model's real usage before trusting the cost"
            )
    if estimate["long_context_calls"]:
        warnings.append(f"{estimate['long_context_calls']} calls exceed the long-context threshold (billed higher)")
    return warnings


def _frames_json(item: PlannedEvent) -> dict[str, Any]:
    """Frame summaries of one event: steps, count and sizes. Never image bytes or base64."""
    if item.request is None or not item.request.frames:
        return {}
    frames = item.request.frames
    return {
        "frames": {
            "count": len(frames),
            "steps": [frame.step for frame in frames],
            "frame_indices": [frame.frame_index for frame in frames],
            "sizes": sorted({f"{frame.width}x{frame.height}" for frame in frames}),
            "jpeg_bytes_total": sum(frame.jpeg_bytes for frame in frames),
            "image_tokens": item.request.image_tokens,
        }
    }


def _plan_json(inputs: Inputs) -> dict[str, Any]:
    events = [
        {
            "event_id": item.event["event_id"],
            "step": item.event["step"],
            "target": item.event["target"],
            "planned": "judged" if item.request is not None else "not_judged",
            "key_points": len(item.request.key_points) if item.request is not None else 0,
            "approx_input_tokens": approx_input_tokens(item.request) if item.request is not None else 0,
            **_frames_json(item),
        }
        for item in inputs.planned
    ]
    return {
        "translation_events": len(inputs.planned),
        "judged": inputs.judged,
        "not_judged": len(inputs.planned) - inputs.judged,
        "max_judged_events": inputs.config.max_judged_events,
        "included_blocks": list(inputs.config.inputs.included()),
        "events": events,
    }


# --- provenance and artifacts ---------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )

    try:
        commit = git("rev-parse", "HEAD")
        status = git("--no-optional-locks", "status", "--porcelain", "--", *GIT_TRACKED_PATHS)
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}
    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _code_hashes() -> dict[str, str]:
    package_dir = Path(__file__).resolve().parent
    return {f"evaluation/{path.name}": _sha256_file(path) for path in sorted(package_dir.glob("*.py"))}


def _translator_field(run_json: dict[str, Any], key: str) -> str | None:
    value = run_json.get(key)
    return value if isinstance(value, str) else None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _video_provenance(inputs: Inputs) -> dict[str, Any] | None:
    """The video files and decoder builds behind the frames (the request hash depends on them)."""
    video = inputs.config.inputs.video
    if not video.include:
        return None
    files = {
        episode_id: {
            "path": str(episode.path),
            "sha256": _sha256_file(episode.path) if episode.path.is_file() else None,
        }
        for episode_id, episode in sorted(video.videos.items())
    }
    return {"videos": files, "av": _package_version("av"), "pillow": _package_version("Pillow")}


def _run_record(inputs: Inputs, *, mode: str, status: str) -> dict[str, Any]:
    run = inputs.run
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "mode": mode,
        "status": status,
        "started_at": utc_now(),
        "ended_at": None,
        "inputs": {
            "config": {"path": str(inputs.config_path), "sha256": _sha256_file(inputs.config_path)},
            "key_points": {
                "path": str(inputs.key_points_path),
                "sha256": _sha256_file(inputs.key_points_path),
                "entries": len(inputs.entries),
            },
            "translator_run": {
                "path": str(run.run_dir),
                "run_json_sha256": run.run_json_sha256,
                "events_jsonl_sha256": run.events_sha256,
                "mode": _translator_field(run.run_json, "mode"),
                "status": _translator_field(run.run_json, "status"),
            },
        },
        "provenance": {
            "git": _git_state(),
            "code_sha256": _code_hashes(),
            "python": sys.version.split()[0],
            "prompt_version": JUDGE_PROMPT_VERSION,
            "openai_base_url": API_BASE_URL,  # the only endpoint make_openai_client targets
            "sdk_environment_set": set_provenance_env_names(),  # names only; proxy URLs can hold credentials
            "system_prompt_sha256": sha256_text(JUDGE_SYSTEM_PROMPT),
            "key_point_rubric_sha256": sha256_text(KEY_POINT_RUBRIC),
            "video": _video_provenance(inputs),
        },
        "config": json.loads(json.dumps(asdict(inputs.config), default=str)),
        "plan": {key: value for key, value in _plan_json(inputs).items() if key != "events"},
        "api_requests_made": 0,
    }


def _write_json(path: Path, payload: object) -> None:
    derived_path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _new_run_dir(args: argparse.Namespace, config: JudgeConfig, mode: str) -> Path:
    """A new, never reused run directory (mode 0700); an existing one is refused (``FileExistsError``)."""
    name = args.run_name or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_judge_{mode}"
    if not SAFE_ID_PATTERN.fullmatch(name):
        raise JudgeConfigError(f"--run-name must match {SAFE_ID_PATTERN.pattern}")
    root = args.output_root if args.output_root is not None else config.output_root
    run_dir = derived_path(Path(root) / name)
    run_dir.mkdir(parents=True, exist_ok=False, mode=RUN_DIR_MODE)
    run_dir.chmod(RUN_DIR_MODE)  # mkdir's mode is masked by the umask
    return run_dir


# --- modes ----------------------------------------------------------------------------------------


def _redact(exc: BaseException) -> dict[str, str]:
    """The LLM part's errors are sanitised by design; everything else goes through ``redacted_error``."""
    if isinstance(exc, (OpenAIClientError, JudgeCallError)):
        return {"type": type(exc).__name__, "message": str(exc)}
    return redacted_error(exc)


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _preview(args: argparse.Namespace, inputs: Inputs) -> int:
    estimate = _estimate(inputs)
    run_dir = _new_run_dir(args, inputs.config, MODE_PREVIEW)
    record = {**_run_record(inputs, mode=MODE_PREVIEW, status=STATUS_PREVIEW), "ended_at": utc_now()}
    warnings = _warnings(inputs, estimate)
    preview = {"api_requests_made": 0, "plan": _plan_json(inputs), "estimate": estimate, "warnings": warnings}
    _write_json(run_dir / "run.json", record)
    _write_json(run_dir / "preview.json", preview)
    first = next((item.request for item in inputs.planned if item.request is not None), None)
    if first is not None:
        example = redact_images(build_responses_body(first.openai_request))  # never base64 on disk
        _write_json(run_dir / "example_judge_request.json", example)
    plan = preview["plan"]
    _print(
        {
            "run_dir": str(run_dir),
            "status": STATUS_PREVIEW,
            "translation_events": plan["translation_events"],
            "judged": plan["judged"],
            "not_judged": plan["not_judged"],
            "approx_input_tokens_total": estimate["input_tokens_total"],
            "approx_cost_usd_upper_bound": estimate["cost_usd_upper_bound"],
            "warnings": warnings,
        }
    )
    print("Preview only. API requests made: 0", file=sys.stderr)
    return EXIT_OK


def _final_record(record: dict[str, Any], summary: JudgeSummary, price: JudgePrice) -> dict[str, Any]:
    counts = {**asdict(summary.counts), "results_written": summary.results_written}
    return {
        **record,
        "status": summary.status,
        "ended_at": utc_now(),
        "counts": counts,
        "usage": summary.usage,
        "approx_cost_usd": _usage_cost(summary.usage, price),
        "unsent": summary.unsent,
        "error": summary.error,
        "api_requests_made": summary.counts.attempted,
    }


def _finish(run_dir: Path, record: dict[str, Any], summary: JudgeSummary, price: JudgePrice) -> int:
    """Final ``run.json``. If it cannot be written, an uncertain or capped run keeps its exit code."""
    final = _final_record(record, summary, price)
    exit_code = EXIT_BY_STATUS[summary.status]
    try:
        _write_json(run_dir / "run.json", final)
    except OSError as exc:
        print(f"error: final run.json not written: {type(exc).__name__} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED if exit_code == EXIT_OK else exit_code
    _print({"run_dir": str(run_dir), "status": summary.status, "counts": final["counts"], "unsent": summary.unsent})
    return exit_code


def _execute(args: argparse.Namespace, inputs: Inputs, openai_client_factory: OpenAIClientFactory) -> int:
    """Order: run dir -> run.json -> paid client (only if something is judged) -> per-event calls."""
    run_dir = _new_run_dir(args, inputs.config, MODE_EXECUTE)
    record: dict[str, Any] = {"schema_version": RUN_SCHEMA_VERSION, "mode": MODE_EXECUTE, "started_at": utc_now()}
    summary: JudgeSummary | None = None
    try:
        record = _run_record(inputs, mode=MODE_EXECUTE, status=STATUS_RUNNING)
        _write_json(run_dir / "run.json", record)
        model = None
        if inputs.judged:
            sdk_client = openai_client_factory(timeout_s=inputs.config.timeout_s)  # the only OPENAI_API_KEY read
            model = OpenAIJudgeModel(OpenAIResponsesClient(sdk_client))
        with JsonlJudgeSink(run_dir) as sink:
            summary = run_judge(  # never raises: every failure becomes a summary status
                inputs.planned, model=model, sink=sink, max_judged_events=inputs.config.max_judged_events
            )
    except INTERRUPTIONS as exc:  # Ctrl-C included: no call is in flight outside run_judge
        error = _redact(exc)
        if summary is not None:  # closing the journals failed after every line was fsynced
            print(f"error: closing the journals failed: {error['type']} ({run_dir})", file=sys.stderr)
            return _finish(run_dir, record, summary, inputs.config.price)
        # before run_judge: nothing was sent; recorded, then a failure exit
        _write_json(run_dir / "run.json", {**record, "status": STATUS_FAILED, "error": error, "ended_at": utc_now()})
        print(f"error: {error['type']}: {error['message']} ({run_dir})", file=sys.stderr)
        return EXIT_FAILED
    return _finish(run_dir, record, summary, inputs.config.price)


def main(
    argv: list[str] | None = None,
    *,
    openai_client_factory: OpenAIClientFactory = make_openai_client,
    frame_source: FrameSource | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute != args.confirm_paid_api:
        parser.error(GATE_MESSAGE)
    try:
        inputs = load_inputs(args.config, args.translator_run, args.key_points, frame_source=frame_source)
        if not args.execute:
            return _preview(args, inputs)
        return _execute(args, inputs, openai_client_factory)
    except INTERRUPTIONS as exc:  # reported without traceback (Ctrl-C included); unknown text is withheld
        error = _redact(exc)
        print(f"error: {error['type']}: {error['message']}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
