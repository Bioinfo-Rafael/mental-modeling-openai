"""``python -m evaluation.assess_cli``: evaluate a Translator run on 3 time points x 4 aspects.

Subcommands ``check``, ``judge``, ``import-ratings`` and ``report`` share one assessment directory
per Translator run, bound to the run's file hashes on first use; every store in it is append-only.
The judge previews unless BOTH ``--execute`` and ``--confirm-paid-api`` are given, and never resends
a journaled request. Files and flags: ``evaluation/README.md``.

Exit codes: 0 done (or preview); 1 deterministic failure, including invalid generation inputs
found by ``check``; 2 usage or gate misuse; 3 a judge call has an uncertain outcome (may have
been billed; never resent); 4 the judge cap stopped the run. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from mtd_rl_xavier import paths
from mtd_rl_xavier.lag.dataset import load_episode
from mtd_rl_xavier.llm.openai_responses import (
    API_BASE_URL,
    OpenAIClientError,
    OpenAIResponsesClient,
    make_openai_client,
    set_provenance_env_names,
)
from mtd_rl_xavier.tools.output_safety import derived_path

from .assess_checks import (
    LEAKAGE_FILE,
    CallCheck,
    check_generation_inputs,
    invalid_targets,
    machine_results,
    match_points,
)
from .assess_data import (
    DEFAULT_CONFIG,
    EVALUATOR_HUMAN,
    EVALUATOR_JUDGE,
    EVALUATOR_MACHINE,
    RESULT_FILES,
    SAFE_ID_PATTERN,
    Annotation,
    AssessConfig,
    AssessError,
    TranslatorRun,
    canonical_json,
    dig,
    load_annotation,
    load_config,
    load_run,
    sha256_file,
    strict_json_loads,
    target_key,
)
from .assess_judge import (
    INTERRUPTIONS,
    JUDGE_PROMPT_VERSION,
    JUDGE_REQUESTS,
    JUDGE_RESPONSES,
    STATUS_CAP_REACHED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_UNCERTAIN,
    JudgeCallError,
    OpenAIAssessJudge,
    PlannedCall,
    plan_calls,
    run_judge_calls,
    system_prompt,
)
from .assess_report import build_report, leakage_summary
from .assess_store import append_jsonl, human_results, load_ratings, read_results, with_supersedes

EXIT_OK, EXIT_FAILED, EXIT_UNCERTAIN, EXIT_CAP_REACHED = 0, 1, 3, 4  # 2 (usage, gate misuse): argparse
EXIT_BY_STATUS = {
    STATUS_COMPLETED: EXIT_OK,
    STATUS_FAILED: EXIT_FAILED,
    STATUS_UNCERTAIN: EXIT_UNCERTAIN,
    STATUS_CAP_REACHED: EXIT_CAP_REACHED,
}
BINDING_FILE, INVOCATIONS_FILE = "run.json", "invocations.jsonl"
BINDING_SCHEMA_VERSION = 1
RUN_DIR_MODE = 0o700  # packets and results hold scenario text and ratings: owner-only
MODE_PREVIEW, MODE_EXECUTE = "preview", "execute"
CHARS_PER_TOKEN = 4  # approximate tokens = ceil(chars / 4); labelled approximate everywhere
TOKENS_PER_MILLION = 1_000_000
COST_DECIMALS = 4
GIT_TIMEOUT_S = 10
GATE_MESSAGE = "the judge always calls a paid API: use BOTH --execute and --confirm-paid-api"
TRANSLATOR_COMPLETED = "completed"

OpenAIClientFactory = Callable[..., Any]
EpisodeLoader = Callable[[Path], pd.DataFrame]


def run_name(value: str) -> str:
    """argparse type: an assessment directory name, never a path (exit 2 otherwise)."""
    if not SAFE_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(f"must match {SAFE_ID_PATTERN.pattern} (a name, not a path)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.assess_cli",
        description="Evaluate Translator output on 3 time points x 4 aspects (machine, human and judge results "
        "are stored separately). The judge writes a preview unless --execute --confirm-paid-api are both given.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="evaluation config (assess.yaml)")
    common.add_argument("--translator-run", type=Path, required=True, help="Translator run directory (v2 events)")
    common.add_argument("--output-root", type=Path, help="override output.root of the config")
    common.add_argument("--name", type=run_name, help="assessment directory name (default: the run directory name)")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", parents=[common], help="leakage check per model call + machine results")
    check.add_argument("--points", type=Path, help="reference annotation YAML (start/change points; private/)")
    judge = commands.add_parser("judge", parents=[common], help="packets and preview; paid calls only with both flags")
    judge.add_argument("--points", type=Path, help="reference annotation YAML (bounds the FUTURE block)")
    judge.add_argument("--execute", action="store_true", help="send the judge calls (needs --confirm-paid-api)")
    judge.add_argument("--confirm-paid-api", action="store_true", help="confirm paid calls (needs --execute)")
    ratings = commands.add_parser("import-ratings", parents=[common], help="validate and append human ratings")
    ratings.add_argument("--ratings", type=Path, required=True, help="human ratings JSONL (private/)")
    report = commands.add_parser("report", parents=[common], help="write a new report-<batch>.json")
    report.add_argument("--rubric-version", help="rubric version to compare (default: the config's)")
    return parser


# --- assessment directory and provenance ---------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_batch_id() -> str:
    """Unique per invocation; part of every result id written by it."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _binding(run: TranslatorRun) -> dict[str, Any]:
    return {"path": str(run.run_dir.resolve()), "files_sha256": run.file_sha256}


def assessment_dir(args: argparse.Namespace, config: AssessConfig, run: TranslatorRun) -> Path:
    """Create (0700) or reuse the run's assessment directory; a directory of another run is refused."""
    name = args.name or run.run_dir.resolve().name
    if not SAFE_ID_PATTERN.fullmatch(name):
        raise AssessError(f"the run directory name is not a safe id; pass --name matching {SAFE_ID_PATTERN.pattern}")
    root = args.output_root if args.output_root is not None else config.output_root
    directory = derived_path(Path(root) / name)
    if not directory.exists():
        directory.mkdir(parents=True, mode=RUN_DIR_MODE)
        directory.chmod(RUN_DIR_MODE)  # mkdir's mode is masked by the umask
        record = {
            "schema_version": BINDING_SCHEMA_VERSION,
            "created_at": utc_now(),
            "scenario_id": run.scenario_id,
            "episode_id": run.episode_id,
            "translator_run": _binding(run),
        }
        _write_new(directory / BINDING_FILE, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return directory
    try:
        bound = strict_json_loads((directory / BINDING_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssessError(f"{directory} exists but has no readable {BINDING_FILE}") from exc
    if not isinstance(bound, dict) or (bound.get("translator_run") or {}).get("files_sha256") != run.file_sha256:
        raise AssessError(f"{directory} belongs to another Translator run (file hashes differ); use another --name")
    return directory


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(paths.ROOT), *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S
        )

    try:
        commit = git("rev-parse", "HEAD")
        status = git("--no-optional-locks", "status", "--porcelain", "--", "evaluation")
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}
    if commit.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _file_input(path: Path | None) -> dict[str, Any] | None:
    return {"path": str(path), "sha256": sha256_file(path)} if path is not None else None


@dataclass(frozen=True)
class Session:
    args: argparse.Namespace
    config: AssessConfig
    run: TranslatorRun
    checks: tuple[CallCheck, ...]
    directory: Path
    batch_id: str

    def log(self, event: str, **fields: Any) -> None:
        """One line of ``invocations.jsonl`` (started / finished)."""
        record = {"batch_id": self.batch_id, "command": self.args.command, "event": event, "at": utc_now(), **fields}
        append_jsonl(self.directory / INVOCATIONS_FILE, [record])

    def started(self, **fields: Any) -> None:
        inputs = {
            "config": _file_input(self.config.path),
            "translator_run": _binding(self.run),
            "points": _file_input(getattr(self.args, "points", None)),
            "ratings": _file_input(getattr(self.args, "ratings", None)),
        }
        package = Path(__file__).resolve().parent
        code = {f"evaluation/{path.name}": sha256_file(path) for path in sorted(package.glob("assess_*.py"))}
        provenance = {"git": _git_state(), "code_sha256": code, "python": sys.version.split()[0]}
        self.log("started", inputs=inputs, rubric_version=self.config.rubric_version, provenance=provenance, **fields)

    def warnings(self) -> list[str]:
        run, warnings = self.run, []
        if run.status != TRANSLATOR_COMPLETED:
            warnings.append(f"translator run status is {run.status!r}, not 'completed': its event log may be partial")
        if run.retrospective:
            warnings.append("retrospective run (future-informed): every record is tagged and reported separately")
        invalid = invalid_targets(self.checks)
        if invalid:
            names = ", ".join(f"{kind}:{target_id}" for kind, target_id in sorted(invalid))
            warnings.append(f"generation input invalid (leak or error) for {names}: their results are not_evaluated")
        return warnings

    def annotation(self) -> Annotation | None:
        return load_annotation(self.args.points, self.run) if self.args.points else None


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _write_new(path: Path, text: str) -> None:
    with derived_path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)


# --- check ----------------------------------------------------------------------------------------


def run_check(session: Session) -> int:
    run, directory, batch_id = session.run, session.directory, session.batch_id
    annotation = session.annotation()
    session.started()
    append_jsonl(
        directory / LEAKAGE_FILE,
        [c.to_json(retrospective=run.retrospective, batch_id=batch_id) for c in session.checks],
    )
    matches = match_points(annotation, run)
    records = machine_results(run, session.config, matches, session.checks, batch_id)
    path = directory / RESULT_FILES[EVALUATOR_MACHINE]
    appended = append_jsonl(path, with_supersedes(records, read_results(path)))
    summary = leakage_summary(session.checks)
    status = STATUS_COMPLETED if summary["leak"] == 0 and summary["error"] == 0 else STATUS_FAILED
    session.log("finished", status=status, leakage=summary, machine_results=appended)
    _print(
        {
            "assessment_dir": str(directory),
            "batch_id": batch_id,
            "leakage": summary,
            "machine_results": appended,
            "reference_points": len(matches),
            "warnings": session.warnings(),
        }
    )
    return EXIT_BY_STATUS[status]


# --- judge ----------------------------------------------------------------------------------------


class JsonlJudgeSink:
    """Appends the judge journal and results; results supersede the active row of the same slot."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._results = read_results(directory / RESULT_FILES[EVALUATOR_JUDGE])

    def write_request(self, record: dict[str, Any]) -> None:
        append_jsonl(self._directory / JUDGE_REQUESTS, [{**record, "written_at": utc_now()}])

    def write_response(self, record: dict[str, Any]) -> None:
        append_jsonl(self._directory / JUDGE_RESPONSES, [{**record, "written_at": utc_now()}])

    def write_results(self, records: list[dict[str, Any]]) -> None:
        stored = with_supersedes(records, self._results)
        append_jsonl(self._directory / RESULT_FILES[EVALUATOR_JUDGE], stored)
        self._results = [*self._results, *stored]


def _episode_frame(run: TranslatorRun, loader: EpisodeLoader) -> pd.DataFrame:
    csv_path, csv_sha256 = (dig(run.run_json, "inputs", "episode", key) for key in ("csv_path", "csv_sha256"))
    if not isinstance(csv_path, str) or not isinstance(csv_sha256, str):
        raise AssessError("run.json inputs.episode needs csv_path and csv_sha256 for the FUTURE block")
    frame = loader(Path(csv_path))
    if frame.attrs.get("csv_sha256") != csv_sha256:
        raise AssessError("the episode CSV differs from the one the Translator run used (csv_sha256)")
    return frame


def _packets_markdown(session: Session, calls: list[PlannedCall]) -> str:
    """D58: human raters get the video's path and SHA-256 only; the judge gets no frames."""
    video = dig(session.run.run_json, "inputs", "episode", "video_path")
    video_path = video if isinstance(video, str) else None
    video_sha256 = sha256_file(Path(video)) if video_path and Path(video).is_file() else None
    lines = [
        f"# Judge packets ({session.batch_id})",
        "",
        f"Rubric {session.config.rubric_version}; config sha256 {session.config.sha256}.",
        "Each packet is exactly the text the judge receives. Rate every listed criterion (human-ratings JSONL).",
        f"Video for human raters (not sent to the judge): {video_path} (sha256 {video_sha256}).",
    ]
    for call in calls:
        lines += ["", f"## {call.packet.event_id}", f"packet sha256 {call.packet.sha256}", ""]
        lines += ["~~~~text", call.packet.text, "~~~~"]
    return "\n".join(lines) + "\n"


def _estimate(planned: tuple[PlannedCall, ...], config: AssessConfig) -> dict[str, Any]:
    sent = planned[: config.judge.max_calls]  # the run stops at the cap
    chars = [len(c.request.instructions) + len(c.packet.text) + len(canonical_json(c.request.schema)) for c in sent]
    tokens = sum(math.ceil(count / CHARS_PER_TOKEN) for count in chars)
    price = config.judge.pricing
    input_cost = tokens * price["input"] / TOKENS_PER_MILLION
    output_cost = len(sent) * config.judge.max_output_tokens * price["output"] / TOKENS_PER_MILLION
    return {
        "approximate": "input tokens ~= ceil(chars / 4) of instructions, packet and schema; no cache hits; output "
        "priced at max_output_tokens per call (upper bound); list price, verify before paid runs",
        "judge_calls": len(sent),
        "input_tokens_total": tokens,
        "cost_usd_upper_bound": round(input_cost + output_cost, COST_DECIMALS),
    }


def _judge_settings(config: AssessConfig) -> dict[str, Any]:
    """The resolved judge configuration, stored in full (not only as a hash)."""
    return {
        **asdict(config.judge),
        "prompt_version": JUDGE_PROMPT_VERSION,
        "system_prompt": system_prompt(config),
        "rubric_version": config.rubric_version,
        "criteria": {name: [asdict(c) for c in items] for name, items in config.criteria.items()},
        "scale": asdict(config.scale),
        "future_evidence": asdict(config.future_evidence),
        "openai_base_url": API_BASE_URL,
        "sdk_environment_set": set_provenance_env_names(),  # names only, never values
    }


def run_judge(session: Session, client_factory: OpenAIClientFactory, loader: EpisodeLoader) -> int:
    directory, config, batch_id = session.directory, session.config, session.batch_id
    run, annotation = session.run, session.annotation()
    invalid = invalid_targets(session.checks)
    events = [event for event in run.events if target_key(event["target"]) not in invalid]
    calls = plan_calls(
        events, run=run, config=config, annotation=annotation, load_frame=lambda: _episode_frame(run, loader)
    )
    journal = read_results(directory / JUDGE_REQUESTS)
    sent = {line["request_sha256"] for line in journal if isinstance(line.get("request_sha256"), str)}
    planned = tuple(call for call in calls if call.request_sha256 not in sent)  # never send a request twice
    mode = MODE_EXECUTE if session.args.execute else MODE_PREVIEW
    plan = {
        "translation_events": len(session.run.events),
        "generation_input_invalid": len(session.run.events) - len(calls),
        "packets": len(calls),
        "already_sent": len(calls) - len(planned),
        "to_send": len(planned),
        "max_calls": config.judge.max_calls,
    }
    estimate, warnings = _estimate(planned, config), session.warnings()
    if not session.args.points:
        warnings.append("no --points: FUTURE blocks end at the next instruction or alarm, down step or max_steps only")
    if len(planned) > config.judge.max_calls:
        warnings.append(f"{len(planned)} calls planned; the run stops after {config.judge.max_calls} (cap_reached)")
    packets = "".join(json.dumps(call.packet.to_json(), ensure_ascii=False) + "\n" for call in calls)
    _write_new(directory / f"packets-{batch_id}.jsonl", packets)
    _write_new(directory / f"packets-{batch_id}.md", _packets_markdown(session, calls))
    session.started(mode=mode, judge=_judge_settings(config), plan=plan, estimate=estimate)
    output = {"assessment_dir": str(directory), "status": STATUS_COMPLETED, "plan": plan}
    if mode == MODE_PREVIEW:
        if planned:
            example = json.dumps(planned[0].payload, indent=2, ensure_ascii=False) + "\n"
            _write_new(directory / f"example_judge_request-{batch_id}.json", example)
        session.log("finished", status=MODE_PREVIEW, api_requests_made=0)
        _print({**output, "status": MODE_PREVIEW, "estimate": estimate, "warnings": warnings})
        print("Preview only. API requests made: 0", file=sys.stderr)
        return EXIT_OK
    if not planned:  # nothing new to send: no client is built
        session.log("finished", status=STATUS_COMPLETED, api_requests_made=0)
        _print({**output, "warnings": warnings})
        return EXIT_OK
    try:
        model = OpenAIAssessJudge(OpenAIResponsesClient(client_factory(timeout_s=config.judge.timeout_s)))
    except INTERRUPTIONS as exc:  # e.g. no OPENAI_API_KEY: nothing was sent
        session.log("finished", status=STATUS_FAILED, api_requests_made=0, error=_error(exc))
        raise
    summary = run_judge_calls(
        planned, model=model, sink=JsonlJudgeSink(directory), run=session.run, config=config, batch_id=batch_id
    )
    counts = asdict(summary.counts)
    details = {"counts": counts, "usage": summary.usage, "unsent": summary.unsent, "error": summary.error}
    session.log("finished", status=summary.status, api_requests_made=counts["attempted"], **details)
    _print({**output, "status": summary.status, **details, "warnings": warnings})
    return EXIT_BY_STATUS[summary.status]


# --- import-ratings and report --------------------------------------------------------------------


def run_import(session: Session) -> int:
    lines, import_sha256 = load_ratings(session.args.ratings)
    path = session.directory / RESULT_FILES[EVALUATOR_HUMAN]
    records = human_results(
        lines,
        run=session.run,
        config=session.config,
        invalid_targets=invalid_targets(session.checks),
        existing=read_results(path),
        import_sha256=import_sha256,
    )
    session.started()
    appended = append_jsonl(path, records)
    session.log("finished", status=STATUS_COMPLETED, ratings=len(lines), records_appended=appended)
    summary = {"ratings": len(lines), "records_appended": appended, "warnings": session.warnings()}
    _print({"assessment_dir": str(session.directory), **summary})
    return EXIT_OK


def run_report(session: Session) -> int:
    stores = {kind: read_results(session.directory / name) for kind, name in RESULT_FILES.items()}
    rubric_version = session.args.rubric_version or session.config.rubric_version
    report = build_report(
        run=session.run,
        config=session.config,
        checks=session.checks,
        stores=stores,
        rubric_version=rubric_version,
        generated_at=utc_now(),
    )
    path = session.directory / f"report-{session.batch_id}.json"
    _write_new(path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    session.log("finished", status=STATUS_COMPLETED, report=path.name)
    _print({"report": str(path), "rubric_version": rubric_version, "records": {k: len(v) for k, v in stores.items()}})
    return EXIT_OK


# --- main -----------------------------------------------------------------------------------------


def _error(exc: BaseException) -> str:
    """Project-controlled messages are printed; anything else only by class name (no traceback)."""
    if isinstance(exc, (AssessError, OSError, OpenAIClientError, JudgeCallError)):
        return f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: details withheld"


def main(
    argv: list[str] | None = None,
    *,
    openai_client_factory: OpenAIClientFactory = make_openai_client,
    episode_loader: EpisodeLoader = load_episode,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "judge" and args.execute != args.confirm_paid_api:
        parser.error(GATE_MESSAGE)
    commands = {
        "check": run_check,
        "judge": lambda session: run_judge(session, openai_client_factory, episode_loader),
        "import-ratings": run_import,
        "report": run_report,
    }
    try:
        config = load_config(args.config)
        run = load_run(args.translator_run)
        checks = check_generation_inputs(run, config)
        session = Session(args, config, run, checks, assessment_dir(args, config, run), new_batch_id())
        return commands[args.command](session)
    except INTERRUPTIONS as exc:  # Ctrl-C included; judge calls handle their own failures
        print(f"error: {_error(exc)}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
