"""Offline, immutable run plans. No API client is constructed here."""
from pathlib import Path
import random
import shutil

from .dataset import lag, build_blocks, make_samples
from .prompts import ACTIVE_CONDITIONS, build_request, load_prompt, render_input, snapshot_prompt
from .storage import HERE, RESULTS, ROOT, digest, file_hash, read_json, safe_name, utc_now, write_json

STAGES = {"develop": "01_develop", "evaluate": "02_evaluate"}


def validate_config(config):
    if config["history"] not in (30, 40):
        raise ValueError("Use H=30 or H=40 for the specified fixed scenes")
    for key in ("stride", "repeats", "max_output_tokens", "timeout_seconds"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["step_tolerance"]) is not int or config["step_tolerance"] < 0:
        raise ValueError("step_tolerance must be a nonnegative integer")
    if config["reasoning_effort"] not in ("none", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported reasoning effort")
    conditions = config["conditions"]
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(ACTIVE_CONDITIONS):
        raise ValueError(f"Choose distinct conditions from {ACTIVE_CONDITIONS}")
    if config.get("decisions_task", "binary") not in ("binary", "aircraft_set"):
        raise ValueError("Unknown Decisions task")


def create_plan(stage, version, name, overrides=None):
    version, name = safe_name(version), safe_name(name)
    config = read_json(HERE / "configs" / f"{stage}.json")
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    validate_config(config)
    if stage == "develop" and config["conditions"] != ["responses_diagnostic"]:
        raise ValueError("Development uses Responses diagnostic output")
    annotations = read_json(HERE / "data/annotations.json")
    csv_path = lag.PROCESSED / (annotations["episode_id"] + ".csv")
    frame = lag.load_episode(csv_path)
    if set(frame.episode_id) != {annotations["episode_id"]} or frame.attrs["dt_s"] != .2:
        raise ValueError("Annotation episode or time unit mismatch")
    samples = make_samples(stage, config, annotations, build_blocks(frame))
    source_prompt = HERE / "prompts" / version
    loaded = load_prompt(source_prompt)
    if "decisions" in config["conditions"] and config.get("decisions_task") == "aircraft_set" and "aircraft_choices.json" not in loaded:
        raise ValueError("This prompt lacks aircraft-selection choices; use v003 or a version containing aircraft_choices.json")
    directory = RESULTS / STAGES[stage] / version / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()  # immutable run name; never silently reuse a run
    try:
        prompt = snapshot_prompt(source_prompt, directory / "prompt")
        write_json(directory / "samples.json", samples)
        write_json(directory / "annotations.json", annotations)
        shutil.copy2(HERE / "configs/pricing.json", directory / "pricing.json")
        (directory / "inputs").mkdir()
        for sid, sample in samples.items():
            (directory / "inputs" / f"{sid}.txt").write_text(
                render_input(prompt, sample["window"]), encoding="utf-8")
        queries = []
        rng = random.Random(config["order_seed"])
        for repeat in range(1, config["repeats"] + 1):
            for sid in samples:  # chronological sliding replay within each repeat
                conditions = list(config["conditions"])
                rng.shuffle(conditions)  # paired cases, randomized API order
                for condition in conditions:
                    user_text = (directory / "inputs" / f"{sid}.txt").read_text()
                    request = build_request(condition, prompt, user_text, config)
                    queries.append({"id": f"r{repeat:02d}_{sid}_{condition}", "sample_id": sid,
                                    "repeat": repeat, "condition": condition,
                                    "request_sha256": digest(request)})
        # Frozen code makes later prompt/code changes visible and supports auditing old runs.
        shutil.copytree(HERE / "decision_experiment", directory / "code",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        frozen_hashes = {str(p.relative_to(directory)): file_hash(p)
                         for p in sorted(directory.rglob("*")) if p.is_file()}
        manifest = {"schema_version": 1, "created_at": utc_now(), "stage": stage,
                    "prompt_version": version, "prompt_sha256": digest(prompt), "run_name": name,
                    "model": "gpt-6-luna", "config": config, "queries": queries,
                    "source": {"csv": str(csv_path.relative_to(ROOT)), "csv_sha256": file_hash(csv_path),
                               "manifest_sha256": file_hash(csv_path.parent / "manifest.json"),
                               "loader_sha256": file_hash(Path(lag.__file__))},
                    "files": frozen_hashes,
                    "limitations": ["Single episode reused for development and evaluation; not a held-out test.",
                                    "Unannotated windows are weak negatives, not verified absence of important events.",
                                    "Window-wide retrospective detection, not future-event prediction."]}
        write_json(directory / "manifest.json", manifest)
        (directory / "manifest.sha256").write_text(file_hash(directory / "manifest.json") + "\n")
    except BaseException:
        # Keep a visibly incomplete directory for debugging; do not turn it into an executable run.
        raise
    return directory


def verify_plan(directory):
    directory = Path(directory)
    expected = (directory / "manifest.sha256").read_text().strip()
    if file_hash(directory / "manifest.json") != expected:
        raise ValueError("Manifest changed after planning; create a new run")
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported plan format")
    for filename, sha in manifest["files"].items():
        if file_hash(directory / filename) != sha:
            raise ValueError(f"Frozen run input changed: {filename}")
    return manifest


def request_for(directory, manifest, query):
    prompt = load_prompt(Path(directory) / "prompt")
    user_text = (Path(directory) / "inputs" / f'{query["sample_id"]}.txt').read_text()
    body = build_request(query["condition"], prompt, user_text, manifest["config"])
    if digest(body) != query["request_sha256"]:
        raise ValueError(f'Request changed since planning: {query["id"]}')
    return body
