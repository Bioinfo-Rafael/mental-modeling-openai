#!/bin/bash
# Offline validation of the core. No mode sends an API request: OPENAI_API_KEY is always unset.
# MODE (first argument):
#   full     the whole core pytest
#   static   git diff --check, ruff check + format check, bash -n of slurm/, imports from outside
#            the repository
#   preview  a preview of every shipped config of both packages with API client factories that
#            raise: the translate configs, then the judge configs over a synthetic run made in job
#            scratch; no preview may write a request or response journal
# SOURCE_ROOT (preview): the checkout whose outputs/ hold the real runs and derived episodes;
# default this repository. Set it with --export=ALL,SOURCE_ROOT=<path> to read another checkout.
#   sbatch slurm/validate.sbatch full
#SBATCH --partition=151-partition
#SBATCH --job-name=validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
unset OPENAI_API_KEY OPENAI_LOG OPENAI_CUSTOM_HEADERS
MODE="${1:-full}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT}"
require_python
log_header "$0" "$MODE" "source_root=$SOURCE_ROOT"

case "$MODE" in
  full)
    "$PY" -m pytest -q
    ;;
  static)
    git diff --check
    "$PY" -m ruff check src tests scripts
    "$PY" -m ruff format --check src tests scripts
    bash -n slurm/common.sh
    for job in slurm/*.sbatch; do bash -n "$job"; done
    (
      cd "$(mktemp -d)"
      "$PY" -B -c 'import evaluation, translate, translate.cli, translate.policy.material'
      for module in evaluation.cli evaluation.assess_cli evaluation.viewer evaluation.clips translate; do
        "$PY" -B -m "$module" --help >/dev/null
      done
      echo "imports and --help from outside the repository: ok"
    )
    ;;
  preview)
    scratch="$(mktemp -d)"
    trap 'status=$?; rm -rf "$scratch"; echo "end=$(date -Is) exit=$status"' EXIT
    "$PY" -B - "$scratch" "$SOURCE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from evaluation.assess_cli import main as assess_main
from evaluation.cli import main as keypoint_main
from translate.cli import Factories
from translate.cli import main as translate_main

scratch, source = Path(sys.argv[1]), Path(sys.argv[2])


def refuse(*_args, **_kwargs):
    raise SystemExit("a preview tried to build an API client")


class CannedClient:  # the synthetic run only: one fixed answer per call, no network
    def __init__(self):
        self.responses = self

    def create(self, **body):
        answer = json.dumps({"intent": "[SYNTHETIC] intent", "evidence": ["[SYNTHETIC] evidence"]})
        output = [{"type": "message", "content": [{"type": "output_text", "text": answer}]}]
        return {"id": "resp_synthetic", "model": body["model"], "status": "completed", "output": output}


def relocated(config):
    """Copies of a config and its scenario whose generated outputs/ inputs are read from SOURCE_ROOT."""
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenario_path = Path(raw["scenario"])
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    episode = dict(scenario["episode"])
    for key in ("csv_path", "video_path"):
        if episode[key] is not None:
            relative = (scenario_path.parent / episode[key]).resolve().relative_to(Path.cwd())
            episode[key] = str(source / relative if relative.parts[0] == "outputs" else relative.resolve())
    scenario_file = scratch / f"{config.stem}.scenario.yaml"
    scenario_file.write_text(yaml.safe_dump({**scenario, "episode": episode}), encoding="utf-8")
    if raw["policy"] is not None:
        raw = {**raw, "policy": {**raw["policy"], "rollout_dir": str(source / raw["policy"]["rollout_dir"])}}
    config_file = scratch / config.name
    config_file.write_text(yaml.safe_dump({**raw, "output": {"root": str(scratch / "translate")}}), encoding="utf-8")
    return config_file, scenario_file


def check(code, what):
    if code != 0:
        sys.exit(f"{what}: exited {code}")


refusing = Factories(detector_client=refuse, openai_client=refuse, preflight=refuse)
configs = sorted(Path("configs/translate").glob("*.yaml"))
for config in configs:
    config_file, scenario_file = relocated(config)
    argv = ["--config", str(config_file), "--scenario", str(scenario_file), "--run-name", config.stem]
    check(translate_main(argv, factories=refusing), str(config))

# A synthetic schema-3 run of the example scenario (canned answers), then every judge config's preview.
raw = yaml.safe_load(Path("configs/translate/seed100601-trajectory.yaml").read_text(encoding="utf-8"))
points = [{"step": 0, "kind": "start", "targets": None}, {"step": 145, "kind": "change", "targets": None}]
points.append({"step": 10, "kind": "request", "targets": [{"kind": "aircraft", "id": "A0100"}]})
raw = {**raw, "scenario": "configs/translate/scenarios/example.yaml", "triggers": {**raw["triggers"], "points": points}}
synthetic = scratch / "synthetic.yaml"
synthetic.write_text(yaml.safe_dump({**raw, "output": {"root": str(scratch / "runs")}}), encoding="utf-8")
argv = ["--config", str(synthetic), "--run-name", "synthetic", "--execute", "--confirm-paid-api"]
check(translate_main(argv, factories=Factories(openai_client=lambda **_: CannedClient())), "synthetic run")
run, judged = scratch / "runs" / "synthetic", scratch / "evaluation"
argv = ["judge", "--translator-run", str(run), "--output-root", str(judged)]
check(assess_main(argv, openai_client_factory=refuse), "configs/evaluation/assess.yaml")
key_points = "src/evaluation/examples/key-points.example.yaml"
for config in sorted(Path("configs/evaluation").glob("judge-text*.yaml")):
    argv = ["--config", str(config), "--translator-run", str(run), "--key-points", key_points]
    argv += ["--output-root", str(judged), "--run-name", config.stem]
    check(keypoint_main(argv, openai_client_factory=refuse), str(config))
legacy_run = source / "outputs/translator/intentcmp-seed100601-ja-A-s045-A0100"  # schema 2: legacy config only
argv = ["judge", "--config", "configs/evaluation/assess-legacy.yaml", "--translator-run", str(legacy_run)]
argv += ["--output-root", str(judged), "--name", "legacy"]
check(assess_main(argv, openai_client_factory=refuse), "configs/evaluation/assess-legacy.yaml")
journals = ("requests.jsonl", "responses.jsonl")
written = sorted(str(p) for root in (scratch / "translate", judged) for p in root.rglob("*") if p.name.endswith(journals))
if written:
    sys.exit(f"a preview wrote {written}")
print(f"previews of {len(configs)} translate configs and every judge config: ok (no request or response journal)")
PY
    ;;
  *)
    echo "MODE must be full, static or preview (got $MODE)" >&2; exit 2
    ;;
esac
