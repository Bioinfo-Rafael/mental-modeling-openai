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
    for job in slurm/*.sbatch slurm/lag/*.sbatch; do bash -n "$job"; done
    (
      cd "$(mktemp -d)"
      "$PY" -B -c 'import evaluation, translate, translate.cli'
      for module in evaluation translate; do
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

from evaluation.cli import main as judge_main
from translate.backends import Reply
from translate.backends.openai import OpenAIBackend
from translate.cli import main as translate_main

scratch, source = Path(sys.argv[1]), Path(sys.argv[2])


def refuse(*_args, **_kwargs):
    raise SystemExit("a preview tried to build an API client")


class CannedBackend:  # the synthetic run only: one fixed answer per call, no network
    request_body = staticmethod(OpenAIBackend.request_body)

    def send(self, body):
        text = json.dumps({"intent": "[SYNTHETIC] intent", "evidence": ["[SYNTHETIC] evidence"]})
        return Reply(text, body["model"], "resp_synthetic", {}, {"id": "resp_synthetic"})


def from_source(path):
    """A generated outputs/ input is read from SOURCE_ROOT; anything else from this checkout."""
    relative = path.resolve().relative_to(Path.cwd())
    return str(source / relative if relative.parts[0] == "outputs" else relative.resolve())


def relocated(config, output):
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenario_path = Path(raw["scenario"])
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario_file = scratch / f"{config.stem}.scenario.yaml"
    scenario_file.write_text(yaml.safe_dump({**scenario, "episode": from_source(scenario_path.parent / scenario["episode"])}))
    config_file = scratch / config.name
    config_file.write_text(yaml.safe_dump({**raw, "scenario": str(scenario_file), "output": str(output)}), encoding="utf-8")
    return config_file


def check(code, what):
    if code != 0:
        sys.exit(f"{what}: exited {code}")


configs = sorted(Path("configs/translate").glob("*.yaml"))
for config in configs:
    argv = ["--config", str(relocated(config, scratch / "translate")), "--run-name", config.stem]
    check(translate_main(argv, backend_factory=refuse, detector_factory=refuse), str(config))

# A synthetic run of the preset config (canned answers, no network), then every judge config's preview.
argv = ["--config", str(relocated(Path("configs/translate/preset.yaml"), scratch / "runs")), "--run-name", "synthetic"]
check(translate_main([*argv, "--execute", "--confirm-paid-api"], backend_factory=lambda model: CannedBackend()), "synthetic run")
run, judged = scratch / "runs" / "synthetic", scratch / "evaluation"
judge_configs = sorted(Path("configs/evaluation").glob("*.yaml"))
for config in judge_configs:  # each with its output in scratch, so the scan reads no real journal
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    (scratch / config.name).write_text(yaml.safe_dump({**raw, "output": str(judged)}), encoding="utf-8")
    check(judge_main(["judge", "--run", str(run), "--config", str(scratch / config.name)], client_factory=refuse), str(config))
journals = ("requests.jsonl", "responses.jsonl")
written = sorted(str(p) for root in (scratch / "translate", judged) for p in root.rglob("*") if p.name.endswith(journals))
if written:
    sys.exit(f"a preview wrote {written}")
print(f"previews of {len(configs)} translate and {len(judge_configs)} judge configs: ok (no request or response journal)")
PY
    ;;
  *)
    echo "MODE must be full, static or preview (got $MODE)" >&2; exit 2
    ;;
esac
