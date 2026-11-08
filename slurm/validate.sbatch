#!/bin/bash
# Offline validation of the core. No mode sends an API request: OPENAI_API_KEY is always unset.
# MODE (first argument):
#   subset            pytest of the Translator, LLM, LAG, methods A/B and judge tests
#   full              the whole core pytest
#   static            git diff --check, ruff check + format check, bash -n of slurm/, imports from
#                     outside the repository, and the stale-path grep against its allowlist
#   golden            request-path manifest, request goldens, the replay of recorded requests, the paid
#                     assess journal and method B resume against an existing checkout's gitignored
#                     outputs (SOURCE_ROOT below),
#                     and the blob IDs of committed results
#   method-a-preview  a method A preview into job scratch; it must write no requests/responses
# SOURCE_ROOT (golden, method-a-preview): the checkout whose outputs/ hold the real runs; default
# this repository. Set it with --export=ALL,SOURCE_ROOT=<path> to check another checkout read-only.
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
MODE="${1:-subset}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT}"
GOLDENS=tests/fixtures/prompt-request-goldens.json
RESULTS_BASE=d8b2d123ca141eb5d9abe4a4cbee0d59968d4363  # last commit before the core/workspace split
ASSESS_RUN=intentcmp-seed100601-ja-A-s045-A0100         # the one paid assess run (2026-11-05)
require_python
log_header "$0" "$MODE" "source_root=$SOURCE_ROOT"

golden_value() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$GOLDENS" "$1"; }

case "$MODE" in
  subset)
    "$PY" -m pytest -q tests/translator tests/llm tests/lag tests/intent_compare tests/weightreading tests/evaluation
    ;;
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
      "$PY" -B -c 'import evaluation, translate, mtd_rl_xavier, mtd_rl_xavier.intent_compare, weightreading'
      for module in evaluation.cli evaluation.assess_cli mtd_rl_xavier.translator.cli \
          mtd_rl_xavier.intent_compare.method_a_cli weightreading; do
        "$PY" -B -m "$module" --help >/dev/null
      done
      echo "imports and --help from outside the repository: ok"
    )
    found="$(mktemp)"
    # No line numbers (-n is dropped on purpose): an edit elsewhere in a file must not break the allowlist.
    git grep -hE -f tests/fixtures/stale-paths.pattern -- README.md pyproject.toml src configs tests slurm scripts \
      ':!tests/fixtures/stale-paths.pattern' ':!tests/fixtures/allowed-stale-paths.txt' > "$found" || true
    diff -u tests/fixtures/allowed-stale-paths.txt "$found"
    echo "stale-path grep: only allowlisted lines"
    ;;
  golden)
    sha256sum -c tests/fixtures/assess-request-path.sha256
    echo "$(golden_value converter_sha256)  tests/fixtures/frozen/to_translator_run_v2.py.txt" | sha256sum -c
    "$PY" -m pytest -q tests/intent_compare/test_migration_goldens.py tests/evaluation/test_migration_goldens.py \
      tests/weightreading/test_wr_intent_goldens.py
    "$PY" -B scripts/check_request_replay.py --source-root "$SOURCE_ROOT"
    "$PY" -B scripts/check_assess_resume.py \
      --translator-run "$SOURCE_ROOT/outputs/translator/$ASSESS_RUN" \
      --journal "$SOURCE_ROOT/outputs/evaluation/assess/$ASSESS_RUN/judge_requests.jsonl" \
      --expect "$(golden_value paid_assess_request_sha256)"
    "$PY" -B scripts/check_method_b_resume.py \
      --runs "$SOURCE_ROOT/outputs/weightreading/intent/runs" --recorded-root "$SOURCE_ROOT"
    "$PY" -m pytest -q tests/weightreading/test_wr_intent_run.py -k 'resume or inputs'
    # Committed experiment results never change (new ones may be added): no modification, deletion or
    # rename. The pathspecs match by shape only; the core never reads these files.
    git diff --exit-code --no-renames --diff-filter=DMT --stat "$RESULTS_BASE" HEAD -- ':(glob)*/*/results/**'
    # The three tracked output trees moved byte-identically with the legacy project.
    diff <(git ls-tree -r "$RESULTS_BASE" -- outputs | awk '{print $3, $4}') \
         <(git ls-tree -r HEAD | awk '$4 ~ /^[^\/]+\/outputs\// {sub("^[^/]+/outputs/", "outputs/", $4); print $3, $4}')
    echo "golden checks: ok"
    ;;
  method-a-preview)
    scratch="$(mktemp -d)"
    trap 'status=$?; rm -rf "$scratch"; echo "end=$(date -Is) exit=$status"' EXIT
    episode="$SOURCE_ROOT/outputs/intent_compare/episode/seed100601"
    mkdir -p "$scratch/out/episode/seed100601"
    cp "$episode/seed100601.csv" "$episode/manifest.json" "$scratch/out/episode/seed100601/"
    "$PY" -B - "$scratch" "$SOURCE_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

scratch, source = Path(sys.argv[1]), Path(sys.argv[2])
raw = yaml.safe_load(Path("configs/intent_compare/compare.yaml").read_text(encoding="utf-8"))
raw = {**raw, "output_root": str(scratch / "out"), "rollout_dir": str(source / raw["rollout_dir"])}
(scratch / "compare.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY
    (cd "$scratch" && "$PY" -B -m mtd_rl_xavier.intent_compare.method_a_cli --config "$scratch/compare.yaml" \
      --run-id migration-preview)
    method_dir="$scratch/out/migration-preview/method_a"
    test -f "$method_dir/preview/preview.json"
    test ! -e "$method_dir/requests.jsonl"
    test ! -e "$method_dir/responses.jsonl"
    find "$method_dir" -type f | sed "s#$scratch#<scratch>#" | sort
    echo "method A preview: ok (no requests.jsonl or responses.jsonl)"
    ;;
  *)
    echo "MODE must be subset, full, static, golden or method-a-preview (got $MODE)" >&2; exit 2
    ;;
esac
