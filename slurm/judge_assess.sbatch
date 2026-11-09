#!/bin/bash
# Per-criterion LLM judge (python -m evaluation.assess_cli) of one Translator run (event schema 2 or 3).
# The judge config defaults to configs/evaluation/assess.yaml; resume the paid 2026-11-05 run with
# --config configs/evaluation/assess-legacy.yaml.
# MODE (first argument): preview (default) = `check` + judge preview (packets, cost estimate, 0 API
# calls, no key read) | paid = `check`, `judge --execute --confirm-paid-api` (one call per event not
# yet in judge_requests.jsonl, at most judge.max_calls), then `report`. Everything after MODE goes
# to assess_cli (`--translator-run` is required; `--points` is passed to check and judge only).
#   sbatch slurm/judge_assess.sbatch preview --translator-run outputs/translator/<run>
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/judge_assess.sbatch paid --translator-run outputs/translator/<run>
# Exit codes of assess_cli pass through: 0 ok or preview, 1 failure, 2 usage, 3 uncertain call, 4 cap.
#SBATCH --partition=151-partition
#SBATCH --job-name=judge_assess
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
ASSESS=("$PY" -m evaluation.assess_cli)
REPORT_ARGS=()                           # report takes every argument except --points <file>
skip_next=0
for arg in "$@"; do
  if [ "$skip_next" = 1 ]; then skip_next=0; continue; fi
  if [ "$arg" = "--points" ]; then skip_next=1; continue; fi
  REPORT_ARGS+=("$arg")
done

log_header "${ASSESS[@]}" judge "$@" "${PAID_FLAGS[@]}"
"${ASSESS[@]}" check "$@"
"${ASSESS[@]}" judge "$@" "${PAID_FLAGS[@]}"
if [ "$MODE" = paid ]; then
  "${ASSESS[@]}" report "${REPORT_ARGS[@]}"
fi
