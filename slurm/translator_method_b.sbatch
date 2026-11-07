#!/bin/bash
# Method B of the intent comparison (weights + activations, `python -m weightreading`).
# MODE (first argument): preview (default; sandbox self-test + `intent-preview`: messages, session
# inputs, cost bound, 0 API calls, no key read) | paid (self-test, `intent-run --execute
# --confirm-paid-api`, then the A/B viewer). Everything after MODE goes to intent-preview /
# intent-run (--compare, --config, --run-id, --resume, ...). --compare and --config default to the
# repo-relative configs/intent_compare/compare.yaml and configs/weightreading/intent_b.yaml: `--resume`
# compares these recorded path strings, so pass them in the same form as the run you resume.
#   sbatch slurm/translator_method_b.sbatch preview --compare configs/intent_compare/compare.ja.yaml
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/translator_method_b.sbatch paid \
#     --compare configs/intent_compare/compare.ja.yaml --run-id seed100601-ja [--resume]
# Sessions run sequentially; the sandbox has no memory cgroup here, so keep paid runs inside a job.
#SBATCH --partition=151-partition
#SBATCH --job-name=translator_method_b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
COMPARE="$(arg_value --compare "$@")"
if [ -z "$COMPARE" ]; then COMPARE=configs/intent_compare/compare.yaml; set -- --compare "$COMPARE" "$@"; fi
CONFIG="$(arg_value --config "$@")"
if [ -z "$CONFIG" ]; then CONFIG=configs/weightreading/intent_b.yaml; set -- --config "$CONFIG" "$@"; fi
case "$CONFIG" in
  *intent_compare/*) echo "--config is method B's own config; pass the compare config as --compare" >&2; exit 2 ;;
esac
RUN_ID="$(arg_value --run-id "$@")"

log_header "$0" "$MODE" "config=$CONFIG"
"$PY" -m weightreading sandbox-selftest --config "$CONFIG"
if [ "$MODE" = preview ]; then
  CMD=("$PY" -m weightreading intent-preview "$@")
else
  CMD=("$PY" -m weightreading intent-run "$@" "${PAID_FLAGS[@]}")
fi
log_header "${CMD[@]}"
"${CMD[@]}"
if [ "$MODE" = paid ]; then
  "$PY" -m mtd_rl_xavier.intent_compare.viewer --config "$COMPARE" ${RUN_ID:+--run-id "$RUN_ID"}
fi
