#!/bin/bash
# Method A of the intent comparison (trajectory-based Translator at fixed steps).
# MODE (first argument): preview (default; prompts + token/cost estimate, 0 API calls, no key read)
# | paid (`--execute --confirm-paid-api`, then the A/B viewer). Everything after MODE goes to
# `python -m mtd_rl_xavier.intent_compare.method_a_cli` (--config, --run-id, --max-output-tokens);
# --config defaults to configs/intent_compare/compare.yaml. Use a new --run-id per paid run.
#   sbatch slurm/translator_method_a.sbatch preview --config configs/intent_compare/compare.ja.yaml
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/translator_method_a.sbatch paid \
#     --config configs/intent_compare/compare.ja.yaml --run-id seed100601-ja
# To judge one call: python -m mtd_rl_xavier.intent_compare.to_translator_run (see README).
#SBATCH --partition=151-partition
#SBATCH --job-name=translator_method_a
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
COMPARE="$(arg_value --config "$@")"
if [ -z "$COMPARE" ]; then COMPARE=configs/intent_compare/compare.yaml; set -- --config "$COMPARE" "$@"; fi
RUN_ID="$(arg_value --run-id "$@")"

CMD=("$PY" -m mtd_rl_xavier.intent_compare.method_a_cli "$@" "${PAID_FLAGS[@]}")
log_header "${CMD[@]}"
"${CMD[@]}"
if [ "$MODE" = paid ]; then
  "$PY" -m mtd_rl_xavier.intent_compare.viewer --config "$COMPARE" ${RUN_ID:+--run-id "$RUN_ID"}
fi
