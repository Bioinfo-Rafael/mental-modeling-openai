#!/bin/bash
# Method B (weights + activations, weightreading intent mode) of the intent-translation comparison.
# Default: sandbox self-test + local preview only (messages, session inputs, token/cost bound; no
# API call, no key read). A PAID run happens only when INTENT_COMPARE_CONFIRM_PAID=yes is exported:
#   mkdir -p outputs/intent_compare/slurm
#   sbatch scripts/intent_compare_run_b.sbatch                                         # preview
#   INTENT_COMPARE_CONFIRM_PAID=yes RUN_ID=seed100601 sbatch --export=ALL scripts/intent_compare_run_b.sbatch
# OPENAI_API_KEY must already be in the submitting environment (read only after the sandbox
# preflight passed); this script never reads or sources a .env file.
# Optional: RUN_ID (default seed<seed>), RESUME=yes (continue an interrupted paid run with the same
# RUN_ID), COMPARE, CONFIG, UV. Sessions run sequentially; the sandbox has no cgroup limits, so keep
# paid runs inside a Slurm job.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=ic-run-b
#SBATCH --output=outputs/intent_compare/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-$HOME/.local/bin/uv}"
COMPARE="${COMPARE:-configs/intent_compare/compare.yaml}"
CONFIG="${CONFIG:-configs/weightreading/intent_b.yaml}"
ARGS=(--compare "$COMPARE" --config "$CONFIG")

"$UV" run --no-sync python -m weightreading sandbox-selftest --config "$CONFIG"
"$UV" run --no-sync python -m weightreading intent-preview "${ARGS[@]}"
if [ "${INTENT_COMPARE_CONFIRM_PAID:-}" = "yes" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY is not set" >&2; exit 2; fi
  RUN_ARGS=("${ARGS[@]}")
  if [ -n "${RUN_ID:-}" ]; then RUN_ARGS+=(--run-id "$RUN_ID"); fi
  if [ "${RESUME:-}" = "yes" ]; then RUN_ARGS+=(--resume); fi
  "$UV" run --no-sync python -m weightreading intent-run "${RUN_ARGS[@]}" --execute --confirm-paid-api
  "$UV" run --no-sync python scripts/intent_compare_viewer.py --config "$COMPARE" ${RUN_ID:+--run-id "$RUN_ID"}
else
  echo "Preview only (set INTENT_COMPARE_CONFIRM_PAID=yes for a paid run)."
fi
