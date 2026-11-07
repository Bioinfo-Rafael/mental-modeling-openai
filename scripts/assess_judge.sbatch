#!/bin/bash
# Per-criterion LLM judge (evaluation.assess_cli) of one Translator v2 run on a compute node.
# Default: `check` + judge PREVIEW only (packets + cost estimate, 0 API requests, no key read).
# A PAID run happens only when ASSESS_CONFIRM_PAID=yes is exported explicitly: `check`, then
# `judge --execute --confirm-paid-api` (one call per event, at most judge.max_calls), then `report`.
#   mkdir -p outputs/evaluation/assess/slurm
#   RUN=outputs/translator/<run> sbatch --export=ALL scripts/assess_judge.sbatch                          # preview
#   set -a; . <key file>; set +a; RUN=outputs/translator/<run> ASSESS_CONFIRM_PAID=yes sbatch --export=ALL scripts/assess_judge.sbatch
# OPENAI_API_KEY must already be in the submitting environment; this script never reads or sources a
# .env or key file and never prints the key.
# Required: RUN (a completed Translator v2 run directory).
# Optional: CONFIG (evaluation config, default evaluation/config/assess.yaml), UV.
# Exit codes of assess_cli are passed through: 0 ok or preview, 1 failure, 2 usage, 3 uncertain call, 4 cap.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=assess-judge
#SBATCH --output=outputs/evaluation/assess/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
unset VIRTUAL_ENV
UV="${UV:-$HOME/.local/bin/uv}"
RUN="${RUN:?set RUN to the Translator v2 run directory (outputs/translator/<run>)}"
CONFIG="${CONFIG:-evaluation/config/assess.yaml}"
ASSESS=("$UV" run --no-sync python -m evaluation.assess_cli)
ARGS=(--config "$CONFIG" --translator-run "$RUN")

"${ASSESS[@]}" check "${ARGS[@]}"
if [ "${ASSESS_CONFIRM_PAID:-}" = "yes" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY is not set" >&2; exit 2; fi
  "${ASSESS[@]}" judge "${ARGS[@]}" --execute --confirm-paid-api
  "${ASSESS[@]}" report "${ARGS[@]}"
else
  "${ASSESS[@]}" judge "${ARGS[@]}"
  echo "Preview only (set ASSESS_CONFIRM_PAID=yes for a paid run)."
fi
