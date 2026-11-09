#!/bin/bash
# The LLM judge of one translation run (python -m evaluation judge), on a compute node.
# MODE (first argument): preview (default; the plan and every request in preview.json, 0 API calls,
# no key read) | paid (`--execute --confirm-paid-api`, then `report`). A paid run never resends a
# request journaled anywhere under the judge's output root. Everything after MODE goes to the judge
# (--run is required; --config, --name).
#   sbatch slurm/evaluate.sbatch preview --run outputs/translate/<run>
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/evaluate.sbatch paid --run outputs/translate/<run>
# Exit codes pass through: 0 ok or preview, 1 failure, 2 usage, 3 uncertain call, 4 cap.
#SBATCH --partition=151-partition
#SBATCH --job-name=evaluate
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
CMD=("$PY" -m evaluation judge "$@" "${PAID_FLAGS[@]}")
log_header "${CMD[@]}"
"${CMD[@]}"
if [ "$MODE" = paid ]; then
  "$PY" -m evaluation report "$@"
fi
