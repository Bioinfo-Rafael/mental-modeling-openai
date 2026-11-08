#!/bin/bash
# Side-by-side HTML viewer of translation runs (python -m evaluation.viewer); no API call.
# Every argument goes to evaluation.viewer: one --column NAME=PATH per run directory or cue file.
#   sbatch slurm/viewer.sbatch --column trajectory=outputs/translate/<run> --column policy=outputs/translate/<run> \
#     --clips outputs/intent_compare/episode/seed100601/clips --language ja --out outputs/translate/<run>/viewer.html
#SBATCH --partition=151-partition
#SBATCH --job-name=viewer
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
require_python
CMD=("$PY" -m evaluation.viewer "$@")
log_header "${CMD[@]}"
"${CMD[@]}"
