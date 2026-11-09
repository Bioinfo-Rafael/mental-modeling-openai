#!/bin/bash
# Create or update the core .venv from uv.lock (the only job that changes the environment).
# Run it before the first job, after pulling a uv.lock change, and while no other core job runs.
#   sbatch slurm/uv_sync.sbatch
#SBATCH --partition=151-partition
#SBATCH --job-name=uv_sync
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
UV="${UV:-$HOME/.local/bin/uv}"
CMD=("$UV" sync --locked --all-extras --reinstall-package mtd-rl-xavier)  # the editable mapping follows pyproject
log_header "${CMD[@]}"
"${CMD[@]}"
