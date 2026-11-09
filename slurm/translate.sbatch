#!/bin/bash
# Translate one episode with one run config (python -m translate), on a compute node (CPU).
# MODE (first argument): preview (default; the schedule and the exact first request, 0 API calls,
# no key read) | paid (`--execute --confirm-paid-api`). A clef_flash config previews here; its paid
# run needs the detector server: slurm/translate_online.sbatch. Everything after MODE goes to
# `python -m translate` (--config is required; --scenario, --run-name).
#   sbatch slurm/translate.sbatch preview --config configs/translate/preset.yaml
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/translate.sbatch paid --config configs/translate/preset.yaml
#SBATCH --partition=151-partition
#SBATCH --job-name=translate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
CONFIG="$(arg_value --config "$@")"
if [ -z "$CONFIG" ]; then echo "pass --config configs/translate/<name>.yaml" >&2; exit 2; fi
SOURCE="$("$PY" -c 'import sys; from pathlib import Path; from translate.config import load_config; print(load_config(Path(sys.argv[1])).config.change.source)' "$CONFIG")"
if [ "$MODE" = paid ] && [ "$SOURCE" = clef_flash ]; then
  echo "$CONFIG uses the clef-flash detector: submit slurm/translate_online.sbatch paid" >&2; exit 2
fi
CMD=("$PY" -m translate "$@" "${PAID_FLAGS[@]}")
log_header "${CMD[@]}"
"${CMD[@]}"
