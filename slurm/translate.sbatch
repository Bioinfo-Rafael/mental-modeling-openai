#!/bin/bash
# Translate one episode with one run configuration (python -m translate), on a compute node.
# Choose inputs, prompt and trigger points in the config (configs/translate/*.yaml); this job runs
# preset-trigger configs. Online configs need the clef-flash detector: use slurm/translate_vllm.sbatch.
# MODE (first argument): preview (default; requests + cost estimate, 0 API calls, no key read)
# | paid (`--execute --confirm-paid-api`).
# Everything after MODE goes to `python -m translate` (--config is required; --run-name, --scenario).
#   sbatch slurm/translate.sbatch preview --config configs/translate/seed100601-trajectory.yaml
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/translate.sbatch paid \
#     --config configs/translate/seed100601-trajectory.yaml --run-name seed100601-trajectory
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
SOURCE="$("$PY" -c 'import sys; from pathlib import Path; from translate.config import load_config; print(load_config(Path(sys.argv[1])).triggers.source)' "$CONFIG")"
if [ "$SOURCE" != preset ]; then
  echo "$CONFIG uses $SOURCE triggers (the clef-flash detector): submit slurm/translate_vllm.sbatch instead" >&2; exit 2
fi

CMD=("$PY" -m translate "$@" "${PAID_FLAGS[@]}")
log_header "${CMD[@]}"
"${CMD[@]}"
