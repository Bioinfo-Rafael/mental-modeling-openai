#!/bin/bash
# D29 key-point LLM judge (python -m evaluation.cli) of one Translator v2 run.
# MODE (first argument): preview (default; run.json + preview.json + example request, 0 API calls)
# | paid (`--execute --confirm-paid-api`: one call per event with human key points; no resume, so a
# re-run re-sends every judged event). Everything after MODE goes to evaluation.cli, which needs
# `--translator-run` and `--key-points private/evaluation/<file>.yaml`.
#   sbatch slurm/judge_keypoints.sbatch preview --translator-run outputs/translator/<run> --key-points private/evaluation/<kp>.yaml
#   set -a; . ./.env; set +a; sbatch --export=ALL slurm/judge_keypoints.sbatch paid --translator-run ... --key-points ...
# Video frames: add `--config configs/evaluation/judge-text-video.yaml` (needs the `video` extra).
#SBATCH --partition=151-partition
#SBATCH --job-name=judge_keypoints
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
set_mode "${1:-preview}"; shift || true
require_python
CMD=("$PY" -m evaluation.cli "$@" "${PAID_FLAGS[@]}")
log_header "${CMD[@]}"
"${CMD[@]}"
