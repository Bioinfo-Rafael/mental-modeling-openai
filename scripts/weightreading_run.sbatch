#!/bin/bash
# weightreading session on a compute node. Default: sandbox self-test + local preview only.
# A PAID run happens only when WEIGHTREADING_CONFIRM_PAID=yes is exported explicitly:
#   mkdir -p outputs/weightreading/slurm
#   sbatch scripts/weightreading_run.sbatch                                  # self-test + preview
#   WEIGHTREADING_CONFIRM_PAID=yes OPENAI_API_KEY=... sbatch --export=ALL scripts/weightreading_run.sbatch
# Optional: SPEC (default lag_baseline_12), CHECKPOINT, TASK_MODE (shown|blind), CONFIG.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=wr-run
#SBATCH --output=outputs/weightreading/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-$HOME/.local/bin/uv}"
MODEL_DIR="${WEIGHTREADING_LAG_MODEL_DIR:-/home/user151005/workspace/mtd-rl-lag/envs/JSBSim/model}"
SPEC="configs/weightreading/${SPEC:-lag_baseline_12}.spec.json"
CHECKPOINT="${CHECKPOINT:-$MODEL_DIR/baseline_model.pt}"
CONFIG="${CONFIG:-configs/weightreading/run.yaml}"
ARGS=(--checkpoint "$CHECKPOINT" --spec "$SPEC" --config "$CONFIG" --task-mode "${TASK_MODE:-shown}")

"$UV" run python -m weightreading sandbox-selftest --config "$CONFIG"
if [ "${WEIGHTREADING_CONFIRM_PAID:-}" = "yes" ]; then
  "$UV" run python -m weightreading run "${ARGS[@]}" --execute --confirm-paid-api
else
  "$UV" run python -m weightreading run "${ARGS[@]}"
  echo "Preview only (set WEIGHTREADING_CONFIRM_PAID=yes for a paid run)."
fi
