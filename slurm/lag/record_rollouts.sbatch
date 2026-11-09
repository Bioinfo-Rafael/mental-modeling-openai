#!/bin/bash
# Record MHI 2v2 team-actor rollouts in mtd-rl-lag on a compute node (CPU): mtd-rl-lag's own
# interpreter (torch + JSBSim) runs scripts/lag/record_rollouts.py. No API is called; nothing in
# either repository is modified. lag/prepare_episode.sbatch then proves the recording against mtd-rl-lag's
# unmodified evaluate_mhi_team.py (byte-identical ACMI) and converts it to an episode CSV.
#   sbatch slurm/lag/record_rollouts.sbatch
# Optional: LAG_ROOT (default ../mtd-rl-lag), MODEL_DIR (default $LAG_ROOT/models/v1.3), MODEL_ALIAS (mhi-v1.3), PROTOCOL
# (per_agent_v1 | mhi_render_v1_3), SEEDS ("100600 100601 100602 100603"), SCENARIO, MAX_STEPS.
#SBATCH --partition=151-partition
#SBATCH --job-name=record_rollouts
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
LAG_PYTHON="$LAG_ROOT/.venv/bin/python"
MODEL_DIR="${MODEL_DIR:-$LAG_ROOT/models/v1.3}"
MODEL_ALIAS="${MODEL_ALIAS:-mhi-v1.3}"
PROTOCOL="${PROTOCOL:-per_agent_v1}"
SCENARIO="${SCENARIO:-2v2/ShootMissile/HierarchyVsbaseline_mhi_v1_3}"
read -r -a SEED_LIST <<< "${SEEDS:-100600 100601 100602 100603}"
CHECKPOINT="$MODEL_DIR/files/actor_latest.pt"
OUT="outputs/rollouts/${MODEL_ALIAS}-$(date -u +%Y%m%dT%H%M%SZ)"  # earlier recordings: outputs/weightreading/rollouts/
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
STEP_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARGS=(--max-steps "$MAX_STEPS"); fi
log_header "$0" "out=$OUT" "protocol=$PROTOCOL" "seeds=${SEED_LIST[*]}" "scenario=$SCENARIO"

SECONDS=0
"$LAG_PYTHON" -B scripts/lag/record_rollouts.py --lag-root "$LAG_ROOT" --checkpoint "$CHECKPOINT" \
  --out "$OUT" --model-alias "$MODEL_ALIAS" --protocol "$PROTOCOL" --scenario "$SCENARIO" \
  --seeds "${SEED_LIST[@]}" "${STEP_ARGS[@]}"
echo "record: ${SECONDS}s; outputs in $OUT"
