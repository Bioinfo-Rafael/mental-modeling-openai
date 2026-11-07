#!/bin/bash
# Record MHI 2v2 team-actor rollouts in mtd-rl-lag and replay them through weightreading's numpy
# reference forward, on a compute node. No API is called; nothing in either repository is modified.
#   sbatch slurm/translator_record_rollouts.sbatch
# 1. record: mtd-rl-lag's own interpreter (torch + JSBSim) runs weightreading_record_rollouts.py;
# 2. cross-check (XCHECK=1): mtd-rl-lag's unmodified evaluate_mhi_team.py replays the same seeds into
#    $LAG_ROOT/LOCAL_ONLY/ (gitignored there; the script refuses outputs outside its worktree);
# 3. replay: an isolated numpy-only uv environment runs weightreading_replay_rollouts.py.
# Optional: LAG_ROOT (default ../mtd-rl-lag), MODEL_DIR (default $LAG_ROOT/models/v1.3), MODEL_ALIAS (mhi-v1.3), PROTOCOL
# (per_agent_v1 | mhi_render_v1_3), SEEDS ("100600 100601 100602 100603"), SCENARIO, MAX_STEPS, XCHECK, UV.
#SBATCH --partition=151-partition
#SBATCH --job-name=translator_record_rollouts
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
UV="${UV:-$HOME/.local/bin/uv}"          # only for the throwaway numpy-only replay env
LAG_PYTHON="$LAG_ROOT/.venv/bin/python"
MODEL_DIR="${MODEL_DIR:-$LAG_ROOT/models/v1.3}"
MODEL_ALIAS="${MODEL_ALIAS:-mhi-v1.3}"
PROTOCOL="${PROTOCOL:-per_agent_v1}"
SCENARIO="${SCENARIO:-2v2/ShootMissile/HierarchyVsbaseline_mhi_v1_3}"
read -r -a SEED_LIST <<< "${SEEDS:-100600 100601 100602 100603}"
CHECKPOINT="$MODEL_DIR/files/actor_latest.pt"
OUT="outputs/weightreading/rollouts/${MODEL_ALIAS}-$(date -u +%Y%m%dT%H%M%SZ)"
XCHECK_ROOT="$LAG_ROOT/LOCAL_ONLY/weightreading_xcheck/${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
STEP_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then STEP_ARGS=(--max-steps "$MAX_STEPS"); fi
log_header "$0" "out=$OUT" "protocol=$PROTOCOL" "seeds=${SEED_LIST[*]}" "scenario=$SCENARIO"

SECONDS=0
"$LAG_PYTHON" -B scripts/weightreading_record_rollouts.py --lag-root "$LAG_ROOT" --checkpoint "$CHECKPOINT" \
  --out "$OUT" --model-alias "$MODEL_ALIAS" --protocol "$PROTOCOL" --scenario "$SCENARIO" \
  --seeds "${SEED_LIST[@]}" "${STEP_ARGS[@]}"
echo "record: ${SECONDS}s"

REPLAY_ARGS=()
if [ "${XCHECK:-1}" = "1" ]; then
  SECONDS=0
  for seed in "${SEED_LIST[@]}"; do
    "$LAG_PYTHON" -B "$LAG_ROOT/scripts/eval/evaluate_mhi_team.py" --checkpoint "$CHECKPOINT" \
      --scenario "$SCENARIO" --protocol "$PROTOCOL" --seed "$seed" "${STEP_ARGS[@]}" --output "$XCHECK_ROOT/$seed"
  done
  REPLAY_ARGS=(--lag-eval-root "$XCHECK_ROOT")
  echo "lag evaluate_mhi_team cross-check runs: ${SECONDS}s"
fi

SECONDS=0
PYTHONPATH=src "$UV" run --isolated --no-project --python 3.12 --with numpy \
  python scripts/weightreading_replay_rollouts.py "$OUT" "${REPLAY_ARGS[@]}"
echo "replay: ${SECONDS}s; outputs in $OUT"
