#!/bin/bash
# Torch parity for weightreading's numpy reference forward pass, on a compute node.
# torch goes into an isolated, throwaway uv environment (CPU wheel); nothing is installed into this
# repository's .venv and no API is called. Usage (from the mtd-rl-xavier root):
#   sbatch slurm/translator_method_b_parity.sbatch
# Optional: TORCH_VERSION (default 2.5.1), LAG_ROOT (default ../mtd-rl-lag; compares its real modules),
# WEIGHTREADING_MHI_MODEL_DIR, UV. Checks the MHI v1.3 and v1.3-1 actors used by method B.
#SBATCH --partition=151-partition
#SBATCH --job-name=translator_method_b_parity
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
UV="${UV:-$HOME/.local/bin/uv}"          # only for the throwaway CPU-torch env
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
MHI_MODELS="${WEIGHTREADING_MHI_MODEL_DIR:-$LAG_ROOT/models}"
OUT="outputs/weightreading/parity/forward-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
log_header "$0" "torch=$TORCH_VERSION" "out=$OUT"

run_case() {  # <name> <checkpoint> <spec>
  "$UV" run --isolated --no-project --python 3.12 \
    --index https://download.pytorch.org/whl/cpu \
    --with "torch==${TORCH_VERSION}" --with numpy \
    python scripts/weightreading_forward_parity.py \
    --checkpoint "$2" --spec "configs/weightreading/$3.spec.json" --lag-root "$LAG_ROOT" --out "$OUT/$1.json"
}

status=0
run_case mhi_v1_3 "$MHI_MODELS/v1.3/files/actor_latest.pt" lag_mappo_2v2_33 || status=1
run_case mhi_v1_3_1 "$MHI_MODELS/v1.3-1/files/actor_latest.pt" lag_mappo_2v2_33 || status=1
if [ "$status" -eq 0 ]; then echo "forward parity OK for both MHI actors; reports in $OUT"; else echo "forward parity FAILED; see $OUT"; fi
exit "$status"
