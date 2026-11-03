#!/bin/bash
# Torch parity check for weightreading's checkpoint reader, on a compute node.
# Usage (from the mtd-rl-xavier root):
#   mkdir -p outputs/weightreading/slurm
#   LAG_PYTHON=/path/to/mtd-rl-lag/env/bin/python sbatch scripts/weightreading_parity.sbatch
# LAG_PYTHON must be an interpreter with torch (the mtd-rl-lag environment). Makes no API call.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=wr-parity
#SBATCH --output=outputs/weightreading/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
: "${LAG_PYTHON:?set LAG_PYTHON to the mtd-rl-lag interpreter (with torch)}"
MODEL_DIR="${WEIGHTREADING_LAG_MODEL_DIR:-/home/user151005/workspace/mtd-rl-lag/envs/JSBSim/model}"
UV="${UV:-$HOME/.local/bin/uv}"
OUT="outputs/weightreading/parity/${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
for model in baseline_model dodge_missile_model; do
  "$UV" run python -m weightreading inspect "$MODEL_DIR/$model.pt" --tensor-hashes > "$OUT/$model.reader.json"
  "$LAG_PYTHON" scripts/weightreading_torch_parity.py "$MODEL_DIR/$model.pt" \
    --expected "$OUT/$model.reader.json" > "$OUT/$model.torch.json"
done
echo "parity OK for both checkpoints; hashes in $OUT"
