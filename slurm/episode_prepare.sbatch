#!/bin/bash
# Episode preparation from a recorded rollout: missile-complete replay, CSV and the episode video (no API calls).
#   ROLLOUT_DIR=outputs/weightreading/rollouts/mhi-v1.3-20261104T010525Z SEED=100601 \
#   EPISODE_DIR=outputs/intent_compare/episode/seed100601 \
#   sbatch --export=ALL slurm/episode_prepare.sbatch
# 1. replay the seed with ACMI on (mtd-rl-lag interpreter, output under the gitignored
#    $LAG_ROOT/LOCAL_ONLY/) and assert bitwise equality of every aircraft field vs the npz;
# 2. run mtd-rl-lag's unmodified evaluate_mhi_team.py --acmi and require a byte-identical ACMI;
# 3. copy the ACMI + reports to EPISODE_DIR and convert npz + ACMI to the 64-column LAG episode CSV
#    (the trajectory input; same format as data/mhi/processed/), scripts/episode_to_csv.py;
# 4. render the full episode with $LAG_ROOT/scripts/render/acmi_to_mp4.py (imageio-ffmpeg binary
#    shimmed onto PATH); human raters watch it.
# Paths must be on the shared filesystem: compute nodes do not see the login node's /tmp.
# Required: ROLLOUT_DIR, SEED, EPISODE_DIR. Optional: LAG_ROOT (default ../mtd-rl-lag), MODEL_DIR,
# UV (only for the throwaway ffmpeg env), DPI.
#SBATCH --partition=151-partition
#SBATCH --job-name=episode_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
require_python
ROLLOUT_DIR="${ROLLOUT_DIR:?set ROLLOUT_DIR to the recorded rollout directory}"
SEED="${SEED:?set SEED to the rollout seed}"
EPISODE_DIR="${EPISODE_DIR:?set EPISODE_DIR to the episode output directory}"
UV="${UV:-$HOME/.local/bin/uv}"
LAG_PYTHON="$LAG_ROOT/.venv/bin/python"
MODEL_DIR="${MODEL_DIR:-$LAG_ROOT/models/v1.3}"
CHECKPOINT="$MODEL_DIR/files/actor_latest.pt"
DPI="${DPI:-100}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORK="$LAG_ROOT/LOCAL_ONLY/episode_prepare/seed$SEED/${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$WORK" "$EPISODE_DIR"
log_header "$0" "seed=$SEED" "work=$WORK" "episode_dir=$EPISODE_DIR"

SECONDS=0
"$LAG_PYTHON" -B scripts/episode_replay.py --lag-root "$LAG_ROOT" --checkpoint "$CHECKPOINT" \
  --npz "$ROLLOUT_DIR/episode_seed$SEED.npz" --seed "$SEED" \
  --acmi-out "$WORK/replay.txt.acmi" --report "$WORK/replay_check.json"
echo "replay + bitwise check passed: ${SECONDS}s"

SECONDS=0
"$LAG_PYTHON" -B "$LAG_ROOT/scripts/eval/evaluate_mhi_team.py" --checkpoint "$CHECKPOINT" \
  --protocol per_agent_v1 --seed "$SEED" --acmi --output "$WORK/evaluate"
cmp "$WORK/replay.txt.acmi" "$WORK/evaluate/episode.acmi"
echo "evaluate_mhi_team.py --acmi is byte-identical to the checked replay: ${SECONDS}s"

cp "$WORK/replay.txt.acmi" "$EPISODE_DIR/replay.txt.acmi"
cp "$WORK/replay_check.json" "$EPISODE_DIR/replay_check.json"
cp "$WORK/evaluate/evaluation.json" "$EPISODE_DIR/lag_evaluation.json"
"$PY" -B scripts/episode_to_csv.py --rollout-dir "$ROLLOUT_DIR" --seed "$SEED" --episode-dir "$EPISODE_DIR"

FF="$("$UV" run --offline --isolated --no-project --python 3.12 --with imageio-ffmpeg \
  python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")"
mkdir -p "$WORK/bin" && ln -sf "$FF" "$WORK/bin/ffmpeg" && export PATH="$WORK/bin:$PATH"

SECONDS=0
"$LAG_PYTHON" -B "$LAG_ROOT/scripts/render/acmi_to_mp4.py" "$EPISODE_DIR/replay.txt.acmi" \
  -o "$EPISODE_DIR/full.mp4" --dpi "$DPI"
echo "render: ${SECONDS}s"

ls -la "$EPISODE_DIR"
