#!/bin/bash
# Missile-complete replay + video clips for the intent-translation comparison (no API calls).
#   mkdir -p outputs/intent_compare/slurm
#   sbatch scripts/intent_compare_video.sbatch
# 1. replay the configured seed with ACMI on (mtd-rl-lag interpreter, output under the gitignored
#    $LAG_ROOT/LOCAL_ONLY/) and assert bitwise equality of every aircraft field vs the npz;
# 2. run mtd-rl-lag's unmodified evaluate_mhi_team.py --acmi and require a byte-identical ACMI;
# 3. copy the ACMI + reports to outputs/intent_compare/episode/seed<seed>/;
# 4. render the full episode with mtd-rl-lag scripts/render/acmi_to_mp4.py (imageio-ffmpeg binary
#    shimmed onto PATH) and trim one clip per translation step.
# Paths must be on the shared filesystem: compute nodes do not see the login node's /tmp.
# Optional: LAG_ROOT, MODEL_DIR, UV, CONFIG, DPI.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=ic-video
#SBATCH --output=outputs/intent_compare/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-$HOME/.local/bin/uv}"
LAG_ROOT="${LAG_ROOT:-/home/user151005/workspace/mtd-rl-lag}"
LAG_PYTHON="$LAG_ROOT/.venv/bin/python"
MODEL_DIR="${MODEL_DIR:-$LAG_ROOT/models/v1.3}"
CHECKPOINT="$MODEL_DIR/files/actor_latest.pt"
CONFIG="${CONFIG:-configs/intent_compare/compare.yaml}"
DPI="${DPI:-100}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

cfg() { "$UV" run --no-sync python -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$CONFIG" "$1"; }
SEED="$(cfg seed)"
ROLLOUT_DIR="$(cfg rollout_dir)"
OUT_ROOT="$(cfg output_root)"
EPISODE_DIR="$OUT_ROOT/episode/seed$SEED"
WORK="$LAG_ROOT/LOCAL_ONLY/intent_compare/seed$SEED/${SLURM_JOB_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$WORK" "$EPISODE_DIR"
echo "host=$(hostname) seed=$SEED work=$WORK episode_dir=$EPISODE_DIR"

SECONDS=0
"$LAG_PYTHON" -B scripts/intent_compare_replay.py --lag-root "$LAG_ROOT" --checkpoint "$CHECKPOINT" \
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

FF="$("$UV" run --offline --isolated --no-project --python 3.12 --with imageio-ffmpeg \
  python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")"
mkdir -p "$WORK/bin" && ln -sf "$FF" "$WORK/bin/ffmpeg" && export PATH="$WORK/bin:$PATH"

SECONDS=0
"$LAG_PYTHON" -B "$LAG_ROOT/scripts/render/acmi_to_mp4.py" "$EPISODE_DIR/replay.txt.acmi" \
  -o "$EPISODE_DIR/full.mp4" --dpi "$DPI"
echo "render: ${SECONDS}s"

"$UV" run --no-sync python scripts/intent_compare_clips.py --config "$CONFIG" --full "$EPISODE_DIR/full.mp4"
ls -la "$EPISODE_DIR" "$EPISODE_DIR/clips"
