#!/bin/bash
# Online translation: `vllm serve` (clef-flash) and `python -m translate` in ONE GPU job. Submit only
# after approval D49 (the separate vLLM env, the pinned download served offline, this short probe).
# Required: VLLM_ENV (contains bin/vllm; never the project .venv), HF_CACHE_DIR (the pinned snapshot),
# SCENARIO (real ones live in private/). Optional: CONFIG (default configs/translate/online.yaml),
# STARTUP_TIMEOUT_S (default 900). MODE: probe (default; server up, /v1/models, server down; no paid
# call) | paid (the online run; OPENAI_API_KEY from the submitting environment).
#   VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=... sbatch --export=ALL slurm/translate_online.sbatch probe
# The server runs under `env -i` with an allow-list, so the paid key never reaches it, and listens only
# on the config's Unix socket (a 0700 directory: the access control; no key over UDS, D39).
#SBATCH --partition=151-partition
#SBATCH --job-name=translate_online
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
set -euo pipefail                        # before source: a missing common.sh must stop the job
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
MODE="${1:-probe}"
case "$MODE" in probe) unset OPENAI_API_KEY ;; paid) require_openai_key ;; *) echo "MODE must be probe or paid" >&2; exit 2 ;; esac
require_python
: "${VLLM_ENV:?set VLLM_ENV}" "${HF_CACHE_DIR:?set HF_CACHE_DIR}" "${SCENARIO:?set SCENARIO}"
CONFIG="${CONFIG:-configs/translate/online.yaml}"
read -r MODEL REVISION SOCKET BUDGET <<< "$("$PY" -c 'import sys; from pathlib import Path; from translate.config import load_config
c = load_config(Path(sys.argv[1])).config.change
if c.source != "clef_flash" or c.socket_path is None: sys.exit("use an online config with change.socket")
print(c.model, c.revision, c.socket_path, c.input_token_budget)' "$CONFIG")"
mkdir -m 700 "$(dirname "$SOCKET")" 2>/dev/null || test -d "$(dirname "$SOCKET")"
if [ -e "$SOCKET" ]; then echo "$SOCKET exists: another server may be running (remove a stale one by hand)" >&2; exit 1; fi
log_header "$VLLM_ENV/bin/vllm" serve "$MODEL" --revision "$REVISION" --uds "$SOCKET" --max-model-len "$BUDGET"
env -i PATH="$VLLM_ENV/bin:/usr/bin:/bin" HOME="$HOME" HF_HOME="$HF_CACHE_DIR" HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$VLLM_ENV/bin/vllm" serve "$MODEL" --revision "$REVISION" \
  --uds "$SOCKET" --max-model-len "$BUDGET" --disable-log-requests &
SERVER=$!
trap 'status=$?; kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; rm -f "$SOCKET"; echo "end=$(date -Is) exit=$status"' EXIT
for _ in $(seq "${STARTUP_TIMEOUT_S:-900}"); do
  curl -sf --unix-socket "$SOCKET" http://localhost/v1/models >/dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "vllm serve exited before it was ready" >&2; exit 1; }
  sleep 1
done
if [ "$MODE" = probe ]; then
  "$PY" -c 'import sys; from pathlib import Path; from translate.config import load_config; from translate.detector import SystemOneClient
SystemOneClient(load_config(Path(sys.argv[1])).config.change).health(); print("detector health: ok (probe only)")' "$CONFIG"
else
  "$PY" -m translate --config "$CONFIG" --scenario "$SCENARIO" "${@:2}" --execute --confirm-paid-api
fi
