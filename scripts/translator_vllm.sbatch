#!/bin/bash
# One-GPU Translator job: `vllm serve` (the clef-flash change-point detector) and the Translator CLI
# run in the SAME job, so no other user or job can reach the server. TEMPLATE ONLY: do not submit it
# before the run-time setup is approved (D49): the separate vLLM env (~10 GB), the pinned clef-flash
# download (~19 GB, served offline from HF_CACHE_DIR) and this short GPU probe.
#
# Required (placeholders; nothing personal is hardcoded):
#   VLLM_ENV      the separate vLLM env directory (contains bin/vllm; never the project .venv)
#   HF_CACHE_DIR  the Hugging Face cache holding the pinned snapshot (e.g. on /lustre)
#   SCENARIO      Translator scenario YAML (real ones live under private/translator/)
# Optional: CONFIG (default configs/translator/run.yaml), MAX_MODEL_LEN (default 16384),
#   STARTUP_TIMEOUT_S (default 900), SHUTDOWN_TIMEOUT_S (default 20), UV.
#
# Default: GPU probe only. The server starts, the detector client's authenticated health check
# runs against it, and the server stops; no paid API call is made.
# A PAID Translator run happens only when TRANSLATOR_CONFIRM_PAID=yes is exported explicitly
# (the default translation layer is the OpenAI Responses API; OPENAI_API_KEY must be set):
#   mkdir -p outputs/translator/slurm
#   VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=... sbatch --export=ALL scripts/translator_vllm.sbatch
#   TRANSLATOR_CONFIRM_PAID=yes VLLM_ENV=... HF_CACHE_DIR=... SCENARIO=... sbatch --export=ALL scripts/translator_vllm.sbatch
#
# Security: the server listens only on the Unix socket from detector.connection.uds_path (default
# /tmp/$USER-vllm/clef-flash.sock, directory mode 0700, one launcher per socket path). A fresh random
# Bearer key is generated for this job as VLLM_API_KEY and passed through the environment only
# (never argv, never printed). Request logging stays off and the launcher accepts no passthrough
# flags (src/mtd_rl_xavier/llm/README.md). Two of your jobs on one node with the same config cannot
# share the socket: the second one stops at the launcher's lock.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=translator-vllm
#SBATCH --output=outputs/translator/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-$HOME/.local/bin/uv}"
VLLM_ENV="${VLLM_ENV:?set VLLM_ENV to the separate vLLM env directory (it contains bin/vllm)}"
HF_CACHE_DIR="${HF_CACHE_DIR:?set HF_CACHE_DIR to the Hugging Face cache holding the pinned snapshot}"
SCENARIO="${SCENARIO:?set SCENARIO to the Translator scenario YAML}"
CONFIG="${CONFIG:-configs/translator/run.yaml}"
# The project's own interpreter, called directly below, so that no `uv` process sits between SLURM's
# signals and the launcher (the launcher stops the server on SIGTERM).
PYTHON="$("$UV" run --no-sync python -c 'import sys; print(sys.executable)')"

# The detector settings come from the Translator config itself, so server and client cannot drift.
# A failing reader fails the assignment, and so the job (set -e).
DETECTOR="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from mtd_rl_xavier.translator.config import load_config

detector = load_config(Path(sys.argv[1])).detector
connection = detector.connection
if connection.uds_path is None:
    sys.exit("this job serves over a Unix socket: set detector.connection.uds_path (D39)")
if connection.api_key_env != "VLLM_API_KEY":
    sys.exit("this job sets VLLM_API_KEY: set detector.connection.api_key_env to VLLM_API_KEY")
values = (
    detector.model,
    detector.revision,
    detector.served_model_name,
    str(connection.uds_path),
    str(detector.input_token_budget),
)
if any(not value or any(char.isspace() for char in value) for value in values):
    sys.exit("detector values must be non-empty and contain no whitespace")
print(*values, sep="\t")
PY
)"
IFS=$'\t' read -r MODEL REVISION SERVED_NAME UDS_PATH BUDGET <<< "$DETECTOR"

# Per-user socket directory, mode 0700 (a concurrent job may create it first). An existing one is
# never changed here; the launcher refuses it unless it is owned by this user with mode exactly 0700.
SOCKET_DIR="$(dirname "$UDS_PATH")"
mkdir -m 700 "$SOCKET_DIR" 2>/dev/null || test -d "$SOCKET_DIR"

# A fresh per-job key, held in this job's environment only. It is never echoed.
VLLM_API_KEY="$("$PYTHON" -c 'import secrets; print(secrets.token_urlsafe(32))')"
if [ -z "$VLLM_API_KEY" ]; then echo "could not generate the per-job key" >&2; exit 1; fi
export VLLM_API_KEY

LAUNCHER=(
  --vllm-executable "$VLLM_ENV/bin/vllm"
  --model "$MODEL"
  --revision "$REVISION"
  --served-model-name "$SERVED_NAME"
  --max-model-len "${MAX_MODEL_LEN:-16384}"
  --dtype bfloat16
  --uds-path "$UDS_PATH"
  --hf-home "$HF_CACHE_DIR"
  --api-key-env VLLM_API_KEY
  --client-input-token-budget "$BUDGET"
  --startup-timeout-s "${STARTUP_TIMEOUT_S:-900}"
  --shutdown-timeout-s "${SHUTDOWN_TIMEOUT_S:-20}"
)

# Preflight (env, cache, key, socket directory) and the exact server argv; the key is never printed.
"$PYTHON" -m mtd_rl_xavier.llm.vllm_server "${LAUNCHER[@]}" --dry-run
if [ "${TRANSLATOR_CONFIRM_PAID:-}" = "yes" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY is not set" >&2; exit 2; fi
  CLIENT=("$PYTHON" -m mtd_rl_xavier.translator.cli --config "$CONFIG" --scenario "$SCENARIO" --execute --confirm-paid-api)
else
  PROBE='import sys
from pathlib import Path
from mtd_rl_xavier.translator.cli import build_system_one_client
from mtd_rl_xavier.translator.config import load_config
with build_system_one_client(load_config(Path(sys.argv[1])).detector) as client:
    client.health()
print("detector health: ok (probe only; set TRANSLATOR_CONFIRM_PAID=yes for a paid Translator run)")'
  CLIENT=("$PYTHON" -c "$PROBE" "$CONFIG")
fi
# exec: the launcher replaces this shell, so SIGTERM from scancel or the time limit reaches it
# directly. It starts the server, waits for authenticated readiness, runs the client, then stops the
# server and its process group (SIGTERM, wait, SIGKILL) and exits with the client's exit code.
exec "$PYTHON" -m mtd_rl_xavier.llm.vllm_server "${LAUNCHER[@]}" -- "${CLIENT[@]}"
