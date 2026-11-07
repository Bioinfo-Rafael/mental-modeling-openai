# slurm/common.sh: sourced by every slurm/*.sbatch. Submit from the repository root;
# Slurm runs a spool copy of the .sbatch, so its own path cannot locate the repository.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"
if ! grep -q '^name = "mtd-rl-xavier"$' pyproject.toml 2>/dev/null; then
  echo "submit from the mtd-rl-xavier repository root (got $REPO_ROOT)" >&2; exit 2
fi
unset VIRTUAL_ENV OPENAI_LOG OPENAI_CUSTOM_HEADERS  # a submit-time venv or SDK debug setting must not leak in
PY="$REPO_ROOT/.venv/bin/python"                   # jobs never run uv sync or uv lock (slurm/uv_sync.sbatch does)
LAG_ROOT="${LAG_ROOT:-$REPO_ROOT/../mtd-rl-lag}"   # the sibling mtd-rl-lag checkout (models, torch env)
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

require_python() {  # checked when used, so slurm/uv_sync.sbatch can run before .venv exists
  [ -x "$PY" ] || { echo "missing $PY: run sbatch slurm/uv_sync.sbatch first" >&2; exit 2; }
}

require_openai_key() {  # paid mode only; never prints the value
  [ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY is not set (see README: .env)" >&2; exit 2; }
}

set_mode() {  # MODE = preview (default, 0 API calls, key dropped) | paid; sets PAID_FLAGS
  MODE="${1:-preview}"
  case "$MODE" in
    preview) unset OPENAI_API_KEY; PAID_FLAGS=() ;;
    paid) require_openai_key; PAID_FLAGS=(--execute --confirm-paid-api) ;;
    *) echo "MODE must be preview or paid (got $MODE)" >&2; exit 2 ;;
  esac
}

arg_value() {  # arg_value --name "$@": the value after --name, or nothing
  local name="$1"; shift
  while [ "$#" -gt 1 ]; do
    if [ "$1" = "$name" ]; then echo "$2"; return 0; fi
    shift
  done
  return 0
}

log_header() {  # "$@" = the command about to run (never holds a key)
  echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${SLURM_CPUS_PER_TASK:-?} start=$(date -Is)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
       "modified=$(git status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
  printf 'command:'; printf ' %q' "$@"; echo
}
trap 'status=$?; echo "end=$(date -Is) exit=$status"' EXIT
