#!/bin/bash
# Offline validation of the independent workspace project, with its own .venv. Submit from the
# repository root:  sbatch workspace/slurm/validate.sbatch <MODE>
# MODE: sync (uv sync of workspace/.venv) | library (tests/ and src/mentalmodeling/tests) |
#   experiments (one pytest per experiment; "no tests" counts as a pass) | static (bash -n, and the
#   rule that nothing here imports the core). No mode runs an experiment or sends an API request.
#SBATCH --partition=151-partition
#SBATCH --job-name=workspace_validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=workspace/logs/slurm/%x-%j.out
set -euo pipefail
WORKSPACE_ROOT="${SLURM_SUBMIT_DIR:-$PWD}/workspace"
cd "$WORKSPACE_ROOT"
if ! grep -q '^name = "mtd-rl-xavier-workspace"$' pyproject.toml 2>/dev/null; then
  echo "submit from the repository root (got ${SLURM_SUBMIT_DIR:-$PWD})" >&2; exit 2
fi
unset VIRTUAL_ENV OPENAI_API_KEY OPENAI_LOG OPENAI_CUSTOM_HEADERS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY="$WORKSPACE_ROOT/.venv/bin/python"
MODE="${1:-library}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=$MODE start=$(date -Is)" \
     "commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
trap 'status=$?; echo "end=$(date -Is) exit=$status"' EXIT
if [ "$MODE" != sync ] && [ ! -x "$PY" ]; then echo "missing $PY: run MODE=sync first" >&2; exit 2; fi

case "$MODE" in
  sync)
    "${UV:-$HOME/.local/bin/uv}" sync --locked --all-extras
    ;;
  library)
    "$PY" -m pytest -q tests src/mentalmodeling/tests
    ;;
  experiments)
    status=0
    for experiment in [0-9]*/; do
      echo "== $experiment"
      "$PY" -m pytest -q "$experiment" || [ $? -eq 5 ] || status=1
    done
    exit "$status"
    ;;
  static)
    bash -n slurm/validate.sbatch
    "$PY" -c 'import importlib.util as u, sys; sys.exit(u.find_spec("mtd_rl_xavier") is not None)'
    if git grep -nE '^\s*(from|import) (mtd_rl_xavier|weightreading|evaluation)(\.|\s|$)' -- . ':(exclude)*/results/**'; then
      echo "the workspace must not import the core" >&2; exit 1
    fi
    echo "static: ok"
    ;;
  *)
    echo "MODE must be sync, library, experiments or static (got $MODE)" >&2; exit 2
    ;;
esac
