#!/bin/bash
# Method A (trajectory-based Translator) of the intent-translation comparison on a compute node.
# Default: local preview only (prompts + token/cost estimate, no API call, no key read).
# A PAID run happens only when INTENT_COMPARE_CONFIRM_PAID=yes is exported explicitly:
#   mkdir -p outputs/intent_compare/slurm
#   sbatch scripts/intent_compare_run_a.sbatch                                         # preview
#   INTENT_COMPARE_CONFIRM_PAID=yes RUN_ID=seed100601 sbatch --export=ALL scripts/intent_compare_run_a.sbatch
# Japanese output (new run id; the English run stays intact):
#   INTENT_COMPARE_CONFIRM_PAID=yes RUN_ID=seed100601-ja COMPARE=configs/intent_compare/compare.ja.yaml \
#     sbatch --export=ALL scripts/intent_compare_run_a.sbatch
# Short Japanese answers (intent <= 200, <= 4 evidence items of <= 80 characters; new run id):
#   INTENT_COMPARE_CONFIRM_PAID=yes RUN_ID=seed100601-ja-short COMPARE=configs/intent_compare/compare.ja-short.yaml \
#     sbatch --export=ALL scripts/intent_compare_run_a.sbatch
# OPENAI_API_KEY must already be in the submitting environment (read only by the shared
# mtd_rl_xavier.llm make_openai_client, as in method B); this script never reads or sources a .env file.
# Optional: RUN_ID (default seed<seed>), COMPARE (the shared compare config, same name as in
# intent_compare_run_b.sbatch; CONFIG is accepted as an alias), MAX_OUTPUT_TOKENS, UV.
#SBATCH -p 151-partition
#SBATCH -n 1
#SBATCH --job-name=ic-run-a
#SBATCH --output=outputs/intent_compare/slurm/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
UV="${UV:-$HOME/.local/bin/uv}"
COMPARE="${COMPARE:-${CONFIG:-configs/intent_compare/compare.yaml}}"
ARGS=(--config "$COMPARE")
if [ -n "${RUN_ID:-}" ]; then ARGS+=(--run-id "$RUN_ID"); fi
if [ -n "${MAX_OUTPUT_TOKENS:-}" ]; then ARGS+=(--max-output-tokens "$MAX_OUTPUT_TOKENS"); fi

"$UV" run --no-sync python scripts/intent_compare_run_a.py "${ARGS[@]}"
if [ "${INTENT_COMPARE_CONFIRM_PAID:-}" = "yes" ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY is not set" >&2; exit 2; fi
  "$UV" run --no-sync python scripts/intent_compare_run_a.py "${ARGS[@]}" --execute --confirm-paid-api
  "$UV" run --no-sync python scripts/intent_compare_viewer.py "${ARGS[@]::2}" ${RUN_ID:+--run-id "$RUN_ID"}
else
  echo "Preview only (set INTENT_COMPARE_CONFIRM_PAID=yes for a paid run)."
fi
