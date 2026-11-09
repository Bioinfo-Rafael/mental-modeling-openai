# Slurm jobs of the core

Every core job sources `slurm/common.sh`. Submit from the repository root; logs go to
`logs/slurm/<job name>-<job id>.out` (the directory is tracked). Jobs use `.venv/bin/python` (3.12);
the system `python3` is never used. The workspace project keeps its own jobs.

| File | What it runs | MODE (first argument) | Resources |
|---|---|---|---|
| `uv_sync.sbatch` | `uv sync --locked --all-extras` (the only job that changes `.venv`) | - | 4 CPU, 30 min |
| `validate.sbatch` | offline checks, never an API call (keys always unset) | `full`, `static`, `preview` | 4 CPU, 2 h |
| `translate.sbatch` | `python -m translate`; refuses `paid` for a clef-flash config | `preview` (default), `paid` | 4 CPU, 12 h |
| `translate_online.sbatch` | `vllm serve` (clef-flash) under `env -i` on the config's Unix socket, then `python -m translate`; GPU, after approval D49 | `probe` (default), `paid` | 1 GPU, 30 min |
| `evaluate.sbatch` | `python -m evaluation judge` (+ `report` when paid) | `preview` (default), `paid` | 2 CPU, 4 h |
| `lag/record_rollouts.sbatch` | records MHI rollouts with mtd-rl-lag's interpreter (`outputs/rollouts/`; CPU) | - | 4 CPU, 1 h |
| `lag/prepare_episode.sbatch` | replays one seed (ACMI, checked byte-for-byte against mtd-rl-lag), converts it to an episode CSV, renders the video | - | 4 CPU, 1 h |

`validate.sbatch preview` previews every shipped config of both packages with client factories that
raise; configs that read generated `outputs/` files (the seed 100601 episode, recorded rollouts) read
them from `SOURCE_ROOT` (default this checkout):
`sbatch --export=ALL,SOURCE_ROOT=<checkout with outputs/> slurm/validate.sbatch preview`.

`lag/prepare_episode.sbatch` takes its inputs from the environment (all required):

```bash
ROLLOUT_DIR=outputs/rollouts/<recording> SEED=100601 EPISODE_DIR=outputs/intent_compare/episode/seed100601 \
  sbatch --export=ALL slurm/lag/prepare_episode.sbatch
```

## Conventions

- `set -euo pipefail` comes before `source slurm/common.sh`, so a missing helper stops the job.
- `common.sh` checks the repository root, unsets `VIRTUAL_ENV`, `OPENAI_LOG` and `OPENAI_CUSTOM_HEADERS`,
  and sets `PY=.venv/bin/python`. Jobs never run `uv sync` or `uv lock` (only `uv_sync.sbatch` does).
- `preview` drops any inherited `OPENAI_API_KEY`; only `paid` adds `--execute --confirm-paid-api`, and
  the Python CLIs keep their own gate, so a paid run needs two explicit opt-ins. Everything after MODE
  goes to the Python CLI, whose `--help` is the reference.
- The sibling mtd-rl-lag checkout defaults to `../mtd-rl-lag` (override with `LAG_ROOT`).
- Every log starts with host, job, commit, modified-file count and the exact command, and ends with
  the end time and exit code. No shell tracing, no environment dumps, no personal paths.

## Submit, follow, find the outputs

```bash
sbatch slurm/uv_sync.sbatch                                   # once, and after uv.lock or package changes
jid=$(sbatch --parsable slurm/translate.sbatch preview --config configs/translate/preset.yaml)
tail -f "logs/slurm/translate-$jid.out"                       # Ctrl-C stops following, not the job
sacct -j "$jid" --format=JobID,State,Elapsed,ExitCode         # sacct -u is refused on this cluster
```

Paid runs read `OPENAI_API_KEY` from the submitting environment. The key lives in the gitignored,
group-shared repo-root `.env`; load it right before the one paid submission and drop it afterwards:

```bash
set -a; . ./.env; set +a
jid=$(sbatch --parsable --export=ALL slurm/translate.sbatch paid --config configs/translate/<config>.yaml)
unset OPENAI_API_KEY
```

No resend: a paid translation run creates its run directory once (a resubmitted job with the default
name stops there); `evaluate.sbatch` never sends a request journaled anywhere under the judge's output
root (its own `requests.jsonl` and the 2026-11-05 `judge_requests.jsonl`) and refuses an event judged
in the same directory with other bytes. Nothing is retried; translation runs have no resume.
