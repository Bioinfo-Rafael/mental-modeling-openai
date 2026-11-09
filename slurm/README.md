# Slurm jobs of the core

Every core job is a `slurm/<area>_<action>.sbatch` file that sources `slurm/common.sh`. Submit from
the repository root; logs go to `logs/slurm/<job name>-<job id>.out` (the directory is tracked, so no
`mkdir` is needed). The workspace project keeps its own jobs in its own `slurm/` directory.

| File | What it runs | MODE (first argument) | Resources |
|---|---|---|---|
| `uv_sync.sbatch` | `uv sync --locked --all-extras --reinstall-package mtd-rl-xavier` (the only job that changes `.venv`) | - | 4 CPU, 30 min |
| `validate.sbatch` | offline checks, never an API call | `full`, `static`, `preview` | 4 CPU, 2 h |
| `translate.sbatch` | `python -m translate` for a preset-trigger config (a code-session config passes the sandbox preflight first) | `preview` (default), `paid` | 4 CPU, 16 GB, 12 h |
| `translate_vllm.sbatch` | vLLM (the clef-flash change-point detector) + `python -m translate` for an online config in one GPU job (awaits approval D49) | `probe` (default), `paid` | 1 GPU, 8 CPU, 30 min |
| `viewer.sbatch` | `python -m evaluation.viewer` (side-by-side HTML of runs or cue files) | - | 1 CPU, 30 min |
| `judge_assess.sbatch` | `evaluation.assess_cli` check + judge (+ report when paid) | `preview` (default), `paid` | 2 CPU, 4 h |
| `judge_keypoints.sbatch` | the D29 key-point judge `evaluation.cli` | `preview` (default), `paid` | 2 CPU, 4 h |
| `episode_record_rollouts.sbatch` | record MHI rollouts in mtd-rl-lag and replay them in numpy (`outputs/rollouts/`) | - | 4 CPU, 1 h |
| `episode_prepare.sbatch` | replay, CSV conversion (`scripts/episode_to_csv.py`) and clips of one rollout seed (mtd-rl-lag) | - | 4 CPU, 1 h |
| `policy_parity.sbatch` | torch vs numpy forward parity of the MHI actors | - | 4 CPU, 1 h |

`episode_prepare.sbatch` takes its inputs from the environment (all required):

```bash
ROLLOUT_DIR=outputs/weightreading/rollouts/mhi-v1.3-20261104T010525Z SEED=100601 \
  EPISODE_DIR=outputs/intent_compare/episode/seed100601 STEPS="45 145 245 315" \
  sbatch --export=ALL slurm/episode_prepare.sbatch
```

## Conventions

- `set -euo pipefail` comes before `source slurm/common.sh`, so a missing helper stops the job.
- `common.sh` checks that it runs in the repository root, unsets `VIRTUAL_ENV`, `OPENAI_LOG` and
  `OPENAI_CUSTOM_HEADERS`, and calls `.venv/bin/python` directly. Jobs never run `uv sync` or `uv lock`
  (only `uv_sync.sbatch` does); three jobs use `uv run --isolated` for throwaway torch, numpy or ffmpeg
  environments that never touch `.venv`.
- `preview` drops any inherited `OPENAI_API_KEY`; only `paid` adds `--execute --confirm-paid-api`, and
  the Python CLIs keep their own gate, so a paid run needs two explicit opt-ins. Everything after MODE
  goes to the Python CLI, whose `--help` is the reference.
- The sibling mtd-rl-lag checkout defaults to `../mtd-rl-lag` (override with `LAG_ROOT`).
- Every log starts with host, job, commit, modified-file count and the exact command, and ends with
  the end time and exit code (except the vLLM job, which `exec`s its launcher: use `sacct`).
- No shell tracing (xtrace), no environment dumps, no personal paths.

## Submit, follow, find the outputs

```bash
sbatch slurm/uv_sync.sbatch                                   # once, and after uv.lock or package changes
jid=$(sbatch --parsable slurm/translate.sbatch preview --config configs/translate/seed100601-trajectory.yaml)
tail -f "logs/slurm/translate-$jid.out"                       # Ctrl-C stops following, not the job
sacct -j "$jid" --format=JobID,State,Elapsed,ExitCode         # sacct -u is refused on this cluster
```

Paid runs read `OPENAI_API_KEY` from the submitting environment. The key lives in the gitignored,
group-shared repo-root `.env`; load it right before the one paid submission and drop it afterwards:

```bash
set -a; . ./.env; set +a
jid=$(sbatch --parsable --export=ALL slurm/translate.sbatch paid --config configs/translate/<config>.yaml --run-name <name>)
unset OPENAI_API_KEY
```

Resume: `judge_assess.sbatch` never re-sends a request already in `judge_requests.jsonl`, and refuses
to judge an event of that directory again with other request bytes (another prompt file or judge
config). Pass the judge config the run was judged with. Translation runs and the D29 judge have no resume: a re-run sends
every call again (the former method B `--resume` was retired on 2026-11-08; its three recorded runs
are complete).
