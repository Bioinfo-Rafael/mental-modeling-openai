# Slurm jobs of the core

Every core job is a `slurm/<area>_<action>.sbatch` file that sources `slurm/common.sh`. Submit from
the repository root; logs go to `logs/slurm/<job name>-<job id>.out` (the directory is tracked, so no
`mkdir` is needed). The workspace project keeps its own jobs in its own `slurm/` directory.

| File | What it runs | MODE (first argument) | Resources |
|---|---|---|---|
| `uv_sync.sbatch` | `uv sync --locked --all-extras` (the only job that changes `.venv`) | - | 4 CPU, 30 min |
| `validate.sbatch` | offline checks, never an API call | `subset`, `full`, `static`, `golden`, `method-a-preview` | 4 CPU, 2 h |
| `translator_vllm.sbatch` | vLLM detector + Translator CLI in one GPU job (awaits approval D49) | `probe` (default), `paid` | 1 GPU, 8 CPU, 30 min |
| `translator_method_a.sbatch` | method A (`mtd_rl_xavier.intent_compare.method_a_cli`), then the viewer | `preview` (default), `paid` | 2 CPU, 4 h |
| `translator_method_b.sbatch` | method B (`weightreading sandbox-selftest`, `intent-preview` / `intent-run`), then the viewer | `preview` (default), `paid` | 4 CPU, 12 h |
| `translator_prepare_episode.sbatch` | replay, CSV conversion and clips of the A/B episode (mtd-rl-lag) | - | 4 CPU, 1 h |
| `translator_record_rollouts.sbatch` | record MHI rollouts in mtd-rl-lag and replay them in numpy | - | 4 CPU, 1 h |
| `translator_method_b_parity.sbatch` | torch vs numpy forward parity of the MHI actors | - | 4 CPU, 1 h |
| `judge_assess.sbatch` | `evaluation.assess_cli` check + judge (+ report when paid) | `preview` (default), `paid` | 2 CPU, 4 h |
| `judge_keypoints.sbatch` | the D29 key-point judge `evaluation.cli` | `preview` (default), `paid` | 2 CPU, 4 h |

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
sbatch slurm/uv_sync.sbatch                                   # once, and after uv.lock changes
jid=$(sbatch --parsable slurm/judge_assess.sbatch preview --translator-run outputs/translator/<run>)
tail -f "logs/slurm/judge_assess-$jid.out"                    # Ctrl-C stops following, not the job
sacct -j "$jid" --format=JobID,State,Elapsed,ExitCode         # sacct -u is refused on this cluster
```

Paid runs read `OPENAI_API_KEY` from the submitting environment. The key lives in the gitignored,
group-shared repo-root `.env`; load it right before the one paid submission and drop it afterwards:

```bash
set -a; . ./.env; set +a
jid=$(sbatch --parsable --export=ALL slurm/judge_assess.sbatch paid --translator-run outputs/translator/<run>)
unset OPENAI_API_KEY
```

Resume: `judge_assess.sbatch` never re-sends a request already in `judge_requests.jsonl`;
`translator_method_b.sbatch paid ... --run-id <id> --resume` continues an interrupted method B run
(pass `--compare` / `--config` exactly as the original run did). Method A and the D29 judge have no
resume: a re-run sends every call again.
