# Agent State

## Main Agent

Claude Code

## Repository Identity

<!-- Managed by /init. Re-run /init to refresh. -->

mtd-rl-xavier translates the behaviour of RL agents into intent statements with LLMs and evaluates those translations with LLM-as-a-judge. The core is two packages: src/translate (one config-driven causal loop: select inputs -> each input renders View(t) -> fill the inline prompt -> one call; trigger points start (step 0), request (a human question) and change (preset steps or the clef-flash detector); configs in configs/translate/) and src/evaluation (one judge engine with its prompt and criteria in configs/evaluation/judge.yaml, a causality audit, human ratings and a report), plus MHI data (data/mhi/), the mtd-rl-lag helpers (scripts/lag/, slurm/lag/) and Slurm jobs (slurm/). workspace/ is an independent uv project and free-experiment area for the legacy LLM-Xavier experiments 00-17, the vendored src/mentalmodeling and the mtd_rl_xavier_workspace library; the core never imports or reads it, and it reads only ../data/mhi.

Macro requirements and design live in [docs/DESIGN.md](docs/DESIGN.md).

## Progress Tracker

Rolling progress summary (latest 5 checkpoints): [PROGRESS.md](../PROGRESS.md)

<!-- Working state below is maintained by workflow skills and manual notes. -->

---

## Current Feature: Simplify the core (2026-11-09)
<!-- orchestra:block-id: simplify-the-core-2026-11-09 -->

### Context

- Goal: the small, readable, config-driven core of the final simplify design (orchestration workspace: .claude/docs/plans/xavier-simple-design-final.md), as eight commits from 4dd05b6 (tag pre-simplify) on branch simplify (worktree ../mtd-rl-xavier-simplify).
- Status: implemented and validated offline (validate full, static, preview); not pushed. Delivery (U-e): a separate PR after PR #10 merges; the owner pushes and opens it.
- Deviations from the design: orchestration workspace .claude/docs/plans/xavier-simplify-deviations.md.

### Decisions

- U-a..U-e of 2026-11-09 (DESIGN.md Key Decisions): request = a human question only; byte-identity machinery deleted; policy input (b) = activations + head weights + act_in; D29 judge deleted; separate PR.
- No paid call was made. Paid calls still need a fresh explicit opt-in: re-judging the 2026-11-05 event under pilot-4 (one call, U-b) and any translation run.
- Post-merge, once, in the main checkout: `OUT=outputs/rollouts/mhi-v1.3-act-in SEEDS=100601 sbatch --export=ALL slurm/lag/record_rollouts.sbatch` (CPU; needed by configs/translate/policy.yaml).
- Open: GPU probe of slurm/translate_online.sbatch after D49; detector threshold calibration; translate resume and retrospective mode are planned extensions; the judge reads no video yet.
