# Agent State

## Main Agent

Claude Code

## Repository Identity

<!-- Managed by /init. Re-run /init to refresh. -->

mtd-rl-xavier translates the behaviour of RL agents into intent statements with LLMs and evaluates those translations with LLM-as-a-judge. The core is two packages: src/translate (one config-driven pipeline: choose inputs -> process them -> complete the prompt -> translate; configs in configs/translate/, triggers online via clef-flash or preset) and src/evaluation (the judges and the viewer), plus MHI data (data/mhi/) and Slurm jobs (slurm/). workspace/ is an independent uv project and free-experiment area for the legacy LLM-Xavier experiments 00-17, the vendored src/mentalmodeling and the mtd_rl_xavier_workspace library; the core never imports or reads it, and it reads only ../data/mhi.

Macro requirements and design live in [docs/DESIGN.md](docs/DESIGN.md).

## Progress Tracker

Rolling progress summary (latest 5 checkpoints): [PROGRESS.md](../PROGRESS.md)

<!-- Working state below is maintained by workflow skills and manual notes. -->

---

## Current Feature: Translator evaluation: 3 time points x 4 aspects
<!-- orchestra:block-id: translator-evaluation-3-time-points-x-4-aspects -->

### Context

- Goal: evaluate LLM Translator output at start / change / request points on elements, correctness, timing and contribution, with machine, human and judge results kept separate.
- Key files: src/evaluation/assess_*.py, configs/evaluation/assess.yaml, src/translate/ (event contract v2), tests/evaluation/test_assess_*.py
- Complexity: COMPLEX (/team-execute)
- Status 2026-11-08: implemented and verified 2026-11-04 (commit 10e34fe, PR #3), merged into main. The repository was then restructured into core and workspace/, and the core into src/translate + src/evaluation, on branch restructure/core-workspace (see DESIGN.md Key Decisions, 2026-11-08).

### Decisions

- D50-D64 in .claude/docs/research/translator-requirements.md (detector decides start/change, alarms and questions as inputs, pilot 1-5 + cannot_judge, no aggregation, 3.0 s / 10.0 s, Japanese judge rationale, simplicity accepted at 2928 lines).
- Open: pilot level wording review, detector threshold calibration, tactic vocabulary, annotated units, rater count, overall aggregation, judge substitution criterion, real annotations and ratings, run-time approvals (vLLM env, clef-flash download, GPU probe).
