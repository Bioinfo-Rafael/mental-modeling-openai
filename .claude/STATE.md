# Agent State

## Main Agent

Claude Code

## Repository Identity

<!-- Managed by /init. Re-run /init to refresh. -->

mtd-rl-xavier is research code that evaluates how language models build mental models of RL agents, with prior-work LLM-Xavier vendored in src/mentalmodeling/ (separate distribution llm-x-evaluator), a shared library in src/mtd_rl_xavier/, and exploratory experiments in workspace/.

Macro requirements and design live in [docs/DESIGN.md](docs/DESIGN.md).

## Progress Tracker

Rolling progress summary (latest 5 checkpoints): [PROGRESS.md](../PROGRESS.md)

<!-- Working state below is maintained by workflow skills and manual notes. -->

---

## Current Feature: Translator evaluation: 3 time points x 4 aspects
<!-- orchestra:block-id: translator-evaluation-3-time-points-x-4-aspects -->

### Context

- Goal: evaluate LLM Translator output at start / change / request points on elements, correctness, timing and contribution, with machine, human and judge results kept separate.
- Key files: evaluation/assess_*.py, evaluation/config/assess.yaml, src/mtd_rl_xavier/translator/ (event contract v2), tests/evaluation/test_assess_*.py
- Complexity: COMPLEX (/team-execute)
- Status 2026-11-04: implemented and verified in worktree ../mtd-rl-xavier-assess; committed 10e34fe on feature/llm-translator-judge; PR #3 to main.

### Decisions

- D50-D64 in .claude/docs/research/translator-requirements.md (detector decides start/change, alarms and questions as inputs, pilot 1-5 + cannot_judge, no aggregation, 3.0 s / 10.0 s, Japanese judge rationale, simplicity accepted at 2928 lines).
- Open: pilot level wording review, detector threshold calibration, tactic vocabulary, annotated units, rater count, overall aggregation, judge substitution criterion, real annotations and ratings, run-time approvals (vLLM env, clef-flash download, GPU probe).
