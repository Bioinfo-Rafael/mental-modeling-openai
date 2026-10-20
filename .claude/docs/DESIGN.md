# Design Document — 要件定義書 (Requirements & Macro Design)

> **Role:** Macro-level requirements and design — *what* this project builds and *why*.
> Written at `/init`, kept current by `/design-tracker` (also invoked from `/checkpointing`).
>
> **Document map:** Shared rules → [rules/](../rules/) ·
> Root contract → [AGENTS.md](../../AGENTS.md) · State → [STATE.md](../STATE.md) ·
> Micro work progress (latest 5 checkpoints) → [PROGRESS.md](../../PROGRESS.md)

## 背景・目的 (Background & Purpose)

<!-- Why does this project exist? What problem does it solve, for whom?
     State the business/technical context and the goal in a few sentences. -->

Research code for evaluating how well language models form a mental model of a trained reinforcement-learning agent from its recorded state/action/reward history, building on LLM-Xavier (Lu et al., TMLR 2024). The experiments ported from `rafael/mental-modeling-openai` reproduce the paper's MountainCar/Pendulum setting with GPT-3.5 and newer OpenAI models, then study few-shot prompting, reward ablation, in-context RL prompting and LAG 2v2 air-combat trajectories. The repository separates tested library code from exploratory experiments so new studies reuse the pipeline without inheriting experiment-specific code.

## スコープ (Scope)

### In Scope

<!-- What this project explicitly delivers. -->

- 

- Raw dataset readers for the pinned `lerrhoo/llmx_data` NPZ episodes and candidate external datasets (`src/mtd_rl_xavier/datasets`).
- Prompt construction that preserves original LLM-Xavier semantics (`src/mtd_rl_xavier/preprocessing`).
- Guarded, resumable OpenAI experiment execution with provenance hashing, and offline analysis (`src/mtd_rl_xavier/experiment`).
- Exploratory experiments with committed results (`workspace/NN_name/`).

### Out of Scope

<!-- What is explicitly NOT covered, to prevent scope creep. -->

- 

- Training RL policies or running live Gym environments; LLM-Xavier consumes recorded trajectories only.
- Editing the vendored LLM-Xavier code in `mentalmodeling/` except to track upstream; its files are part of the semantics hash.

## 機能要件 (Functional Requirements)

<!-- What the system must do. Each requirement gets a stable ID (FR-1, FR-2, ...). -->

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Build prompts from recorded RL trajectories with original LLM-Xavier semantics (next/last action, state direction, Joint variants) | Must | src/mtd_rl_xavier/preprocessing, mentalmodeling/llm_x |
| FR-2 | Send paid OpenAI queries only through explicit execute paths, with resumable journals and provenance | Must | src/mtd_rl_xavier/experiment/common.py, split_execution.py |
| FR-3 | Analyse saved responses offline (accuracy, continuous metrics, confusion matrices, figures) without API calls | Must | experiment/common_analysis, workspace/*/analysis |
| FR-4 | Inspect raw datasets and count prompt tokens locally | Should | src/mtd_rl_xavier/tools, workspace/notebooks |

## 非機能要件 (Non-Functional Requirements)

<!-- Quality attributes: performance, availability, security, maintainability, etc.
     Prefer measurable targets in the Metric column. -->

| Category | Requirement | Metric / Target |
|----------|-------------|-----------------|
| Performance | | |
| Availability | | |
| Security | | |
| Maintainability | New library modules pass the full ruff rule set and have tests in tests/ | uv run poe lint and uv run poe test pass |
| Reproducibility | Plans regenerate the saved query identities from pinned data and unchanged semantics code | make_plan reproduces saved query_id sets (verified for 03_1, 05, 06) |
| Cost safety | No paid request outside documented execute paths; SDK retries disabled | Previews/dry-runs make 0 API calls |

## アーキテクチャ (Architecture)

<!-- High-level architecture: components, data flow, boundaries.
     Add a diagram or description here. -->

### Agent Roles

| Agent | Role | Responsibilities |
|-------|------|------------------|
| | | |

## 技術選定 (Tech Stack & Rationale)

<!-- Chosen technologies and why. Record alternatives considered. -->

| Area | Technology | Rationale | Alternatives Considered |
|------|------------|-----------|-------------------------|
| Environment / packaging | uv + setuptools src layout; mentalmodeling/ as editable path dependency | Project rule (.claude/rules/dev-environment.md); one install exposes llm_x and mtd_rl_xavier | pip + setuptools flat packages (original branch) |
| LLM API | openai SDK (Chat Completions; Batch API for LAG 14-17) | Used by all ported experiments |  |
| Lint | ruff (E,W,F,I,B,UP) with per-file ignores for ported modules | Enforce rules on new code without reformatting ported code or dropping star re-exports | Reformat all ported code now |
| Tests / tasks | pytest + poethepoet (test, test-workspace, lint) | Workspace experiments reuse test file names, so each runs as its own pytest session | Single pytest run with importlib mode |

## 制約 (Constraints)

<!-- Technical, organizational, regulatory, or resource constraints. -->

- 

- Saved results are immutable and tracked in git. They record repo-relative paths in the pre-restructure layout; read them via `mtd_rl_xavier.paths.resolve_recorded`.
- Plans refuse to reuse, resume or merge results produced under different prompt/scoring code (`semantics_sha256`, see `experiment/common.py`).
- Paid API: `workspace/09`-`12` send requests when `run.py` runs without arguments; other experiments need explicit execute flags.
- Experiments need `data/llmx_data` at revision `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`; the dataset registry reads generated inventories in `outputs/dataset_inventory/`.

## Key Decisions

<!-- Durable architectural/design decisions. Append-only log. -->

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| Layout: mentalmodeling/ (vendored LLM-Xavier, code only), src/mtd_rl_xavier/ (shared library), workspace/ (exploratory experiments) | User request: separate prior-work code, curated code and messy experiments with low cognitive load | Keep the original flat layout; LLM-Xavier as a git submodule | 2026-10-20 |
| Experiments live at workspace/NN_name (same depth as experiments/NN_name), imported as workspace.NN_name | Existing parents[k] root computations stay valid; only module names change | workspace/experiments/NN_name (shifts every root computation) | 2026-10-20 |
| Saved results are ported byte-identical, including results/ (user choice); old-layout paths are resolved by paths.resolve_recorded | Results are immutable evidence and hash-protected | Rewrite paths inside results; leave results on the original branch | 2026-10-20 |
| RESTRUCTURED_SEMANTICS maps the branch-head semantics hash to the post-restructure hash | Only import/path lines changed; make_plan reproduces identical query identities for 03_1, 05, 06 | Require the original branch for any re-analysis of 05/06 | 2026-10-20 |
| No CLAUDE.md; root AGENTS.md stays the agent contract, project context lives in .claude/STATE.md and this file | User decision during /init (Orchestra rule) | CLAUDE.md importing @AGENTS.md | 2026-10-20 |

## TODO / Open Questions

<!-- Open design questions and deferred decisions for this project. -->

- 
