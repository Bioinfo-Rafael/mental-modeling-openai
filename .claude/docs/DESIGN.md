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

Research code for evaluating how well language models form a mental model of a trained reinforcement-learning agent from its recorded state/action/reward history, building on LLM-Xavier (Lu et al., TMLR 2024). The experiments ported from `rafael/mental-modeling-openai` reproduce the paper's MountainCar/Pendulum setting with GPT-3.5 and newer OpenAI models, then study few-shot prompting, reward ablation, in-context RL prompting and LAG 2v2 air-combat trajectories. Later work on `main` adds weight reading (an LLM explains a policy from its raw weights), an LLM Translator that turns LAG trajectories into intent statements, an LLM judge and assessment for those translations, and a comparison of trajectory-based and weights-based intent translation (intent_compare). The repository separates tested library code from exploratory experiments so new studies reuse the pipeline without inheriting experiment-specific code.

## スコープ (Scope)

### In Scope

<!-- What this project explicitly delivers. -->

- Raw dataset readers for the pinned `lerrhoo/llmx_data` NPZ episodes and candidate external datasets (`src/mtd_rl_xavier/datasets`).
- Prompt construction that preserves original LLM-Xavier semantics (`src/mtd_rl_xavier/preprocessing`).
- Guarded, resumable OpenAI experiment execution with provenance hashing, and offline analysis (`src/mtd_rl_xavier/experiment`).
- Exploratory experiments with committed results (`workspace/NN_name/`).
- Weight reading: an LLM explains an RL policy from its raw weights without a forward pass (`src/weightreading/`, `configs/weightreading/`).
- LLM Translator: two-layer intent translation of LAG trajectories (`src/mtd_rl_xavier/translator/`, `llm/`, `lag/`, `configs/translator/`).
- LLM judge and the 3-point x 4-aspect assessment of Translator output (`evaluation/`).
- intent_compare: trajectory-based (method A, `scripts/intent_compare_run_a.py`) vs. weights-and-activations-based (method B, `weightreading intent-run`) intent translation on recorded MHI 2v2 rollouts, with an HTML viewer (`configs/intent_compare/`, `scripts/intent_compare_*`).

### Out of Scope

<!-- What is explicitly NOT covered, to prevent scope creep. -->

- Training RL policies. LLM-Xavier experiments consume recorded trajectories only; live environments run only to record or replay rollouts of existing LAG checkpoints (`scripts/weightreading_record_rollouts.py` runs `MultipleCombatEnv`, `scripts/intent_compare_replay.py`) with the external mtd-rl-lag interpreter, never inside this project's `.venv`.
- Editing the vendored LLM-Xavier code in `src/mentalmodeling/` except to track upstream; its files are part of the semantics hash.

## 機能要件 (Functional Requirements)

<!-- What the system must do. Each requirement gets a stable ID (FR-1, FR-2, ...). -->

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Build prompts from recorded RL trajectories with original LLM-Xavier semantics (next/last action, state direction, Joint variants) | Must | src/mtd_rl_xavier/preprocessing, src/mentalmodeling/llm_x |
| FR-2 | Send paid OpenAI queries only through explicit execute paths, with resumable journals and provenance | Must | src/mtd_rl_xavier/experiment/common.py, split_execution.py |
| FR-3 | Analyse saved responses offline (accuracy, continuous metrics, confusion matrices, figures) without API calls | Must | experiment/common_analysis, workspace/*/analysis |
| FR-4 | Inspect raw datasets and count prompt tokens locally | Should | src/mtd_rl_xavier/tools, workspace/notebooks |
| FR-5 | Let an LLM explain an RL policy from its raw weights in a sandbox, without forward passes, and audit the transcript | Should | src/weightreading |
| FR-6 | Translate LAG trajectories into intent statements per target with a change-point detector and a translation model, using only information at or before t | Must | src/mtd_rl_xavier/translator, llm, lag |
| FR-7 | Judge and assess Translator output (key-point judge; 3 points x 4 aspects) with machine, human and judge results stored separately | Must | evaluation/ |
| FR-8 | Compare trajectory-based and weights-based intent translation on the same recorded rollouts and render a side-by-side viewer | Should | scripts/intent_compare_*, configs/intent_compare |

## 非機能要件 (Non-Functional Requirements)

<!-- Quality attributes: performance, availability, security, maintainability, etc.
     Prefer measurable targets in the Metric column. -->

| Category | Requirement | Metric / Target |
|----------|-------------|-----------------|
| Security | API keys only from environment variables (or a hidden prompt); customer text stays in gitignored `private/`; local model servers bind only to a 0700 Unix socket or 127.0.0.1 | No secrets or `private/` content in git; previews read no credential |
| Repository size | Existing tracked results stay; new large generated artifacts (images, videos, request/response journals, HTML viewers, dry runs) go to Git LFS or external storage, not plain git | Storage location TBD; no new large binaries in plain git |
| Maintainability | New library modules pass the full ruff rule set and have tests in tests/ | uv run poe lint and uv run poe test pass |
| Reproducibility | Plans regenerate the saved query identities from pinned data and unchanged semantics code | make_plan reproduces saved query_id sets (verified for 03_1, 05, 06) |
| Cost safety | No paid request outside documented execute paths; SDK retries disabled | Previews/dry-runs make 0 API calls |

## アーキテクチャ (Architecture)

<!-- High-level architecture: components, data flow, boundaries.
     Add a diagram or description here. -->

### Agent Roles

| Agent | Role | Responsibilities |
|-------|------|------------------|
| Change-point detector (System One model on local vLLM) | Translator layer 1 | Each tick, decide none / start / change for a target from state, instructions, alarms and requests at or before t |
| Translation model (OpenAI Responses) | Translator layer 2 | Write the intent statement for a target when the detector, a new instruction or a request triggers it |
| Judge (OpenAI Responses) | evaluation/ | Score each translation event per key point and overall, with rationale |
| Weight reader (OpenAI, code-execution loop) | src/weightreading | Read policy weights (and, for intent_compare method B, replay activations) and explain the policy or its intent |

- Translator (src/mtd_rl_xavier/translator/) writes events.jsonl/requests.jsonl/responses.jsonl (event schema v2); the D29 judge (evaluation/judge.py) and the 3x4 assessment (evaluation/assess_*.py) read them; plans in .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md and feature-translator-evaluation-three-points-four-aspects-plan.md.

## 技術選定 (Tech Stack & Rationale)

<!-- Chosen technologies and why. Record alternatives considered. -->

| Area | Technology | Rationale | Alternatives Considered |
|------|------------|-----------|-------------------------|
| Environment / packaging | uv + setuptools src layout; src/mentalmodeling/ as editable path dependency (own distribution, excluded from the main wheel) | Project rule (.claude/rules/dev-environment.md); one install exposes llm_x and mtd_rl_xavier | pip + setuptools flat packages (original branch) |
| LLM API | openai SDK (Chat Completions; Batch API for LAG 14-17) | Used by all ported experiments |  |
| Lint | ruff (E,W,F,I,B,UP) with per-file ignores for ported modules | Enforce rules on new code without reformatting ported code or dropping star re-exports | Reformat all ported code now |
| Tests / tasks | pytest + poethepoet (test, test-workspace, lint) | Workspace experiments reuse test file names, so each runs as its own pytest session | Single pytest run with importlib mode |
| Change-point detector (local) | Cloudflare/clef-flash @ fb4417d31c1d0f9e7e77f1141b40aaede7491e00 on vLLM >= 0.33.0 POST /v1/systemone (separate env, not the project .venv) | System One decision model natively served by vLLM (D26); client budgets the state because vLLM silently truncates it | OpenJev-4B (not supported by vLLM, own wev serve) |
| Translation / judge API | OpenAI Responses API with strict json_schema output, reasoning.effort, store=False, no temperature | gpt-6.1 models; structured per-key-point scores; frames sent as input_image items | Chat Completions |
| Judge video frames | PyAV (av) + Pillow as optional extra `video` (pending dependency approval) | D31; frame source is pluggable so tests need no decoder | ffmpeg subprocess; OpenCV |

## 制約 (Constraints)

<!-- Technical, organizational, regulatory, or resource constraints. -->

- Saved results are immutable; the ones already tracked in git stay as they are (no history rewrite). New large generated artifacts go to Git LFS or external storage, not plain git (location TBD). Saved results record repo-relative paths in the pre-restructure layout; read them via `mtd_rl_xavier.paths.resolve_recorded`.
- Plans refuse to reuse, resume or merge results produced under different prompt/scoring code (`semantics_sha256`, see `experiment/common.py`).
- Paid API: `workspace/09`-`12` send requests when `run.py` runs without arguments; `workspace/14`-`17` send OpenAI Batch requests with `--execute` alone; every other entry point needs `--execute --confirm-paid-api` (the Translator only when a layer is paid). See the table in the root README.
- Experiments need `data/llmx_data` at revision `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`; the dataset registry reads generated inventories in `outputs/dataset_inventory/`.

## Key Decisions

<!-- Durable architectural/design decisions. Append-only log. -->

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| Layout: src/mentalmodeling/ (vendored LLM-Xavier, code only; at mentalmodeling/ until bd72bed moved it under src/ on 2026-11-04), src/mtd_rl_xavier/ (shared library), workspace/ (exploratory experiments) | User request: separate prior-work code, curated code and messy experiments with low cognitive load | Keep the original flat layout; LLM-Xavier as a git submodule | 2026-10-20 |
| Experiments live at workspace/NN_name (same depth as experiments/NN_name), imported as workspace.NN_name | Existing parents[k] root computations stay valid; only module names change | workspace/experiments/NN_name (shifts every root computation) | 2026-10-20 |
| Saved results are ported byte-identical, including results/ (user choice); old-layout paths are resolved by paths.resolve_recorded | Results are immutable evidence and hash-protected | Rewrite paths inside results; leave results on the original branch | 2026-10-20 |
| RESTRUCTURED_SEMANTICS maps the branch-head semantics hash to the post-restructure hash | Only import/path lines changed; make_plan reproduces identical query identities for 03_1, 05, 06 | Require the original branch for any re-analysis of 05/06 | 2026-10-20 |
| No CLAUDE.md; root AGENTS.md stays the agent contract, project context lives in .claude/STATE.md and this file | User decision during /init (Orchestra rule) | CLAUDE.md importing @AGENTS.md | 2026-10-20 |
| LLM as a Translator = one Translator with two layers (change-point detector + translation model), run offline as a chronological loop per instruction target (aircraft or team); translate at step 0, detector tick every 5 steps, re-translate on detected/external change, new instruction or request | User requirement interview 2026-11-04 (D7, D9, D10, D17, D24, D28); details in .claude/docs/research/translator-requirements.md and .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md | Single-model translator; real-time loop; video-LLM timestamping | 2026-11-03 |
| Reusable parts in src/mtd_rl_xavier/{llm,lag,translator}/, judge and its YAML in evaluation/ (configs in evaluation/config/); the LLM call parts (OpenAI Responses, System One, local vLLM chat, vLLM launcher) are separate modules that never import each other, wired only in the CLIs | User asked for separate parts, API-based and local-vLLM-based calls kept apart (D2, D25) | Everything under evaluation/; a shared LLM base class or registry | 2026-11-03 |
| Judge = OpenAI gpt-6.1-sol, one call per translation event; per key point 0/1/2 plus overall 1-5, each with rationale; YAML decides which of six inputs (instructions, video frames, timestamps, trajectory, translated intent, key points) are sent and how they are rendered | Judge sees the same information as the human rater (D13, D15, D20, D29, D30); translation default gpt-6.1-terra avoids self-preference bias (D18) | One judge call per episode; single overall score | 2026-11-03 |
| Paid OpenAI calls preview by default and need --execute --confirm-paid-api; local servers bind only to a 0700 Unix socket or 127.0.0.1 with the API key from an env var; run outputs and real scenarios with customer text live in gitignored trees | Repo paid-gate convention (D14); vllm serve binds all interfaces by default; customer instructions may go to OpenAI (D11) but never into git | Execute by default; bind 0.0.0.0 on the shared node | 2026-11-03 |
| Translator triggers follow the user's point definitions: the System One detector asks one choice question (none/start/change) each tick, with instructions, alarms and request questions at or before t in its state; a target's first translation is a forced `initial` bootstrap; later instructions only feed the detector | D50/D51/D61: start = a planned new tactic begins, change = tactic change caused by an external factor; the detector owns the start/change judgement | Separate evaluation-only reference points with unchanged triggers; forced re-translation on every instruction | 2026-11-04 |
| Evaluation of Translator output on 3 points x 4 aspects lives in evaluation/assess_*.py with one commented config evaluation/config/assess.yaml; machine, human and judge results are stored separately and never overwrite each other; unset scales/thresholds give not_evaluated with a reason, never 0 | User request eval-3points-4aspects-request.md; D53-D59 (pilot 1-5 + cannot_judge, no aggregation, 3.0 s / 10.0 s, Japanese judge rationale, simplest defaults) | Extend the D29 key-point judge; hard-coded thresholds | 2026-11-04 |
| Generation inputs are checked per model call for future information (payload hash, every rendered step and message <= t, no episode/scenario id, declared visibility); retrospective runs are a tagged ablation reported separately | The user's strictest constraint: an explanation at t uses only information at or before t; D46 keeps retrospective as an ablation | Trust the pipeline without a per-call check | 2026-11-04 |
| MHI v1.3 match data (ACMI, recordings, CSVs) in workspace/13_LAG/data/ is tracked in git | User decision: sharing it through the repository is allowed; 14-17 and the Translator/judge configs read it | Move it to gitignored private/ and purge history | 2026-11-04 |
| Generated artifacts: results already tracked stay as they are; new large artifacts (images, videos, request/response journals, HTML viewers, dry runs) go to Git LFS or external storage | User decision: about 1 GB of generated files is already in plain git; stop the growth without rewriting history | Keep committing everything; rewrite history to move old results | 2026-11-04 |
| New main-side experiments are numbered from 21; 18-20 are reserved for the unported experiments on rafael/mental-modeling-openai, which is left as is for now | User decision: the legacy branch already uses 18, 19 (twice) and 20 | Continue at 18 and rename on port; a prefix for main-only experiments | 2026-11-04 |

## TODO / Open Questions

<!-- Open design questions and deferred decisions for this project. -->

- Where new large generated artifacts are stored (Git LFS or an external store) is not decided; nothing is set up yet.
- Whether and how the post-snapshot work on rafael/mental-modeling-openai (experiments 18-20, changes to 13 and 17) is ported.
- LLM Translator + judge (feature llm-judge-intent-translation-eval): open questions are listed in .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md section "Open Questions For The User" (dependency approval, mid-episode first instruction, judging all events, caps, detector threshold, judge video window, dead-target policy).
