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

Research code that translates the behaviour of reinforcement-learning agents into intent statements with LLMs and evaluates those translations with LLM-as-a-judge. The core is two packages: `src/translate`, one config-driven pipeline (choose inputs -> process them -> complete the prompt -> translate) that covers the two-layer online Translator (M1) and the former trajectory-based (A) and weights-and-activations-based (B) methods as configurations, and `src/evaluation`, the two judges (the D29 key-point judge and the 3-point x 4-aspect assessment). The legacy LLM-Xavier work (Lu et al., TMLR 2024: mental modelling, few-shot prompting, reward ablation, in-context RL, LAG trajectories) is kept, not deleted, as the independent `workspace/` project, which is also the free-experiment area. The core and `workspace/` do not depend on each other.

## スコープ (Scope)

### In Scope

<!-- What this project explicitly delivers. -->

- `src/translate`: one translation pipeline. A run config (`configs/translate/*.yaml`) selects the inputs (instructions, alarms, question, previous translation, trajectory, policy weights and replay activations), the prompt file (`configs/translate/prompts/`, whose protocol decides the call: one strict-schema Responses request or a sandboxed code session), and the trigger source: online (the clef-flash change-point detector decides start/change as the episode advances; first instructions and human requests force initial/request) or preset (steps chosen in advance, each with its kind).
- `src/evaluation`: the D29 key-point judge, the 3-point x 4-aspect assessment (`assess_*.py`), and the side-by-side viewer of translation runs.
- MHI v1.3 match data in `data/mhi/` (read-only) and Slurm jobs in `slurm/` (`common.sh`; MODE `preview`/`paid`, `probe`/`paid` for vLLM; logs in `logs/slurm/`).
- `workspace/`: an independent uv project (own `pyproject.toml`, `uv.lock`, `.venv`) with the legacy LLM-Xavier experiments 00-17, the vendored `src/mentalmodeling`, and the `mtd_rl_xavier_workspace` legacy library (experiment, preprocessing, datasets, tools, lag batch). It reads only `../data/mhi` from outside and is the place for free experiments. The core does not guarantee it.

### Out of Scope

<!-- What is explicitly NOT covered, to prevent scope creep. -->

- Training RL policies. Live environments run only to record or replay rollouts of existing LAG checkpoints (`scripts/policy_record_rollouts.py` runs `MultipleCombatEnv`, `scripts/episode_replay.py`) with the external mtd-rl-lag interpreter, never inside the core `.venv`.
- Importing or reading `workspace/` from the core, or importing the core from `workspace/`. Legacy LLM-Xavier experiments are not part of the core's tests, jobs or guarantees.
- Weights-only policy explanation (the former `weightreading run/evaluate`) and local vLLM chat (`llm/vllm_chat.py`): both deleted 2026-11-08.

## 機能要件 (Functional Requirements)

<!-- What the system must do. Each requirement gets a stable ID (FR-1, FR-2, ...). -->

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-1 | Legacy, `workspace/` only: build prompts from recorded RL trajectories with original LLM-Xavier semantics (next/last action, state direction, Joint variants) | Could | workspace/src/mtd_rl_xavier_workspace/preprocessing, workspace/src/mentalmodeling/llm_x |
| FR-2 | Send paid OpenAI queries only through explicit paid paths (core: `--execute --confirm-paid-api`, Slurm MODE `paid`), with resumable journals and provenance where a resume contract exists | Must | core: src/translate/llm, src/evaluation/, slurm/common.sh; legacy: workspace/src/mtd_rl_xavier_workspace/experiment |
| FR-3 | Legacy, `workspace/` only: analyse saved responses offline (accuracy, continuous metrics, confusion matrices, figures) without API calls | Could | workspace/src/mtd_rl_xavier_workspace/experiment, workspace/*/analysis |
| FR-4 | Legacy, `workspace/` only: inspect raw datasets and count prompt tokens locally | Could | workspace/src/mtd_rl_xavier_workspace/tools, workspace/notebooks |
| FR-5 | Removed 2026-11-08: weights-only explanation of an RL policy (`weightreading run/evaluate`); the policy input of the translation pipeline (FR-6) remains | - | deleted |
| FR-6 | Translate a LAG episode into intent statements per target at start, change and request points, online (clef-flash) or at preset points, from config-selected inputs with a config-provided prompt, using only information at or before t | Must | src/translate, configs/translate |
| FR-7 | Judge and assess Translator output (key-point judge; 3 points x 4 aspects) with machine, human and judge results stored separately | Must | evaluation/ |
| FR-8 | Compare input configurations (e.g. trajectory vs policy weights and activations) on the same recorded rollouts with a side-by-side viewer; every configuration writes the judges' input format | Should | configs/translate/seed100601-*.yaml, src/evaluation/viewer.py |

## 非機能要件 (Non-Functional Requirements)

<!-- Quality attributes: performance, availability, security, maintainability, etc.
     Prefer measurable targets in the Metric column. -->

| Category | Requirement | Metric / Target |
|----------|-------------|-----------------|
| Security | API keys only from environment variables (or a hidden prompt); customer text stays in gitignored `private/`; local model servers bind only to a 0700 Unix socket or 127.0.0.1 | No secrets or `private/` content in git; previews read no credential |
| Repository size | Existing tracked results stay; new large generated artifacts (images, videos, request/response journals, HTML viewers, dry runs) go to Git LFS or external storage, not plain git | Storage location TBD; no new large binaries in plain git |
| Maintainability | New library modules pass the full ruff rule set and have tests in tests/ | uv run poe lint and uv run poe test pass |
| Reproducibility | The assess judge skips requests by `request_sha256` (its prompts stay in code); the shipped configs reproduce the former prompts byte for byte and the recorded A/B runs replay with identical requests (`scripts/check_request_replay.py`); the former method B `--resume` is retired (2026-11-08; its three runs are complete); the legacy `workspace/` plans reproduce saved query_id sets from pinned data (03_1, 05, 06) | `sbatch slurm/validate.sbatch golden`; workspace tests |
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
| Code-session translator (OpenAI, code-execution loop) | src/translate/call_session.py, sandbox/, policy/ | The policy input: read policy weights and replay activations in the sandbox to write the intent statement |

- The pipeline (src/translate/) writes events.jsonl/requests.jsonl/responses.jsonl (event schema v2) for every configuration; each event records the human-visible material (scenario texts and the trajectory window) and the translator's inputs, so the D29 judge (src/evaluation/judge.py) and the 3x4 assessment (src/evaluation/assess_*.py) judge every configuration the same way; plans in .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md and feature-translator-evaluation-three-points-four-aspects-plan.md.

## 技術選定 (Tech Stack & Rationale)

<!-- Chosen technologies and why. Record alternatives considered. -->

| Area | Technology | Rationale | Alternatives Considered |
|------|------------|-----------|-------------------------|
| Environment / packaging | uv + setuptools; core packages `translate` and `evaluation` (both under src/) installed explicitly; `workspace/` is a separate uv project with its own lock and `.venv` | Project rule (.claude/rules/dev-environment.md); `python -m evaluation.*` works from any directory; core tests never see workspace code | One project with src/mentalmodeling as a path dependency (until 2026-11-08) |
| LLM API | openai SDK, Responses API for the core; Chat Completions and the Batch API only in `workspace/` (experiments 01-17) | Core judge and Translator need strict json_schema output and image input; legacy experiments were ported unchanged |  |
| Lint | ruff (E,W,F,I,B,UP) over `src tests scripts` in the core | Enforce rules on all core code | Per-file ignores for ported legacy modules (now only in workspace/) |
| Tests / tasks | pytest + poethepoet in each project (core: `test`, `lint`; workspace: `test-library`, `test-experiments`); core Slurm `validate.sbatch` modes `subset`, `full`, `static`, `golden`, `translate-preview` | Core tests run without API access; workspace experiments reuse test file names, so workspace runs each as its own pytest session | Single pytest run across both projects |
| Change-point detector (local) | Cloudflare/clef-flash @ fb4417d31c1d0f9e7e77f1141b40aaede7491e00 on vLLM >= 0.33.0 POST /v1/systemone (separate env, not the project .venv) | System One decision model natively served by vLLM (D26); client budgets the state because vLLM silently truncates it | OpenJev-4B (not supported by vLLM, own wev serve) |
| Translation / judge API | OpenAI Responses API with strict json_schema output, reasoning.effort, store=False, no temperature | gpt-6.1 models; structured per-key-point scores; frames sent as input_image items | Chat Completions |
| Judge video frames | PyAV (av) + Pillow as optional extra `video` (declared in pyproject.toml and locked in uv.lock) | D31; frame source is pluggable so tests need no decoder | ffmpeg subprocess; OpenCV |

## 制約 (Constraints)

<!-- Technical, organizational, regulatory, or resource constraints. -->

- Saved results are immutable; the ones already tracked in git stay as they are (no history rewrite). New large generated artifacts go to Git LFS or external storage, not plain git (location TBD). Saved legacy results record repo-relative paths in the pre-restructure layout; `workspace/` resolves them with `mtd_rl_xavier_workspace.paths.resolve_recorded`.
- Dependency direction: the core never imports or reads `workspace/`; `workspace/` never imports the core and reads only `../data/mhi`. `tests/test_core_boundary.py` checks the core side. In `workspace/`, plans refuse to reuse, resume or merge results produced under different prompt/scoring code (`semantics_sha256`).
- Paid API in the core: every entry point previews by default and needs `--execute --confirm-paid-api`; Slurm MODE `paid` adds both flags and `preview` drops any inherited `OPENAI_API_KEY`. The key lives in the gitignored, group-shared repo-root `.env` (mode 0660 on purpose) and is loaded only right before a paid submission. Legacy entry points in `workspace/` are weaker (09-12 send with no arguments; 14-17 send Batch requests with `--execute` alone); see `workspace/README.md`.
- Legacy `workspace/` experiments need `data/llmx_data` at revision `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd` (downloaded locally, gitignored). The core needs only `data/mhi/` (read-only).
- The host is a Slurm login node: heavy work runs as Slurm jobs submitted from the repository root; the login node is for editing and git only.

## Key Decisions

<!-- Durable architectural/design decisions. Append-only log. -->

| Decision | Rationale | Alternatives Considered | Date |
|----------|-----------|------------------------|------|
| Layout: src/mentalmodeling/ (vendored LLM-Xavier, code only; at mentalmodeling/ until bd72bed moved it under src/ on 2026-11-04), src/mtd_rl_xavier/ (shared library), workspace/ (exploratory experiments) (superseded 2026-11-08, see below) | User request: separate prior-work code, curated code and messy experiments with low cognitive load | Keep the original flat layout; LLM-Xavier as a git submodule | 2026-10-20 |
| Experiments live at workspace/NN_name (same depth as experiments/NN_name), imported as workspace.NN_name (location superseded 2026-11-08: workspace/ is now an independent project, see below) | Existing parents[k] root computations stay valid; only module names change | workspace/experiments/NN_name (shifts every root computation) | 2026-10-20 |
| Saved results are ported byte-identical, including results/ (user choice); old-layout paths are resolved by paths.resolve_recorded | Results are immutable evidence and hash-protected | Rewrite paths inside results; leave results on the original branch | 2026-10-20 |
| RESTRUCTURED_SEMANTICS maps the branch-head semantics hash to the post-restructure hash | Only import/path lines changed; make_plan reproduces identical query identities for 03_1, 05, 06 | Require the original branch for any re-analysis of 05/06 | 2026-10-20 |
| No CLAUDE.md; root AGENTS.md stays the agent contract, project context lives in .claude/STATE.md and this file | User decision during /init (Orchestra rule) | CLAUDE.md importing @AGENTS.md | 2026-10-20 |
| LLM as a Translator = one Translator with two layers (change-point detector + translation model), run offline as a chronological loop per instruction target (aircraft or team); translate at step 0, detector tick every 5 steps, re-translate on detected/external change, new instruction or request | User requirement interview 2026-11-04 (D7, D9, D10, D17, D24, D28); details in .claude/docs/research/translator-requirements.md and .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md | Single-model translator; real-time loop; video-LLM timestamping | 2026-11-03 |
| Reusable parts in src/mtd_rl_xavier/{llm,lag,translator}/, judge and its YAML in evaluation/ (configs in evaluation/config/) (paths superseded 2026-11-08: src/translate/{llm,lag}/, src/evaluation/ with src/evaluation/config/, see below); the LLM call parts (OpenAI Responses, System One, local vLLM chat, vLLM launcher) are separate modules that never import each other, wired only in the CLIs (the local vLLM chat part was deleted 2026-11-08, see below) | User asked for separate parts, API-based and local-vLLM-based calls kept apart (D2, D25) | Everything under evaluation/; a shared LLM base class or registry | 2026-11-03 |
| Judge = OpenAI gpt-6.1-sol, one call per translation event; per key point 0/1/2 plus overall 1-5, each with rationale; YAML decides which of six inputs (instructions, video frames, timestamps, trajectory, translated intent, key points) are sent and how they are rendered | Judge sees the same information as the human rater (D13, D15, D20, D29, D30); translation default gpt-6.1-terra avoids self-preference bias (D18) | One judge call per episode; single overall score | 2026-11-03 |
| Paid OpenAI calls preview by default and need --execute --confirm-paid-api; local servers bind only to a 0700 Unix socket or 127.0.0.1 with the API key from an env var; run outputs and real scenarios with customer text live in gitignored trees | Repo paid-gate convention (D14); vllm serve binds all interfaces by default; customer instructions may go to OpenAI (D11) but never into git | Execute by default; bind 0.0.0.0 on the shared node | 2026-11-03 |
| Translator triggers follow the user's point definitions: the System One detector asks one choice question (none/start/change) each tick, with instructions, alarms and request questions at or before t in its state; a target's first translation is a forced `initial` bootstrap; later instructions only feed the detector | D50/D51/D61: start = a planned new tactic begins, change = tactic change caused by an external factor; the detector owns the start/change judgement | Separate evaluation-only reference points with unchanged triggers; forced re-translation on every instruction | 2026-11-04 |
| Evaluation of Translator output on 3 points x 4 aspects lives in evaluation/assess_*.py with one commented config evaluation/config/assess.yaml; machine, human and judge results are stored separately and never overwrite each other; unset scales/thresholds give not_evaluated with a reason, never 0 | User request eval-3points-4aspects-request.md; D53-D59 (pilot 1-5 + cannot_judge, no aggregation, 3.0 s / 10.0 s, Japanese judge rationale, simplest defaults) | Extend the D29 key-point judge; hard-coded thresholds | 2026-11-04 |
| Generation inputs are checked per model call for future information (payload hash, every rendered step and message <= t, no episode/scenario id, declared visibility); retrospective runs are a tagged ablation reported separately | The user's strictest constraint: an explanation at t uses only information at or before t; D46 keeps retrospective as an ablation | Trust the pipeline without a per-call check | 2026-11-04 |
| MHI v1.3 match data (ACMI, recordings, CSVs) in workspace/13_LAG/data/ is tracked in git (superseded 2026-11-08: the data moved to data/mhi/, see below) | User decision: sharing it through the repository is allowed; 14-17 and the Translator/judge configs read it | Move it to gitignored private/ and purge history | 2026-11-04 |
| Generated artifacts: results already tracked stay as they are; new large artifacts (images, videos, request/response journals, HTML viewers, dry runs) go to Git LFS or external storage | User decision: about 1 GB of generated files is already in plain git; stop the growth without rewriting history | Keep committing everything; rewrite history to move old results | 2026-11-04 |
| New main-side experiments are numbered from 21; 18-20 are reserved for the unported experiments on rafael/mental-modeling-openai, which is left as is for now | User decision: the legacy branch already uses 18, 19 (twice) and 20 | Continue at 18 and rename on port; a prefix for main-only experiments | 2026-11-04 |
| The repository keeps only translator and judge code (core: M1 Translator, method A, method B, judges in evaluation/, data/mhi/, slurm/). The legacy LLM-Xavier work (experiments 00-17, src/mentalmodeling, the mtd_rl_xavier_workspace library) moves to workspace/ as an independent uv project and free-experiment area; the core never imports or reads workspace/, and workspace/ reads only ../data/mhi. Not deleted. | Owner decision 2026-11-08: the research is now Translator and judge; legacy code must stay available without constraining the core or its tests | Delete the legacy work; keep one project with a shared library | 2026-11-08 |
| MHI v1.3 match data lives in data/mhi/ (read-only), shared by the core and by workspace/ experiments 14-17; jobs live in slurm/ with common.sh, MODE preview/paid (probe/paid for vLLM), logs in logs/slurm/ | One data location outside both projects keeps the dependency direction one-way; the single MODE argument keeps a paid run an explicit opt-in | Keep the data in workspace/13_LAG/data; per-job flags instead of MODE | 2026-11-08 |
| Delete the weights-only explainer W (weightreading run/evaluate) and llm/vllm_chat.py; rescue the method A to Translator v2 converter (intent_compare.to_translator_run) and the assess judge job slurm/judge_assess.sbatch. Method B keeps only sandbox-selftest, intent-preview and intent-run | Owner decision 2026-11-08: the repository keeps only translator and judge code; the converter and judge_assess.sbatch connect method A to the judge | Move W and vllm_chat to workspace/ | 2026-11-08 |
| .env (OPENAI_API_KEY, repo root) stays group-shared at mode 0660; it is gitignored, loaded only right before a paid submission and unset afterwards, and is never moved or chmod-ed | Owner decision 2026-11-08; the paid gate (preview by default, --execute --confirm-paid-api) still applies | Moving the file or tightening its mode | 2026-11-08 |
| The root README.md is written in Japanese (workspace/README.md is Japanese too) at the owner's request; this is an exception to the English-docs rule. Code, identifiers, comments, commands and the other technical documents stay English | Owner request 2026-11-08 | English README per the language rule | 2026-11-08 |

## TODO / Open Questions

<!-- Open design questions and deferred decisions for this project. -->

- Where new large generated artifacts are stored (Git LFS or an external store) is not decided; nothing is set up yet.
- Whether and how the post-snapshot work on rafael/mental-modeling-openai (experiments 18-20, changes to 13 and 17) is ported into `workspace/`.
- LLM Translator + judge: open questions are listed in .claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md section "Open Questions For The User" (mid-episode first instruction, judging all events, caps, detector threshold, judge video window, dead-target policy). Real-data M1 runs still await the vLLM environment and GPU probe approval (D49).
- The preset trigger kinds of the reproduction configs (45 initial, 145/245/315 change) are assumptions; set them from an annotation before judging.
- The judge heading "TRAJECTORY THE TRANSLATOR SAW" is shown for every event, also for translations that read only the policy; changing it is a new rubric version (judge prompts stay in code).
- "Online" is the causal per-step loop over a recorded episode with scenario requests revealed at their step; a live request channel is not built.
| One config-driven pipeline in src/translate (choose inputs -> process -> prompt -> translate) replaces M1, method A and method B; src/ holds only translate/ and evaluation/ (evaluation depends on translate, never the reverse) | Owner request 2026-11-08: methods A and B differ only in what they reference; "method A" is meaningless as a name | Keep three packages; a third shared package | 2026-11-08 |
| Trigger kinds stay initial/start/change/request; trigger source online (clef-flash) or preset (steps and kinds in the config) | Owner decision 2026-11-08: translation runs at start, change and request points, online or chosen in advance | A new `scheduled` kind with event schema v3 | 2026-11-08 |
| The judge is method-independent: every event records the human-visible material and the judge prompt is the same for every configuration; judge prompts stay in code | Owner decision 2026-11-08; the paid assess resume depends on judge prompt bytes | Per-method judge variants; not_evaluated for policy-only translations | 2026-11-08 |
| Translation prompt text lives in prompt files (configs/translate/prompts/) filled from the selected inputs; shipped configs reproduce the former prompts byte for byte (goldens + replay) | Owner decision 2026-11-08: prompt ablations by config | Two prompt templates in code | 2026-11-08 |
| The former method B `--resume` is retired | Owner decision 2026-11-08: its three recorded runs are complete; results stay readable | Keep a legacy resume adapter and the old config paths | 2026-11-08 |
| Judge prompts live in prompt files under configs/evaluation/prompts/, named by the judge configs in configs/evaluation/ (supersedes "judge prompts stay in code"); code only renders the material into the templates; placeholders checked at load, strict YAML. assess-legacy.yaml (pilot-2, assess-judge-v2) and judge-text-legacy.yaml (judge-text-v3) reproduce the former request bytes; the defaults (assess-judge-v3, judge-text-v4) call the trajectory the human-visible telemetry | Owner decision 2026-11-08: the judge follows the translate prompt-file pattern; a policy-input translation must not be labelled with the trajectory the Translator saw. request_sha256 covers only the request body, so the paid 2026-11-05 run resumes with the legacy config (check_assess_resume); an event already journaled with other bytes is refused, not re-billed | Keep prompts in code; one prompt file shared by both judges; edit the legacy prompt in place | 2026-11-08 |
| Trigger kinds are exactly start (開始点, first instruction or preset), change (変化点, clef-flash; merges the former detector start and change) and request (要求点), event schema 3; detector options none/change; seed100601 presets 45 -> start (supersedes "trigger kinds stay initial/start/change/request"). Schema 2 runs stay readable: an explicit per-version table (src/evaluation/trigger_kinds.py) and a per-config trigger map (assess-legacy: initial,start -> start; default pilot-3: initial -> start, start -> change) | Owner decision 2026-11-08: three kinds; the paid run (schema 2) must resume byte for byte under the legacy config, and old kinds are mapped explicitly, never guessed | Rewrite old event logs; infer kinds from trigger sources; keep four kinds | 2026-11-08 |
