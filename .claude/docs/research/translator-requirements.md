# LLM as a Translator + LLM-as-a-judge: requirement decisions

> **Copy.** The canonical decision log is kept on the project's Google Drive. This file is a
> repository copy so that the `Dn` references in the code and READMEs (e.g.
> `src/mtd_rl_xavier/translator/README.md`, `evaluation/README.md`) resolve. If the two
> disagree, the Google Drive version wins; update this copy to match.

Decision log from the requirement interview with the user (2026-11-04). Facts behind the questions:
[`translator-contract-facts.md`](archive/translator-contract-facts.md),
[`feature-llm-judge-intent-translation-eval-codebase.md`](archive/feature-llm-judge-intent-translation-eval-codebase.md),
[`llm-judge-intent-translation-eval.md`](archive/llm-judge-intent-translation-eval.md) (in `archive/`).

## Decided

| # | Topic | Decision |
|---|-------|----------|
| D1 | Order of work | Define the Translator first, then fit the judge to it (option A). |
| D2 | Modularity | Keep parts separate. API-based LLM calls and local vLLM-server-based LLM calls are separate parts. |
| D3 | Translation direction | Aircraft behaviour (trajectory) -> natural-language intent. The instruction is context: what the human asked the policy to do. |
| D4 | Request point (要求点) | The moment a human asks for a translation. It may arrive at any time. |
| D5 | Visibility | Whether the Translator may see the trajectory after the translation time (online vs. retrospective) is a config switch. |
| D6 | Change point (変化点) | A motion change the previous intent translation cannot explain. Two sources must both be supported: given externally, or judged by the Translator itself from the trajectory. |
| D7 | Translator structure | One `Translator` class with two layers: a change-point detection model and a translation model. Not real-time (offline is fine). Any model can be assigned to each layer. |
| D8 | Default models | Detection layer: "Openjev 4B" (exact model id unconfirmed) run locally on vLLM. Translation layer: a strong OpenAI model (exact default unconfirmed). Judge: always OpenAI `gpt-6.1-sol`. |
| D9 | Detection-layer input | Trajectory + previous translation. Detection and translation therefore alternate in time order (sequential loop): translate at the start point, then at each detection tick ask "is there an unexplained change?", re-translate on yes. |
| D10 | Request-point behaviour | Re-translate on the spot: call the translation model with the trajectory up to the request time and the previous translation. |
| D11 | Customer instructions | The user states the 43 customer instructions MAY be sent to OpenAI (translation layer and judge). Note: `work/evaluation-data-handoff/.../source-documents.csv` still labels them "do not send"; that label is now out of date per the user. |
| D12 | Development data | Build on the three no-instruction 2v2 episodes in `workspace/13_LAG` with provisional instructions; swap in instruction-conditioned recordings later. Keep the episode loader replaceable. |
| D13 | Judge output | Per-key-point score with rationale + overall score with rationale. |
| D14 | Paid calls | Preview by default; explicit flags for real calls (mirror weightreading `--execute --confirm-paid-api`). |
| D15 | Judge config | YAML under `evaluation/config/` declares which judge inputs are passed and how each is rendered. |

| D16 | Start point (開始点) | Episode start (step 0). The instruction is given at episode start; the first translation is produced there. |
| D17 | Translation unit | Per instruction target: the single aircraft for a single-aircraft instruction, the whole team for a team instruction. |
| D18 | Default translation model | `gpt-6.1-terra` (kept different from the judge `gpt-6.1-sol` to avoid self-preference bias). |
| D19 | Detection model | OpenJev (https://github.com/alanhuangyoo/OpenJev), 4B, run locally. The local vLLM runner is its own part, and OpenJev is its own part. The user expects vLLM may already support running such "System 1" models; research pending in `openjev-vllm.md`. |

| D20 | Judge unit | One judge call per translation event (start point, change point, or request point). Key points are prepared per event. |
| D21 | Instructions per episode | Instructions may change mid-episode; each instruction carries its issue step. (The first instruction is issued at the start point, step 0.) |
| D22 | Trajectory rendering | 17_commentary_LAG JSON lines (one line per step per object) by default; the renderer is a separate, swappable part and each layer may choose its own renderer. |
| D23 | Output language | Config switch. |

| D24 | Instruction change | A newly issued instruction is itself a change point (re-translate with the new instruction). |
| D25 | Code placement | Reusable parts under `src/mtd_rl_xavier/` (LLM calls: API / local server; LAG episode loading; trajectory rendering; Translator layers + pipeline) with tests in `tests/`; the judge and its YAML configs under `evaluation/`. |

Facts from `openjev-vllm.md`: OpenJev-4B (`alanhuangya/OpenJev-4B`) is a System-One decision model
(typed questions `choice` / `noul` / `score` -> calibrated probabilities; no text generation; English;
state <= 4096 tokens). vLLM >= 0.32 exposes `POST /v1/systemone` with the same request shape but does
not support OpenJev's architecture; OpenJev ships `wev serve` (FastAPI, same endpoint, no auth).

| D26 | Detector default model | Switched from OpenJev-4B to Cloudflare clef-flash, a vLLM-supported System One decision model, served by `vllm serve` (details pending in `openjev-vllm.md`, section "Clef-Flash on vLLM"). The System One client stays server-agnostic (`base_url`). |
| D27 | Detector trajectory | Short window of raw data (17-format subset: target aircraft and main opponents, key fields); window length and fields in config; token count checked before sending. |
| D28 | Detection tick | Every 5 steps (1 s), configurable. |

| D29 | Judge scales | Per key point 0/1/2 (2 = core purpose and main action reproduced, 1 = partial, 0 = missing or different purpose; same as 17's human rubric) + overall 1-5. |
| D30 | Judge video | `workspace/13_LAG/data/viz/action_videos/*.mp4` (20 fps, 1920x1080, frame = step), the same video the human rater watches. Frame/step mapping per video in config. |
| D31 | Video decoder | Add PyAV (`av`) as an optional extra via uv. |

Facts from `openjev-vllm.md` "Clef-Flash on vLLM": `Cloudflare/clef-flash` is 9B (Qwen3.5-9B based,
~19 GB bf16, Apache-2.0, pin revision `fb4417d31c1d0f9e7e77f1141b40aaede7491e00`); runs on stock vLLM
>= 0.33.0 with `vllm serve Cloudflare/clef-flash --max-model-len 16384`; over-long states are silently
truncated from the END (budget min(16384, max_model_len)) so the client must budget and check usage;
`vllm serve` binds all interfaces by default -> use `--uds` (0700 dir) or `--host 127.0.0.1` plus
`VLLM_API_KEY`; vLLM lives in a separate env; HF cache on /lustre.

## Pending user approval at run time (not needed to implement)

- ~19 GB weight download, a separate vLLM env, one short `--gres=gpu:1` probe job.
- A small labelled set of LAG windows to check clef-flash change-point accuracy before fixing the threshold.

| D32 | Translation-layer window | Always the last N steps (default N = 60 = 12 s, configurable), plus the previous translation text. |
| D33 | Default visibility | Online (no steps after the translation time). Retrospective (+M steps after) is the config alternative. |

## Open

- None blocking design. Key points (採点の要点) will be supplied by the user later.

## Decided at plan approval (2026-11-04)

| # | Topic | Decision |
|---|-------|----------|
| D34 | Plan + dependencies | Plan `.claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md` approved; P0 adds `httpx` and `pandas` as core dependencies and the optional `video` extra (`av`, `Pillow`). |
| D35 | Mid-episode first instruction | Allowed: a target may receive its first instruction after step 0; it is translated at the issue step with trigger `instruction` and no previous translation. |
| D36 | Partial judging | Allowed: the key-points file may cover a subset of the run's translation events. Every key-point entry must reference an existing event (unknown or duplicate event ids are errors); events without key points are recorded as not judged, with no API call. Supersedes the plan's "bijection with all events". |
| D37 | Dead targets | Once a target is down (a team: all its aircraft), stop translating it; later triggers are logged as `trigger_suppressed` with no model call. |

## Decided at review (2026-11-04)

| # | Topic | Decision |
|---|-------|----------|
| D38 | Review fix scope | Fix the recommended set before any paid run: explicit OpenAI base_url, judge prompt-injection fencing, JSONL split on "\n" only + duplicate-key rejection, config URL/secret leak, uncertain status kept on interrupt/final-write failure, High test gaps (budget non-fit, --run-name traversal), small fixes (`/private/` ignored whole, etc.). Reports: `.claude/docs/research/review-{security,quality,tests}-llm-judge-intent-translation-eval.md`. |
| D39 | Detector endpoint default | UDS under a per-user 0700 directory is the default in `configs/translator/run.yaml`; loopback TCP stays selectable by config. |
| D40 | File permissions | Keep the current umask (007); the 0700 run directory is the protection. No umask change. |
| D41 | Over-long intent | Keep the current policy: an intent over `translation.max_intent_chars` fails the run (limit configurable). |

## Decided after the review fixes (2026-11-04)

| # | Topic | Decision |
|---|-------|----------|
| D42 | Translation output language | Japanese (`translation.output_language` in `configs/translator/run.yaml`). The user chose it knowing clef-flash is English-only and receives the previous translation as part of its state. |
| D43 | Translation prompt | Keep `translator-intent-v1` as written. |
| D44 | SDK environment variables | Refuse to start while `OPENAI_LOG` is set (it logs request bodies). Proxy, CA-bundle and `OPENAI_ORG_ID`/`OPENAI_PROJECT_ID` variables stay honoured; run provenance records which of them were set (names only, never values). |
| D45 | Evaluation framework | The user asked for a "3 time points x 4 aspects" evaluation of Translator output (`eval-3points-4aspects-request.md`, verbatim). It supersedes the open question on the judge's overall 1-5 anchors. It goes through /feature (design + plan approval) before implementation. |
| D46 | Retrospective mode | Kept as an important ablation (not dropped). Online stays the default and the only "<= t"-compliant mode; retrospective runs must be declared, flagged as future-informed by the leakage check, and reported separately from online results. Implementation (L2) not started yet. |
| D47 | Commit | Commit this feature's files only to a new branch `feature/llm-translator-judge`, without switching the user's checked-out branch (`feature/weightreading`) or touching weightreading / intent_compare work. |
| D48 | Later phases | Start L1 (vllm_chat, vllm_server launcher, sbatch) and L3 (judge video frames) now, in parallel with the evaluation-feature design. |
| D49 | Run-time setup | vLLM env (~10 GB), clef-flash download (~19 GB) and the GPU probe job: ask again after L1 is done. |

## Decided for the evaluation feature (2026-11-04)

Plan `.claude/docs/plans/feature-translator-evaluation-three-points-four-aspects-plan.md` approved, with these decisions:

| # | Topic | Decision |
|---|-------|----------|
| D50 | Point definitions | Align the Translator to the user's definitions: 開始点 = the target begins a new tactic as planned; 変化点 = the tactic changes because of an external factor (alarm, instruction). The System One detector decides start/change points; supersedes the trigger semantics of D6/D16/D24. |
| D51 | Alarms and questions | Scenarios carry alarms and request question text; pass them to the Translator (detector state and translation prompt) at or before t. The detector must receive them so it can identify change points. |
| D52 | Translator output | Stays free text `{intent, evidence}` (no structured v2). |
| D53 | Aspect scales | Pilot scale 1-5 plus "cannot judge" (recorded as not_evaluated), versioned; level wording drafted as a pilot for the user to review. The D29 key-point judge stays separate. |
| D54 | Overall result | Not aggregated for now (not_evaluated/aggregation_unset). |
| D55 | Request timing | Response deadline 3.0 s, minimum question interval 10.0 s (config values). |
| D56 | Latency | Add monotonic `latency_s` to Translator `responses.jsonl`. |
| D57 | Contribution and language | Humans rate 判断への寄与 1-5 after reading the explanation (no paired study). Judge rationales in Japanese. |
| D58 | Defaults | a_t inclusive; no window matching; hindsight change-point timing not_evaluated; alias outcome check (~10 lines); judge cap 100 per run; pseudonymous rater ids; keep every rating; segment ends at the next start/change point; video stored as hashes only; no pytest-cov; tactic vocabulary, annotated units, rater count null until given. All in clearly commented, easy-to-edit config. |
| D59 | Simplicity | The user asks for the simplest possible implementation. |
| D60 | Where | Separate git worktree `../mtd-rl-xavier-assess` on `feature/llm-translator-judge` (after L1/L3 are committed there), with its own `.venv`. |
| D61 | Revised evaluation plan | Approved as revised (2026-11-04): later instructions only feed the detector (no forced re-translation); scenario `external_change_points` removed (change references live in the evaluation annotation); forced `initial` translation at a target's first instruction; `max_intent_chars` 600; event schema v2. |
| D62 | Simplification pass | After E2, simplify `evaluation/assess_*.py` (about 3.2k lines vs the ~1.3k estimate) toward <= 2k lines without dropping any requested capability or test. |
| D63 | Simplification result | Accepted at 2928 lines (from 3229); further cuts would weaken requested validation, provenance or error clarity. |
| D64 | Delivery | Committed `10e34fe` to `feature/llm-translator-judge` and opened PR #3 against `main` (https://github.com/matsuoinstitute/mtd-rl-xavier/pull/3). |
| D65 | Score anchors | Every 1-5 score gets concrete, observable anchors per criterion (about 20 criteria, with LAG examples); humans and the judge see the same anchors. |
| D66 | Anchor scope | Both judges: the 3x4 assessment (judge + packets.md) and the existing D29 key-point judge's overall 1-5. |
| D67 | Judge rationale | Every rationale must state which level's conditions are met and why the next level up is not; output format (score + rationale) unchanged. Design consulted with sol (Codex). |
| D68 | Anchor follow-ups | change_delta / consistent_with_latest are not_applicable without a previous explanation; timing.change_in_time stays one criterion; one judge call per event. Shipped as PR "Anchor every 1-5 judge score" (feature/judge-score-anchors). |
