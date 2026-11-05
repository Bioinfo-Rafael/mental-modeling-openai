# Codebase Scan: Translator evaluation, 3 time points x 4 aspects

Date: 2026-11-04. Scope: affected-area scan for /feature MODE=existing (design only).
Requirement (verbatim Japanese): `eval-3points-4aspects-request.md`. Decisions D1-D49:
`translator-requirements.md` (D46 retrospective kept as an ablation; D47 branch
`feature/llm-translator-judge`; D48 L1/L3 now in parallel; D49 run-time setup later).
Everything below was read from the working tree on 2026-11-04 (all Translator/judge code is
uncommitted; 879 tests pass after small-fixer's changes). Read-only scan: no source, test or
config file was edited, no paid call, no network.

## 1. Areas the feature touches

| Area | Files | Role for this feature |
|---|---|---|
| Translator event contract | `src/mtd_rl_xavier/translator/{schema,pipeline,translation,detector,config,cli}.py`, `README.md` | Producer of everything evaluated. `events.jsonl` (`translation_event`, `detector_call`, `target_closed`, `trigger_suppressed`), `requests.jsonl` (exact payload, written before send), `responses.jsonl` (raw), `run.json` (provenance, resolved config incl. `pipeline.visibility`) |
| Judge package | `evaluation/{configuration,schemas,rendering,judge,cli}.py`, `evaluation/config/judge-text.yaml`, `evaluation/README.md` | Reused infrastructure: strict YAML (`read_strict_yaml`), strict JSONL reader (`load_translator_run`), Responses request building, `JudgeModel`/`OpenAIJudgeModel`, write-before-send JSONL sink, paired paid gate, exit codes 0-4, provenance |
| LAG data | `src/mtd_rl_xavier/lag/{dataset,rendering}.py` | Loader (`load_episode`, hash-verified), `last_complete_step`, `render_lag17_jsonl`, `render_detector_compact` (needed to build *evaluation* windows that may extend after t) |
| LLM part | `src/mtd_rl_xavier/llm/openai_responses.py` | `OpenAIRequest`, `build_responses_body`, `make_openai_client` (only reader of `OPENAI_API_KEY`) |
| In flight (D48, do not depend on final shape) | `src/mtd_rl_xavier/llm/{vllm_chat,vllm_server}.py`, `scripts/translator_vllm.sbatch`, `evaluation/config/judge-text-video.yaml`, edits in `evaluation/{rendering,cli}.py` | Video block (`FrameSource`, PyAV) lands in the existing judge; the new aspect judge should consume it through the same block interface later |
| Out of scope (user's concurrent work) | `src/weightreading/`, `configs/weightreading/`, `configs/intent_compare/`, `scripts/`, `tests/weightreading/`, `tests/intent_compare/` | Untouched |

## 2. Existing contracts the feature must conform to

- **Event ids**: `event_id = <scenario_id>:<kind>:<target_id>:<step:06d>`; also the `call_id` in
  `requests.jsonl`/`responses.jsonl`; `request_sha256` = SHA-256 of canonical JSON of the exact
  payload. This join is the only authoritative link from an event to its exact generation input.
- **Trigger reasons** (`schema.py:30-42`): `start`, `instruction`, `external_change`,
  `detector_change`, `request`; each `{kind, source, reference_id, probability}`;
  `reference_id` = instruction / request / change id or detector `call_id`. Request and
  instruction points can therefore be matched to events *exactly* (no time window needed).
- **Translator output** (`translation.py:123-133`, D43 prompt `translator-intent-v1`): strict
  `{"intent": str, "evidence": [str, ...]}` free text, now written in Japanese (D42). No
  structured element fields, no "unknown" field, no question text.
- **Scenario** (`config.py`): `instructions[{instruction_id, issue_step, target, text}]`,
  `requests[{request_id, step, targets, source}]` (no question text),
  `external_change_points[{change_id, step, target, source}]` (no cause, no content; only a
  source label). No alarm records exist anywhere.
- **Judge** (D29): key points 0/1/2 + overall 1-5, one call per event, partial judging (D36),
  `not_judged` for events without key points, prompt-injection envelope `judge-text-v2` (D38).
- **Visibility**: `pipeline.visibility.mode` accepts only `online` with `future_steps: 0`;
  `retrospective` is rejected "not implemented in this phase" (`config.py:44`, `:268`, `:321`).
  `run.json` records the resolved config, so the mode is machine-readable per run (D46 hook).

## 3. Conventions (to follow exactly)

- Strict YAML: exact key sets, unknown/missing/duplicate keys fail, `bool` is never an int, no
  default model in code; `read_strict_yaml` in `evaluation/configuration.py` is reusable.
- Frozen dataclasses, pure functions, new objects instead of mutation (`{**d, k: v}` style).
- Composition roots only in CLIs; Protocols on the consumer side; clients built after the gate.
- Paid gate: preview by default; judge needs `--execute --confirm-paid-api`; preview writes
  `api_requests_made: 0`; run dirs `mkdir(exist_ok=False, mode=0o700)` via `derived_path()`;
  `--run-name` must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`.
- Exit codes: 0 ok/preview, 1 deterministic failure, 2 usage/gate, 3 uncertain, 4 cap_reached.
- Provenance: git commit + dirty flag, SHA-256 of code files, config/inputs hashes, prompt
  version + SHA-256, model, usage. Errors as `{type, message}` with redaction, no tracebacks.
- JSONL: split on `"\n"` only, duplicate JSON keys rejected (`strict_json_loads`), finite JSON.
- Sensitive data: real scenarios / key points under gitignored `/private/`; outputs under
  gitignored `/outputs/translator/`, `/outputs/evaluation/`; committed examples `[PLACEHOLDER]`.
- Tests: `tests/<area>/` with `*_fakes.py` + `conftest.py`; real 13_LAG episode as session
  fixture (`tests/evaluation/conftest.py` writes a fake-layer Translator run with 5 events);
  scripted SDK clients; `httpx.MockTransport`; offline only; names `test_<target>_<condition>_...`.
- Lint: `uv run poe lint` = `ruff check src tests evaluation`; line length 120; py310 target.
- Root `evaluation` is a top-level package run as `python -m evaluation.cli` from the repo root.

## 4. Dependencies

- Runtime deps already declared: `httpx`, `pandas`, `numpy`, `openai`, `PyYAML`; optional
  `video` extra (`av`, `Pillow`). No statistics library (scipy is only in `data`/`estimation`
  extras). Agreement statistics (Cohen's kappa, weighted kappa, exact agreement) are a few dozen
  lines of pure Python, so no new dependency is needed.
- `pytest-cov` is NOT installed; "80 % coverage" can only be measured after approving it.
- Downstream consumers of the Translator contract today: `evaluation/` only.

## 5. Data that exists vs. data the request needs

| Needed by the request | Exists? | Where / gap |
|---|---|---|
| Trajectory <= t and > t | Yes | 13_LAG processed CSV (3 episodes, 293-495 steps); evaluator can render future windows from the same hash-verified CSV |
| Instructions with issue step | Provisional only | Scenario YAML; 13_LAG has no instruction-conditioned recordings (`translator-contract-facts.md`) |
| Alarms (警報) | **No** | Not in any recording or schema |
| Tactic vocabulary, tactic start/end/switch labels | **No** | Undecided by the user; two incompatible candidate vocabularies exist in mtd-rl-lag (not reference values) |
| Reference 開始点 / 変化点 / 要求点 | **No** | Must be annotated; current scenario change points are trigger inputs, not references |
| Question text for 要求点 | **No** | `RequestPoint` has no text; the Translator prompt has no question |
| Who-targets-whom (誰を狙っているか) | **No ground truth** | `launcher_id` names the launcher, never the target; 13_LAG README states missile targets/hits are not recorded |
| Wall-clock latency of a call | Coarse only | `written_at` in requests/responses has 1-second resolution (`translator/cli.py:337`, `evaluation/judge.py:134`) |
| Human ratings, rater ids, rubric versions | **No** | No storage exists |
| "Before the AI acts" action onset | Partly | Learned-aircraft actions recorded per step (A team only); onset of a *tactic* needs annotation |

## 6. Leakage audit of the current generation inputs (read + empirical)

Method: read every code path that builds a model input (`translation.py:render_user_text`,
`detector.py:render_detector_state`, `lag/rendering.py`, `pipeline.py`), the 13_LAG derivation
(`workspace/13_LAG/lag_data/prepare.py`), and ran an offline check (scratch script, no API):
real `render_lag17_jsonl` + `build_prompt` at steps 0, 5, 46, 136, 200, 293 of all three
episodes; max `"step"` in the full prompt equals t every time, and none of the episode id,
`_win`, `_loss`, `seed100100`, `ic1006`, `mhi_v1.3` appears. Also checked `missiles_remaining`
around all 17 launches.

| # | Finding | Evidence | Severity |
|---|---|---|---|
| A1 | Episode alias is the only episode name in the prompt, but the guard is weak: only `alias != episode_id` and safe-id. An alias such as `loss_01` or `ours_seed100100_ic100601_loss_x` passes and reaches the model | `translator/config.py:548-549`, `translation.py:161`; 13_LAG README `:77` says the id carries the outcome | Medium (latent; committed example uses `episode_01`) |
| A2 | Re-translation *timing* can be oracle: `external_change_points` are generation inputs placed by an annotator (`source: placeholder-annotation`); if placed with hindsight, the event's timing and the content benefit from future knowledge. The point's content is not rendered | `config.py:586-596`, `pipeline.py:163-164` | High for timing validity; no text leak |
| A3 | Current-step action a_t is in the window: the line for step t renders `action` of step t (chosen from s_t, 13_LAG README `:55-58`). It is "at t", so allowed under an inclusive reading, but it matters for "explained before the AI acts" | `lag/rendering.py:136` | Semantics decision, not a leak |
| A4 | No leak: `missiles_remaining` is ACMI `Missiles` at the same step; at every launch step it still shows the pre-launch count and drops at t+1 (17/17 launches). `missile_launched` (result of a_t) and `shoot_action` are not rendered | `prepare.py:174`, `lag/rendering.py:35-37,137` | None |
| A5 | No leak: `last_step` (episode length) bounds the loop and clips windows, but online windows end at `step <= last_step`, so clipping never changes the text; the length is never rendered | `pipeline.py:125`, `lag/rendering.py:113` | None |
| A6 | No leak in prompts: target closure uses whole-episode down steps (`episode_facts`) but acts only when `step >= down`; `target_closed` / `trigger_suppressed` are never rendered. The preview schedule uses future down steps (preview only, no model input) | `pipeline.py:124-132, 220-240, 436` | None |
| A7 | No leak: instructions in effect filter `issue_step <= step`; previous translation is the target's own earlier one | `pipeline.py:171-177`, `translation.py:149-153` | None |
| A8 | No leak: missile `launcher_id` comes from the episode-wide launch table, but a missile row exists only from its first ACMI appearance (launch step + 1) | `prepare.py:151`, README §2.2 | None |
| A9 | Ids in records, not in payloads: `episode_id`, `scenario_id`, `event_id` appear in `events.jsonl` / `requests.jsonl` records and `run.json`; payloads use the alias only. The judge never renders ids | `pipeline.py:361-375, 500-512` | None (keep it that way; check mechanically) |
| A10 | Retrospective mode (D5/D33/D46) leaks by design when built (L2); today rejected at config load | `config.py:44, 268-322` | By design; must be flagged, never accepted silently |
| A11 | Existing tests pin window *metadata* (`end_step == step`) but never parse the rendered text; most pipeline tests use fake renderers. No test proves the absence of steps > t in the payload | `tests/translator/test_tr_pipeline.py:180-190` | Test gap |
| A12 | Judge (evaluation input) today shows nothing after the event (`timestamps` scope `through_event`); L3 video will show post-event frames by design | `evaluation/rendering.py:178-199` | Evaluation side, allowed but must be labelled |

## 7. Integration points for the new feature

1. Read a Translator run through `evaluation.schemas.load_translator_run` (strict) plus
   `requests.jsonl` (exact payload per `event_id`, verified against `request_sha256`).
2. Resolve the episode CSV from the event/run provenance and verify `csv_sha256` before rendering
   any evaluation window (future allowed there only).
3. Reuse `evaluation.judge` (`JudgeModel`, `OpenAIJudgeModel`, sink, gate, caps). `run_judge`
   is coupled to `parse_judge_output` and the key-point `result_record`, so the aspect judge
   needs either an injected parser/record builder or its own short loop over the same parts.
4. Reuse `read_strict_yaml`, `SAFE_ID_PATTERN`, `redacted_error`, `canonical_json`,
   `sha256_text`, `derived_path`, run-directory and provenance helpers.
5. The new feature must not change the Translator in phase 1 unless the user approves
   (D43 prompt; event contract consumed by the existing judge tests).

## 8. Test-suite facts

- `uv run poe test` covers `tests/` + `src/mentalmodeling/tests`; 879 pass (coordinator).
- Known unrelated failures documented in the root README (external candidate datasets etc.).
- `tests/evaluation/conftest.py` builds a real-episode, fake-layer Translator run (5 events) once
  per session; the new tests can extend `eval_fakes.write_translator_run` with a scenario that
  has one start, one instruction-caused change and one request.
