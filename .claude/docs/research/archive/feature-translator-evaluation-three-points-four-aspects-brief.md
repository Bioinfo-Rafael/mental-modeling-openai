## Feature Brief: Translator evaluation, 3 time points x 4 aspects

Mode: /feature MODE=existing. Requirement (authoritative, verbatim Japanese):
`.claude/docs/research/eval-3points-4aspects-request.md`. Decisions D1-D49:
`.claude/docs/research/translator-requirements.md`. Codebase scan:
`.claude/docs/research/feature-translator-evaluation-three-points-four-aspects-codebase.md`.
Plan: `.claude/docs/plans/feature-translator-evaluation-three-points-four-aspects-plan.md`.

### Current State
- Architecture: the offline two-layer Translator (`src/mtd_rl_xavier/translator/`) writes one
  run directory per scenario: `events.jsonl` (`translation_event`, `detector_call`,
  `target_closed`, `trigger_suppressed`), `requests.jsonl` (exact payload, written before
  sending, joined to events by `event_id` and `request_sha256`), `responses.jsonl` and `run.json`
  (provenance and resolved config, including `pipeline.visibility`). The judge
  (`evaluation/`) reads that run and scores each event against human key points: 0/1/2 per key
  point plus an overall 1-5 (D29), with partial judging (D36). Its run loop is coupled to that
  key-point schema.
- Relevant files: `src/mtd_rl_xavier/translator/{schema,pipeline,translation,config,cli}.py`,
  `src/mtd_rl_xavier/lag/{dataset,rendering}.py`, `src/mtd_rl_xavier/llm/openai_responses.py`,
  `evaluation/{configuration,schemas,rendering,judge,cli}.py`, `evaluation/config/judge-text.yaml`,
  `tests/{translator,evaluation}/`. In flight now (D48): `llm/vllm_chat.py`,
  `llm/vllm_server.py`, `scripts/translator_vllm.sbatch`, judge video frames
  (`evaluation/config/judge-text-video.yaml`, `evaluation/{rendering,cli}.py`).
- Patterns: strict YAML with exact key sets (`read_strict_yaml`), frozen dataclasses and pure
  functions, Protocols on the consumer side, clients built only in CLI composition roots after
  the paid gate, preview by default with `--execute --confirm-paid-api`, write-before-send
  JSONL journals, owner-only run directories under the gitignored `outputs/evaluation/`, real
  customer text only under the gitignored `private/`, exit codes 0/1/2/3/4, provenance hashes,
  offline tests that use fakes and the real 13_LAG example episode.
- Data gaps: 13_LAG has no instructions, alarms, tactic labels, question texts or missile-target
  ground truth. Call timestamps have 1-second resolution. The Translator output is free text
  `{intent, evidence}` in Japanese (D42, prompt fixed by D43).

### Feature Goal
Evaluate every Translator explanation at three kinds of time point (開始点 start, 変化点
change, 要求点 request) on four aspects (要素 elements, 正しさ correctness, タイミング timing,
判断への寄与 contribution to human decisions). Generation inputs may use only information up
to t, evaluation inputs may use the future, and the two are kept apart and checked mechanically.
Machine, human and judge results are stored separately with full traceability, and every
undecided threshold, scale or aggregation stays configurable with an explicit `not_evaluated`
behaviour when unset.

### Scope
(Revision 2, D50-D60; simplest implementation, editable defaults.)
- Include:
  - **Translator event contract v2** (D50/D51/D56/D58):
    - the System One detector decides 開始点 and 変化点 with one `choice` question (none/start/change);
    - a non-detector `initial` bootstrap translation at each target's first instruction; later
      instructions only feed the detector;
    - scenario `alarms` and request `question` text, shown to both layers at or before t;
    - prompt `translator-intent-v2` (additive);
    - `latency_s` in `responses.jsonl`;
    - an alias outcome check;
    - a budget preflight that includes the message texts (no truncation);
    - `processing_s` on each translation event (request deadline measure);
    - `external_change_points` moved from the scenario into the evaluation annotation.
  - **Evaluation** (5 modules, one YAML config):
    - a leakage check per model call (generation input = exact payload);
    - optional start/change reference annotation, matched inside its segment;
    - machine timing and structural checks, plus human and judge timing ratings;
    - one judge packet and call per event (pilot 1-5 + `cannot_judge`; FUTURE block for
      start/change only, cut before the next external factor or change point);
    - human-rating import (append-only);
    - a report with human-judge agreement and a separate retrospective section;
  - a placeholder end-to-end fixture with one start, one change and one request;
  - README sections.
- Exclude:
  - window matching, calibration manifests and judge substitution;
  - overall aggregation (D54);
  - structured output (D52);
  - L2 generation; L1/L3 themselves;
  - paid runs, real annotation and human studies;
  - video frames for the aspect judge (hashes only);
  - weightreading and intent_compare work.

### Complexity Classification (from Codex)
- Classification: COMPLEX (two subsystems: Translator contract v2 and the evaluation), with each
  piece small. Route: `/team-execute` with two parallel tracks (T1 and E1) from P0.
- Estimated files:
  - 13 new: `evaluation/assess_{data,checks,judge,report,cli}.py`, `evaluation/config/assess.yaml`,
    2 examples, 5 test files;
  - about 15 modified: 6 Translator modules + README, 2 Translator configs, about 5 Translator
    test files, the D29-judge fixtures, the evaluation README section.
- Estimated LOC:
  - evaluation: about 1.2-1.4k production + 0.2k YAML/examples, about 1.1k test;
  - Translator: about +0.3k / -0.1k production, about +0.4k test;
  - total about 1.6k production and 1.5k test (revision 1: about 22 new files, 2.6-4.2k).
- Implementation route: `/team-execute`. P0 (lead) worktree -> (T1 Translator || E1 evaluation,
  in parallel; E1 holds its own v2 constants) -> E2 integration -> F gate. Codex scope consult
  `.claude/logs/codex/20261104T040205Z-eval-scope.md`; revision 2 validation in the plan.

### Integration Points
- Translator run directory: read through `evaluation.schemas.load_translator_run` (strict), plus
  `requests.jsonl` for the exact generation payload. Each payload is verified against the event's
  `request_sha256`, and `run.json` supplies the declared visibility mode.
- Trigger reasons (contract v2): `start`/`change` come from the detector, `initial` from a
  target's first instruction, `request` from the scenario. An event with several reasons is judged
  on the union of their point kinds' criteria. Requests match by
  `request_id`. Annotated start/change points match the first same-kind event inside their segment
  (no tolerance window).
- LAG data: `lag.dataset.load_episode` (hash-verified against the event's `csv_sha256`) and
  `lag.rendering.render_detector_compact` build the FUTURE block (evaluation input only).
- Judge: a local adapter over `mtd_rl_xavier.llm.openai_responses` and `make_openai_client`, with
  the paired gate, write-before-send, cap 100 and no resend of uncertain calls. The D29 key-point
  judge and the L3 video judge stay separate; they follow the contract bump through the
  `translator.schema` constants.
- Video: path and SHA-256 only (D58).
- L2 (future): `run.json` `pipeline.visibility.{mode, future_steps}` declares a retrospective run.

### Risks
- Detector competence: clef-flash must classify none/start/change; accuracy on LAG is unknown and
  the threshold is provisional. Mitigation: every probability is logged; the evaluation's
  missed/excess timing measures it.
- An instruction the detector misses produces no translation (D50). Mitigation: it is counted as a
  missed 変化点.
- Detector budget: messages compete with the trajectory. Mitigation: exact preflight before any
  call; `max_intent_chars` 600; text limits in config.
- Contract break (event schema 2). Mitigation: only preview runs exist; T1b updates the D29-judge
  fixtures in the same phase.
- No real annotations or ratings yet, so timing and human cells stay `not_evaluated` with explicit
  reasons.
- Uncalibrated judge: contribution scores are shown beside human scores, never instead of them.
- Commit order (D60): the worktree needs L1/L3 committed first.

### Success Criteria
- Every minimum test in the request passes, mapped one-to-one in the plan's Test Plan:
  - no info after t in the Translator input; future info is detected; a missing required element
    is detected;
  - start, change and request use different rules; a start explanation is not penalised after an
    external change;
  - an unsupported assertion is detected or sent to the judge;
  - latency and frequency settings are configurable; an unset threshold is never silently
    replaced;
  - human and judge results stay separate; not_evaluated differs from 0;
  - all existing tests pass.
- A placeholder fixture with one start, one change and one request point runs offline end to end:
  leakage check, machine checks, human import, judge preview (0 API calls), report.
- `uv run poe test` and `uv run poe lint` pass. The judge preview reports `api_requests_made: 0`.
- A declared retrospective (D46) fixture run is tagged `retrospective` and appears only in the
  report's retrospective section; an undeclared future step is a leak.
