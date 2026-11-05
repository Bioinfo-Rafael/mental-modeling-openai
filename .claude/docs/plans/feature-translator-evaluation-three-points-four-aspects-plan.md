# Plan: Translator evaluation, 3 time points x 4 aspects (revision 2, simplified)

Date: 2026-11-04. Status: revision 2 of the approved plan; nothing implemented.
Requirement (authoritative, verbatim Japanese): `.claude/docs/research/eval-3points-4aspects-request.md`.
Decisions: `.claude/docs/research/translator-requirements.md`; this revision applies **D50-D60**:
- the Translator is aligned to the user's point definitions;
- alarms and questions become inputs;
- the user asked for the simplest implementation and editable defaults:
  「最大限シンプルな実装を心掛けてね」「基本は既定でいい．がちゃんとconfigで編集しやすくしておいて」.

Brief: `.claude/docs/research/feature-translator-evaluation-three-points-four-aspects-brief.md`.
Codebase scan and leakage evidence: `.claude/docs/research/feature-translator-evaluation-three-points-four-aspects-codebase.md`.
Codex consults (read-only, `--caller eval-designer`) are under `.claude/logs/codex/`: revision 1 in
`20261104T04*-eval-*.md`; revision 2 under "Validation Verdict".

## Revision 2 Delta (from the approved revision 1)

- **Translator in scope (D50/D51).**
  - Triggers are redefined: the System One detector decides 開始点 (`start`) and 変化点 (`change`)
    with one `choice` question.
  - A target's first instruction gives a non-detector `initial` bootstrap translation.
  - Later instructions only feed the detector.
  - `external_change_points` leave the scenario and become evaluation references.
  - Scenarios gain `alarms` and request `question` text. Both reach the detector state and the
    translation prompt at or before t.
  - New prompt version `translator-intent-v2`, additive only.
  - Output stays `{intent, evidence}` (D52).
  - Event contract v2 (below). `latency_s` added (D56). Alias outcome check (D58).
- **Matching** is no longer by exact instruction id. Start/change reference points match the first
  same-kind detector event inside their annotated segment (no tolerance window, D58). Requests
  still match by `request_id`. An event with several reasons is judged on the union of its point
  kinds' criteria.
- **Timing** is measured by machine and also rated 1-5 by humans and the judge (the 4-aspect flow).
  The request deadline is checked on `processing_s` (question to usable answer).
- **Removed (D59):**
  - window matching;
  - the calibration manifest;
  - the cohort framework (now one `retrospective` tag);
  - `UpToTSubject`/`FutureSubject` and the packet-horizon machinery (one packet and one judge call
    per event);
  - `PacketBlockEncoder`, the run-stage ledger, `init`, `export-human`, the human-only `*_validity`
    criteria;
  - the `forbidden_token_patterns` and `trigger_observability` rules (replaced by the Translator
    alias check and the removal of `external_change_points` from the scenario);
  - pytest-cov.
- **Scales (D53).** A pilot 1-5 scale plus `cannot_judge` (and `not_applicable` for conditional
  elements). Draft level wording per aspect is in the config. No overall aggregation (D54).
- **Defaults set (D55/D57/D58):**
  - deadline 3.0 s; question interval 10.0 s;
  - judge rationale in Japanese; judge cap 100;
  - a_t inclusive; segment ends at the next start/change point;
  - video by hash only;
  - humans rate contribution 1-5 after reading the explanation.
- **One config file** `evaluation/config/assess.yaml` holds every editable evaluation value, with
  the unset behaviour written inline. Translator values stay in `configs/translator/run.yaml`.
- **Size.**
  - Evaluation: 5 new modules (was 6), 13 new files in total (was about 22), about 1.3k
    production LOC (was 2.6-4.2k).
  - Translator: about +0.3k LOC across existing files.
- **Where (D60).** The worktree `../mtd-rl-xavier-assess` on `feature/llm-translator-judge`, after
  L1/L3 are committed there; P0 is a lead step.
- **Ownership** is split into a Translator track (T1) and an evaluation track (E1). They run in
  parallel from P0, because E1 uses its own v2 constants; E2 checks that they match.

## Complexity

- Classification: COMPLEX. The Translator trigger change and the evaluation span two subsystems,
  but each piece is small. Route: `/team-execute` with two parallel tracks (T1 and E1 from P0).
- **New files (13):**
  - `evaluation/assess_{data,checks,judge,report,cli}.py`;
  - `evaluation/config/assess.yaml`;
  - `evaluation/examples/{assess-points.example.yaml,human-ratings.example.jsonl}`;
  - `tests/evaluation/{assess_fakes,test_assess_checks,test_assess_judge,test_assess_data_report,test_assess_e2e}.py`.
- **Modified files (about 15):**
  - Translator: `src/mtd_rl_xavier/translator/{schema,config,detector,translation,pipeline,cli}.py`
    and `README.md`; `configs/translator/{run,scenario.example}.yaml`; `tests/translator/*` (about
    5 files);
  - contract consumers: `tests/evaluation/{eval_fakes,conftest}.py` and the D29-judge tests that pin
    event ids;
  - `evaluation/README.md` (one section).
- **LOC:**
  - evaluation: about 1.2-1.4k production Python plus about 0.2k YAML and examples, and about
    1.1k test;
  - Translator: about +0.3k / -0.1k production and about +0.4k test (net of rewritten tests).
  - Total about 1.6k production and 1.5k test.
- **No change to:** `src/mtd_rl_xavier/{llm,lag}/` (`SystemOneClient` already parses `choice`
  answers with probabilities, `llm/system_one.py:124-140,497-501`); the D29 judge modules (they
  import `TRIGGER_PRECEDENCE` and `EVENT_SCHEMA_VERSION` from `translator.schema`, so their
  validation follows the bump); `pyproject.toml`, `uv.lock`, `.gitignore`.

## Translator Event Contract v2 (the interface both tracks build against)

### Point semantics (D50)

| Point | Trigger kind / source | When |
|---|---|---|
| Bootstrap (not a detector decision) | `initial` / `first_instruction` | Forced at a target's first instruction step (step 0, or later per D35). It is needed because the detector compares against a previous intent. Its content is evaluated with the start rules; its **start** timing (machine start timing and `timing.start_in_time`) is `not_evaluated/forced_trigger`. A coinciding `request` keeps its own request timing |
| 開始点 (new tactic as planned) | `start` / `detector` | Detector label `start` |
| 変化点 (external factor) | `change` / `detector` | Detector label `change`: alarm, instruction or observed event |
| 要求点 | `request` / scenario `source` | Forced at the request step (a human asked); the question text goes into the prompt |

- **Removed kinds:**
  - `instruction`: a later instruction only feeds the detector (D50 supersedes D24);
  - `detector_change`;
  - `external_change`: D50 supersedes D6, so hindsight-placed change points stop being generation
    triggers. They move to the evaluation annotation as references, which may use hindsight. With
    no hindsight trigger left, D58's "hindsight change-point timing not_evaluated" never applies.
- **Precedence:** `initial > start > change > request` (`start` and `change` are exclusive by
  construction). Coinciding reasons are all kept, and the evaluator uses every one (see
  Architecture rule 1).
- **Ticks:** the fixed grid every 5 steps, unchanged (D28). A target is ticked once it has a
  translation.

### Detector question: one `choice`, three options

```yaml
question:
  id: point_kind
  type: choice
  instructions: >-
    Classify the target's behaviour in the trajectory window against its previous intent
    translation, its instructions in effect and the recent messages listed in the state.
  criteria:
    none:   The target continues the tactic described by its previous intent translation.
    start:  The target begins a new tactic that follows its existing plan or instruction, with no new external cause.
    change: The target's tactic changes because of an external factor (an alarm, a new instruction or an observed event).
```

- **Decision rule:** `label = answer.choice` (the argmax), and the event triggers when
  `label != none` and `p(label) >= threshold`. `threshold: 0.65` stays provisional (unchanged
  value).
- **Why one `choice` and not two `noul` questions:**
  - the three outcomes are mutually exclusive, so a single distribution cannot say "start" and
    "change" at once;
  - it is still one call per tick;
  - one threshold and one parsed answer;
  - `SystemOneClient` already supports it, and clef-flash scores a fixed question set.
- **Adapter change:** `SystemOneDetectionModel` requires one `choice` question with exactly these
  option names, in place of today's one `noul` question.

### Scenario v2 (`schema_version: 2`)

```yaml
schema_version: 2
scenario_id: placeholder-episode-01
episode: {episode_id: mhi_v1.3_ic100601_win, alias: episode_01, csv_path: ..., video_path: ...}
instructions:            # unchanged
  - {instruction_id: placeholder-aircraft-001, issue_step: 0, target: {kind: aircraft, id: A0100}, text: "[PLACEHOLDER] ..."}
alarms:                  # NEW: external warnings, shown to every target sharing an aircraft with `targets`
  - {alarm_id: placeholder-alarm-001, step: 12, targets: [{kind: aircraft, id: A0100}], text: "[PLACEHOLDER] ..."}
requests:                # NEW field `question` (required, non-blank)
  - {request_id: placeholder-request-001, step: 10, targets: [{kind: aircraft, id: A0100}], source: placeholder-human, question: "[PLACEHOLDER] ..."}
# external_change_points: REMOVED in v2 (hindsight; references now live in the evaluation annotation)
```

Rules, in addition to today's:
- ids are unique across instructions, alarms and requests;
- at most one request per target per step;
- an alarm, like a request, needs every target to be instructed at or before its step;
- texts must stay within the configured character limits;
- **alias outcome check (D58):** the alias may share no token (split on `_ . -`) with `episode_id`,
  and may contain none of `win`, `loss`, `lose`, `won`, `lost` (case-insensitive).

### Model inputs (everything at or before t)

- **Detector state:** today's text plus two sections.
  - `Instructions in effect` for the target: its own and, for an aircraft, its team's, each with
    its issue step.
  - `Recent messages`: **all** alarms and questions for this target issued in
    `[t - messages_lookback_steps + 1, t]`, most recent last. Nothing is truncated.
  - The config requires `messages_lookback_steps >= tick_interval_steps`, so every message reaches
    at least one detector call while the target is up. The default 25 covers 5 ticks.
- **Translation prompt `translator-intent-v2`.** These changes are additive to v1:
  - an `Alarms` section: **every** alarm for this target issued at or before t (scenarios hold
    few), so the first translation after an alarm always shows it;
  - for request events, a `Question from a human (step s)` section;
  - one system-prompt sentence: "When the user message contains a question, answer it at the
    granularity it asks for, using only the recorded data up to the current step; if the data
    cannot answer it, say so instead of guessing."

### Records (`events.jsonl` `schema_version: 2`)

- **`translation_event`:**
  - `input.prompt_version: translator-intent-v2`;
  - `input.alarms: [{alarm_id, step, targets, text}]`;
  - `input.question: {request_id, step, text} | null`;
  - trigger kinds as above;
  - `processing_s`: monotonic seconds from the start of processing this target at this step (before
    any detector call) until the parsed event record is complete, immediately before its own
    fsynced write. A field cannot include the write of the record that contains it; that write is
    one append, the same for every event. This is the request deadline's measure: question received
    to usable answer.
- **`detector_call`:**
  - `input.instructions_in_effect`, `input.messages: [{kind: alarm|question, id, step, text}]`;
  - `input.question` is the choice question;
  - `output: {label, probabilities: {none, start, change}, threshold, triggered}`.
- **`responses.jsonl`:** each record gains `latency_s` (`time.perf_counter()` around the model
  call, rounded to 3 decimals) (D56). It is present on every attempted call (`ok`, `error`,
  `uncertain`) and is finite and non-negative.
- **`run.json`:** `config.pipeline.visibility.{mode: online, future_steps: 0}` must be present in
  every run. It already is today (`translator/cli.py:406`); a T1 test pins it.
- **Unchanged:** `target_closed`, `trigger_suppressed`, ids, `request_sha256`, write-before-send.
- **L2 (D46, not built):** a retrospective run declares its non-online mode and `future_steps` in
  the same `run.json` key. The trajectory `end_step` of each record shows the visible end.

### Budget preflight

- The detector preflight renders the exact state of every scheduled tick, including the actual
  instruction, alarm and question texts. The scenario is known before the run, so this is exact.
  It then adds the intent reserve and must stay below `input_token_budget`.
- The limits in config bound each text at load.
- Default limits:
  - `max_instruction_chars: 300`, `max_alarm_chars: 200`, `max_question_chars: 200`;
  - `messages_lookback_steps: 25` (no message cap: the exact preflight fails instead of
    truncating).
- `max_intent_chars` drops from 1000 to **600**.
  - Today's worst bound is 14310 of 16384, of which 4000 is the reserve for a 1000-character
    intent.
  - 600 frees 1600 bytes for messages; Japanese text is 3 bytes per character.
  - The over-long-intent policy (D41) is unchanged.
- If a real scenario does not fit, the run fails in preflight before any call, as today. The
  README lists the knobs.

## Architecture (evaluation)

```text
Translator run v2 (events.jsonl, requests.jsonl, responses.jsonl, run.json)
  + optional reference annotation (start/change segments; private/, may use hindsight)
  + optional human ratings JSONL (private/)
  + hash-verified episode CSV (evaluation-side future evidence only)
  -> assess_data:   strict loading of config / run / annotation / ratings; result records; JSONL writers
  -> assess_checks: (1) leakage check per model call  (2) point matching + segments  (3) machine metrics
  -> assess_judge:  one packet (text) + one judge call per translation event; strict 1-5 parse
  -> assess_report: per kind x aspect x evaluator; timing summary; human-judge agreement
  -> outputs/evaluation/assess/<run>/  machine_results.jsonl | judge_results.jsonl | human_results.jsonl | report-*.json
```

1. **Subject = translation event.**
   - Point kinds come from **all** of its trigger reasons: `initial` and `start` count as `start`,
     `change` as `change`, `request` as `request`.
   - The event is judged once, on the union of those kinds' criteria. A criterion shared by two
     kinds is asked once.
   - Each result lists `subject.point_kinds`, and the report counts it under each kind.
   - Reference points (optional annotation) are used only for timing: missed, duplicate, excess,
     delay and before-action.
   - The segment ends at the next annotated start/change point; that end bounds the future
     evidence.
2. **Generation input vs evaluation input.**
   - Generation input = the exact `requests.jsonl` payload of each call (both layers), verified
     against `request_sha256`.
   - Evaluation input = the judge packet. It is rebuilt from the event's own recorded inputs (all
     at or before t) and, for start/change events only, a labelled `FUTURE` block rendered from the
     CSV.
   - The packet never feeds the Translator, and the Translator never reads the annotation.
3. **Future block bound.** It covers steps `(t, end)` with
   `end = min(next annotated start/change point of the target, next instruction or alarm for the
   target, target down step, last step + 1, t + 1 + future_max_steps)`.
   - The packet also carries a `TIMELINE` block: this target's translation events (step, kinds,
     `processing_s`). Entries after t sit inside the FUTURE section.
   - The judge and raters are told that only criteria with `reference: future_behaviour` and the
     timing criteria may use FUTURE. `observable_records` criteria must not.
   - So behaviour after a later external factor can never count against a start explanation
     (request: 開始点の正しさ, 3rd bullet).
   - Request events never get a future block; their correctness is about the records up to t.
4. **One result schema for machine, human and judge, in three files.** It covers the 11 traced
   fields. `status` is `evaluated` or `not_evaluated` with a `reason`; `cannot_judge` and
   `not_applicable` are reasons, so they are never 0. Re-ratings append with `supersedes`, and
   every rating is kept (D58).
5. **Timing has two layers.**
   - The machine produces measurements and verdicts against config thresholds.
   - Humans and the judge also rate one timing criterion per point kind, on the same 1-5 scale.
     This follows the request's flow 「人が4観点ごとに判定する」 and 「同じ4観点をJudgeで評価」.
   - An `initial` event's start timing is `not_evaluated/forced_trigger` from every evaluator. Other
     timing criteria of a coinciding reason (for example `timing.request_in_time`) are evaluated
     normally.
6. **Retrospective (D46).** A run whose `run.json` declares a non-online visibility is tagged
   `retrospective: true` on every record, and the report shows it in a separate section. An
   undeclared future step is a leak.
7. **No overall aggregation (D54).** Each event gets one `overall` row per evaluator with
   `not_evaluated/aggregation_unset`.
8. **The judge** uses an adapter (about 40 lines) over `llm/openai_responses`.
   - Model `gpt-6.1-sol` (D8); rationale in Japanese (D57); cap 100 per run.
   - Preview by default; `--execute --confirm-paid-api`; write-before-send; uncertain calls are
     never resent.
   - It does not reuse `evaluation/judge.py`, whose request type is tied to D29 key points.
9. **Video (D58).** Only the path and SHA-256 are recorded for human raters; no frames go to the
   aspect judge. The L3 video judge stays a separate D29 component.

## Module Structure

| Path | Responsibility (user's split) | ~LOC |
|---|---|---:|
| `evaluation/assess_data.py` | Strict loading of `assess.yaml`, the Translator run (reuses `evaluation.schemas.load_translator_run`, joins `requests.jsonl`/`responses.jsonl`), the annotation and the ratings. Result-record builder; append-only JSONL writer; human-rating import and validation. [human storage; per-axis results and rationale storage] | 300 |
| `evaluation/assess_checks.py` | `check_generation_inputs()` [input time-range validation]; `match_points()` and `segment_end()` [points and segments]; `machine_results()`: structural elements, delay, missed/duplicate/excess, before-action lead, request latency, question interval, utterance count [machine checks] | 320 |
| `evaluation/assess_judge.py` | `build_packet()` (text blocks, future block via `lag.rendering.render_detector_compact` on strided steps), output schema, strict parser, OpenAI adapter, `run_judge_calls()` (gate, write-before-send, cap). [judge execution] | 300 |
| `evaluation/assess_report.py` | Counts and means per point kind x aspect x criterion x evaluator, per `reference`; timing summary; leakage summary; exact agreement and quadratic weighted kappa (pure Python); retrospective section [aggregation and report] | 160 |
| `evaluation/assess_cli.py` | Composition root: `check`, `judge`, `import-ratings`, `report`; provenance; exit codes 0-4 | 200 |
| `evaluation/config/assess.yaml` | Every editable evaluation value, commented, unset behaviour inline (below) | 150 |
| `evaluation/examples/assess-points.example.yaml` | Placeholder annotation | 25 |
| `evaluation/examples/human-ratings.example.jsonl` | Placeholder ratings | 5 |

Responsibilities are separated at function level inside these modules, with one clearly
commented section per responsibility. The thresholds and rubric live only in config, so they can
be changed without touching code.

## Data Formats

Real annotations and ratings go under the gitignored `private/evaluation/`; outputs under the
gitignored `outputs/evaluation/assess/<run-name>/` (0700). Strict JSON/YAML rules apply as in the
existing code: exact keys, no duplicate keys, finite numbers, safe ids.

### Generation input (verified, not copied)

The generation input is the `requests.jsonl` payload itself. `check` writes one line per model
call to `leakage.jsonl`:

```json
{"call_id": "placeholder-episode-01:aircraft:A0100:000010", "layer": "translation", "target": {"kind": "aircraft", "id": "A0100"},
 "step": 10, "visible_end_step": 10, "request_sha256": "...", "retrospective": false,
 "status": "ok", "problems": []}
```

`status` is `ok`, `leak` or `error`; `problems` is a list of `{check, detail}`.

### Reference annotation (optional; start/change references, may use hindsight)

```yaml
schema_version: 1
scenario_id: placeholder-episode-01
annotator: annotator-01            # pseudonymous id
points:                            # list every start/change point of each target you annotate
  - {point_id: p-start-001,  kind: start,  target: {kind: aircraft, id: A0100}, step: 0,  action_onset_step: 3}
  - {point_id: p-change-001, kind: change, target: {kind: aircraft, id: A0100}, step: 14, action_onset_step: null}
```

- A point's segment is `[step, next start/change point of the same target)`, ending at the target's
  episode end if there is none.
- **Matching:** the first detector `start`/`change` event of the same kind in the segment. A point
  at a target's `initial` step is skipped for timing (`not_evaluated/forced_trigger`).
- Former scenario `external_change_points` belong here, as `kind: change` points.
- `action_onset_step` is optional; when it is null, before-action is
  `not_evaluated/no_action_onset`.

### Result record (machine, judge and human share it; three files)

```json
{"schema_version": 1, "result_id": "judge:<run>:placeholder-episode-01:aircraft:A0100:000010:correctness.unknown_not_asserted",
 "evaluator": {"kind": "judge", "model": "gpt-6.1-sol", "response_model": "gpt-6.1-sol", "prompt_version": "assess-judge-v1",
               "prompt_sha256": "...", "reasoning_effort": "medium", "request_sha256": "...", "response_id": "resp_..."},
 "subject": {"scenario_id": "placeholder-episode-01", "episode_id": "mhi_v1.3_ic100601_win", "target": {"kind": "aircraft", "id": "A0100"},
             "target_aircraft": ["A0100"], "step": 10, "time_s": 2.0, "event_id": "placeholder-episode-01:aircraft:A0100:000010",
             "point_kinds": ["request"], "reference_point_id": null},
 "input_range": {"start_step": 0, "end_step": 10, "request_sha256": "...", "future_end_step": null},
 "translator_output": {"intent": "[PLACEHOLDER]", "evidence": ["[PLACEHOLDER]"]},
 "aspect": "correctness", "criterion_id": "correctness.unknown_not_asserted", "reference": "observable_records",
 "status": "evaluated", "score": 4, "measurement": null, "rationale": "（日本語の根拠）", "reason": null,
 "rubric_version": "pilot-1", "config_sha256": "...", "retrospective": false, "supersedes": null}
```

- **Traceability.** The request's 11 fields map as follows:
  - scenario, aircraft and time: `subject`;
  - point kind: `subject.point_kinds`;
  - Translator input range: `input_range`;
  - original output: `translator_output`;
  - per-axis result: `aspect`, `criterion_id`, `score`/`measurement`;
  - rationale: `rationale`;
  - evaluator kind: `evaluator.kind`;
  - rubric and config: `rubric_version`, `config_sha256`;
  - judge model, prompt and conditions: `evaluator`;
  - overall: the event's `overall` row;
  - not-evaluable reason: `status`/`reason`.
- **Machine records:** `evaluator: {kind: machine, version}`, a numeric `measurement` (for example
  `{"delay_steps": 4}` or `{"latency_s": 2.41, "deadline_s": 3.0}`) and `score: null`. Pass/fail
  lives in `measurement.passed` when a threshold is set.
- **Human records:** `evaluator: {kind: human, rater_id, import_sha256}`.
- **Overall row:** `aspect: overall`, `criterion_id: overall`, `status: not_evaluated`,
  `reason: aggregation_unset` (D54).

### Human rating input (one line per rating)

```json
{"rating_id": "r-0001", "rater_id": "rater-01", "event_id": "placeholder-episode-01:aircraft:A0100:000010",
 "criterion_id": "contribution.question_resolved", "judgement": "4", "rationale": "...", "rubric_version": "pilot-1",
 "rated_at": "2026-11-05T09:00:00+00:00", "supersedes": null}
```

- `judgement` is one of `"1"`-`"5"`, `cannot_judge` or `not_applicable` (conditional elements
  only).
- `import-ratings` validates the whole file before appending anything. It checks:
  - the event and criterion exist;
  - the rubric version matches;
  - duplicates: the same (rater, event, criterion, rubric version) without `supersedes` is an
    error.
- Raters read `packets.md`, the same text the judge receives (D57: after reading the explanation).

### Judge packet and output

- `packets.jsonl` and the readable `packets.md` hold, per event: a header (target, aircraft, step,
  point kinds), instructions in effect, alarms, the question, the previous translation, the exact
  trajectory window the Translator saw, the Translator output, the `TIMELINE` (events up to t,
  with `processing_s`) and, for start/change events, the labelled `FUTURE (evaluation only)`
  block (future telemetry plus later timeline entries). Each packet has a SHA-256.
- The judge's resolved config (model, effort, tokens, prompt text and version, rubric version) is
  stored in the assessment run's `run.json`, not only as a hash.
- Free texts are quoted as JSON strings: the injection envelope from `judge-text-v2`.
- Output (strict json_schema): `{"criteria": [{"criterion_id": <enum>, "rationale": str,
  "judgement": "1".."5" | "cannot_judge" | "not_applicable"}]}`, with exactly the event's
  criteria.
- The parser accepts `not_applicable` only for conditional elements. Nothing is repaired or
  retried.

### Report (`report-<UTC stamp>.json`, a new file each time)

The report has `online` and `retrospective` sections. Each contains:
- `leakage`: calls ok/leak/error;
- `timing`: matched/missed/duplicates/excess, delays, latency pass/fail, interval violations;
- `cells[point_kind][aspect][criterion][evaluator]`: n evaluated, mean, distribution, and
  not_evaluated counts by reason, with correctness grouped by `reference`;
- `agreement[criterion]`: human-judge and human-human pairs, exact agreement, quadratic weighted
  kappa. Below 2 pairs or with no variance the value is `null` with a reason, never 0;
- `overall: not_evaluated/aggregation_unset`.

## Config (`evaluation/config/assess.yaml`, every editable value in one place)

```yaml
# Evaluation of Translator output: 3 time points x 4 aspects. Edit values here; nothing is hard-coded.
# null = undecided: the affected check reports the raw measurement and records not_evaluated/<key>_unset.
schema_version: 1
rubric_version: pilot-1          # bump whenever you edit criteria or scale wording below

timing:
  request_deadline_s: 3.0        # D55. 要求点: processing_s <= this passes (see contract v2). null => reported, no verdict
  min_question_interval_s: 10.0  # D55. Requests of one target closer than this => protocol warning. null => no check
  start_min_lead_steps: null     # 開始点: explanation >= N steps before annotated action_onset_step. null => lead reported only
  change_max_delay_steps: null   # 変化点 "in time": delay <= N steps. null => delay reported only
  max_events_per_segment: null   # "too many": start/change events of one kind in one annotated segment. null => counts only
  max_utterances_per_target: null  # per scenario. null => counts only

future_evidence:                 # evaluation input only (start/change events); never shown to the Translator
  fields: [alive, state_valid, lon_deg, lat_deg, alt_m, yaw_deg, tas_mps, vn_mps, ve_mps, vu_mps,
           altitude_action_id, heading_action_id, speed_action_id]
  stride_steps: 5
  max_steps: 120                 # the window is cut here and the cut is recorded (future_end_step)
  float_precision: 5

leakage:
  allow_action_at_t: true        # D58: the action recorded at step t counts as "<= t"

judge:
  model: gpt-6.1-sol             # D8 (the only accepted value)
  reasoning_effort: medium
  max_output_tokens: 25000
  timeout_s: 1800
  max_calls: 100                 # D58: per run; the run stops before call 101
  rationale_language: Japanese   # D57
  pricing: {input: 2.00, cached_input: 0.10, output: 10.00}   # USD per 1M tokens, preview estimate only

open_items:                      # D58: null until you give them; informational, nothing depends on a guess
  tactic_vocabulary: null        # null => tactic names are not checked
  annotated_units: null          # null => whatever targets the annotation lists
  expected_raters: null          # null => agreement computed over whatever ratings exist

scale:                           # D53 pilot, for you to review. Applies to judge and human criteria.
  values: ["1", "2", "3", "4", "5"]
  cannot_judge: 材料から判断できない（not_evaluated として記録し、0点とは扱わない）
  not_applicable: 場面上不要、またはデータがない（条件付き要素のみ）
  levels:
    elements:      {"5": 明確かつ具体的に述べている, "4": 述べているが一部あいまい, "3": 部分的・間接的にしか読み取れない, "2": ほとんど読み取れない, "1": 述べていない}
    correctness:   {"5": 参照できる記録と完全に整合し根拠のない断定がない, "4": おおむね整合し軽微な不正確さのみ, "3": 一部が整合しない、または重要な点が不確か, "2": 大部分が記録と整合しない, "1": 記録と矛盾する、または別の目的を述べている}
    timing:        {"5": 十分な余裕をもって説明された, "4": 対応できる時点で説明された, "3": ぎりぎり間に合った、または軽微な出しすぎ, "2": 遅れた、または出しすぎで読みにくい, "1": 間に合わなかった、見落とした、または過剰に繰り返した}
    contribution:  {"5": 判断が大きく容易になる, "4": かなり容易になる, "3": いくらか役立つ, "2": ほとんど役立たない, "1": 役立たない、または誤解を招く}

criteria:   # judge and humans rate every criterion. reference (correctness only): observable_records | future_behaviour.
            # FUTURE may be used only by future_behaviour criteria and timing.* criteria.
  elements:                      # every point kind; certainty expressions are deliberately not a criterion
    - {id: elements.what_now, text: 今何をしているかが述べられているか}
    - {id: elements.why, text: なぜそうするか（近い将来に実現したい状況を含めてよい）が述べられているか}
    - {id: elements.target, conditional: true, text: 誰を狙っているか}
    - {id: elements.wingman_role, conditional: true, text: 僚機との役割分担}
    - {id: elements.instruction_relation, conditional: true, text: 指示との関係}
    - {id: elements.priority, conditional: true, text: 重視点}
  start:
    - {id: correctness.start_plan, reference: observable_records, text: これから何を、何のためにするかが説明されているか}
    - {id: correctness.start_future, reference: future_behaviour, text: 説明した計画と FUTURE ブロック内の実際の行動が一致するか（FUTURE は次の外的要因・変化点の前で切れている）}
    - {id: timing.start_in_time, text: AIが行動する前、または人が介入できる余裕を残して説明できたか（TIMELINE と FUTURE を参照）}
    - {id: contribution.intent_gap, text: 自分の意図とAIの意図のずれに気づきやすくなるか}
    - {id: contribution.next_plan, text: 次の計画を理解しやすくなるか}
    - {id: contribution.delegation, text: 任せる・確認する・指示するの判断がしやすくなるか}
  change:
    - {id: correctness.change_cause, reference: observable_records, text: 何が起きたかが警報・指示・観測の記録と一致するか}
    - {id: correctness.change_delta, reference: observable_records, text: 以前の計画から何を変更したかが説明されているか}
    - {id: correctness.change_future, reference: future_behaviour, text: 新しい計画が FUTURE ブロックの記録と矛盾しないか}
    - {id: timing.change_in_time, text: 人が対応できる時点までに説明でき、短い揺れや重複を報告しすぎていないか（TIMELINE と FUTURE を参照）}
    - {id: contribution.change_reason, text: 変化の理由を理解しやすくなるか}
    - {id: contribution.next_plan, text: 次の計画を理解しやすくなるか}
    - {id: contribution.instruction_reflected, text: 指示が反映されたか確認しやすくなるか}
    - {id: contribution.delegation, text: 任せる・確認する・指示するの判断がしやすくなるか}
  request:
    - {id: correctness.question_granularity, reference: observable_records, text: 質問に対応する粒度で答えているか}
    - {id: correctness.records_to_t, reference: observable_records, text: 回答が時刻 t 以前の記録と一致するか}
    - {id: correctness.unknown_not_asserted, reference: observable_records, text: 記録から判断できない内容を断定せず「分からない」と扱えているか}
    - {id: correctness.consistent_with_latest, reference: observable_records, text: 最新の説明と不必要に矛盾していないか}
    - {id: timing.request_in_time, text: 問いを受けてから十分速く応答できたか（processing_s を参照）}
    - {id: contribution.question_resolved, text: 疑問が解消されるか}
    - {id: contribution.delegation, text: 任せる・確認する・指示するの判断がしやすくなるか}

output:
  root: outputs/evaluation/assess
```

- **Translator-side values** live in `configs/translator/run.yaml`:
  - the detector `question` (choice) and `threshold`;
  - `messages_lookback_steps` (must be >= `tick_interval_steps`);
  - `max_instruction_chars`, `max_alarm_chars`, `max_question_chars`;
  - `max_intent_chars: 600`.
- **Loader rules:**
  - the config fails on unknown keys;
  - a criterion's aspect is its id prefix (`elements.`, `correctness.`, `timing.`, `contribution.`);
  - `correctness.*` criteria must carry a `reference` from the two listed values. The RL model's
    unobservable internal reason is not one of them, so it can never be a correctness reference.
  - Certainty expressions have no criterion, so they are never required.
- **Timing** is both machine-measured (the `timing:` thresholds) and rated by humans and the judge
  (the `timing.*` criteria). They are stored as different evaluator kinds and never merged.

## Leakage Audit Findings (revision 2 status)

| # | Finding (file:line in the current tree) | Revision 2 handling |
|---|---|---|
| A1 | The alias guard is only `alias != episode_id` (`translator/config.py:548-549`); the alias reaches the prompt (`translation.py:161`) | Fixed in the Translator: the D58 alias outcome check (token overlap and outcome words); test |
| A2 | `external_change_points` are hindsight-placed forced triggers (`config.py:586-596`, `pipeline.py:163-164`) | Removed from the scenario (D50 supersedes D6). They become annotation references on the evaluation side, where hindsight is allowed |
| A3 | The action at step t is in the window (`lag/rendering.py:136`) | Inclusive (D58, `leakage.allow_action_at_t: true`); before-action is measured against annotated onset |
| A4-A9 | No leak: `missiles_remaining` is causal (17/17 launches), `last_step` is never rendered, closure is causal, instructions are filtered to issue step <= t, `launcher_id` is causal, ids appear only in records (citations in the codebase scan §6) | New Translator tests assert the same for alarms and questions: none after t reaches either layer |
| A10 | Retrospective leaks by design (`config.py:44,268,321`) | A declared run is tagged `retrospective` and reported separately; an undeclared one is a leak |
| A11 | No test parses payload text (`tests/translator/test_tr_pipeline.py:180-190`) | New Translator test with real renderers parses every payload of both layers; the evaluator re-checks every run |
| A12 | L3 video renders 20 steps after the event (`evaluation/rendering.py:304`, `judge-text-video.yaml:47-48`) | The aspect judge uses no video frames (D58: hashes only); the L3 judge remains a separate, evaluation-side component |
| A13 (new) | Alarms and questions are new generation inputs | Detector and translation inputs filter `step <= t`; nothing is truncated (delivery guaranteed by `lookback >= tick`); budget preflight exact; checked again by `check` |

**`check` (per model call, both layers):**
1. The call joins exactly one event record, and the canonical payload SHA-256 equals the record's
   `request_sha256`.
2. Every trajectory line in the payload parses, with `step <= t` and `relative_step <= 0`.
3. Every instruction, alarm and question shown has a step <= t; the previous translation's step is
   < t.
4. The run's own `episode_id` and `scenario_id` do not appear in the payload.
5. `run.json` declares online with `future_steps: 0`, or a retrospective mode, which sets the tag
   and moves the bound in check 2 to `t + future_steps`.

A `leak` or `error` on any call of a target turns every result for that target's events into
`not_evaluated/generation_input_invalid`.

## Machine vs Human vs Judge Boundary

| Aspect | Machine | Human | Judge |
|---|---|---|---|
| 要素 | `intent` non-blank, `evidence` non-empty | Each element, 1-5 / `not_applicable` | Same criteria and scale as humans |
| 正しさ | None (it only decides which evidence the packet may contain) | All criteria of the event's kind | Same; `future_behaviour` criteria use only the FUTURE block |
| タイミング | Measurements and threshold verdicts: delay, missed, duplicate, excess, before-action lead, request `processing_s` vs 3.0 s, question interval vs 10.0 s, utterance count. An `initial` event's start timing is `not_evaluated/forced_trigger`; its coinciding request timing is evaluated | One `timing.*` criterion per point kind, 1-5 | Same as humans |
| 判断への寄与 | Never | 1-5 after reading the packet (D57); no ratings => `not_evaluated/no_human_rating` | Rated too, shown in its own column next to the human scores with agreement. Never merged and never a substitute until you decide a calibration rule |

**Calibration, simplified (request flow 1-5):**
1. Humans rate.
2. `import-ratings` stores the ratings and their rationales.
3. You adjust the criterion or scale wording in `assess.yaml` and bump `rubric_version`.
4. Re-run `judge`.
5. The report shows human-judge agreement per criterion and rubric version.

## Implementation Phases

| Phase | Owner | Files | Depends on | Acceptance commands |
|---|---|---|---|---|
| P0 Worktree and baseline (D60) | lead | `git worktree add ../mtd-rl-xavier-assess feature/llm-translator-judge` after L1/L3 are committed there; `uv sync --all-extras` in it; record the base SHA | L1/L3 committed | `git -C ../mtd-rl-xavier-assess rev-parse HEAD` recorded; `uv run pytest tests/translator tests/evaluation -q` green |
| T1 Translator contract v2 (parallel with E1) | `translator-core` | `src/mtd_rl_xavier/translator/{schema,config,detector,translation,pipeline,cli}.py`, `translator/README.md`, `configs/translator/{run,scenario.example}.yaml`, `tests/translator/**`; contract-consumer fixups in `tests/evaluation/{eval_fakes,conftest}.py` and D29-judge tests that pin event ids. One atomic, green migration | P0 | `uv run pytest tests/translator tests/evaluation -q`; `uv run ruff check src/mtd_rl_xavier/translator tests/translator tests/evaluation`; `uv run python -m mtd_rl_xavier.translator.cli --scenario configs/translator/scenario.example.yaml` (preview, 0 API calls) |
| E1 Evaluation core (parallel with T1) | `assess-core` | `evaluation/assess_{data,checks,judge,report,cli}.py`, `evaluation/config/assess.yaml`, `evaluation/examples/assess-points.example.yaml`, `evaluation/examples/human-ratings.example.jsonl`, `tests/evaluation/{assess_fakes,test_assess_checks,test_assess_judge,test_assess_data_report}.py`. Built against the v2 contract in this plan: `assess_data.py` holds its own v2 constants (schema version 2, kinds `initial/start/change/request`), and the tests use a small hand-written v2 run in `assess_fakes.py`. Nothing imported from the changing Translator code | P0 | `uv run pytest tests/evaluation/test_assess_checks.py tests/evaluation/test_assess_judge.py tests/evaluation/test_assess_data_report.py -q`; `uv run ruff check evaluation/assess_*.py tests/evaluation/assess_fakes.py tests/evaluation/test_assess_*.py`; `uv run python -m evaluation.assess_cli --help` |
| E2 Integration | `assess-core` | `tests/evaluation/test_assess_e2e.py` (real pipeline v2, fake layers, example scenario, plus a test that `assess_data`'s v2 constants equal `translator.schema`'s); `evaluation/README.md` section | T1 + E1 | `uv run pytest tests/evaluation -q` |
| F Final gate (no edits) | lead | none | E2 | `uv run poe test`; `uv run poe lint`; `.venv/bin/python .claude/skills/_shared/verify_delegation.py --base <P0 SHA> --forbid-outside src/mtd_rl_xavier/translator --forbid-outside configs/translator --forbid-outside tests/translator --forbid-outside tests/evaluation --forbid-outside evaluation`; diff review for weakened tests and placeholders |

Ownership:
- `translator-core` owns `src/mtd_rl_xavier/translator/**`, `configs/translator/**`,
  `tests/translator/**`, plus only the contract-consumer fixups named above.
- `assess-core` owns `evaluation/assess*`, `evaluation/config/assess.yaml`, the two examples,
  `tests/evaluation/assess*`, `tests/evaluation/test_assess*` and the README section.
- Nobody edits `llm/`, `lag/`, the D29 judge modules, `pyproject.toml` or `uv.lock`.

Translator steps, test first; the phase lands as one green change:
1. **T1-1** Contract constants and dataclasses: kinds `initial/start/change/request`, `Alarm`,
   `RequestPoint.question`, `EVENT_SCHEMA_VERSION = 2`.
2. **T1-2** Scenario v2 loader: alarms, question, limits, alias check; `external_change_points`
   removed.
3. **T1-3** Choice detector adapter, state messages and decision rule.
4. **T1-4** Prompt `translator-intent-v2`.
5. **T1-5** Pipeline:
   - the `initial` bootstrap;
   - no instruction or external-change trigger;
   - detector start/change;
   - alarms and question in the inputs;
   - budget preflight with the exact texts;
   - `latency_s` and `processing_s`.
6. **T1-6** CLI preview estimate and README contract.
7. **T1-7** Fix the contract consumers' fixtures.

Evaluation steps, test first:
1. **E1-1** `assess_data` loaders and result writer.
2. **E1-2** `check_generation_inputs`.
3. **E1-3** `match_points`, `segment_end`, `machine_results`.
4. **E1-4** Packet, schema, parser and judge loop.
5. **E1-5** Report and agreement.
6. **E1-6** CLI.

## File Changes

| File | Change | Phase |
|---|---|---|
| `src/mtd_rl_xavier/translator/schema.py` | modify: trigger kinds, `Alarm`, `RequestPoint.question`, precedence, `EVENT_SCHEMA_VERSION = 2` | T1 |
| `src/mtd_rl_xavier/translator/config.py` | modify: scenario v2 (alarms, question, limits, alias check; `external_change_points` removed); run config choice question, lookback, text limits | T1 |
| `src/mtd_rl_xavier/translator/detector.py` | modify: choice adapter, state sections, decision rule | T1 |
| `src/mtd_rl_xavier/translator/translation.py` | modify: prompt v2 (alarms, question, one sentence) | T1 |
| `src/mtd_rl_xavier/translator/pipeline.py` | modify: triggers, inputs, preflight, `latency_s`, `processing_s` | T1 |
| `src/mtd_rl_xavier/translator/cli.py` | modify: preview estimate with messages | T1 |
| `src/mtd_rl_xavier/translator/README.md` | modify: contract v2 | T1 |
| `configs/translator/run.yaml`, `configs/translator/scenario.example.yaml` | modify | T1 |
| `tests/translator/*` | modify + new tests | T1 |
| `tests/evaluation/{eval_fakes,conftest}.py`, D29-judge tests pinning event ids | modify (contract fixups only) | T1 |
| `evaluation/assess_{data,checks,judge,report,cli}.py` | new | E1 |
| `evaluation/config/assess.yaml` | new | E1 |
| `evaluation/examples/assess-points.example.yaml`, `evaluation/examples/human-ratings.example.jsonl` | new | E1 |
| `tests/evaluation/{assess_fakes,test_assess_checks,test_assess_judge,test_assess_data_report}.py` | new | E1 |
| `tests/evaluation/test_assess_e2e.py` | new | E2 |
| `evaluation/README.md` | modify: one section | E2 |

## Test Plan (one row per user minimum test)

| # | User minimum test | Test |
|---|---|---|
| 1 | 時刻 t より後の情報がTranslator入力に混入しない | `tests/translator/test_tr_pipeline.py::test_no_payload_contains_information_after_its_step` uses real renderers, real 13_LAG and the v2 example scenario with alarms/questions. It parses every detector and translation payload and asserts every trajectory step, instruction, alarm and question step is <= t. `test_assess_e2e.py` asserts that `check` reports all calls `ok` |
| 2 | 未来情報の混入を検出できる | `test_assess_checks.py::test_check_detects_future_information`, parametrised over a t+1 trajectory line, a future alarm/question/instruction in the payload, the own `episode_id`, an undeclared future range, and a payload hash mismatch. Each gives `leak` or `error`, and the target's results become `not_evaluated/generation_input_invalid`. A declared retrospective run gives `ok` + `retrospective: true` |
| 3 | 必須要素の欠落を検出できる | `test_assess_checks.py::test_structural_elements_detect_blank_intent_or_empty_evidence` (machine) and `test_assess_judge.py::test_missing_why_is_recorded` (a fake judge returns `"1"` for `elements.why`, stored with rationale) |
| 4 | 開始点・変化点・要求点で異なる評価規則が使われる | `test_assess_judge.py::test_each_point_kind_gets_its_own_criteria` (including the union for an event with `change` + `request` reasons) and `test_assess_checks.py::test_timing_rules_differ_by_point_kind` |
| 5 | 開始点の後に外的要因で計画が変わった場合、開始時の説明を不当に減点しない | `test_assess_judge.py::test_start_future_block_ends_before_next_external_factor`: an alarm or instruction after the start, or the next annotated change point, ends the FUTURE block. No post-change step appears in the packet |
| 6 | 記録にない内容を断定した回答を検出またはJudgeへ送れる | `test_assess_judge.py::test_request_packet_sends_unknown_not_asserted_without_future_block` |
| 7 | 応答時間と発言頻度の設定値を変更できる | `test_assess_checks.py::test_deadline_interval_and_utterance_limits_follow_config`: the same run under two configs (deadline 3.0 vs 2.0 on `processing_s`; interval 10.0 vs 5.0; utterance cap) gives different verdicts |
| 8 | 未確定の閾値が未設定でも、黙って恣意的な値を使わない | `test_assess_checks.py::test_null_thresholds_report_raw_values_without_verdict`, parametrised over every nullable timing key: `not_evaluated/<key>_unset` with the measurement kept |
| 9 | 人評価とJudge評価が上書きされず別々に残る | `test_assess_data_report.py::test_human_and_judge_results_are_separate_and_append_only` (separate files; a re-rating appends with `supersedes`; earlier bytes are unchanged) |
| 10 | 評価不能が0点と区別される | `test_assess_data_report.py::test_cannot_judge_is_not_a_score`: `cannot_judge` is stored as `not_evaluated` with a null score, excluded from means and counted by reason; the agreement statistic with <2 pairs is null, not 0 |
| 11 | 既存テストがすべて通る | Phase F: `uv run poe test` |
| fixture | 開始点・変化点・要求点を1件ずつ通した実行例 | `test_assess_e2e.py::test_example_scenario_start_change_request_offline`. Example scenario v2 on the real episode, fake layers: `initial` at 0, a fake detector `start` at tick 5, a request with a question at 10, an alarm at 12, and a fake detector `change` at tick 15. It runs `check` -> `judge` preview (0 calls) and a fake execute -> `import-ratings` (example) -> `report` |

Translator tests (T1):
- the choice adapter and decision rule (threshold boundaries);
- the `initial` bootstrap at the first instruction, with no detector before it;
- a later instruction forces no translation;
- the detector `start`/`change` triggers;
- a v1 field `external_change_points` is rejected by the v2 loader;
- precedence;
- alarms/questions only <= t; every message reaches at least one detector state;
  `messages_lookback_steps < tick_interval_steps` is a config error;
- `processing_s` and `latency_s` are present, finite and non-negative;
- `run.json` declares `pipeline.visibility`;
- scenario v2 validation (limits, unique ids, one request per target per step, alias check);
- the preflight fails when a text pushes the state over budget;
- `latency_s` recorded on ok, error and uncertain;
- the D29 judge still loads a v2 run.

## Dependencies To Approve

None. No runtime or dev dependency is added (pytest-cov is not added, per D58).

## Risks

- **No explanation until the first detector start/change.** Apart from the `initial` bootstrap and
  requests, a target is explained again only when the detector fires. A quiet detector means few
  explanations; the evaluation reports it as missed points and low utterance counts.
- **Detector competence.** clef-flash must now classify start, change and none from compact
  telemetry, English framing and Japanese texts (D42). Out-of-domain accuracy is unknown, and the
  0.65 threshold is provisional. Every probability is logged; the evaluation's missed/excess
  timing shows the effect.
- **No translation after an instruction the detector misses.** This follows D50 (the detector
  decides), and the evaluation counts it as a missed 変化点.
- **Budget.** Messages compete with the trajectory for the 16384 budget, so `max_intent_chars`
  drops to 600 for headroom. A real scenario with long texts can fail preflight (before any call);
  the knobs are in `run.yaml`.
- **Contract break.**
  - `EVENT_SCHEMA_VERSION 2`: v1 runs (only previews exist today) are no longer accepted by either
    judge.
  - T1 updates the D29-judge fixtures in the same phase, so the suite stays green.
- **Sparse first reports.** No real annotations or ratings exist yet. Timing missed/excess and
  every human cell stay `not_evaluated` until they do.
- **Uncalibrated judge.** Judge contribution scores are uncalibrated and are only shown beside
  human scores, never in their place.
- **One packet per event.** The FUTURE block shares a packet with the up-to-t criteria of
  start/change events.
  - The judge and raters are told that `observable_records` criteria must not use FUTURE.
  - A pure request event never gets a FUTURE block.
  - An event with `request` and `change` reasons does get one, and the instruction above is its
    only guard.
- **Cost.** The judge call per event adds a compact FUTURE block (stride 5, at most 120 steps) to
  the translation window. The preview estimates tokens and cost, and the cap is 100 per run.
- **Commit order (D47/D60).** The worktree needs L1/L3 committed first; the coordinator orders
  this.

## Validation Verdict

Revision 1 PASSED after three rounds (`20261104T043356Z-eval-validate.md` ->
`20261104T045431Z-eval-revalidate.md`). Revision 2: see the update below.

**Revision 2: PASS** after two revision rounds. All consults were read-only, under
`.claude/logs/codex/`.

| Round | File | Result |
|---|---|---|
| Validation | `20261104T051538Z-eval-revalidate.md` | NEEDS_REVISION, 7 blocking issues |
| Re-validation 1 | `20261104T052444Z-eval-revalidate.md` | NEEDS_REVISION: 6 resolved, 2 consistency issues |
| Re-validation 2 | `20261104T052825Z-eval-revalidate.md` | **PASS**, no blocking issue |

The seven blocking issues of the first round, and how they were applied:

1. **Forced start and `external_change` triggers contradicted D50.**
   - The first translation is now a non-detector `initial` bootstrap.
   - `external_change_points` moved to the annotation.
2. **Messages could be silently capped (D51).**
   - No cap: `lookback >= tick` guarantees delivery, and the exact preflight fails instead of
     truncating.
3. **Humans and the judge did not rate timing.** Added one `timing.*` criterion per point kind.
4. **Coinciding triggers lost criteria.** Point kinds now come from all reasons, and the event is
   judged on their criterion union.
5. **T1a was not a stable parallel boundary.**
   - T1 is now one atomic, green migration.
   - E1 uses its own v2 constants, and E2 checks they match.
6. **Visibility was required only of L2 runs.** It is now required in every run's `run.json`; it is
   already written there today.
7. **The request deadline measured only the model call.** It now uses `processing_s`; `latency_s`
   stays per call.

The second round fixed the scope of the `initial` timing exclusion and the `processing_s` endpoint
wording.

Designer calls Codex rated sound:
- later instructions only feed the detector;
- one `choice` question with a threshold;
- `max_intent_chars` 600;
- one packet per event;
- segment matching, which is consistent with "no window matching".

Not run: no implementation exists, so no test, lint or build has been executed against this design.

## Decisions For The User

D50-D60 settle everything the implementation needs; **nothing blocks implementation**. The
designer decided the items below within D50/D58, checked by Codex in revision 2:

1. **First translation of a target:** a forced `initial` bootstrap. It is not a detector decision.
   Its content is evaluated with the start rules; its start timing is
   `not_evaluated/forced_trigger`. A coinciding request is timed normally.
2. **A later instruction** only feeds the detector (D50 supersedes D24).
3. **`external_change_points`** are removed from the scenario (D50 supersedes D6) and become
   `kind: change` references in the evaluation annotation.
4. **Detector question:** one `choice` (`none`/`start`/`change`), threshold 0.65 (provisional).
5. **Messages:** every alarm and question within the 25-step lookback, no cap. The exact preflight
   fails instead of truncating. Text limits:
   - instruction 300 characters;
   - alarm 200;
   - question 200;
   - `max_intent_chars` 1000 -> 600.
6. **Judge:** one packet and one call per event, judged on the union of the event's point kinds;
   video not sent (D58).
7. **Matching:** the first same-kind detector event inside an annotated segment.
8. **Timing:** humans and the judge rate one timing criterion per point kind; the request deadline
   applies to `processing_s`.

Pilot items for later review, which do not block:
- the scale level wording and criterion texts in `assess.yaml` (D53);
- the detector threshold after a labelled check (D49 run-time setup).

## Out Of Scope

- L2 retrospective generation (evaluator-side tagging only).
- L1/L3 themselves.
- Paid runs, real annotation and human studies.
- Window matching, calibration manifests and judge substitution.
- Structured Translator output (D52).
- Overall aggregation (D54).
- Video frames for the aspect judge.
- Changes to `llm/`, `lag/`, the D29 judge modules, weightreading and intent_compare.

## Completion Report Mapping (完了条件)

| Requested item | Source |
|---|---|
| 調査した既存設計 | Codebase scan + this plan (contract v2, architecture) |
| 採用した実装方針 | Revision 2 Delta + Architecture |
| 変更したファイル | File Changes, verified by `verify_delegation.py` in phase F |
| 評価データの形式 | Data Formats |
| 実行方法 | `evaluation/README.md` section (`python -m evaluation.assess_cli check | judge | import-ratings | report`) |
| テスト内容と結果 | Test Plan + phase F output |
| 機械判定と人・Judge判定の境界 | Machine vs Human vs Judge Boundary |
| 設定可能にした項目 | `assess.yaml` + the `run.yaml` keys listed under Config |
| 残っている未決事項 | `open_items` in `assess.yaml` (vocabulary, units, rater count) + pilot wording |
| 今回の実装範囲外としたもの | Out Of Scope |
