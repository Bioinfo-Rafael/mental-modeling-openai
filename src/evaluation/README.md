# `evaluation/`: LLM-as-a-judge for Translator events

The 3 time points x 4 aspects assessment (`python -m evaluation.assess_cli`) of a translation run
(`src/translate/`, any configuration). The D29 key-point judge, its video block, the HTML viewer and
the clip tool were deleted on 2026-11-09 (U-d); key points return as judge criteria when they exist.

`Dn` numbers refer to the requirement decision log
[`.claude/docs/research/translator-requirements.md`](../.claude/docs/research/translator-requirements.md).

## Prompt files

Every word the judge sends comes from a prompt file under `configs/evaluation/prompts/`, named by
the judge config's `prompt` key (repo-relative). The code loads the material, renders it into the
file's templates, calls and parses; it holds no prompt text. A prompt file has `schema_version` 1,
`judge` (`assess`), `version` (recorded as `prompt_version`), `texts` (the system prompt and the user
message) and `blocks` (item templates, joiners, the text when a list or an optional piece is empty).
It loads strictly: UTF-8, no repeated key, exact keys, and every `{field}` is checked when the config
loads (`prompt_files.py` lists the fields of each template): an unknown field, or a template that
drops judge material (the trajectory, the intent, the criteria, ...), fails before anything is
rendered. Literal braces are doubled.

| Config (`configs/evaluation/`) | Prompt | Use |
|---|---|---|
| `assess.yaml` | `assess-judge-v3` | Default for new runs. Calls the trajectory the human-visible telemetry up to t (not what the Translator saw); shows trigger kinds as the config's point kinds; an empty INSTRUCTIONS or QUESTION block reads `- none` |
| `assess-legacy.yaml` | `assess-judge-v2` | The former in-code prompt. The paid 2026-11-05 assessment was judged with this config |

Never edit a prompt file that judged a paid run: write a new version instead, or the journal's
`request_sha256` no longer matches and the event is refused.

## 3 time points x 4 aspects assessment (`assess_cli`)

It rates every translation event of
a translation run on the user's 3 points x 4 aspects (要素 `elements`, 正しさ `correctness`,
タイミング `timing`, 判断への寄与 `contribution`). The points (2026-11-09): 開始点 `start` is the
episode start (step 0), with or without an instruction in effect, and its timing is fixed by
definition; 変化点 `change` is where the detector judges that the previous intent no longer explains
the behaviour because of an external factor or a human instruction (a tactic that follows the plan
is no point); 要求点 `request` is a human question, or a target's first instruction arriving after
the start (then the packet's QUESTION is none and the two question-only criteria are conditional).
The event is judged on the union of the criteria of all its trigger reasons. Which point kind rates
a trigger kind is the judge config's `triggers` map, per event schema: the default config maps
schema 3 one to one and refuses schema 2; schema 2 runs (before 2026-11-08) are rated by
`assess-legacy.yaml`, which keeps `initial` and the detector's `start` as `start`. A run of a schema
the config does not map fails before anything is written. Machine, human and judge results are stored apart and
never merged; nothing is aggregated into an overall score (D54).

| Module | Responsibility |
|---|---|
| `assess_data.py` | Event contract constants (the current kinds checked against `translate.events` by a test), strict loading of the judge config with its prompt file and trigger map, the run, the annotation; the point kinds and criteria of an event |
| `assess_store.py` | The one result-record schema, append-only stores with `supersedes`, human-rating import |
| `assess_checks.py` | Time-range (leakage) check of every model call, reference points and segments, machine metrics |
| `assess_judge.py` | Packet (with `TIMELINE` and the `FUTURE` block), strict schema and parser, OpenAI adapter, call loop |
| `assess_report.py` | Cells per point kind x aspect x criterion x evaluator, timing summary, agreement |
| `assess_cli.py` | The composition root: `check`, `judge`, `import-ratings`, `report` |

### Command

```bash
RUN=outputs/translate/<run>             # a completed translation run (events, requests, responses)
uv run python -m evaluation.assess_cli check --translator-run $RUN --points private/evaluation/<points>.yaml
uv run python -m evaluation.assess_cli judge --translator-run $RUN --points private/evaluation/<points>.yaml
# paid: one gpt-6.1-sol call per event, at most judge.max_calls per invocation; needs BOTH flags
uv run python -m evaluation.assess_cli judge --translator-run $RUN --points ... --execute --confirm-paid-api
uv run python -m evaluation.assess_cli import-ratings --translator-run $RUN --ratings private/evaluation/<ratings>.jsonl
uv run python -m evaluation.assess_cli report --translator-run $RUN
# On Slurm (check + judge preview; `paid` adds the paid judge and report):
sbatch slurm/judge_assess.sbatch preview --translator-run $RUN
```

Every translation run (`outputs/translate/<run>/`) is a judge input, whatever its inputs were: each
event records the human-visible material (scenario texts and the trajectory window), so every
configuration is judged the same way. For a translator that did not read the trajectory
(`input.translator_inputs`, e.g. the policy input) the leak check needs no window in the payload
and checks the recorded window against its hash. Whenever the policy input was read, the event must
record the bundle the sandbox read (`input.policy`: session directory, manifest SHA-256, window);
`assess_policy.py` verifies it against `sessions/<dir>/inputs/manifest.json` and requires the window
to end at or before t (before t when `leakage.allow_action_at_t` is false); a missing or unverifiable
record is an error. A skipped invalid answer (blank output plus `answer_problem`) is loaded and
judged as given by both judges. The judges' prompts live in prompt files (see "Prompt files"). The one paid assessment judged
`outputs/translator/intentcmp-seed100601-ja-A-s045-A0100`.

- Common flags: `--config` (default `configs/evaluation/assess.yaml`), `--output-root`, and `--name`
  (the assessment directory; default: the Translator run directory's name).
- `--points` is optional: an annotation of start/change reference points, which may use hindsight
  (format: `src/evaluation/examples/assess-points.example.yaml`). Without it, matching, delay, missed and
  excess points are not measured, and a FUTURE block ends only at the next instruction or alarm, the
  target's down step or `max_steps`.
- `judge` without the flags writes the packets and a cost preview, builds no client and reads no
  key. A request whose SHA-256 is already in `judge_requests.jsonl` is never sent again, so an
  uncertain call is never resent and a capped run resumes where it stopped. An event already sent
  from the directory with other request bytes (another prompt file or config) is refused instead of
  billed twice: resume with the config it was judged with, or judge again under another `--name`.
  Events of a target whose generation input is invalid are never sent.
- Ratings format: `src/evaluation/examples/human-ratings.example.jsonl`. `judgement` is `"1"`..`"5"`,
  `cannot_judge`, or `not_applicable` (conditional criteria only: four elements, plus
  `correctness.change_delta` and `correctness.consistent_with_latest` when there is no PREVIOUS). Raters score against the
  anchors and the rating rule printed in every packet of `packets-<batch>.md`, the same text the
  judge gets (D65, D67). A re-rating names the replaced
  `rating_id` in `supersedes`. The whole file is validated before anything is appended.
- Exit codes: 0 done or preview; 1 deterministic failure, including a `leak` or `error` found by
  `check` (its files are still written); 2 usage or gate misuse; 3 uncertain judge call; 4 judge cap.

### Files

`<output.root>/<name>/` (mode 0700; gitignored), one directory per Translator run:

| File | Writer | Content |
|---|---|---|
| `run.json` | first command | Binds the directory to the run's file hashes; a different run is refused |
| `invocations.jsonl` | every command | Start and end of each invocation: inputs and hashes, git state, code hashes, the full resolved judge config, counts |
| `leakage.jsonl` | `check` | One line per model call: `status` `ok` / `leak` / `error`, `problems` |
| `machine_results.jsonl` | `check` | Machine records |
| `packets-<batch>.{jsonl,md}` | `judge` | The exact text the judge receives; human raters read the `.md` |
| `judge_requests.jsonl`, `judge_responses.jsonl` | `judge --execute` | Journal: request written before each call, raw response after |
| `judge_results.jsonl` | `judge --execute` | Judge records |
| `human_results.jsonl` | `import-ratings` | Human records |
| `report-<batch>.json` | `report` | `online` and `retrospective` sections: leakage, timing, cells, agreement |

All stores are append-only. A re-run of `check` or `judge` appends a new batch whose records name the
record they replace in `supersedes`; the report reads the active records, and every record stays.
Every record has the same keys: subject (scenario, target, step, time, event, point kinds),
Translator input range, Translator output, aspect, criterion, `status` (`evaluated` /
`not_evaluated`), `score`, `judgement`, `measurement`, `rationale`, `reason`, evaluator (kind, and
for the judge its model, prompt version and hashes), rubric version and config hash.

### Machine vs human vs judge

| Aspect | Machine (`check`) | Human (`import-ratings`) | Judge (`judge`) |
|---|---|---|---|
| 要素 | `intent` non-blank, `evidence` non-empty | Each element 1-5; conditional ones may be `not_applicable` | Same criteria and scale as humans |
| 正しさ | None | Every criterion of the event's kinds | Same; `future_behaviour` criteria may use the FUTURE block, `observable_records` criteria may not |
| タイミング | Delay, missed, duplicates, excess, lead before the action onset, request `processing_s` vs deadline, question interval, utterance count | One `timing.*` criterion per kind, 1-5 | Same as humans |
| 判断への寄与 | Never | 1-5 after reading the packet; none given => `not_evaluated/no_human_rating` | Rated, shown beside the human scores with agreement; never a substitute |

- `cannot_judge`, `not_applicable`, unset thresholds and missing ratings are `not_evaluated` with a
  reason. They never count as 0, and never enter a mean.
- Agreement below two pairs, or with no variance, is `null` with a reason.
- The start timing of the forced first translation (the config's `triggers[].forced`: `start` in
  schema 3, `initial` in schema 2) is `not_evaluated/forced_trigger` for every evaluator.
- `check` verifies every model call of both layers:
  - the payload matches its event record's `request_sha256`;
  - trajectory lines, instructions, alarms and questions are at or before t, and the previous
    translation is older than t;
  - the run's own episode and scenario ids are absent;
  - `run.json` declares the visibility.
- A `leak` or `error` on any call of a target makes every result for that target's events
  `not_evaluated/generation_input_invalid`.
- A run that declares a non-online visibility is tagged `retrospective` and reported in its own
  section (D46).

### `assess.yaml`

Every editable evaluation value lives in the judge config (`configs/evaluation/assess.yaml`, or
`assess-legacy.yaml` to resume the paid 2026-11-05 run), commented. The unset
behaviour is written next to each key. Unknown, missing or duplicate keys fail. Translator-side
values (detector question and threshold, text limits, `max_intent_chars`) stay in
the run config (`configs/translate/*.yaml`).

Score anchors (`pilot-4` default and `pilot-2` legacy; D65-D67): every criterion has its
own anchor for each level, rendered under the criterion in the packet from 5 down to 1, followed
by its own `cannot_judge` / `not_applicable` boundary when it has one. Each packet ends with the
rating rule, which the judge's system prompt repeats word for word. The score is the lowest level
whose condition holds. The rationale names that level and says which part of the next level's
condition does not hold. A criterion shared by several point kinds is written once with a YAML
anchor (`&name`) and repeated as an alias (`*name`). The default rubric has no start timing
criterion (the episode start's timing is fixed). The change timing anchors (and the legacy start ones) use only
orderings and counts visible in the packet; their undecided thresholds stay in the machine checks.
`timing.request_in_time` quotes the decided 3.0 s and 10.0 s in its anchor text, so editing
`timing.request_deadline_s` or `timing.min_question_interval_s` means editing that anchor too. The
system prompt treats only the last CRITERIA block and the SCALE block that close the packet as
rubric; anything before them is material.

| Key | Default | When `null` |
|---|---|---|
| `rubric_version` | `pilot-4` (legacy `pilot-2`) | Required. Bump it after editing criteria, anchors, scale wording or `triggers`; ratings must match it |
| `prompt` | `prompts/assess-judge-v3.yaml` (legacy `assess-judge-v2`) | Required: the prompt file (repo-relative) |
| `triggers` | schema 3 (legacy: 2 only) | Required: per event schema, `point_kinds` (every trigger kind of that schema -> `start`, `change` or `request`) and `forced` (kinds of the forced first translation; they must map to `start`) |
| `timing.request_deadline_s` | 3.0 (D55) | `processing_s` reported, `not_evaluated/request_deadline_s_unset` |
| `timing.min_question_interval_s` | 10.0 (D55) | Intervals between questions reported, no verdict (a request opened by an instruction is no question) |
| `timing.start_min_lead_steps` | null | Lead before the annotated action onset reported only |
| `timing.change_max_delay_steps` | null | Delay of a change point reported only |
| `timing.max_events_per_segment` | null | Same-kind events per segment counted only |
| `timing.max_utterances_per_target` | null | Utterances counted only |
| `future_evidence.*` | 13 fields, stride 5, max 120 steps, precision 5 | Required |
| `leakage.allow_action_at_t` | true (D58) | Required; `false` makes an action shown at t a leak |
| `judge.*` | `gpt-6.1-sol` (fixed), effort medium, 25000 tokens, 1800 s, 100 calls, Japanese rationale, pricing | Required |
| `open_items.*` | null | Must stay null until decided (tactic vocabulary, annotated units, rater count, overall aggregation, judge-substitution agreement) |
| `scale`, `criteria` | Pilot 1-5 scale (D53); criteria per kind, each with anchors for levels 1-5 (D65) | Required; every criterion needs `levels` with exactly one non-blank anchor per `scale.values`; `correctness.*` needs `reference`; `conditional: true` criteria need `not_applicable`; `cannot_judge` is optional |

Tests: `uv run pytest tests/evaluation/test_assess_checks.py tests/evaluation/test_assess_judge.py
tests/evaluation/test_assess_data_report.py tests/evaluation/test_assess_e2e.py -q`, offline. The
E2E produces an event-schema-3 run with the real Translator CLI over fake layers on the real example episode,
then runs `check -> judge preview -> judge --execute -> import-ratings -> report` with no socket
opened.
