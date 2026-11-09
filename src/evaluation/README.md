# `evaluation`: the LLM judge of translation runs

One engine: per translation event, the same human-visible material plus the criteria of the event's
point kinds go into one `gpt-6.1-sol` call (D8, D20), which returns one verdict per criterion. The
judge treats every translation the same way, whatever the translator read (decision 4).

```text
run (events.jsonl) -> audit -> material per event -> one request -> verdict lines (results.jsonl)
human ratings (import-ratings) + results -> report.json (cells, agreement, D55 deadline)
```

| Module | Responsibility |
|---|---|
| `config.py` | `judge.yaml` as pydantic models (defaults, unknown keys fail); placeholder checks |
| `material.py` | Run loader (event schema 3), the event's user text: material, TIMELINE, FUTURE, criteria |
| `judge.py` | Request and verdict schema, strict parse, the no-resend guards, the call loop |
| `audit.py` | Causality audit of every journaled request body of the translation run |
| `results.py` | Result lines, the human-ratings import, the report |
| `cli.py` | `judge`, `import-ratings`, `report`; the paid gate; the only place a client is built |

## Commands

```bash
.venv/bin/python -m evaluation judge --run outputs/translate/<run>                       # preview
.venv/bin/python -m evaluation judge --run outputs/translate/<run> --execute --confirm-paid-api
.venv/bin/python -m evaluation judge --run outputs/translate/<run> --execute --confirm-paid-api --only-event <event_id>
.venv/bin/python -m evaluation import-ratings --run outputs/translate/<run> --ratings private/evaluation/<file>.jsonl
.venv/bin/python -m evaluation report --run outputs/translate/<run>
```

On Slurm: `sbatch slurm/evaluate.sbatch preview|paid --run outputs/translate/<run>` (paid also writes
the report). Every subcommand takes `--config` (default `configs/evaluation/judge.yaml`) and
`--name` (default: the config's file stem). Exit codes: 0 ok or preview, 1 failure, 2 usage or gate,
3 a call has an uncertain outcome (it may have been billed; it is never resent), 4 the call cap.

## Config (`configs/evaluation/judge.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `model` | `{name: gpt-6.1-sol, reasoning_effort: medium, max_output_tokens: 25000, timeout_s: 1800}` | The judge model (D8: only `gpt-6.1-sol`) |
| `material` | `{past_steps: 60, future_steps: 120, future_stride: 5, fields: [...]}` | The trajectory window up to t; the FUTURE block |
| `rubric_version` | required | Recorded in every result line and part of every request (the verdict schema's name `verdict_<rubric_version>`) |
| `rationale_language` | `Japanese` | D57 |
| `scale` | `values` 1-5; `cannot_judge`, `not_applicable` texts required | D53 |
| `timing.request_deadline_s` | `3.0` | D55: a request event passes when `processing_s` is at most this |
| `limits.max_calls` | `100` | Per invocation; the run stops before call `max_calls + 1` (exit 4) |
| `output` | `outputs/evaluation` | Repo-relative root of the judgement directories |
| `prompt` | `system`, `user`, `rating_rule`, `future` required | Every word of the request (see below) |
| `criteria` | required | `{id, kinds, text, levels, conditional, not_applicable, cannot_judge, may_use_future}` |

A criterion's aspect is its id prefix (`elements`, `correctness`, `timing`, `contribution`; any prefix
works). `levels` must be exactly the scale values. `conditional: true` allows `not_applicable` and
needs its text. An event is rated on every criterion of any of its kinds, each once; the verdict
schema is generated from exactly those criteria. A new criterion is a new list entry, no code.

Placeholders are `str.format` bare names, checked at load (literal braces are doubled):

- `system`: `{rating_rule}`, `{rationale_language}`; `rating_rule`: `{rationale_language}`;
- `user` (must use the material): `{instructions}`, `{alarms}`, `{question}`, `{previous_translation}`,
  `{trajectory}`, `{intent}`, `{evidence}`, `{timeline}`, `{future}`, `{criteria}`, `{scale}`; may use
  `{target}`, `{target_aircraft}`, `{step}`, `{time_s}`, `{kinds}`, `{cannot_judge}`, `{not_applicable}`,
  `{rating_rule}`;
- `future` (start and change events only; `{future}` is empty otherwise): `{rows}` and
  `{last_step}`, `{end_step}`, `{cut}`, `{stride_steps}`, `{fields}`, `{later}`.

None carries the trigger source, a detector probability, the translation model, the translator's
inputs or a request hash. Free texts (instructions, alarms, questions, intent, evidence) arrive as
quoted JSON strings, so they cannot leave their envelope.

## Material

- Instructions in effect, alarms and the question come from the event record (issued at or before t).
- The trajectory is rendered from the episode CSV (its SHA-256 must match the run's): lag17 JSON lines
  `[max(0, t - past_steps + 1), t]`, the same for every translation.
- TIMELINE: this target's translations up to t with their kinds and `processing_s`.
- FUTURE (start and change only): every `future_stride` steps after t, `fields` of the target's aircraft
  then the others, cut at the next instruction or alarm the target's later calls listed, its down step,
  the episode end or `future_steps`, whichever comes first (`cut` names it).

## The judgement directory: `<output>/<translation run>/<name>/`

| File | Content |
|---|---|
| `run.json` | Resolved config and its SHA-256, the run's `run.json`/`events.jsonl` SHA-256 (a changed run is refused), status, plan (`packets`, `already_sent`, `to_send`, `refused`), counts, error |
| `preview.json` | Preview: the plan and every planned request body (the exact material human raters can read) |
| `audit.jsonl` | Per journaled translation or detector request: `ok` or `leak` with its problems |
| `requests.jsonl` / `responses.jsonl` | `call_id`, `request_sha256`, `body` (fsynced before the send) / status, latency, raw, error |
| `results.jsonl` | One line per (event, criterion): `judgement`, `score` (null for cannot_judge / not_applicable), `rationale`, `rubric_version`, `request_sha256` |
| `ratings.jsonl`, `report.json` | Imported human ratings (`supersedes` names a replaced rating); the report |

## Safety

- Paid gate: `--execute` and `--confirm-paid-api` together or neither (exit 2). A preview builds no
  client, reads no key and writes no request or response journal.
- No resend: a paid run creates `outputs/evaluation/.paid-judge.running` (in this repository,
  whatever `output` says) with `O_EXCL` for its whole duration (if it exists, the run stops and names
  the file; a crashed run leaves it: remove it by hand once no judge runs). Under that marker it
  collects every `request_sha256` (and the hash of every journaled body) from all `requests.jsonl` and
  `judge_requests.jsonl` below `outputs/evaluation/` and below the configured output root, the
  2026-11-05 assess journal `outputs/evaluation/assess/<run>/judge_requests.jsonl` included, and never
  sends a known one, nor the same bytes twice in one run (a run that repeats an event is refused).
  A paid run from a git worktree (`.git` is a file) is refused, naming the main checkout: run paid
  judgements there, where `outputs/evaluation/` holds the journals. A journal line it cannot read
  stops every paid run until repaired. An event already
  judged in the same directory with other bytes is refused (judge it under another `--name`). A
  deliberate replicate needs new bytes: bump `rubric_version` or change the prompt. An event sent
  before whose verdict is not in this directory (an invalid verdict, or results kept elsewhere) is
  never resent automatically and keeps the run `failed`, naming it.
- Recovery of a stuck event (deviation (c), 2026-11-09): `--only-event <event_id>` with both paid
  flags re-sends exactly that event's request, once, when the user asks for it. It refuses an event
  that is not stuck: it must have been sent from this judgement directory with the same bytes, have
  no verdict here, and appear in no other journal (where its verdict may live). The request line
  carries `"resend": {"requested_by": "user", "option": "--only-event", "previous_sends": N}`, and
  `run.json` names the event. No other event is planned or sent; a later plain run still sends no
  journaled request. The re-send is billed again, also after an `uncertain` outcome; if it fails
  too, the event is still stuck and each further re-send needs its own `--only-event` call.
  The scan follows directory symlinks below the roots (each directory once) and stops on an
  unreadable directory.
- Every request line is fsynced before its call; nothing is retried. An HTTP or connection error, a
  refusal or an incomplete response ends the run `failed`; a timeout, an unknown status, an
  unreadable response or any other error ends it `uncertain`. A journaled request is never sent
  again, whatever its outcome.
- Causality: the audit re-reads every request body the translation run journaled: each trajectory
  line must have `step <= t`, no text of a message issued after t may appear, and neither the run's
  episode id nor its scenario id. A leaking target is excluded from judging and reported.
- `OPENAI_API_KEY` is read only after the gate, by the client factory; judgement directories are
  owner-only.

## Human ratings

`import-ratings` reads JSONL lines with exactly `rating_id`, `rater_id`, `event_id`, `criterion_id`,
`judgement`, `rationale`, `rubric_version`, `rated_at` (ISO 8601 with offset) and `supersedes` (the
rater's rating it replaces, or null); example: `src/evaluation/examples/human-ratings.example.jsonl`.
The criterion must belong to the event's kinds, `not_applicable` only to a conditional criterion,
and the rubric must be the config's. A second rating of the same slot needs `supersedes`.

`report.json`: per point kind, aspect and criterion, the judge's and the humans' `n`, `n_scored`,
`mean`, `distribution` and `not_scored`; per criterion the human-judge and human-human exact agreement
and quadratic weighted kappa (`null` below 2 pairs or without variance); the D55 request deadline
pass rate. cannot_judge and not_applicable never count as 0; leaking targets are excluded.
