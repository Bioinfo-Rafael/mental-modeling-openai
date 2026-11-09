# `translate`: one config-driven translation loop

Turns the behaviour of RL agents in a recorded LAG episode into intent statements, one target
(`aircraft:<id>` or `team:<id>`, D17) at a time. One run config decides everything:

```text
select inputs (inputs:) -> each input renders View(t) -> fill the prompt (prompt:) -> one call (model:)
```

| Module | Responsibility |
|---|---|
| `config.py` | `load_yaml` (a duplicate key fails); pydantic `RunConfig` and `Scenario` (defaults; unknown keys fail) |
| `episode.py` | The CSV checked against its manifest, targets, down steps, `View(t)`, the two renderers |
| `inputs.py` | `INPUTS`: `instructions`, `alarms`, `question`, `recent_messages`, `previous_translation`, `trajectory`, `policy_trace` |
| `prompts.py` | Placeholder checks and the one-pass fill |
| `triggers.py` | The forced `start` and `request`, `CHANGE_SOURCES` (`preset`), the tick grid, the schedule |
| `detector.py` | clef-flash: the state text, the System One client (Unix socket or loopback + key), the budget |
| `backends/` | `Prompt`, `Reply`, `BackendError`; `openai.py`: the pinned client, the body, the parse |
| `journal.py` | Run directory, journals (fsync before each send), shared helpers |
| `pipeline.py` | The causal loop |
| `cli.py` | Flags, the paid gate, preview or execute, exit codes |

## Command

```bash
.venv/bin/python -m translate --config configs/translate/preset.yaml                         # preview
.venv/bin/python -m translate --config configs/translate/preset.yaml --execute --confirm-paid-api
```

Flags: `--config` (required), `--scenario` (overrides the config's; real ones live in `private/`),
`--run-name` (default `<config stem>-<sha256(config)[:8]>`), `--execute` with `--confirm-paid-api`
(both or neither). Exit codes: 0 preview or completed, 1 failed, 2 usage or gate, 3 uncertain (a
call may have been billed; never resend it blindly), 4 a cap stopped the run. On Slurm:
`slurm/translate.sbatch preview|paid` (CPU) and `slurm/translate_online.sbatch probe|paid` (the
clef-flash server in the same GPU job; after approval D49).

## Trigger points (2026-11-09)

- `start`: step 0. Every scenario target is translated there, with or without an instruction in
  effect (the `instructions` input then reads `(no instruction in effect)`).
- `request`: a human question to the target, at its step. Instructions and alarms never trigger a
  translation; they are prompt context and detector input (D4, D61, D50).
- `change`: the change source's decision. `preset`: every live target at each listed step.
  `clef_flash`: on the tick grid (`step > 0`, `step % every_steps == 0`, D28) every live target that
  has a translation gets exactly one detector call (D9); a change when the choice is `change` and
  `p(change) >= threshold`. A change that follows the plan is no change point.

Coinciding reasons give one translation listing all of them (`kinds`, `trigger_reasons`). A target
is closed at its down step (aircraft: first `alive == 0`; team: its last aircraft, D37); its later
reasons are logged as `trigger_suppressed`. A new change source is one function in
`CHANGE_SOURCES` plus a `change.source` value in the config models.

## Config

```yaml
scenario: configs/translate/scenarios/example.yaml   # repository-relative
change: {source: preset, steps: [145, 245, 315]}     # or source: clef_flash (configs/translate/online.yaml)
inputs: {instructions: {}, alarms: {}, question: {}, previous_translation: {}, trajectory: {past_steps: 60}}
prompt: {name: intent-v3-en, system: "...", user: "..."}   # name default: the config's file stem
model: {backend: openai, name: gpt-6.1-sol, reasoning_effort: high, max_output_tokens: 25000}   # timeout_s 1800
answer: {max_intent_chars: 1000, on_invalid: skip}   # defaults 600, stop (D41)
limits: {max_translations: 250, max_detector_calls: 500}   # defaults
output: outputs/translate                           # default
```

`clef_flash` keys (defaults in brackets): `every_steps` [5], `threshold` [0.65], `model`, `revision`,
`socket` [`/tmp/{user}-vllm/clef-flash.sock`; `{user}` is the only placeholder] or `base_url`
(`http://127.0.0.1:<port>` or `[::1]`) plus `api_key_env` (a variable NAME), `timeout_s` [30],
`input_token_budget` [16384; must equal the server's `--max-model-len`], `question` (`instructions`,
`none`, `change`), `inputs` (the detector's own inputs) and `state` (the state template).

Placeholders (`str.format`, bare names, literal braces doubled): `{step}`, `{time_s}`, `{target}`,
`{target_aircraft}`, `{max_intent_chars}` and one per selected input. A config fails to load on an
unknown placeholder or on a selected input its prompt does not use; the same holds for `change.state`
and `change.inputs`. A prompt ablation is an edited copy of a config, no code.

## Inputs (`View(t)` x target x options -> text)

| Input | Options [default] | Text |
|---|---|---|
| `instructions` | - | One line per instructed target, its latest instruction (issue step, quoted text); this target's marked `(this target)`; none at all: `(no instruction in effect)` |
| `alarms` | - | Alarms issued by t that reach the target (an alarm reaches every target sharing an aircraft with an addressee), oldest first, else `(none)` |
| `question` | - | The quoted question to the target arriving at t, else `(none)` |
| `recent_messages` | `past_steps` [25] | Alarms reaching the target and questions to it in `[t - past_steps + 1, t]` |
| `previous_translation` | `evidence` [true] | The target's latest valid translation before t, else `(none: this is the first translation for this target)` |
| `trajectory` | `past_steps` [60], `objects` [`all`, or `target_and_opponents`], `fields` [none], `precision` [5] | `[max(0, t - past_steps + 1), t]`: every entity in the 17 format (D22, D32), or the chosen fields of the chosen objects |
| `policy_trace` | `rollout`, `seed` (required); `arrays` [`obs, gru_out, act_in, logits, actions`], `head_weights` [true], `past_steps` [60], `precision` [4] | Per aircraft of the target that the recorded policy flies, one JSON line per step of the window ending at t with the chosen arrays; then one line per logit with `W_head` (13 x 128) and `b_head`, so that `logits = W_head @ act_in + b_head` |

`policy_trace` is the policy input of U-c option (b) (2026-11-09): the recorded activations, the
action-head weights and the per-step head input `act_in`. It replaces the former "weights +
activations" code session, which read the whole network in a sandbox: the translator now sees an
honest slice of the weights with per-logit contributions, not arbitrary computation on all of them.
A rollout needs `act_in` and `head_weights.npz`; record one with
`OUT=outputs/rollouts/mhi-v1.3-act-in SEEDS=100601 sbatch --export=ALL slurm/lag/record_rollouts.sbatch`
(CPU; the recording is deterministic and reproduces the 2026-11-04 one).

`View(t)` holds only rows with `step <= t`, messages issued by t and earlier translations; an input
receives nothing else, so it cannot leak (the poison test in `tests/translate/test_inputs.py` runs
every registered input on two episodes equal through t). A new input is one decorated function in
`inputs.py` (with an options model) plus its key in `inputs:` and its placeholder in the prompt.

## Scenario

```yaml
id: example-01                      # never shown to a model
episode: ../../../data/mhi/processed/mhi_v1.3_ic100601_win.csv   # relative to the scenario file
targets: [aircraft:A0100, aircraft:A0200, team:A]
messages:                           # default []; each revealed at its step
  - {id: question-001, step: 10, kind: question, to: [aircraft:A0100], text: "..."}   # kind: instruction | question | alarm
```

Rules: ids unique; `to` names only listed targets; at most one question per target per step; steps
within `[0, last complete step]` and targets that exist in the episode (checked against the episode).

## Run directory `<output>/<run name>/` (owner-only, created once; a preview writes `<name>-preview/`, never into a directory that holds a journal, and a paid run name may not end in `-preview`)

| File | Content |
|---|---|
| `run.json` | mode, status (`preview`, `running`, `completed`, `failed`, `uncertain`, `cap_reached`), times, git commit and dirty flag, the resolved config and scenario, `prompt.name` + SHA-256, episode CSV path and SHA-256, detector model and revision, counts per layer, error |
| `events.jsonl` | Event schema 3, one line each: `translation_event`, `detector_call`, `target_closed`, `trigger_suppressed` |
| `requests.jsonl` | One line per call of either layer, fsynced before the send: `call_id`, `layer`, `step`, `target`, `request_sha256`, `body` |
| `responses.jsonl` | `call_id`, `status` (`ok`, `error`, `uncertain`), `latency_s`, `raw`, `error` |
| `preview.json` | The forced translations, the detector-call count (an upper bound when a skipped invalid answer leaves a target without a translation), the translation bound, caps, warnings, approximate input tokens and the exact first request, the detector question and state template and the worst state bound |

A `translation_event` holds `event_id` (`<scenario>:<kind>:<id>:<step:06d>`, also the `call_id`),
`kinds`, `trigger_reasons` (`kind`, `source`, `reference_id`, `probability`), the human-visible
`input` (`target_aircraft`, `instructions_in_effect`, `alarms`, `question`, `previous_translation`,
the 60-step `trajectory` window with its SHA-256, whatever the translator read, and
`translator_inputs`), `output` (`intent`, `evidence`), `answer_problem` (a skipped invalid answer),
`prompt`, `model`, `usage`, `request_sha256` and `processing_s` (from the start of step t, when a
question arrives, to the record: it includes the calls of targets processed before it at t). A `detector_call` holds the state text, the
question, `output` (`label`, every probability, `threshold`, `triggered`), the raw response and usage.

## Safety

- Paid gate: `--execute` and `--confirm-paid-api` together or neither (exit 2). A preview builds no
  backend, reads no key and starts no detector client.
- No resend: the run directory is created before the backend (which reads `OPENAI_API_KEY`), so a
  resubmitted paid job with the default name stops at `mkdir`; a replicate passes `--run-name`.
  Every request line is fsynced before its call; nothing is retried. An HTTP or connection error, a
  refusal or an incomplete response ends the run `failed`; a timeout, an unknown status or an
  unreadable response ends it `uncertain`.
- The OpenAI client targets `https://api.openai.com/v1` only, has SDK retries off and refuses to
  start while `OPENAI_LOG` or `OPENAI_CUSTOM_HEADERS` is set. Errors record the class and project
  message only.
- The detector server is local: a Unix socket whose directory is owned by you with mode 0700 (the
  access control; no key), or a loopback TCP root with a Bearer key from a variable NAME. Every
  scheduled state is bounded before the run (with a worst-case previous intent; the detector's
  `previous_translation` must not include evidence), each state again before its send, and the
  server's reported input tokens after (vLLM truncates over-long states silently).
- Run directories are owner-only; real scenarios and outputs stay in gitignored trees.
