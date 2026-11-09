# `translate`: one config-driven translation pipeline

Turns the behaviour of RL agents in a recorded LAG episode into natural-language intent statements,
one target (aircraft or team) at a time. Every run follows the same four steps, all chosen by one
run config (`configs/translate/*.yaml`):

1. **choose inputs** (`inputs:`): `instructions`, `alarms`, `question`, `previous_translation`,
   `trajectory`, `policy`;
2. **process inputs** (`inputs.py`, `policy/`): each selected input becomes named values for the
   step t, built only from information at or before t;
3. **complete the prompt** (`prompts.py`): the prompt file named by `prompt.file` is filled with
   those values and `prompt.vars`; the prompt text lives in config, so prompt ablations need no
   code change;
4. **translate**: the prompt file's `protocol` decides the call: `one_shot` (one strict-schema
   OpenAI Responses request) or `code_session` (the model reads the policy's weights and replay
   activations with Python in a sandbox, then answers in JSON).

When to translate is the trigger source (`triggers:`): **online** (every target is translated at
the episode start; a human question, or a target's first instruction arriving after the start,
forces a translation; the clef-flash change-point detector decides every tick whether the previous
intent still explains the behaviour) or **preset** (the steps are chosen in advance in the config,
each with its kind). Trigger kinds are exactly three (2026-11-09):

- `start` (開始点): the episode start, step 0, every target's first translatable step (every input
  window clips at step 0). An episode may start with or without an instruction in effect; the start
  is translated either way.
- `change` (変化点): the detector judges that the previous intent translation no longer explains the
  behaviour, because of an external factor (an alarm, an observed event), a human instruction or
  similar. A tactic that follows the plan is no change: the plan is already in the previous intent.
- `request` (要求点): a human question, or a target's first instruction arriving after the start. A
  later instruction for a target that already has one forces nothing; it only reaches the detector
  (D61).

Every run writes `events.jsonl` (schema 3) for the judges. Each event records the human-visible
material (scenario texts and the trajectory window) whatever the translator read, so the judges
evaluate every configuration the same way.

| Module | Responsibility |
|---|---|
| `config.py` | Strict run config (schema 2): triggers, inputs, prompt, call, answer, session, detector, output; checks the prompt against the selected inputs |
| `scenario.py` | Scenario YAML, scenario rules, episode facts, what a target sees at step t |
| `triggers.py` | Online and preset triggers, the deterministic schedule plan, the detector budget |
| `inputs.py` | Input processors: slot -> prompt values; the event's human-visible `input` record |
| `prompts.py` | Prompt files: load, placeholder check, list/optional blocks, follow-up turns |
| `call_one_shot.py` | One strict-schema Responses call, its parser and the OpenAI adapter |
| `call_session.py`, `session_backend.py` | One code session per slot: inputs bundle, preflight, loop, audit, answer; the Responses backend |
| `answers.py` | Answer limits and validation, the same for both calls |
| `pipeline.py` | The chronological loop: triggers -> inputs -> prompt -> call -> answer -> journal -> event |
| `detector.py` | clef-flash detection layer (System One state, budget bound, adapter) |
| `run_files.py` | JSONL journals (written before each send) and the `cues.jsonl` projection |
| `events.py` | Event and scenario types, event ids, canonical JSON (frozen: it defines `request_sha256`) |
| `cli.py` | The only composition root: gate, preview, clients, run directory, provenance |
| `llm/`, `lag/` | LLM clients (OpenAI Responses, System One, vLLM launcher, retry); LAG episode loader and renderers |
| `policy/`, `sandbox/` | Checkpoint reader, spec, float64 replay and its renderings; the unshare sandbox, preflight, reply protocol, audit |

## Shipped configurations

| Config | Inputs | Prompt | Triggers |
|---|---|---|---|
| `online.example.yaml` | instructions, alarms, question, previous translation, trajectory | `translator-intent-v2` | online (clef-flash), example scenario |
| `seed100601-trajectory{,-ja,-ja-short}.yaml` | instructions, alarms, question, previous translation, trajectory | `translator-intent-v2` | preset 45 / 145 / 245 / 315 |
| `seed100601-policy{,-ja,-ja-short}.yaml` | instructions, previous translation, policy | `intent-b-v1` (code session) | preset 45 / 145 / 245 / 315 |

The trajectory and policy configs reproduce the two former comparison methods byte for byte (the
prompts and request bodies are pinned by `tests/translate/test_request_goldens.py`, and
`scripts/check_request_replay.py` re-runs the recorded runs through this pipeline). The preset
kinds of the reproduction configs (45 `start`, the others `change`) are assumptions: the
recorded runs had none. Step 45 is not the episode start (step 0 is translatable too); the four
steps were chosen for large movement changes, and a preset run translates exactly its points. `prompts/translator-intent-v1.yaml` is the 2026-11-04 trajectory prompt,
kept only to reproduce those recorded runs.

## Command

```bash
# Preview (default): no client, no credential, no network. Writes run.json, preview.json and
# example_translation_request.json (the exact first request).
.venv/bin/python -m translate --config configs/translate/seed100601-trajectory.yaml
# Paid run (keys from the environment only; OPENAI_API_KEY lives in the group-shared repo-root .env).
.venv/bin/python -m translate --config configs/translate/seed100601-policy.yaml --run-name <name> --execute --confirm-paid-api
```

On Slurm: `slurm/translate.sbatch preview|paid` (preset triggers) and `slurm/translate_vllm.sbatch
probe|paid` (online triggers: the job serves clef-flash on a local vLLM; awaits approval D49).

| Flag | Meaning |
|---|---|
| `--config PATH` | Run config (required) |
| `--scenario PATH` | Scenario YAML; overrides the config's `scenario` (real ones live in `private/translator/`) |
| `--output-root DIR` | Overrides `output.root` |
| `--run-name NAME` | Run directory name (default `<UTC stamp>_<scenario_id>_<preview|execute>`); a name, never a path |
| `--execute`, `--confirm-paid-api` | Both, or neither: every configuration makes paid translation calls |

A code-session config passes the fail-closed sandbox preflight before any client or key exists.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Preview written, or run `completed` |
| 1 | Deterministic failure: config, scenario, episode data, rendering, budget, sandbox preflight, existing run directory, local server, protocol, refusal, incomplete response, invalid answer under `on_invalid: stop` |
| 2 | Usage or gate misuse (argparse), including an invalid `--run-name` |
| 3 | `uncertain`: a call may have reached the server and been billed. Never resend it blindly |
| 4 | `cap_reached`: a call cap stopped the run; `events.jsonl` is partial |

## Config

Every key is required and explicit; unknown and duplicate keys fail (`configs/translate/*.yaml`
are commented examples).

| Section | Content |
|---|---|
| `scenario` | Scenario file (repository-relative), or null to require `--scenario` |
| `triggers` | `source: online` + `tick_interval_steps`, or `source: preset` + `points: [{step, kind, targets}]` (`targets: null` = every translation target) |
| `visibility` | `{mode: online, future_steps: 0}` (retrospective is not implemented) |
| `limits`, `scenario_limits` | Call caps; maximum scenario text lengths |
| `detector` | The clef-flash System One settings (online only; null for preset) |
| `inputs` | The selected inputs; each must be used by the prompt, and the prompt may use only values they provide |
| `trajectory` | `{renderer: lag17-jsonl, past_steps}`: the window every event records |
| `policy` | `{rollout_dir, seed, spec, checkpoint, past_steps}` when `policy` is selected |
| `prompt` | `{file, vars}`: the prompt file and its free variables |
| `translation` | `{backend: openai_responses, model, reasoning_effort, max_output_tokens, timeout_s, retry: default | none}`; `none` sends every request once (a 5xx is reported, not resent) |
| `answer` | `{max_intent_chars, max_evidence_items, max_evidence_chars, on_invalid: stop | skip}` |
| `session` | Code session only: `{max_runs, max_total_tokens, max_protocol_errors, answer_at_fraction, sandbox}` |
| `output`, `pricing` | Output root; USD per 1M tokens for the preview estimate |

### Inputs and the values they provide

| Input | Values (prompt placeholders) |
|---|---|
| always | `episode_alias`, `step`, `time_s`, `t_s`, `step_seconds`, `target_label`, `target_aircraft`, `max_intent_chars` |
| `instructions` | `instructions` (every instruction in effect, through `blocks.instructions`; its `empty` text when none is), `primary_instruction` (this target's own instruction through `blocks.primary_instruction`, `{text}` or its `empty` text; required when the prompt uses it) |
| `alarms` | `alarms` (alarms for the target at or before t, through `blocks.alarms`) |
| `question` | `question` (the request at t, through `blocks.question`) |
| `previous_translation` | `previous_translation` (the target's latest valid answer, through `blocks.previous_translation`) |
| `trajectory` | `trajectory`, `trajectory_start`, `trajectory_end`, `trajectory_points`, `trajectory_anchor` |
| `policy` | `policy_alias`, `policy_task`, `policy_architecture`, `policy_inputs`, `policy_outputs`, `policy_parameters`, `policy_activations`, `window_start`, `window_end`, `window_length` |
| code session | `max_runs`, `timeout_s`, `stdout_cap_chars`, `stderr_tail_chars` |

### Prompt files

`configs/translate/prompts/*.yaml`: `version` (recorded as `prompt_version`), `protocol`,
`output_schema` (`intent_evidence`, one-shot only), `system` and `user` (`str.format` templates:
`{t_s:g}` formats a number, literal braces are doubled), `blocks` (item template, joiner and the
text when empty, for the list and optional inputs) and `turns` (code session: the follow-up
messages). A new prompt is a new file with a new `version`; the judges' prompts are not
configurable (they stay in `src/evaluation`, because the paid assess resume depends on their bytes).

## Run directory

`<output.root>/<run name>/`, created with `mkdir(exist_ok=False)` and mode `0700`.

| File | Content |
|---|---|
| `run.json` | Mode, status, times, inputs (config, prompt, scenario, episode, policy) with SHA-256, git commit + dirty flag, SHA-256 of every `translate` module, renderer versions, resolved config (`config.pipeline.visibility` is where the judges read the visibility), schedule, detector budget, call counts, `unsent`, `error`, `api_requests_made` (every request sent: detector calls plus every OpenAI request, each code-session turn and each retried attempt included; `counts.openai_requests` is the OpenAI part) |
| `preview.json`, `example_translation_request.json` | Preview only: schedule, budget, the SHA-256 of every request known in advance, an approximate cost bound; the first request |
| `events.jsonl` | The event log (contract below) |
| `requests.jsonl` | One record per call, written and fsynced **before** the call is sent (`call_id`, `layer`, `step`, `target`, exact `payload`) |
| `responses.jsonl` | One record per sent call (`status`, `latency_s`, `raw` body or the session summary, `error`) |
| `cues.jsonl` | Derived after the run: one line per translation slot (answer or `skipped` with the reason) for `python -m evaluation.viewer` |
| `sessions/<sSSSS_AIRCRAFT>/` | Code session only: `inputs/` (read-only `W` and `A`), `message.txt`, `transcript.jsonl`, `tool_calls.jsonl`, `state.json`, `final_answer.md`, `preflight.json`, `audit.json` |

A request without a response line means the process died mid-call: treat that call as uncertain.

## Pipeline rules (both trigger sources)

The unit of translation is one exact target: `aircraft:<id>` or `team:<id>` (D17). Per step
`0 .. last_complete_step` (340 / 293 / 495 for the committed episodes):

0. Every translation target (each target an instruction or a request names) exists from step 0,
   with or without an instruction in effect.
1. Instructions issued at this step are applied (in `instruction_id` order). A later instruction
   replaces the target's current one; the target keeps its previous translation (D21). A later
   instruction forces no translation: it only reaches the detector state (D61).
2. Every target that is down is closed once (`target_closed`). Aircraft: first step with
   `alive == 0`. Team: the step its last aircraft is down. A closed target never reopens (D37).
3. Targets are processed in canonical `(kind, id)` order (aircraft before team):
   - online triggers: `start` at step 0 for every target (source `episode_start`; the bootstrap
     the detector needs as its previous intent), `request` at a scenario request (D4, D10; the
     question goes into the prompt) and `request` at a target's first instruction when it arrives
     after the start (source `instruction`, reference the instruction id; no question);
   - preset triggers: the config's `triggers.points`, each step with its kind (source `preset`;
     a preset `request` point takes the source and id of the scenario request it answers);
   - a closed target only logs its triggers as `trigger_suppressed` (no model call);
   - online only, on a tick (`step > 0` and `step % tick_interval_steps == 0`, a fixed grid that requests never
     move, D28) a target that already has a translation gets exactly one detector call, even when
     other triggers exist (D9). Decision rule: `label` is the model's choice; the tick triggers
     when `label != none` and `p(label) >= threshold`, adding a `change` reason;
   - if any trigger remains, the target gets exactly one translation. Coinciding reasons are all
     kept; the primary trigger follows `start > change > request`.
4. A target first instructed after step 0 is translated at the start without an instruction, then
   at its issue step as a `request` (supersedes D35); its detector ticks start after its start
   translation.

Scenario change points are not triggers: hindsight-placed change references live in the
evaluation annotation, never in the scenario (D50 supersedes D6).

Windows (online visibility, D33): the trajectory window `[max(0, step-59), step]` with `lag17-jsonl` (D22, D32;
`trajectory.past_steps`), recorded in every event and inserted into the prompt when selected;
detector `[max(0, step-9), step]` with `lag-detector-compact` over the target's aircraft, then the
opposing team's aircraft (D27). Downed aircraft keep rendering with `alive == 0`.

Caps: immediately before each call, if the layer's sent count equals its cap, nothing is sent and
the run stops with `cap_reached`; `run.json.unsent` names the layer, target, step and cap. The
preview reports the exact detector call count (detector calls never depend on model outputs), the
mandatory translations, the translation upper bound (mandatory plus one per tick without a scenario
trigger) and the headroom; a bound above a cap is only a warning.

Invalid answers: `answer.on_invalid: stop` ends the run `failed`; `skip` writes the
`translation_event` with a blank `output` and `answer_problem`, the target keeps its last valid
translation, and the run continues.

Errors: a failed call is never resent beyond the narrow `translate.llm.retry` policy
(`translation.retry: default`: HTTP 5xx and non-timeout connection failures only). The response line records the error, the run stops, and
`run.json` gets `failed` or `uncertain`. If journaling itself fails while a sent call has no
response line yet, the run stops as `uncertain` with that call's id.

## Detector layer

State string sent to `/v1/systemone` (exact; logged as `state_text`, state version 2):

```text
Target: aircraft A0100 (aircraft A0100). Opponents: B0100, B0200.
Previous intent translation for this target (from step <s>):
<the target's own previous intent; evidence is not included>

Instructions in effect for this target:
- aircraft A0100, issued at step <i>: <its own instruction text>
- team A, issued at step <j>: <its team's instruction text (aircraft targets only)>

Recent messages (alarms and human questions up to this step, most recent last):
- question at step <q>: <question text>
- alarm at step <w>: <alarm text>
(or "- none")

Trajectory window: steps <a>-<b>, 0.2 s per step, one JSON line per step, newest last; relative_step = step - <b>. Fields: <fields>. null = unavailable.
<compact window text>
```

- Question: one `choice` question `point_kind` with exactly the options `none` and `change`
  (config `detector.question`; the adapter rejects any other question set, the former third option
  `start` included).
- Instructions in effect: the target's own current instruction and, for an aircraft, its team's;
  each with its issue step (all issued at or before t).
- Recent messages: **all** alarms and questions for this target issued in
  `[t - messages_lookback_steps + 1, t]`, oldest first. An alarm is for every target that shares
  an aircraft with one of its targets (an alarm for aircraft A0100 reaches team A too); a
  question is for the request's own targets. Nothing is truncated or capped. The config requires
  `messages_lookback_steps >= tick_interval_steps`, so every message reaches at least one
  detector call while the target is up (default 25 = 5 ticks). Boundary: a message issued after
  the last tick the target gets (after the episode's last tick, or between the target's last tick
  and its down step) reaches no detector call; a question still forces its own translation.

- `float_precision: 5` (about 1 m). With 3 decimals, lon/lat resolve about 100 m while an aircraft
  moves about 50 m per step: in `mhi_v1.3_ic100601_win`, 45 of 340 consecutive A0100 positions are
  identical at 3 decimals and none at 5 (`tests/translate/test_tr_budget.py`).
- Budget fit is proved before any call: for every scheduled tick, the client's conservative bound
  (UTF-8 bytes + fixed reserves) of the exact state, the scenario's actual instruction, alarm and
  question texts included, plus a worst-case intent (4 bytes x `answer.max_intent_chars`)
  must stay below `input_token_budget`. The scenario is known before the run, so this is exact;
  a scenario that does not fit fails in preflight, before any directory, client or call. The
  knobs: shorter scenario texts (`scenario_limits`), a shorter `detector.messages_lookback_steps`,
  `detector.window` (steps, fields), or a lower `answer.max_intent_chars`.
- With the default config (`max_intent_chars: 600`, reserve 2400) the example scenario's worst
  bound is 13158 (team A, margin 3226). A dense worst case over every tick of all three episodes
  (`test_tr_budget.py`: every target instructed at step 0 with a 300-character Japanese text, a
  200-character Japanese alarm per team and a 200-character Japanese question per aircraft in every
  25-step lookback) peaks at 14620 (team targets, margin 1764) against 16384.
- `input_token_budget` must equal the server's real cap, `min(16384, --max-model-len)` for
  `vllm serve Cloudflare/clef-flash` (4096 for OpenJev `wev serve`); the config rejects > 16384.
- `threshold: 0.65` is provisional until a labelled LAG check.
- `detector_call.output` = `{label, probabilities: {none, change}, threshold, triggered}`.

## Code-session safety (the policy input)

- **Checkpoint reader** (`policy/torch_reader.py`): stdlib-only allow-list unpickler, no torch. It accepts
  only `OrderedDict`, `_rebuild_tensor_v2`, `_rebuild_parameter` and `torch.*Storage`, and rejects
  every other global, compressed or unknown zip members, bad offsets, strides or sizes, and any
  dtype other than float32.
- **Sandbox** (`sandbox/executor.py`): `unshare` user, mount, pid and net namespaces; a tmpfs root holding
  only `/usr`, the interpreter, numpy, `/work` and the read-only inputs, entered via `pivot_root`;
  `env -i`, `python -I`, RLIMIT_AS/FSIZE/NPROC/CPU, a timeout, a `/work` size cap and an output cap.
  There is no aggregate memory cgroup limit on these nodes, so run paid sessions inside a Slurm
  job. The preflight (`sandbox/preflight.py`) must pass every check, or no request is sent.
- **Audit** (`sandbox/audit.py`): flags nonlinearities, LayerNorm-like expressions, recurrence loops and
  torch imports; `audit.json` records `mode: code-session` and the prompt version.
- **Reference forward** (`policy/reference.py`): numpy float64, mirrors mtd-rl-lag (Linear -> ReLU ->
  LayerNorm, GRU gates `[r; z; n]`). Torch parity: `slurm/policy_parity.sbatch`.

## `events.jsonl` contract (schema_version 3)

One JSON object per line, in emission order. Four record types only. `requests.jsonl` and
`responses.jsonl` records carry the same `schema_version`. Schema 3 (2026-11-08) has three trigger
kinds, with the meanings of 2026-11-09 (no schema 3 run was written before them); schema 2 runs (kinds `initial`, `start`, `change`, `request`, where `initial` was the first
instruction and `start` a detector label) stay readable by the judges through an explicit table
(`src/evaluation/trigger_kinds.py`). A v1 run is not accepted.
Every record has:

| Field | Meaning |
|---|---|
| `schema_version` | `3` |
| `sequence` | 1, 2, 3, ... without gaps, in file order |
| `record_type` | `translation_event`, `detector_call`, `target_closed`, `trigger_suppressed` |
| `scenario_id`, `episode_id` | From the scenario |
| `step`, `time_s` | Step and simulation seconds (`round(step * 0.2, 1)`) |
| `target` | `{"kind": "aircraft" | "team", "id": "<id>"}` |

Trigger reason object: `{"kind", "source", "reference_id", "probability"}`.

| `kind` | Point | `source` | `reference_id` | `probability` |
|---|---|---|---|---|
| `start` | 開始点: the episode start, step 0 (forced, with or without an instruction in effect; also the bootstrap the detector needs as its previous intent) | `episode_start` (`preset` for a preset point) | `episode-start` (`preset-step-<s>`) | `null` |
| `change` | 変化点: the previous intent no longer explains the behaviour, because of an external factor or a human instruction (progress along the plan is none) | `detector` (`preset`) | the detector `call_id` (`preset-step-<s>`) | `p(change)` (`null`) |
| `request` | 要求点: a human asked, or a target's first instruction arrived after the start | the request's `source` (`instruction`) | the request id (the instruction id) | `null` |

Lists are in precedence order (`start > change > request`), then by `reference_id`.

### `translation_event`

| Field | Meaning |
|---|---|
| `event_id` | `<scenario_id>:<kind>:<target_id>:<step:06d>`; unique (one translation per target per step); also the `call_id` in `requests.jsonl` / `responses.jsonl` |
| `request_sha256` | SHA-256 of the canonical JSON (sorted keys, `,`/`:` separators, UTF-8) of the exact request `payload` journaled under `call_id == event_id`: the full prompt; for a code session, its first request (the later turns are in `sessions/<slot>/transcript.jsonl`) |
| `episode` | `{episode_id, alias, csv_sha256}` |
| `primary_trigger` | Trigger kind with the highest precedence |
| `trigger_reasons` | Every coinciding reason |
| `input.prompt_version` | The prompt file's `version` (e.g. `translator-intent-v2`, `intent-b-v1`) |
| `input.target_aircraft` | The target's aircraft ids, as named in the prompt |
| `input.instructions_in_effect` | Every instructed target's current instruction, canonical target order: `{instruction_id, issue_step, target, text}` |
| `input.primary_instruction_id` | The instruction of this exact target; `null` before its first one |
| `input.alarms` | Every alarm for this target issued at or before `step`, oldest first: `{alarm_id, step, targets, text}` |
| `input.question` | `{request_id, step, text}` for a question's request event, else `null` (also for a request opened by an instruction) |
| `input.trajectory` | `{renderer, renderer_version, start_step, end_step, anchor_step, object_ids, fields, text, sha256}`: the trajectory window a human rater sees, always recorded; when `trajectory` is a translator input, `text` is exactly the window text in the prompt; steps are the clipped, actually rendered ones |
| `input.previous_translation` | `{event_id, step, intent, evidence}` of this target, or `null` |
| `input.output_language` | The prompt's `output_language` var, if any |
| `input.translator_inputs` | The selected inputs (what the translator read). Absent in runs before 2026-11-08, which read every listed input |
| `input.policy` | Policy input only: `{network_alias, checkpoint_sha256, rollout_sha256, session_dir, inputs_manifest_sha256, window}`; the last three describe the bundle the sandbox read (`sessions/<session_dir>/inputs/manifest.json`, its SHA-256 and the window it states), which the assess leak check verifies |
| `output` | `{intent, evidence}` as parsed |
| `model` | `{backend, protocol, model, reasoning_effort, max_output_tokens, response_model, response_id}`; `protocol` is `one_shot` or `code_session` |
| `answer_problem` | Only when the answer was invalid and `answer.on_invalid` is `skip`: why (then `output` is blank) |
| `usage` | `{input_tokens, cached_input_tokens, output_tokens, reasoning_tokens}` |
| `processing_s` | Monotonic seconds from the start of processing this target at this step (before any detector call) until this parsed record is complete, immediately before its own fsynced write (one append, the same for every event). The request deadline's measure: question received to usable answer |

The trajectory text is all the judge needs verbatim. The full prompt (system prompt + user
message) is the `payload` of the request line whose `call_id` equals `event_id`; `request_sha256`
lets a consumer check that join.

### `detector_call`

| Field | Meaning |
|---|---|
| `call_id` | `<scenario_id>:detector:<kind>:<target_id>:<step:06d>` |
| `request_sha256` | SHA-256 of the canonical JSON of the exact `/v1/systemone` body (`model`, `state`, `questions`) |
| `input.trajectory` | Compact window, same keys as above |
| `input.previous_translation` | `{event_id, step, intent}` the state was built from |
| `input.instructions_in_effect` | The target's own instruction and, for an aircraft, its team's: `{instruction_id, issue_step, target, text}` |
| `input.messages` | Every alarm and question for this target in the lookback, oldest first: `{kind: alarm \| question, id, step, text}` |
| `input.state_text` | The exact state string sent |
| `input.question` | Full question object `{id, type: choice, instructions, criteria: {none, change}}` |
| `output` | `{label, probabilities: {none, start, change}, threshold, triggered}`; `triggered = label != none and probabilities[label] >= threshold` |
| `raw_response` | The full server response body |
| `model` | `{backend, model, revision, served_model_name, response_model}` |
| `usage` | `{input_tokens, output_tokens}` |

### `target_closed`

`reason` (`target_down`), `down_step`, `aircraft_down_steps` (`{aircraft id: step or null}`),
`last_event_id` (the target's last translation, or `null`). `step` is when the pipeline closed
it: the down step, or the issue step of an instruction for an aircraft that was already down.

### `trigger_suppressed`

`reason` (`target_down`), `down_step`, `primary_trigger`, `trigger_reasons`. No model call is made
and nothing about it is judged.

### Examples (trajectory and state text truncated)

`translation_event` for the example request at step 10, after a detector `change` at step 5:

```json
{"schema_version": 3, "sequence": 7, "record_type": "translation_event", "scenario_id": "placeholder-episode-01",
 "episode_id": "mhi_v1.3_ic100601_win", "step": 10, "time_s": 2.0, "target": {"kind": "aircraft", "id": "A0100"},
 "event_id": "placeholder-episode-01:aircraft:A0100:000010",
 "episode": {"episode_id": "mhi_v1.3_ic100601_win", "alias": "episode_01", "csv_sha256": "812a44c4..."},
 "primary_trigger": "request",
 "trigger_reasons": [{"kind": "request", "source": "placeholder-human", "reference_id": "placeholder-request-001", "probability": null}],
 "input": {"prompt_version": "translator-intent-v2", "target_aircraft": ["A0100"],
           "instructions_in_effect": [
             {"instruction_id": "placeholder-aircraft-001", "issue_step": 0, "target": {"kind": "aircraft", "id": "A0100"}, "text": "[PLACEHOLDER] Perform the assigned single-aircraft task."},
             {"instruction_id": "placeholder-team-001", "issue_step": 0, "target": {"kind": "team", "id": "A"}, "text": "[PLACEHOLDER] Perform the assigned team task."}],
           "primary_instruction_id": "placeholder-aircraft-001",
           "alarms": [],
           "question": {"request_id": "placeholder-request-001", "step": 10, "text": "[PLACEHOLDER] What is aircraft A0100 trying to do now?"},
           "trajectory": {"renderer": "lag17-jsonl", "renderer_version": "1", "start_step": 0, "end_step": 10, "anchor_step": 10,
                          "object_ids": ["A0100", "A0200", "B0100", "B0200"], "fields": ["object_id", "...", "missiles_remaining"],
                          "text": "{\"step\":0,\"time_s\":0.0,\"relative_step\":-10,\"entities\":[...]}\n...", "sha256": "..."},
           "previous_translation": {"event_id": "placeholder-episode-01:aircraft:A0100:000005", "step": 5,
                                    "intent": "[PLACEHOLDER] Earlier intent.", "evidence": ["[PLACEHOLDER] Evidence."]},
           "output_language": "Japanese"},
 "output": {"intent": "[PLACEHOLDER] Answer and intent.", "evidence": ["[PLACEHOLDER] Evidence."]},
 "model": {"backend": "openai_responses", "model": "gpt-6.1-terra", "reasoning_effort": "medium", "max_output_tokens": 4000,
           "response_model": "gpt-6.1-terra", "response_id": "resp_..."},
 "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
 "request_sha256": "...", "processing_s": 2.41}
```

`detector_call` at tick 15, after the question at 10 and the alarm at 12:

```json
{"schema_version": 3, "sequence": 9, "record_type": "detector_call", "scenario_id": "placeholder-episode-01",
 "episode_id": "mhi_v1.3_ic100601_win", "step": 15, "time_s": 3.0, "target": {"kind": "aircraft", "id": "A0100"},
 "call_id": "placeholder-episode-01:detector:aircraft:A0100:000015", "request_sha256": "...",
 "input": {"trajectory": {"renderer": "lag-detector-compact", "renderer_version": "1", "start_step": 6, "end_step": 15, "anchor_step": 15,
                          "object_ids": ["A0100", "B0100", "B0200"], "fields": ["alive", "...", "vu_mps"], "text": "...", "sha256": "..."},
           "previous_translation": {"event_id": "placeholder-episode-01:aircraft:A0100:000010", "step": 10, "intent": "[PLACEHOLDER] Answer and intent."},
           "instructions_in_effect": [
             {"instruction_id": "placeholder-aircraft-001", "issue_step": 0, "target": {"kind": "aircraft", "id": "A0100"}, "text": "[PLACEHOLDER] Perform the assigned single-aircraft task."},
             {"instruction_id": "placeholder-team-001", "issue_step": 0, "target": {"kind": "team", "id": "A"}, "text": "[PLACEHOLDER] Perform the assigned team task."}],
           "messages": [
             {"kind": "question", "id": "placeholder-request-001", "step": 10, "text": "[PLACEHOLDER] What is aircraft A0100 trying to do now?"},
             {"kind": "alarm", "id": "placeholder-alarm-001", "step": 12, "text": "[PLACEHOLDER] Warning about the situation of aircraft A0100."}],
           "state_text": "Target: aircraft A0100 (aircraft A0100). Opponents: B0100, B0200.\n...",
           "question": {"id": "point_kind", "type": "choice", "instructions": "...", "criteria": {"none": "...", "change": "..."}}},
 "output": {"label": "change", "probabilities": {"none": 0.1, "change": 0.9}, "threshold": 0.65, "triggered": true},
 "raw_response": {"model": "clef-flash", "answers": {"point_kind": {"type": "choice", "choice": "change", "probabilities": {"...": 0}}}, "usage": {"...": 0}},
 "model": {"backend": "system_one", "model": "Cloudflare/clef-flash", "revision": "fb4417d31c1d0f9e7e77f1141b40aaede7491e00",
           "served_model_name": "clef-flash", "response_model": "clef-flash"},
 "usage": {"input_tokens": 0, "output_tokens": 0}}
```

## Scenario

`configs/translate/scenarios/example.yaml` documents the scenario format (`schema_version: 2`)
with placeholder text: `instructions`, `alarms` (`{alarm_id, step, targets, text}`) and
`requests` (`{request_id, step, targets, source, question}`). The v1 key
`external_change_points` is rejected. Scenario rules: exact keys; ids unique across
instructions, alarms and requests; at least one translation target (a target an instruction or a
request names; no instruction need be issued at step 0); at most one instruction and at most one
request per exact target per step; every alarm target is a translation target; every text within
`scenario_limits` (longer fails the load,
nothing is truncated); all steps within `[0, last_complete_step]`; aircraft and team ids must
exist in the episode. Alias outcome check (D58): `episode.alias` is lowercased and split into
alphanumeric tokens (`episode_id` likewise); no token may be an outcome word (`win`, `loss`,
`lose`, `won`, `lost`), and no token may be shared with `episode_id`. So `window_01` and
`close_call` pass, while `ep-win` or a reused seed or IC token fail. Paths resolve relative to the scenario file. A
scenario is always loaded against the run config (`load_scenario(path, limits=...)`), and
`scenario.validate_scenario` re-runs the same rules, so a `Scenario` built in code is held to them
too.

### Detector endpoint

The default (D39) is a Unix socket in a per-user `0700` directory:

```yaml
connection:
  base_url: null
  uds_path: /tmp/{user}-vllm/clef-flash.sock   # {user} = your user name (getpass.getuser())
  api_key_env: VLLM_API_KEY                     # optional over UDS (null = no key)
```

- `{user}` is the only placeholder. No environment variable is expanded: `$USER`, `${...}` or any
  other `{...}` is a config error. The committed file names no user.
- Create the directory before the server starts: `mkdir -m 700 /tmp/$USER-vllm`, then
  `vllm serve ... --uds /tmp/$USER-vllm/clef-flash.sock`. The client checks the directory (owner,
  mode `0700`, no symlink) when it is built, so a preview works without the directory or socket.
- Loopback TCP stays selectable: `base_url: http://127.0.0.1:<port>`, `uds_path: null` and a key
  variable. **On a shared node another user may own the port** (bound before vLLM starts, or
  after it exits), and the client would send its Bearer key and every detector state to that
  process. Use TCP only on a node allocated to this job.
- The config runs the System One client's own checks at load time (`checked_tcp_url`,
  `checked_env_name`): a URL with userinfo, a path, query or fragment, a host other than
  `127.0.0.1` / `::1`, or an `api_key_env` that is not a POSIX variable NAME (`[A-Z_][A-Z0-9_]*`,
  so a pasted key is rejected) fails the preview before anything is written. Messages never echo
  the URL or the value.

## Sensitive data

- Real scenarios (customer instructions, D11) live under `private/translator/`; the whole
  `private/` tree is gitignored. Committed examples use `[PLACEHOLDER]` text only.
- Outputs go under `outputs/translate/` (gitignored as a whole tree). Event logs, requests and
  previews contain exact model-visible text; run directories are created owner-only (`0700`).
- Keys are never written: the config holds only the NAME of the detector key variable;
  `OPENAI_API_KEY` is read by `make_openai_client()` after the gate. That client always targets
  `https://api.openai.com/v1` (recorded as `provenance.openai_base_url`): `OPENAI_BASE_URL` is
  ignored, and `OPENAI_CUSTOM_HEADERS` and `OPENAI_LOG` (it logs request bodies) must be unset:
  the client refuses to start, naming the variable only. Proxy (`HTTPS_PROXY`, `HTTP_PROXY`,
  `ALL_PROXY`, `NO_PROXY`, lowercase too), CA-bundle (`SSL_CERT_FILE`, `SSL_CERT_DIR`) and
  `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` stay honoured; `provenance.sdk_environment_set` in
  `run.json` lists which of them were set (names only, never values). Preview builds no client,
  so it neither refuses nor warns about `OPENAI_LOG`. Error records carry the
  exception class name and, for project-controlled errors only, the message; never tracebacks or
  response bodies (raw bodies live in `responses.jsonl` only).
- Local server requirements (UDS in a 0700 directory, or `127.0.0.1` plus a Bearer key, same SLURM
  job): see `llm/README.md`.

## Tests

`.venv/bin/python -m pytest tests/translate tests/policy tests/sandbox -q` (via
`slurm/validate.sbatch subset`) runs offline: no network, GPU, paid call or real key.
