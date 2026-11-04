# `mtd_rl_xavier.translator`: two-layer LLM Translator

Turns a LAG trajectory into natural-language intent statements, one target at a time (D3, D7).
Two layers alternate in time order (D9):

- the **detector layer** asks a System One model, on a fixed tick grid, whether the target's
  motion is still explained by its previous intent;
- the **translation layer** asks an OpenAI model for a new intent whenever a trigger fires.

The authoritative output is `events.jsonl`. The judge (`evaluation/`) consumes it as is, so the
trajectory the judge shows is byte-identical to what the translation layer saw (D1, D20).

| Module | Responsibility |
|---|---|
| `schema.py` | Frozen scenario, target, trigger, translation and state types; event-id and record helpers |
| `config.py` | Strict run-config and scenario YAML loading (exact key sets, no default model) |
| `detector.py` | `DetectionModel` Protocol, `DetectorLayer`, exact state string, budget bound, System One adapter |
| `translation.py` | `IntentModel` Protocol, `TranslationLayer`, prompt, strict output schema and parser, OpenAI adapter |
| `pipeline.py` | Episode facts, scenario checks, deterministic schedule, budget preflight, the `Translator` loop |
| `cli.py` | The only composition root: gate, preview, client construction, run directory, provenance, sink |

The layers depend on Protocols only. `cli.py` alone builds `SystemOneClient` and the OpenAI SDK
client. The production adapters are `SystemOneDetectionModel(client, question_id=...)` and
`OpenAIIntentModel(OpenAIResponsesClient(sdk_client), model=..., reasoning_effort=...,
max_output_tokens=...)`; tests run them over `httpx.MockTransport` and a scripted SDK stand-in.

## Command

```bash
# Preview (default): no client, no credential, no network. Writes run.json + preview.json.
uv run python -m mtd_rl_xavier.translator.cli --scenario configs/translator/scenario.example.yaml

# Real run with the default config (local detector + paid OpenAI translation layer). Keys come
# from the environment only. Never type them on the command line (shell history, sbatch files):
# keep them in a 0600 file outside the repository and export them from it, for example
#   set -a; . ~/.config/mtd-rl-xavier/keys.env; set +a     # defines VLLM_API_KEY and OPENAI_API_KEY
# or type them hidden: read -rs OPENAI_API_KEY && export OPENAI_API_KEY
uv run python -m mtd_rl_xavier.translator.cli \
  --scenario private/translator/<scenario>.yaml --execute --confirm-paid-api
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Run config (default `configs/translator/run.yaml`) |
| `--scenario PATH` | Scenario YAML (required) |
| `--output-root DIR` | Overrides `output.root` |
| `--run-name NAME` | Run directory name (default `<UTC stamp>_<scenario_id>_<preview|execute>`). Must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`: a name, never a path; anything else is a usage error (exit 2) before anything is read or created |
| `--execute` | Call the layers. Alone it is enough only for a config without a paid layer |
| `--confirm-paid-api` | Required together with `--execute` when any layer is paid (`openai_responses`) |

Gate: `--confirm-paid-api` without `--execute` is a usage error. `--execute` without
`--confirm-paid-api` is a usage error when the config has a paid layer (every valid config in this
phase has one, because the translation layer is `openai_responses` only).

Execute order: gate -> config, scenario and hash-verified episode -> schedule and detector budget
preflight -> detector client (reads the key variable named in the config) -> authenticated
`GET /v1/models` health check -> new run directory -> OpenAI client (`make_openai_client`, the
only reader of `OPENAI_API_KEY`) -> pipeline. A missing or wrong detector fails before any paid
client exists.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Preview written, or run `completed` |
| 1 | Deterministic failure: config, scenario, episode data, rendering, budget, existing run directory, local server, protocol, refusal, incomplete response, invalid translation output |
| 2 | Usage or gate misuse (argparse), including an invalid `--run-name` |
| 3 | `uncertain`: a call may have reached the server and been billed (timeout, 5xx, unreadable response, unexpected exception or Ctrl-C during a call, or a sent call whose response could not be journaled). Never resend it blindly. Kept even if closing the journals or the detector client, or writing the final `run.json`, fails |
| 4 | `cap_reached`: a call cap stopped the run; `events.jsonl` is partial |

## Run directory

`<output.root>/<run name>/`, created with `mkdir(exist_ok=False)` and mode `0700`; an existing
directory is refused. Nothing in it is ever rewritten except `run.json` (status updates).

| File | Content |
|---|---|
| `run.json` | Mode, status, start/end time, input paths + SHA-256 (config, scenario, episode CSV, manifest), git commit + dirty flag, SHA-256 of every `.py` in `translator/`, `lag/`, `llm/`, renderer versions, detector state version, question-set SHA-256, prompt version, OpenAI base URL, system-prompt and output-schema SHA-256, resolved config (detector endpoint included), schedule, budget fit, call counts, `unsent` (cap), `error` (type + message only), `api_requests_made` |
| `preview.json` | Preview only: schedule, caps and headroom, warnings, budget fit, approximate token and cost estimate, `api_requests_made: 0` |
| `example_translation_request.json` | Preview only: the exact Responses body of the first mandatory translation |
| `events.jsonl` | The event log (contract below) |
| `requests.jsonl` | One record per call, written and fsynced **before** the call is sent: `call_id`, `layer`, `step`, `target`, exact `payload` (System One body or Responses body), `written_at` |
| `responses.jsonl` | One record per sent call: `call_id`, `layer`, `status` (`ok` / `error` / `uncertain`), `raw` response body (kept even when parsing fails), `error` (`type`, `message`), `written_at` |

A request without a response line means the process died mid-call: treat that call as uncertain.
All three JSONL files are standard JSON: a non-finite number in a received body (a lax server's
`NaN`) is stored as the string `"NaN"`, `"Infinity"` or `"-Infinity"`; nothing else is changed.

Failure recording: once the run directory exists every failure ends with a terminal `run.json`
status and the call counts. Error records hold `{type, message}`. The message is kept for errors
whose text this project or the LLM parts control (`ValueError`, `OSError`, sanitised client
errors); any other exception records only its class name. No traceback is written or printed.
Ctrl-C is handled the same way: during a call the call is journaled `uncertain` (exit 3); with no
call in flight the run ends `failed` (exit 1). A lone surrogate in a received body (an escaped
`"\ud800"`) is journaled as the JSON escape `\ud800`, so a complete response is never reported
`uncertain` just because it could not be encoded.

## Pipeline rules

The unit of translation is one exact target: `aircraft:<id>` or `team:<id>` (D17). Per step
`0 .. last_complete_step` (340 / 293 / 495 for the committed episodes):

1. Instructions issued at this step are applied (in `instruction_id` order). A later instruction
   replaces the target's current one; the target keeps its previous translation (D21).
2. Every instructed target that is down is closed once (`target_closed`). Aircraft: first step with
   `alive == 0`. Team: the step its last aircraft is down. A closed target never reopens (D37).
3. Targets are processed in canonical `(kind, id)` order (aircraft before team):
   - scenario triggers: `start` (instruction at step 0), `instruction` (instruction at step > 0,
     D24), `external_change` (D6), `request` (D4, D10);
   - a closed target only logs its triggers as `trigger_suppressed` (no model call);
   - on a tick (`step > 0` and `step % tick_interval_steps == 0`, a fixed grid that requests never
     move, D28) a target that already has a translation gets exactly one detector call, even when
     other triggers exist (D9); `probability >= threshold` adds a `detector_change` trigger;
   - if any trigger remains, the target gets exactly one translation. Coinciding reasons are all
     kept; the primary trigger follows `start > instruction > external_change > detector_change > request`.
4. A target first instructed after step 0 is translated at its issue step with trigger
   `instruction` and no previous translation; its detector ticks start after that (D35).

Windows (online mode, D33): translation `[max(0, step-59), step]` with `lag17-jsonl` (D22, D32);
detector `[max(0, step-9), step]` with `lag-detector-compact` over the target's aircraft, then the
opposing team's aircraft (D27). Downed aircraft keep rendering with `alive == 0`.

Caps: immediately before each call, if the layer's sent count equals its cap, nothing is sent and
the run stops with `cap_reached`; `run.json.unsent` names the layer, target, step and cap. The
preview reports the exact detector call count (detector calls never depend on model outputs), the
mandatory translations, the translation upper bound (mandatory plus one per tick without a scenario
trigger) and the headroom; a bound above a cap is only a warning.

Errors: a failed call is never retried. The response line records the error, the run stops, and
`run.json` gets `failed` or `uncertain`. If journaling itself fails while a sent call has no
response line yet, the run stops as `uncertain` with that call's id.

## Detector layer

State string sent to `/v1/systemone` (exact; logged as `state_text`):

```text
Target: team A (aircraft A0100, A0200). Opponents: B0100, B0200.
Previous intent translation for this target (from step <s>):
<the target's own previous intent; evidence is not included>

Trajectory window: steps <a>-<b>, 0.2 s per step, one JSON line per step, newest last; relative_step = step - <b>. Fields: <fields>. null = unavailable.
<compact window text>
```

- `float_precision: 5` (about 1 m). With 3 decimals, lon/lat resolve about 100 m while an aircraft
  moves about 50 m per step: in `mhi_v1.3_ic100601_win`, 45 of 340 consecutive A0100 positions are
  identical at 3 decimals and none at 5 (`tests/translator/test_tr_budget.py`).
- Budget fit is proved before any call: for every scheduled tick, the client's conservative bound
  (UTF-8 bytes + fixed reserves) of the exact state framing plus a worst-case intent
  (4 bytes x `translation.max_intent_chars`) must stay below `input_token_budget`. With the default
  config the worst bound over every tick of all three episodes is 14310 (team targets, margin 2074)
  and 12334 (single aircraft, margin 4050) against 16384. Of that, 1407 is the question and
  framing reserve and 4000 the intent reserve.
- `input_token_budget` must equal the server's real cap, `min(16384, --max-model-len)` for
  `vllm serve Cloudflare/clef-flash` (4096 for OpenJev `wev serve`); the config rejects > 16384.
- `threshold: 0.65` is provisional until a labelled LAG check.

## Translation layer

- Prompt version `translator-intent-v1`. The system prompt describes LAG, the data format and the
  task; the user message carries the episode alias (never the episode id, scenario id or event
  ids), the step and time, the target and its aircraft, every instruction in effect (the target's
  own one marked), the target's previous translation (intent and evidence) or "none", and the
  17-format window verbatim between `BEGIN WINDOW DATA` / `END WINDOW DATA`.
- Output: strict JSON schema `intent_translation` = `{"intent": string, "evidence": [string, ...]}`
  (`minItems: 1`, no other keys). The parser accepts exactly one JSON object with exactly those
  keys, a non-blank intent of at most `max_intent_chars` characters and a non-empty list of
  non-blank evidence strings, all valid UTF-8 (a lone surrogate escape is rejected). Nothing is
  repaired, coerced or retried; text is kept unchanged.
- Model, `reasoning_effort`, `max_output_tokens`, `timeout_s` and the output language come from
  the config only. `store` is always false; no temperature or sampling parameter is sent.
- The committed config sets `translation.output_language: Japanese` (D42). clef-flash is
  English-only and receives the previous (now Japanese) translation in its state. The detector
  budget stays proved for any language: the intent reserve is 4 UTF-8 bytes per character.

## `events.jsonl` contract (schema_version 1)

One JSON object per line, in emission order. Four record types only. Every record has:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `sequence` | 1, 2, 3, ... without gaps, in file order |
| `record_type` | `translation_event`, `detector_call`, `target_closed`, `trigger_suppressed` |
| `scenario_id`, `episode_id` | From the scenario |
| `step`, `time_s` | Step and simulation seconds (`round(step * 0.2, 1)`) |
| `target` | `{"kind": "aircraft" | "team", "id": "<id>"}` |

Trigger reason object: `{"kind", "source", "reference_id", "probability"}`. `kind` is one of
`start`, `instruction`, `external_change`, `detector_change`, `request`. `source` is `scenario`
(start / instruction), `detector`, or the scenario's request / change source label.
`reference_id` is the instruction, request or change id, or the detector `call_id`.
`probability` is set only for `detector_change`. Lists are in precedence order.

### `translation_event`

| Field | Meaning |
|---|---|
| `event_id` | `<scenario_id>:<kind>:<target_id>:<step:06d>`; unique (one translation per target per step); also the `call_id` in `requests.jsonl` / `responses.jsonl` |
| `request_sha256` | SHA-256 of the canonical JSON (sorted keys, `,`/`:` separators, UTF-8) of the exact request `payload` journaled under `call_id == event_id`: the full system prompt and user message |
| `episode` | `{episode_id, alias, csv_sha256}` |
| `primary_trigger` | Trigger kind with the highest precedence |
| `trigger_reasons` | Every coinciding reason |
| `input.prompt_version` | `translator-intent-v1` |
| `input.target_aircraft` | The target's aircraft ids, as named in the prompt |
| `input.instructions_in_effect` | Every instructed target's current instruction, canonical target order: `{instruction_id, issue_step, target, text}` |
| `input.primary_instruction_id` | The instruction of this exact target |
| `input.trajectory` | `{renderer, renderer_version, start_step, end_step, anchor_step, object_ids, fields, text, sha256}`: `text` is the exact window text in the prompt; steps are the clipped, actually rendered ones |
| `input.previous_translation` | `{event_id, step, intent, evidence}` of this target, or `null` |
| `input.output_language` | Configured output language |
| `output` | `{intent, evidence}` as parsed |
| `model` | `{backend, model, reasoning_effort, max_output_tokens, response_model, response_id}` |
| `usage` | `{input_tokens, cached_input_tokens, output_tokens, reasoning_tokens}` |

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
| `input.state_text` | The exact state string sent |
| `input.question` | Full question object `{id, type, instructions, criteria: {true, false}}` |
| `output` | `{probability, threshold, changed}`; `changed = probability >= threshold` |
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

### Example (trajectory and state text truncated)

```json
{"schema_version": 1, "sequence": 6, "record_type": "translation_event", "scenario_id": "placeholder-episode-01",
 "episode_id": "mhi_v1.3_ic100601_win", "step": 10, "time_s": 2.0, "target": {"kind": "aircraft", "id": "A0100"},
 "event_id": "placeholder-episode-01:aircraft:A0100:000010", "request_sha256": "...",
 "episode": {"episode_id": "mhi_v1.3_ic100601_win", "alias": "episode_01", "csv_sha256": "812a44c4..."},
 "primary_trigger": "request",
 "trigger_reasons": [{"kind": "request", "source": "placeholder-human", "reference_id": "placeholder-request-001", "probability": null}],
 "input": {"prompt_version": "translator-intent-v1", "target_aircraft": ["A0100"],
           "instructions_in_effect": [
             {"instruction_id": "placeholder-aircraft-001", "issue_step": 0, "target": {"kind": "aircraft", "id": "A0100"}, "text": "[PLACEHOLDER] Perform the assigned single-aircraft task."},
             {"instruction_id": "placeholder-team-001", "issue_step": 0, "target": {"kind": "team", "id": "A"}, "text": "[PLACEHOLDER] Perform the assigned team task."}],
           "primary_instruction_id": "placeholder-aircraft-001",
           "trajectory": {"renderer": "lag17-jsonl", "renderer_version": "1", "start_step": 0, "end_step": 10, "anchor_step": 10,
                          "object_ids": ["A0100", "A0200", "B0100", "B0200"], "fields": ["object_id", "...", "missiles_remaining"],
                          "text": "{\"step\":0,\"time_s\":0.0,\"relative_step\":-10,\"entities\":[...]}\n...", "sha256": "..."},
           "previous_translation": {"event_id": "placeholder-episode-01:aircraft:A0100:000000", "step": 0,
                                    "intent": "[PLACEHOLDER] Initial intent.", "evidence": ["[PLACEHOLDER] Evidence."]},
           "output_language": "Japanese"},
 "output": {"intent": "[PLACEHOLDER] Updated intent.", "evidence": ["[PLACEHOLDER] Evidence."]},
 "model": {"backend": "openai_responses", "model": "gpt-6.1-terra", "reasoning_effort": "medium", "max_output_tokens": 4000,
           "response_model": "gpt-6.1-terra", "response_id": "resp_..."},
 "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}}
```

## Config and scenario

`configs/translator/run.yaml` is the full default; every key is required, unknown keys fail and
duplicate YAML keys fail (in the scenario too).
Phase limits: `pipeline.visibility.mode` accepts only `online` (with `future_steps: 0`);
`retrospective` and `translation.backend: vllm_chat` are rejected as "not implemented in this
phase" (LATER, L2). `pricing` must have an entry for `translation.model` (preview estimates only).

`configs/translator/scenario.example.yaml` documents the scenario format with placeholder text.
Scenario rules: exact keys; ids unique across instructions, requests and change points; at least
one instruction at step 0 (D16); at most one instruction per exact target per step; every request
or change point references a target already instructed at or before its step; all steps within
`[0, last_complete_step]`; aircraft and team ids must exist in the episode; `episode.alias` must
differ from `episode_id`. Paths resolve relative to the scenario file. `pipeline.validate_scenario`
re-runs the same rules, so a `Scenario` built in code is held to them too.

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
- Outputs go under `outputs/translator/` (gitignored as a whole tree). Event logs, requests and
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
  job): see `src/mtd_rl_xavier/llm/README.md`.

## Tests

`uv run pytest tests/translator -q` runs offline: no network, GPU, paid call or real key.
`test_tr_budget.py` renders every tick of all three episodes (about 17 s).
