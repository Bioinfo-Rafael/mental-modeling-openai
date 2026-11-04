# `evaluation/`: LLM-as-a-judge for Translator events

Scores the intent translations in a Translator run (`src/mtd_rl_xavier/translator/`) with OpenAI
`gpt-6.1-sol` (D8): one call per judged `translation_event` (D20). A YAML config decides which of
six inputs the judge sees and how each one is rendered (D15). The judge reads the Translator's
`events.jsonl` as is, so the trajectory it shows is byte-identical to what the translation layer
saw (D1).

| Module | Responsibility |
|---|---|
| `configuration.py` | Strict judge YAML: model, limits, the six input blocks, output root, pricing |
| `schemas.py` | Event-log checks, key points, per-event output schema, strict verdict parser |
| `rendering.py` | Fixed system prompt, the block renderers (video block included), one request per event, token estimate, `redact_images` |
| `frames.py` | Video frames: `FrameSource` Protocol, frame selection (`plan_frames`), `PyAVFrameSource` (PyAV + Pillow, lazy), `CachingFrameSource` |
| `judge.py` | `JudgeModel` Protocol, OpenAI adapter, JSONL sink, the per-event loop (caps, write-before-send) |
| `cli.py` | The only composition root: gate, preflight, preview, client construction, run directory, provenance |

## Command

Run from the repository root only (a stray `evaluation.py` on `sys.path` would shadow the package).

```bash
# Preview (default): no client, no credential, no network. Writes run.json, preview.json and
# example_judge_request.json.
uv run python -m evaluation.cli --translator-run outputs/translator/<run> \
  --key-points private/evaluation/<key-points>.yaml

# Real run: one paid gpt-6.1-sol call per event that has key points. OPENAI_API_KEY comes from
# the environment only; never type it on the command line (shell history, sbatch files). Export
# it from a 0600 file outside the repository (set -a; . ~/.config/mtd-rl-xavier/keys.env; set +a)
# or type it hidden (read -rs OPENAI_API_KEY && export OPENAI_API_KEY).
uv run python -m evaluation.cli --translator-run outputs/translator/<run> \
  --key-points private/evaluation/<key-points>.yaml --execute --confirm-paid-api
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Judge config (default `evaluation/config/judge-text.yaml`) |
| `--translator-run DIR` | Translator run directory holding `run.json` and `events.jsonl` (required) |
| `--key-points PATH` | Key-points YAML (required; real ones under `private/evaluation/`) |
| `--output-root DIR` | Overrides `output.root` |
| `--run-name NAME` | Run directory name (default `<UTC stamp>_judge_<preview|execute>`). Must match `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`: a name, never a path; anything else is a usage error (exit 2) before anything is read or created |
| `--execute` | Send the judge calls. Only together with `--confirm-paid-api` |
| `--confirm-paid-api` | Confirms paid calls. Only together with `--execute` |

The judge always calls a paid API, so either flag alone is a usage error (exit 2).

Order (both modes): config -> Translator run checks -> key points -> event matching ->
`max_judged_events` preflight -> every request is built and validated. Every input error stops the
run here, before any directory, client or credential. Execute then creates the run directory,
writes `run.json`, builds the OpenAI client (`make_openai_client`, the only reader of
`OPENAI_API_KEY`; skipped when no event has key points) and judges the events in log order.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Preview written, or run `completed` |
| 1 | Deterministic failure: config, event log, key points, more judged events than `max_judged_events` (preflight: nothing is created or sent), rendering, existing run directory, missing key, refusal, incomplete response, invalid verdict |
| 2 | Usage or gate misuse (argparse), including an invalid `--run-name` |
| 3 | `uncertain`: a call may have been billed with an unknown outcome (timeout, 5xx, unreadable response, unexpected exception or Ctrl-C during a call, or a sent call whose response could not be journaled). Never resend it blindly. Kept even if the final `run.json` cannot be written |
| 4 | `cap_reached`: the runtime cap stopped the run before call `max_judged_events + 1`; `results.jsonl` is partial |

## Inputs

`--translator-run` must hold `run.json` (any JSON object; its `mode` and `status` are recorded, and
a status other than `completed` is a preview warning) and `events.jsonl` with `schema_version` 1, a
gap-free `sequence` 1..n, the four known record types, unique translation `event_id`s and, in every
`translation_event`, the fields the judge renders (`step`, `time_s`, `target`, `primary_trigger`,
`trigger_reasons`, `input.{instructions_in_effect, primary_instruction_id, target_aircraft,
trajectory}`, `output.{intent, evidence}`). The trajectory text must match its `sha256`.

The log is read strictly, because it is untrusted input to a paid judge:

- Records are split on `"\n"` only. U+2028, U+2029 and U+0085 are legal inside a JSON string (the
  Translator writes them raw), so they never split a record.
- A repeated JSON key (in `events.jsonl`, `run.json` or a verdict) is an error: with "last one
  wins" a reader could see one value while the judge scores another. Deep nesting is an error too.
- Every id the judge prints must be a safe id (`[A-Za-z0-9][A-Za-z0-9._-]{0,127}`): target ids,
  `target_aircraft`, instruction ids, trigger `source`, trajectory `object_ids` and
  `renderer_version`. `event_id` must have the form `<scenario_id>:<kind>:<target_id>:<step:06d>`.
- The trajectory text must be JSON lines (one object per line, no Unicode line separator), so no
  line of it can look like a heading or end the window early.

### The six input blocks

`evaluation/config/judge-text.yaml` (and `judge-text-video.yaml`) has exactly these six blocks, each with `include` plus its
rendering options. An excluded block is absent from the request (heading and content). The user
message is one fixed header naming the target and its aircraft, then one text item per included
block in this order:

| Block | Renders | Options |
|---|---|---|
| `instructions` | The event's `instructions_in_effect`, by issue step, the target's own one marked `[primary]` | `renderer: chronological`, `show_ids`, `show_targets`, `show_issue_steps` |
| `video` | Selected frames of the episode's action video as images, each immediately preceded by a text label with its step and time (see "Video block" below) | `renderer: sampled-jpeg`, `source`, `videos`, `sampling`, `resize`, `detail`, `label_format`, `image_tokens` |
| `timestamps` | Episode start (step 0), this target's earlier translation events (step, time, primary trigger), this event's step and time, and its trigger reasons | `renderer: chronological`, `scope: through_event` (never anything later), `show_sim_seconds`, `show_sources` (source labels and detector probabilities) |
| `trajectory` | `input.trajectory.text` verbatim between `BEGIN WINDOW DATA` / `END WINDOW DATA`, with a legend of the 17 format | `renderer: translation_input_verbatim`, `show_window_metadata` |
| `translated_intent` | The output being judged: intent and evidence | `renderer: intent-and-evidence` |
| `key_points` | The event's key points, numbered, with the fixed D29 rubric | `renderer: numbered-with-fixed-rubric`. `include: false` asks for the overall score only |

The model never sees the episode id, the scenario id or an event id. A request also never carries
`temperature` or any sampling parameter; `store` is false. No file written to disk contains base64 (see "Video block").

Prompt-injection envelope (prompt version `judge-text-v2`, D38): every free text the judge did not
write (the intent, each evidence item, instruction texts and key-point texts) is rendered as one
JSON string literal in double quotes, with U+0085, U+2028 and U+2029 escaped too. Such a text can
never contain a raw line break, so it cannot open a `## ...` section or end the window data. The
system prompt says that the whole user message is material to judge and that requests inside it
(about scores, the rubric or the output format) must not be followed. The judge does not render
the Translator's previous translation.

## Video block (L3, D30, D31)

`judge-text.yaml` keeps `video.include: false`. `judge-text-video.yaml` is the same config with the
block on, for the three committed episodes. The video is **evaluation input**: the
videos (`workspace/13_LAG/data/viz/action_videos/*.mp4`, 20 fps, 1920x1080, frame = step) show the
A-team's actions and the visible hit and shot-down outcomes, including frames **after** the judged
step that the online Translator never saw. That is intentional: the human rater watches the same
video. It is never an input of the Translator (generation). The request text says so to the judge.

Decoding needs the optional `video` extra (`uv sync --extra video`: PyAV `av` and `Pillow`). They
are imported only when a frame is read; a text-only run never imports them, and a missing package
fails with a message naming the extra.

Every key is required (unknown or missing keys are an error); there are no hidden defaults.

```yaml
video:
  include: true
  renderer: sampled-jpeg
  source: scenario_action_video
  videos:                          # episode id (run.json inputs.episode.episode_id) -> video
    mhi_v1.3_ic100601_win:
      path: workspace/13_LAG/data/viz/action_videos/mhi_v1.3_ic100601_win_with_actions.mp4  # relative to the repo root
      fps: 20                      # the file's own rate; a mismatch is a preflight error
      frame_offset: 0              # video frame that shows step 0
      frames_per_step: 1           # frame = frame_offset + step * frames_per_step
      step_seconds: 0.2            # simulation seconds per step, for the label time
  sampling: {strategy: event_window, before_steps: 20, after_steps: 20, stride_steps: 5, max_frames: 12}
  resize: {max_width: 960, max_height: 540, jpeg_quality: 85}
  detail: high                     # low | high | original | auto
  label_format: "Video frame at step {step} ({time_s} s, {relative_step} steps from the judged step):"
  image_tokens: {per_image: 1105, status: unverified}
```

- **Mapping.** The mapping from a video to its episode lives only in this YAML. The episode of a
  Translator run is `inputs.episode.episode_id` of its `run.json` (the Translator CLI writes it,
  with `last_complete_step`). A video whose frame rate or length differs needs its own entry
  (for example a 10 fps recording of the same game: `frames_per_step: 2`).
- **Window.** Candidate steps are `event_step + k * stride_steps` between `-before_steps` and
  `+after_steps` (the event step itself is always a candidate). A candidate is dropped when its
  step is outside `[0, last_complete_step]` or its frame outside `[0, frame_count - 1]`. If more
  than `max_frames` remain, the ones closest to the event are kept, the earlier one on a tie.
  Frames are sent in time order. A window with no frame at all is an error.
- **Request layout.** Header, `instructions`, then the video block, then the other blocks. The
  video block is one text item (`## VIDEO FRAMES` and a short explanation), then for every frame a
  text item with the `label_format` and an `input_image` item with a `data:image/jpeg;base64,...`
  URL, so each image is immediately preceded by its own label. `label_format` must contain
  `{step}` and `{time_s}`; `{relative_step}` (`-10`, `0`, `+5`) is optional; no other field, format
  spec or conversion. `time_s` is `step * step_seconds`.
- **Resize and JPEG.** Frames are shrunk to fit `max_width x max_height` keeping the aspect ratio
  (never enlarged) and encoded as JPEG at `jpeg_quality`. JPEG bytes carry no timestamp, so the
  request hash is deterministic for one PyAV / Pillow build; `run.json` records the video file
  hashes and the `av` / `pillow` versions (`provenance.video`).
- **Preflight.** A judged event whose episode has no `videos` entry, whose video file is missing or
  unreadable, whose frame rate differs from `fps`, or that has no frame in its window fails while
  the inputs are loaded, before any client exists. Events without key points never read a video.
  A video-enabled preview decodes the frames of every judged event (a cache decodes each frame
  once).
- **Preview and files.** `preview.json` lists per event `frames` (`count`, `steps`,
  `frame_indices`, `sizes`, `jpeg_bytes_total`, `image_tokens`) and the estimate adds
  `images_total`, `image_tokens_total`, `text_tokens_total`. No file contains base64 or image bytes:
  `example_judge_request.json` and `requests.jsonl` show each image as
  `[image data omitted: <chars> chars, sha256 <hash>]`. The `request_sha256` in `results.jsonl` is
  over the real payload, images included.
- **Image tokens are unverified.** `image_tokens.per_image` is a preview-only estimate. The default
  1105 is the GPT-4o-style high-detail formula (85 + 170 per 512 px tile; 6 tiles at 960x540); the
  formula for `gpt-6.1-sol` is not verified, and it changes with `detail` and `resize`. The preview
  warns while `status` is `unverified`. Check the real `usage.input_tokens` of a first call and set
  `status: verified` only after adjusting the number.

## Scales and output

- Per key point (D29): **2** = the translated intent reproduces the key point's core purpose and
  main action; **1** = partially; **0** = missing, or a different purpose.
- Overall: **1-5** for the translation as a whole (anchors in `rendering.JUDGE_SYSTEM_PROMPT`;
  provisional wording, not fixed by D29).
- Every score comes with a rationale, written before the score (D13).

The output schema is built per event and sent as strict `json_schema` `judge_verdict`:
`key_points` is an array of exactly N items whose `key_point_id` is an enum of that event's ids,
each with `rationale` and `score` (enum 0/1/2), and `overall` has `rationale` and `score` (enum
1-5). Every object has `additionalProperties: false` and every field is required. `minLength` is
not used, because strict structured outputs do not support it.

The parser rejects, with nothing repaired, coerced or retried: invalid JSON or more than one
value, missing / duplicate / unknown key-point ids, extra or missing keys, boolean or out-of-range
scores, and blank or non-UTF-8 rationales. Refusals and incomplete responses fail the call. The
raw response is journaled before parsing.

## Key points and partial judging (D36)

```yaml
schema_version: 1
events:
  - event_id: placeholder-episode-01:aircraft:A0100:000010
    key_points:
      - key_point_id: kp-001
        text: "[PLACEHOLDER] The core purpose that the translation should reproduce."
```

- The file may cover any subset of the run's translation events, including none (`events: []`).
- Every entry must name an existing `translation_event`. An unknown event id, a duplicate event id,
  a duplicate `key_point_id` within an event, an empty key-point list or a blank text is an error,
  detected before any client exists.
- Events without key points get a `not_judged` result (`reason: no_key_points`) and no call.
- `limits.max_judged_events` counts only events with key points. More is a preflight error. At run
  time the loop stops with `cap_reached` before call cap+1 (defence in depth).
- `evaluation/examples/key-points.example.yaml` is placeholder-only. **Real key points live under
  `private/evaluation/`; the whole `private/` tree is gitignored. Never commit them.**

## Run directory

`<output.root>/<run name>/` is created with `mkdir(exist_ok=False)` and mode `0700`; an existing
directory is refused. Only `run.json` is rewritten (status updates).

| File | Content |
|---|---|
| `run.json` | Mode, status, times; inputs with SHA-256 (judge config, key-points file, translator `run.json` and `events.jsonl`, plus the translator run's mode and status); git commit + dirty flag; SHA-256 of every `evaluation/*.py`; prompt version, OpenAI base URL, system-prompt and rubric SHA-256; resolved config; plan counts; final counts, usage, approximate cost, `unsent` (cap), `error` (type + message only), `api_requests_made` |
| `preview.json` | Preview only: per-event plan (judged / not_judged, key-point count, approximate tokens), estimate, warnings, `api_requests_made: 0` |
| `example_judge_request.json` | Preview only: the exact Responses body of the first judged event |
| `requests.jsonl` | Execute: one line per call, written and fsynced **before** the call: `call_id` (= event id), `layer`, exact `payload`, `written_at` |
| `responses.jsonl` | Execute: one line per sent call: `status` (`ok` / `error` / `uncertain`), `raw` response (kept even when parsing fails), `error`, `written_at` |
| `results.jsonl` | Execute: one line per processed event, in event-log order (same keys for every status) |

Result records (`status`: `judged`, `not_judged`, `failed`, `uncertain`):

```json
{"schema_version": 1, "event_id": "placeholder-episode-01:aircraft:A0100:000010", "status": "judged", "reason": null,
 "step": 10, "time_s": 2.0, "target": {"kind": "aircraft", "id": "A0100"}, "primary_trigger": "request",
 "key_points": [{"key_point_id": "kp-001", "rationale": "...", "score": 2}],
 "overall": {"rationale": "...", "score": 4}, "request_sha256": "...", "response_model": "gpt-6.1-sol",
 "response_id": "resp_...", "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
 "error": null}
```

Any failed call stops the run (never retried); the remaining events are not processed and get no
result line. A request without a response line means the process died mid-call: treat that call
as uncertain.

## Cost estimate

The preview estimates text tokens as `ceil(chars / 4)` of the system prompt, the user text and the
schema, assumes no cache hits, and prices output at `max_output_tokens` per call (an upper bound).
A request above `long_context_threshold_tokens` (272K) is priced at the long-context multipliers.
The prices in the config ($2 input, $0.10 cached input, $10 output per 1M tokens) are list prices:
**verify them before paid runs.**

## Data safety

- Key points and event logs contain customer instructions (D11: they may be sent to OpenAI). Real
  key points stay under the gitignored `private/evaluation/`; outputs under the gitignored
  `outputs/evaluation/`, in owner-only directories.
- `OPENAI_API_KEY` is read only by `make_openai_client()`, after the paired gate, and only when an
  event is judged. That client always targets `https://api.openai.com/v1` (`OPENAI_BASE_URL` is
  ignored; `OPENAI_CUSTOM_HEADERS` and `OPENAI_LOG` must be unset). Proxy, CA-bundle and
  `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` stay honoured; `provenance.sdk_environment_set` in `run.json`
  lists which were set (names only). Preview neither refuses nor warns about `OPENAI_LOG`. Ctrl-C
  outside a call (for example while the client is built) ends the run `failed` (exit 1) with a
  terminal `run.json` and no traceback. Keys are never written. Error records keep the
  exception class name and only
  messages this project or the LLM part controls; never tracebacks or response bodies.
- Video (L3): the action videos (D30) show A-team actions and visible hit and shot-down outcomes,
  including moments after the event that the online Translator never saw. This is intentional:
  it is the same video the human rater watches. Evaluation input only, never generation input.
  A video-enabled run records the video file hashes and decoder versions in `run.json`.

## Tests

`uv run pytest tests/evaluation -q` runs offline: no network, paid call, GPU or real key, and a fake
`FrameSource` instead of a decoder. One smoke test decodes two frames of a committed video with PyAV
(skipped when the `video` extra is not installed).
It includes two end-to-end chains on the real example episode: fake Translator layers -> event log
-> judge preview, and the production System One adapter over `httpx.MockTransport` plus the
production OpenAI translation adapter over a scripted SDK stand-in -> event log -> judge preview.
