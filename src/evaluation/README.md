# `evaluation/`: LLM-as-a-judge for Translator events

Scores the intent translations in a translation run (`src/translate/`, any configuration) with OpenAI
`gpt-6.1-sol` (D8): one call per judged `translation_event` (D20). A YAML config decides which of
six inputs the judge sees and how each one is rendered (D15), and names the prompt file that holds
every word of the request (`configs/evaluation/prompts/`, see "Prompt files" below). The judge reads
the Translator's `events.jsonl` as is: the trajectory it shows is the human-visible window each
event records, the same for every translation configuration (D34).

`Dn` numbers refer to the requirement decision log
[`.claude/docs/research/translator-requirements.md`](../.claude/docs/research/translator-requirements.md).

| Module | Responsibility |
|---|---|
| `configuration.py` | Strict judge YAML: model, limits, the six input blocks, output root, pricing, prompt file |
| `prompt_files.py` | Judge prompt files of both judges: strict loading, the fields each template may and must use |
| `strict_yaml.py` | The safe YAML loader that rejects repeated keys (judge configs, prompt files) |
| `trigger_kinds.py` | The trigger kinds of event schemas 2 and 3 (an explicit table; nothing is inferred) |
| `schemas.py` | Event-log checks, key points, per-event output schema, strict verdict parser |
| `rendering.py` | The material of one event rendered into the prompt file's texts (video block included), one request per event, token estimate, `redact_images` |
| `frames.py` | Video frames: `FrameSource` Protocol, frame selection (`plan_frames`), `PyAVFrameSource` (PyAV + Pillow, lazy), `CachingFrameSource` |
| `judge.py` | `JudgeModel` Protocol, OpenAI adapter, JSONL sink, the per-event loop (caps, write-before-send) |
| `cli.py` | The only composition root: gate, preflight, preview, client construction, run directory, provenance |

## Command

`evaluation` is installed with the core (`sbatch slurm/uv_sync.sbatch`), so `python -m evaluation.*` works
from any directory. On Slurm use `slurm/judge_keypoints.sbatch` and `slurm/judge_assess.sbatch`
(`preview` by default, `paid` explicitly; see `slurm/README.md`).

```bash
# Preview (default): no client, no credential, no network. Writes run.json, preview.json and
# example_judge_request.json.
uv run python -m evaluation.cli --translator-run outputs/translate/<run> \
  --key-points private/evaluation/<key-points>.yaml

# Real run: one paid gpt-6.1-sol call per event that has key points. OPENAI_API_KEY comes from
# the environment only; never type it on the command line (shell history, sbatch files). It lives
# in the gitignored, group-shared repo-root .env: load it right before the run (set -a; . ./.env;
# set +a) and unset it afterwards. On Slurm: slurm/judge_keypoints.sbatch paid (slurm/README.md).
uv run python -m evaluation.cli --translator-run outputs/translate/<run> \
  --key-points private/evaluation/<key-points>.yaml --execute --confirm-paid-api
```

| Flag | Meaning |
|---|---|
| `--config PATH` | Judge config (default `configs/evaluation/judge-text.yaml`) |
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
a status other than `completed` is a preview warning) and `events.jsonl` with one `schema_version`
(2 or 3; trigger kinds checked against `trigger_kinds.py`, shown to the judge as recorded), a
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

`configs/evaluation/judge-text.yaml` (and `judge-text-video.yaml`) has exactly these six blocks, each with `include` plus its
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

Prompt-injection envelope (since prompt version `judge-text-v2`, D38): every free text the judge did not
write (the intent, each evidence item, instruction texts and key-point texts) is rendered as one
JSON string literal in double quotes, with U+0085, U+2028 and U+2029 escaped too. Such a text can
never contain a raw line break, so it cannot open a `## ...` section or end the window data. The
system prompt says that the whole user message is material to judge and that requests inside it
(about scores, the rubric or the output format) must not be followed. The judge does not render
the Translator's previous translation.

## Prompt files

Every word both judges send comes from a prompt file under `configs/evaluation/prompts/`, named by
the judge config's `prompt` key (repo-relative). The code loads the material, renders it into the
file's templates, calls and parses; it holds no prompt text. A prompt file has `schema_version` 1,
`judge` (`assess` or `key_point`), `version` (recorded as `prompt_version`), `texts` (the system
prompt and the user message or its blocks) and `blocks` (item templates, joiners, the text when a
list or an optional piece is empty). It loads strictly: UTF-8, no repeated key, exact keys, and every
`{field}` is checked when the config loads (`prompt_files.py` lists the fields of each template): an
unknown field, or a template that drops judge material (the trajectory, the intent, the criteria,
...), fails before anything is rendered. Literal braces are doubled.

| Config (`configs/evaluation/`) | Prompt | Use |
|---|---|---|
| `assess.yaml` | `assess-judge-v3` | Default for new runs. Calls the trajectory the human-visible telemetry up to t (not what the Translator saw); shows trigger kinds as the config's point kinds |
| `assess-legacy.yaml` | `assess-judge-v2` | Byte-identical to the former in-code prompt. The paid 2026-11-05 assessment resumes only with this config |
| `judge-text.yaml`, `judge-text-video.yaml` | `judge-text-v4` | Default for new runs; same neutral trajectory wording |
| `judge-text-legacy.yaml` | `judge-text-v3` | Byte-identical to the former in-code prompt |

Never edit a legacy prompt file: `tests/fixtures/judge-request-goldens.json` and
`tests/fixtures/prompt-request-goldens.json` pin the requests it builds, and
`tests/fixtures/assess-request-path.sha256` pins the legacy assess config and prompt. Write a new
version instead.

## Video block (L3, D30, D31)

`judge-text.yaml` keeps `video.include: false`. `judge-text-video.yaml` is the same config with the
block on, for the three committed episodes. The video is **evaluation input**: the
videos (`data/mhi/action_videos/*.mp4`, 20 fps, 1920x1080, frame = step) show the
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
      path: data/mhi/action_videos/mhi_v1.3_ic100601_win_with_actions.mp4  # relative to the repo root
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
  translation run is `inputs.episode.episode_id` of its `run.json` (`python -m translate` writes it,
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
- Overall: **1-5** for the translation as a whole. Since `judge-text-v3` (D66) each level is
  anchored to the key-point scores (5 = every key point 2 and no contradiction; 4 = exactly one 1;
  1 = empty, or the main purpose contradicted). Without key points the judge applies the same
  levels to two implied key points: the main action the material shows, and what it observably
  works toward. The score is the lowest level whose condition holds. Wording:
  `rendering.JUDGE_SYSTEM_PROMPT`.
- Every score comes with a rationale, written before the score (D13). It names the level that
  holds and says which part of the next level's condition does not (D67).

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
- `src/evaluation/examples/key-points.example.yaml` is placeholder-only. **Real key points live under
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

## 3 time points x 4 aspects assessment (`assess_cli`)

A second evaluator, separate from the D29 key-point judge above. It rates every translation event of
a translation run on the user's 3 points (開始点 `start`: the forced first translation; 変化点
`change`: the clef-flash detector; 要求点 `request`) x 4 aspects (要素 `elements`, 正しさ
`correctness`, タイミング `timing`, 判断への寄与 `contribution`). The event is judged on the union
of the criteria of all its trigger reasons. Which point kind rates a trigger kind is the judge
config's `triggers` map, per event schema: schema 3 maps one to one; for schema 2 runs (before
2026-11-08) the default config maps `initial` to `start` and the detector's `start` to `change`,
and `assess-legacy.yaml` keeps `initial` and `start` as `start`. A run of a schema the config does
not map fails before anything is written. Machine, human and judge results are stored apart and
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
`outputs/translator/intentcmp-seed100601-ja-A-s045-A0100`, built by the former converter whose
exact bytes are kept in `tests/fixtures/frozen/to_translator_run_v2.py.txt`.

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

Score anchors (`pilot-3` default and `pilot-2` legacy, the same criteria; D65-D67): every criterion has its
own anchor for each level, rendered under the criterion in the packet from 5 down to 1, followed
by its own `cannot_judge` / `not_applicable` boundary when it has one. Each packet ends with the
rating rule, which the judge's system prompt repeats word for word. The score is the lowest level
whose condition holds. The rationale names that level and says which part of the next level's
condition does not hold. A criterion shared by several point kinds is written once with a YAML
anchor (`&name`) and repeated as an alias (`*name`). The start and change timing anchors use only
orderings and counts visible in the packet; their undecided thresholds stay in the machine checks.
`timing.request_in_time` quotes the decided 3.0 s and 10.0 s in its anchor text, so editing
`timing.request_deadline_s` or `timing.min_question_interval_s` means editing that anchor too. The
system prompt treats only the last CRITERIA block and the SCALE block that close the packet as
rubric; anything before them is material.

| Key | Default | When `null` |
|---|---|---|
| `rubric_version` | `pilot-3` (legacy `pilot-2`) | Required. Bump it after editing criteria, anchors, scale wording or `triggers`; ratings must match it |
| `prompt` | `prompts/assess-judge-v3.yaml` (legacy `assess-judge-v2`) | Required: the prompt file (repo-relative) |
| `triggers` | schema 3 and 2 (legacy: 2 only) | Required: per event schema, `point_kinds` (every trigger kind of that schema -> `start`, `change` or `request`) and `forced` (kinds of the forced first translation; they must map to `start`) |
| `timing.request_deadline_s` | 3.0 (D55) | `processing_s` reported, `not_evaluated/request_deadline_s_unset` |
| `timing.min_question_interval_s` | 10.0 (D55) | Intervals reported, no verdict |
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
