# LLM-as-a-Judge with `gpt-6.1-sol` on Simulation Video

Research date: 2026-11-04. Scope: how to call OpenAI `gpt-6.1-sol` as a judge that must
"see" a simulation video. No API calls were made. Official OpenAI sources were used
unless a line is explicitly marked **[third-party]**.

**Verdict: native video input is not supported. Send sampled frames as `input_image`
items through the Responses API, each preceded by an `input_text` timestamp label.**

---

## Model Availability

- `gpt-6.1-sol` exists. Its exact API model id is `gpt-6.1-sol`, and it is the only
  snapshot or alias listed (no dated snapshot). Released 2026-09-29 (API changelog,
  system-card addendum dated 2026-09-29).
- It sits in the GPT-6 flagship line: `gpt-6-astra` ($10/$50), `gpt-6.1-sol` ($2/$10),
  `gpt-6.1-terra` ($1/$5), and `gpt-6-luna` ($0.10/$0.50) per 1M input/output tokens.
- Context window 1,050,000; max input 922,000; max output 128,000. Knowledge cutoff
  2026-04-30.
- Pricing per 1M tokens (Standard, input <= 272K): input $2.00, cached input $0.10,
  output $10.00. Above 272K input tokens the **whole request** costs 2x input and 1.5x
  output ($4.00 / $0.20 / $15.00). Batch and Flex are 50% off ($1.00 / $0.05 / $5.00).
  Fast mode is 2x and Ultrafast 6x.
- Rate limits (Standard): Build 5,000 RPM / 1M TPM; Launch 10,000 RPM / 4M TPM; Grow
  15,000 RPM / 40M TPM.
- The repo currently uses `gpt-5.6-sol` (`configs/weightreading/run.yaml`), and its
  pricing table has no `gpt-6.1-sol` entry. A judge config needs its own pricing row:
  `gpt-6.1-sol: {input: 2.00, cached_input: 0.10, output: 10.00}`.
- Related: the changelog positions `gpt-6.1-terra` (released 2026-10-26, half the price)
  for "visual reasoning". It may be worth an A/B against Sol as the judge (not
  evaluated here).

## Endpoint & Request Shape

- Supported endpoints: Responses (`v1/responses`), Chat Completions
  (`v1/chat/completions`), and Batch (`v1/batch`). Not supported: Realtime, Assistants,
  fine-tuning, Videos, and others.
- Tool calling requires the Responses API. A judge needs no tools, so either endpoint
  works, but Responses is OpenAI's recommended path and matches the repo's existing
  `OpenAIResponsesBackend` (`src/weightreading/backends.py`).
- Features: streaming, structured outputs, function calling, image input, prompt
  caching, and reasoning tokens. Fine-tuning and predicted outputs are not supported.

### Python SDK

- Latest `openai` release: **3.43.0 (2026-11-03)**. The 3.x line started at 3.0.0
  (2026-08-12), whose only breaking change is "HTTPX2 is now the default HTTP client,
  and `httpx` is no longer installed automatically."
- The repo pins `openai>=1,<3`. The installed and locked version is **2.54.0
  (2026-08-11)**, the last 2.x release. 2.x receives no further updates.
- 2.54.0 predates GPT-6. It has no `gpt-6.1-sol` literal; that literal arrived in
  3.21.0 (2026-09-29). This does **not** block calls because `ResponsesModel` is
  `Union[str, ...]`. Verified in the installed package: 2.54.0 already has
  `ReasoningEffort` incl. `"xhigh"` and `"max"`, `input_image.detail` incl.
  `"original"`, `client.responses.parse(text_format=...)`, `prompt_cache_options`, and
  `FilePurpose` incl. `"vision"` and `"user_data"`.
- Fields added after 2.54.0 can be sent with `extra_body={...}`. Raising the pin to
  `<4` is optional. It changes the HTTP client (httpx to HTTPX2), so the repo's plain
  `openai.OpenAI(api_key=..., max_retries=0, timeout=...)` construction should keep
  working, but this was not tested.

Call shape (works on 2.54.0; mirrors the repo's backend conventions):

```python
from pydantic import BaseModel, Field

class CriterionScore(BaseModel):
    criterion: str
    evidence: str            # cite frame timestamps, e.g. "t=12.0s"
    score: int = Field(ge=1, le=5)

class JudgeVerdict(BaseModel):
    criteria: list[CriterionScore]
    overall_pass: bool
    reason: str

content = [{"type": "input_text", "text": task_and_intent_text}]
for t_s, data_url in frames:                      # data_url = "data:image/jpeg;base64,..."
    content.append({"type": "input_text", "text": f"Frame at t={t_s:.1f}s"})
    content.append({"type": "input_image", "image_url": data_url, "detail": "high"})

response = client.responses.parse(
    model="gpt-6.1-sol",
    instructions=RUBRIC_TEXT,                     # static, so it can be prompt-cached
    input=[{"role": "user", "content": content}],
    text_format=JudgeVerdict,                     # -> text.format json_schema, strict
    reasoning={"effort": "medium"},               # low | medium (default) | high | xhigh | max
    max_output_tokens=25_000,                     # covers reasoning + visible output
    store=False,
)
if response.status == "incomplete":               # e.g. reason "max_output_tokens"
    ...
verdict = response.output_parsed                  # None on refusal; check refusal first
```

Raw `responses.create` equivalent: `text={"format": {"type": "json_schema",
"name": "judge_verdict", "strict": True, "schema": {...}}}`.

Chat Completions equivalent (if ever needed): content parts
`{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}`,
`response_format={"type": "json_schema", ...}`, `reasoning_effort="medium"`. Function
calling is unavailable there for `gpt-6.1-sol`. `input_file` with non-PDF files is
Responses-only.

## Video Input (native or frames)

**Native video input: NOT supported.** The evidence:

- The `gpt-6.1-sol` model page lists input "text and image", output "text", and
  "Unsupported modalities: audio, video."
- The models overview says "All latest OpenAI models support text and image input, text
  output." No flagship lists video input.
- The Responses API reference shows no `input_video` content type. The input content
  parts are `input_text`, `input_image`, and `input_file`.
- `input_file` accepts PDF, office documents, spreadsheets, and text/code. Video
  formats (`.mp4` etc.) are not listed. The 50 MB per-request file limit applies.
- The API changelog (2026-09-01 to 2026-10-30) has no entry announcing video input
  for Responses or Chat Completions.
- The openai-python changelog has no `input_video` type in any 2.x or 3.x release. The
  "video" entries concern the Videos (generation) API, e.g. "deprecate Sora video APIs"
  in 3.0.x and polling helpers.
- An openai-node GitHub issue (#1778) explicitly proposes `input_video` as a *new*
  feature, which implies it does not exist. **[third-party gateways]** Vercel AI
  Gateway, NanoGPT, and others accept `input_video` or MP4 in `input_file`, but that is
  their own pre-processing, not OpenAI's API.
- No per-second or per-frame video pricing exists for text models.

**Therefore: sample frames and send them as images.** The official cookbook ("GPT with
vision for video understanding", now marked *archived*) does exactly this. It reads
the video with OpenCV, JPEG-encodes frames, base64-encodes them, and sends every 25th
frame as `input_image` data URLs, noting "we don't need to send every frame for GPT to
understand what's going on." It gives no guidance on timestamps, detail, or resizing.

Frame-sampling recommendations (engineering judgment, not official OpenAI guidance):

- Use uniform sampling at a fixed rate (e.g. 1-2 fps) for short simulation clips. Add
  event-anchored frames when the simulator logs key events (contact, goal reached,
  failure), because the judge cannot see what falls between frames.
- Put an `input_text` label before each frame ("Frame 7/32, t=12.0s"). Images are
  otherwise identified only by position in the content list (the official image-evals
  cookbook relies on order plus a one-line role description). **[third-party]** Simon
  Willison's `llm-video-frames` plugin (2025-05-05) samples 1 fps by default and can
  burn "filename hh:mm:ss" timestamps into each frame with ffmpeg (`?timestamps=1`).
  Text labels plus optional burned-in timestamps are complementary.
- Pass simulator state that is not visible (numeric trajectories, success flags) as
  text alongside the frames. The judge should not have to infer it from pixels.
- Use a fixed resolution and `detail` level so per-frame tokens are predictable.

## Image Limits

From the official "Images and vision" guide (applies to Responses and Chat
Completions):

- Formats: PNG, JPEG, WEBP, and non-animated GIF. Animated GIFs are not a video
  workaround.
- Up to **1,500 images per request**. Up to **512 MB total payload per request**. Up to
  30,000 patches per image after resizing; images over that are rejected, not resized.
- Image sources: an `image_url` (https URL or base64 data URL), or a `file_id` from the
  Files API (SDK purposes include `vision` and `user_data`). A Files API upload is not
  required.
- `detail`: `low`, `high`, `original`, `auto` (default `auto`). "`low` does not always
  use fewer tokens than `high`." `original` is for dense or spatially sensitive images.
- Token cost on patch-based models: `patches = ceil(w/32) * ceil(h/32)`. Images are
  scaled down to fit the model's patch budget if one applies. Billable tokens =
  patches x model multiplier (1.2 for `gpt-6-astra` and `gpt-5.6-sol/terra/luna`).
  Image tokens are billed as ordinary input tokens; there is no separate image price.
- Documented sizing: `gpt-6-astra` `low` fits 512x512, and `high` allows 2,500
  patches with a 65,535 px maximum side. `gpt-5.6-*` `high` fits 2048x2048 within
  2,500 patches. **`gpt-6.1-sol` is not listed** in the sizing or multiplier tables
  (see Unverified Items).

Worked estimate, **assuming** the Astra/5.6 values (multiplier 1.2, 2,500-patch `high`
budget) also apply to `gpt-6.1-sol`:

| Frame size | Patches | Tokens/frame | 32 frames | 64 frames |
|------------|---------|--------------|-----------|-----------|
| 512x512    | 256     | ~308         | ~9.9K     | ~19.7K    |
| 640x360    | 240     | ~288         | ~9.2K     | ~18.4K    |
| 1280x720   | 920     | ~1,104       | ~35K      | ~71K      |

At $2/1M input, 64 frames at 720p cost about $0.14 of input per judgment. Keep the total
request under **272K input tokens** to avoid the 2x long-context price on the whole
request.

## Structured Outputs & Reasoning Params

- Structured outputs are supported for `gpt-6.1-sol` (model page feature list).
  Responses: `text.format = {type: "json_schema", name, schema, strict: true}`. The
  SDK helper is `client.responses.parse(text_format=PydanticModel)` -> `output_parsed`.
- Schema rules: every field in `required` (optional = `["type", "null"]`);
  `additionalProperties: false` on every object; the root must be an object, not
  `anyOf`. Limits: 5,000 properties, 10 nesting levels, 1,000 enum values. Unsupported:
  `allOf`, `not`, `if/then/else`, `dependentRequired`, `dependentSchemas`. Supported:
  `pattern`, `format`, `minimum/maximum`, `minItems/maxItems`, `enum`, `anyOf`.
- Refusals produce a `refusal` field that "does not necessarily follow the schema".
  Check for it before parsing.
- `reasoning.effort` for `gpt-6.1-sol`: `low`, `medium` (default), `high`, `xhigh`,
  `max`. It **does not support `none` or `minimal`**; the migration advice is to use
  `low` instead. Chat Completions uses `reasoning_effort`.
- **temperature / top_p:** the latest-model guide says "When reasoning effort is not
  `none`, remove `temperature`, `top_p`, and `top_logprobs`." Because `gpt-6.1-sol`
  cannot use `none`, **never send temperature or top_p**. The GPT-6 Astra changelog
  entry also says "no custom temperature/top_p or logprobs." Whether the API rejects
  them with HTTP 400 or ignores them is not documented. Judge repeatability therefore
  comes from the rubric, strict schema, and repeated sampling, not temperature 0.
- Max output tokens: `max_output_tokens` (Responses) covers reasoning, visible output,
  and formatting tokens. If exhausted, `status="incomplete"` with reason
  `max_output_tokens`, possibly before any visible output. OpenAI recommends
  "reserving at least 25,000 tokens for reasoning and outputs" at first. Chat
  Completions uses `max_completion_tokens`.
- `store=False` (as in the repo) is fine. Reasoning items then carry `encrypted_content`
  by default; `include=["reasoning.encrypted_content"]` is still accepted but no longer
  required. A single-turn judge does not need to replay reasoning.
- Vision-specific advice (GPT-5.4 vision cookbook): "Raise reasoning effort when the
  image is readable but the answer requires combining multiple regions." `high`/`xhigh`
  helps more than higher detail once images are readable. Temporal reasoning across
  frames is analogous; start with `medium`, compare `high` on a calibration set.
- Migration note: `prompt_cache_retention` is deprecated in favour of
  `prompt_cache_options.ttl`. Put the static rubric in `instructions` first so it can be
  cached across judgments.

## Judge Best Practices

Official OpenAI guidance ("Evaluation best practices", "Graders", and the image-evals
cookbook):

- Prefer pairwise comparison or pass/fail: "Use pairwise comparison or pass/fail for
  more reliability." Graders guide: "Produce a smooth score, not a pass/fail stamp" for
  RL-style rewards. Pick based on use (gating vs. reward signal).
- Reason before scoring: "Add reasoning and chain-of-thought as reasoning before scoring
  improves eval performance." With a reasoning model this is internal; still put an
  `evidence` field before `score` in the schema, because output keys follow schema
  order.
- Known biases: position bias and verbosity bias. Randomise order in pairwise mode and
  control for length.
- Calibrate against human labels before optimising cost: scale up only once the judge
  "consistently agrees with human annotations". Keep anchor examples re-scored
  periodically to catch drift. Compare judge and human results to detect grader hacking.
- Use detailed rubrics, scored per criterion with anchored levels (e.g. 5/3/0-2). "Base
  all scores on concrete visual observations." "If uncertain between two scores, choose
  the lower one." Keep hard gates separate from graded metrics and do not average them
  into the verdict. Split overloaded rubrics into one call per metric.
- Use few-shot examples of great, fair, and poor answers (Graders guide).
- Check stability by running several candidates through the judge and confirming
  consistent grading.
- Video- or frame-sequence-specific judge guidance from OpenAI: **none found**. The
  frame-sampling and timestamp-labelling advice in the Video Input section is
  engineering judgment plus a third-party tool, not OpenAI guidance.
- Note: OpenAI's hosted Graders (Evals / RFT `score_model`) are being deprecated, their
  `score_model` content is typed as `str`, and their model list is old (gpt-4o/o3 era).
  Use a plain Responses call with a strict schema instead.

## Sources

All accessed 2026-11-04.

Official OpenAI:
- Model page, GPT-6.1 Sol: https://developers.openai.com/api/docs/models/gpt-6.1-sol
- Models overview: https://developers.openai.com/api/docs/models
- Latest-model / migration guide: https://developers.openai.com/api/docs/guides/latest-model
- Images and vision: https://developers.openai.com/api/docs/guides/images-vision
- File inputs: https://developers.openai.com/api/docs/guides/file-inputs
- Responses create reference: https://developers.openai.com/api/reference/resources/responses/methods/create
- Reasoning guide: https://developers.openai.com/api/docs/guides/reasoning
- Structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs
- Pricing: https://developers.openai.com/api/docs/pricing
- API changelog: https://developers.openai.com/api/docs/changelog
- Evaluation best practices: https://developers.openai.com/api/docs/guides/evaluation-best-practices
- Graders: https://developers.openai.com/api/docs/guides/graders
- Cookbook, video understanding (archived): https://developers.openai.com/cookbook/examples/gpt_with_vision_for_video_understanding
- Cookbook, image evals (LLM-as-judge): https://developers.openai.com/cookbook/examples/multimodal/image_evals
- Cookbook, GPT-5.4 vision tips: https://developers.openai.com/cookbook/examples/multimodal/document_and_multimodal_understanding_tips
- Announcement (HTTP 403 to fetcher; seen only via search snippet): https://openai.com/index/introducing-gpt-6-1-sol/
- System card addendum (not read): https://deploymentsafety.openai.com/gpt-6-1-sol/gpt-6-1-sol.pdf
- openai-python releases: https://github.com/openai/openai-python/releases
- openai-python changelog: https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md
- PyPI history: https://pypi.org/project/openai/#history
- Local: installed `openai` 2.54.0 in `.venv` (type definitions inspected directly).

Third-party (labelled where used):
- openai-node issue #1778 (proposal for `input_video`): https://github.com/openai/openai-node/issues/1778
- Vercel AI Gateway video input: https://vercel.com/docs/ai-gateway/inputs-and-tools/video-input
- Simon Willison, llm-video-frames (2025-05-05): https://simonw.substack.com/p/feed-a-video-to-a-vision-llm-as-a

## Unverified Items

- TODO: image sizing for `gpt-6.1-sol` (patch budget per `detail`, `low` box, and image
  token multiplier). It is not in the vision guide's sizing or multiplier tables. The
  estimates above assume the documented `gpt-6-astra`/`gpt-5.6-*` values (1.2x,
  2,500-patch `high`). Verify with one cheap request and read `usage.input_tokens`.
- TODO: whether sending `temperature`/`top_p` to `gpt-6.1-sol` returns HTTP 400 or is
  ignored. The docs only say to remove them.
- TODO: whether the 1,500-image / 512 MB limits hold for `gpt-6.1-sol` specifically.
  The guide says "Other models and specialized variants can use different limits."
- TODO: the system card and announcement were not read (403 / not fetched), so no
  official vision or video benchmark numbers for `gpt-6.1-sol` were checked.
- TODO: whether `gpt-6.1-terra` ("visual reasoning") is a better or cheaper judge than
  Sol for frame sequences. It is not evaluated.
- Not tested: SDK 3.x (HTTPX2) compatibility with the repo; the current `<3` pin
  (2.54.0) is sufficient for this judge.
- The archived video cookbook may reference outdated models. Its frame approach is
  still the only official video recipe found.
