# Codebase scan: LLM-as-a-judge for LAG intent translations (`evaluation/`)

Date: 2026-11-04. Mode: existing codebase, read-only scan. Scanner: general-purpose-opus.
Repository: `/home/user151005/workspace/mtd-rl-xavier` (paths below are repo-relative unless absolute).
No files were modified except this one. No API calls were made. `.env` was not read.

Feature under design: a new top-level package `evaluation/` (currently an empty directory)
holding an LLM-as-a-judge (`gpt-6.1-sol`) that scores "intent translations" produced by an
"LLM as a Translator" for LAG episodes. YAML files in `evaluation/config/` choose which of six
inputs reach the judge and how each is rendered: instruction text, simulation video,
start/change/request timestamps (開始点・変化点・要求点), trajectory, translated intent,
and scoring key points (採点の要点).

---

## 1. Relevant Areas

| Area | Path | Why it matters |
|---|---|---|
| LAG raw data | `workspace/13_LAG/data/raw/20261005_LAG_2v2_ACMI/` (3 `.acmi`, `actions/*.csv`, `recordings/*.mp4`) | Source of trajectory and video. Raw files are immutable (hashed in `data/raw/manifest.json`). |
| LAG integrated CSV + loader | `workspace/13_LAG/data/processed/{*.csv,manifest.json,README.md}`, `workspace/13_LAG/lag_data/{dataset.py,definitions.py,prepare.py}` | Canonical trajectory: 64 columns, 1 row = 1 step x 1 object, SHA-256-verified loader `load_episode()`. |
| Video/step mapping | `workspace/13_LAG/viewer/model.js` (`videoFrame()`), `workspace/13_LAG/viewer/prepare.py`, `workspace/13_LAG/data/viz/render_action_videos.py` | The only existing code that maps video frames to simulation steps. |
| Trajectory-to-prompt renderers | 14: `workspace/14_mental-modeling_LAG/lag_experiment/prompts.py` (`_render_block`); 15/16 near-copies; 17: `workspace/17_commentary_LAG/commentary_experiment/{dataset.py,prompts.py}` | Two existing text formats for trajectories (section 3.2). |
| Closest Translator precursor | `workspace/17_commentary_LAG/` "tactical" variant: `commentary_experiment/tactical_{system,user}.txt`, `results/tactical_batch_v1/` | Per-aircraft intent narratives (JSON with keys A0100/A0200/B0100/B0200) from a 61-step window. |
| Closest judge/rubric precedents | `workspace/17_commentary_LAG/commentary_experiment/evaluation.py`; `workspace/17_commentary_LAG/results/tactical_batch_v1/evaluation/model_comparison_20261019/*.txt`; `workspace/07_view_rawdata/reasoning_score_analysis/Prompt.md` | Human rubric template, 0/1/2 scoring against an annotation, and a 1-4 LLM-as-a-Judge rubric (run interactively, no code). |
| OpenAI Responses API backend | `src/weightreading/backends.py`, `src/weightreading/cli.py`, `src/weightreading/config.py`, `configs/weightreading/run.yaml` | Newest, typed, tested pattern: Responses API, YAML config, paid gate. Best template for `evaluation/`. |
| Chat Completions + journaling | `src/mtd_rl_xavier/experiment/common.py`; `workspace/17_commentary_LAG/commentary_experiment/{execution.py,batch.py}` | Plan hashing, write-before-send journals, Batch API, redaction. |
| Shared paths/safety | `src/mtd_rl_xavier/paths.py`, `src/mtd_rl_xavier/tools/output_safety.py` (`derived_path`) | Repo root and write guards; README forbids computing the repo root from `__file__` in library code. |
| Tooling | `pyproject.toml` (pytest, ruff, poe), `tests/conftest.py`, `tests/weightreading/` | Where new tests and lint coverage must be wired. |
| Cross-repo context | `/home/user151005/workspace/.claude/docs/DESIGN.md`; `/home/user151005/workspace/work/evaluation-data-handoff/output/20261102-second-round-paste/` | Declares mtd-rl-xavier as the "LLM as a Translator" repo; draft data contract for instructions/annotations. |

---

## 2. Existing Patterns

### 2.1 OpenAI calls

Two coexisting styles. Neither uses structured outputs, image input, or video input anywhere
(`grep` for `response_format|json_schema|input_image|image_url|input_file|responses.parse` found nothing).

**A. Responses API (newest; `src/weightreading/backends.py`)**
- `client.responses.create(input=list(history), model=..., reasoning={"effort": ...}, max_output_tokens=..., store=False, include=["reasoning.encrypted_content"])`.
- Reads `response.output_text`, `response.output` (dumped with `model_dump(mode="json", exclude_none=True)`), `response.status`, `response.id`, and usage (`input_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens`, `input_tokens_details.cached_tokens`) into frozen dataclasses `Usage` and `BackendReply`.
- `request_record` stores the request options plus `input_sha256` (`history_sha256`). Credentials are never stored.
- Protocol class `ChatBackend` so tests pass a fake. `import openai` is lazy, inside `make_openai_client()`.
- `make_openai_client()`: key from `os.environ["OPENAI_API_KEY"]` only, else `RuntimeError`; `openai.OpenAI(api_key=..., max_retries=0, timeout=1800.0)`. Constant `SDK_MAX_RETRIES = 0  # a retried paid request may be billed twice`.

**B. Chat Completions (`common.py`, workspace 14-17)**
- `client.chat.completions.create(model=..., messages=[system, user], reasoning_effort="medium")` for gpt-5.6-*; `temperature=0` only for GPT-3.5 (`common.api_request_options`).
- 14-17 also use the Batch API (`ENDPOINT = '/v1/chat/completions'`, `workspace/17_commentary_LAG/commentary_experiment/batch.py`) with per-model JSONL shards, a tiktoken `o200k_base` estimate x1.1 (no tiktoken cache is present on this host, so the first use would need network), and `--adopt` recovery.
- The output contract is free text containing JSON, parsed strictly after the call: `parse_intentions()` in 17 rejects duplicate keys, `finish_reason != 'stop'`, wrong key set, and empty strings, and records `valid/error` instead of repairing.

**Model naming and config**
- Aliases `sol/terra/luna` map to `gpt-5.6-sol/-terra/-luna` (`workspace/14_.../config.py`, `workspace/17_.../config.py`). `gpt-6-astra` appears only in pricing tables (`configs/weightreading/run.yaml`, `workspace/notebooks/01_token_estimation.py`).
- **`gpt-6.1-sol` appears nowhere in the repo.** There is no pricing entry and no record of request options for it.
- Reasoning: Chat Completions sends a top-level `reasoning_effort`; Responses sends `reasoning.effort`. Allowed values are validated in `weightreading/config.py`: `("none","minimal","low","medium","high","xhigh")`.
- weightreading policy: "the code has no default model id". The model must be set in YAML, and a `pricing:` table (USD per 1M tokens: input/cached_input/output) feeds an offline cost preview (`estimate.py`, chars/4 approximation, explicitly labelled).

**API key handling** (two conventions)
- Env-only: weightreading and `common.py` (`OPENAI_API_KEY must be set in the environment`).
- Env, falling back to `getpass`: 14-17 and 09-12 (`os.environ.get("OPENAI_API_KEY") or getpass.getpass(...)`).
- Redaction: `common.redact()` masks dict keys matching `authorization|api.?key|secret|...`, the literal key value, `sk-...`, and `Bearer ...`; 17 has `_redact_error()`. HTTP debug loggers are silenced (`quiet_sdk_logging`).

**Retries/timeouts**: SDK `max_retries=0` everywhere. Timeout is 180 s in 17 sync and 1800 s in weightreading. There is no automatic resend; uncertain (sent, no response) queries need explicit flags (`--retry-uncertain` in 03).

### 2.2 Paid-call safety

- Strongest form (weightreading `cli.main`, `common._run`): paid calls require **both** `--execute` and `--confirm-paid-api`. One flag alone is `parser.error` (exit 2). Without either, the run writes a local preview (`message.txt`, `preview.json` with `api_requests_made: 0`) and prints `Preview only. API requests made: 0` to stderr.
- The client is built only after the gate (and after preflight in weightreading), through an injected `client_factory`. Tests assert `factory.calls == 0` on preview paths.
- Weaker forms exist: 17 sends with `--execute` alone. 09-12 send with **no** arguments (documented as a hazard in `README.md` and `workspace/README.md`). The new package should follow the weightreading form.
- Compute-node convention: paid runs go through `scripts/weightreading_run.sbatch`, which sends only when `WEIGHTREADING_CONFIRM_PAID=yes`. The login node is for previews and tests.

### 2.3 Result immutability and provenance

- 14-17 (`execution.py`): `plan_id = sha256(canonical JSON of plan)`; per-query `request_sha256` and `query_id`; `source_sha256` records the hashes of the code, data CSV, manifest, and prompt `.txt` files. Before any send, `execute()` re-hashes them and refuses with "Source changed since planning". `save_plan()` refuses a different plan in the same directory ("select a new --output name"), and refuses edited prompt previews.
- Journals: `api/requests.jsonl` is appended (with fsync) **before** the send, and `api/responses.jsonl` after it, including errors. Then `api/summary.json` (`planned/attempted/responses_saved/unresolved/status`) and an `fcntl` run lock (`.lock`). Atomic `write_json` uses a temp file plus `replace`.
- weightreading: a new UTC-stamped directory per run (`mkdir(exist_ok=False)`); `run.json` holds provenance (`provenance.code_hashes()` = SHA-256 of every package `.py`, `git_state()` commit and dirty flag, environment, input hashes); `transcript.jsonl` holds per-turn usage. `derived_path()` refuses writes under `data/` and `mentalmodeling/`.
- Repo rules (`README.md` Conventions): results are immutable and tracked in git under `workspace/*/results/`; analyses write to new directories and refuse to overwrite. `outputs/**/*.{csv,json,jsonl}` are gitignored.
- Human-review precedent: `rubric_template.json` is written once and never overwritten; `reference.json` records `contents_verified: false, included_in_prompt: false`; the 07 Prompt.md states "evaluate reasoning, not correctness; ignore length/fluency/confidence".

### 2.4 Config loading

- `src/weightreading/config.py`: `yaml.safe_load` into **frozen dataclasses** with hand-written validation. It requires exact key sets per section (`_section(raw, key, allowed)`), rejects unknown top-level keys, checks positive ints with `type(v) is int`, checks choices against tuples, and raises `ConfigError(ValueError)`. There are no hidden defaults: the model must be explicit.
- 14: `@dataclass(frozen=True) class Experiment` with a `validate()` method. 17: module constants plus prompt templates as `.txt` files filled with `str.format(**fields)`.
- **pydantic is not used in repo code.** It is installed only transitively (2.14.1 via openai) and is not a declared dependency. PyYAML (`PyYAML>=6`) is declared. jsonschema 4.26.0 and jinja2 3.1.6 are present transitively.

### 2.5 Tests, lint, packaging

- `pyproject.toml`: `testpaths = ["tests", "mentalmodeling/tests"]`, `pythonpath = ["."]`, `addopts = "-ra -p no:cacheprovider"`, `python_files = ["test_*.py"]`. `poe test` = `pytest`. `poe test-workspace` loops over `workspace/[0-9]*/`. `poe lint` = `ruff check src tests`.
- Ruff: `target-version = "py310"`, `line-length = 120`, rules `E,W,F,I,B,UP`, `E501` ignored, `force-exclude = true`, `extend-exclude = ["mentalmodeling", "workspace"]`. `evaluation/` is therefore lint-eligible but **not covered by `poe lint`** unless the task is changed to `ruff check src tests evaluation`.
- Packaging: setuptools `packages.find where=["src"]` includes only `mtd_rl_xavier*` and `weightreading*`. A root-level `evaluation/` is **not installed**. It is importable under pytest through `pythonpath = ["."]`, and through `uv run python -m evaluation` from the repo root only. Today `evaluation/` has no `__init__.py`, so it resolves as a namespace package; add `__init__.py`. Only package-scoped `evaluation.py` modules exist in workspace 14-17, so there is no current top-level collision. Still, the generic name can be shadowed by any script directory that has an `evaluation.py` at `sys.path[0]`.
- Test style (`tests/weightreading/`): `from __future__ import annotations`, typed tests, AAA layout, a `conftest.py` with fixtures, and a helper module imported by bare name (`wr_helpers`; no `__init__.py`, default rootdir import mode). OpenAI fakes:
  - `ScriptedClient` exposes `.responses.create(**kwargs)`, records the kwargs, and returns simple objects with `output_text/output/usage/status/id`.
  - `RecordingFactory` counts client constructions.
  - `monkeypatch.delenv("OPENAI_API_KEY")` asserts the key requirement.
  - Gate tests: one flag gives exit 2; no flags give a preview and `factory.calls == 0`.
- Workspace tests (17) prepend `sys.path` with the experiment dir and `13_LAG` and fake Batch clients with `SimpleNamespace(..., model_dump=...)`.
- Known pre-existing failures are documented in `README.md` (external datasets, one test each in 16/17).

---

## 3. LAG Data Formats

### 3.1 Episodes available

Three 2v2 matches. A0100/A0200 are the learned policy (Red); B0100/B0200 are the rule-based pursuer (Blue). The step interval is 0.2 s, with a 200 s cap per match. Episode IDs contain win/loss strings, so 17 uses the neutral alias `episode_01`.

| episode_id | CSV rows | last CSV step | last ACMI step | recording |
|---|---:|---:|---:|---|
| `mhi_v1.3_ic100601_win` | 2,124 | 340 (68.0 s) | 341 (68.2 s) | 960x940, 171 frames, 17.1 s |
| `ours_seed100100_ic100601_loss` | 1,893 | 293 (58.6 s) | 294 (58.8 s) | 960x674, 148 frames, 14.8 s |
| `ours_seed100100_ic100606_win` | 3,016 | 495 (99.0 s) | 496 (99.2 s) | 960x742, 249 frames, 24.9 s |

### 3.2 Trajectory

- **Raw ACMI** (Tacview text, UTF-8 BOM): `#<sec>` time lines, then `ID,T=lon|lat|alt|roll|pitch|yaw,Name=F16,Color=Red,TAS=..,Health=..,Missiles=..`. Missile IDs are the launcher ID plus a digit (`A01002`). `-ID` marks removal. `Type=Misc+Explosion,Name=SHOTDOWN` marks a shoot-down effect.
- **Raw actions CSV**: one row per step per aircraft (4 aircraft), with state (lon/lat/alt/roll/pitch/yaw, tas/vn/ve/vu), 4 action labels plus 13 probabilities, and `missile_launched`. Actions are blank for Blue and for downed aircraft.
- **Integrated 64-column CSV** (canonical; `workspace/13_LAG/data/processed/README.md`, schema `1.0`): primary key `(episode_id, step, object_id)`. Columns:
  - `entity_type` (`aircraft|missile|effect`), `team`, `launcher_id`, `alive`, `state_valid`, `state_source`.
  - `action_status` (`recorded|not_recorded|inactive|state_only|not_applicable`).
  - The 10 `STATE_FEATURES` (`lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg, tas_mps, vn_mps, ve_mps, vu_mps`).
  - `missiles_remaining`; `*_action` labels and `*_action_id` (altitude 0 up/1 hold/2 down; heading 0 left30 .. 4 right30; speed 0 up/1 hold/2 down; shoot 0 no/1 yes); `p_*` probabilities; `acmi_event` (`update|remove|none`); provenance columns.
  - Missing values stay blank or NA and are never filled.
- **Loader**: `load_episode(path)` verifies the CSV SHA-256 against the adjacent `manifest.json`, types the columns, and sets `frame.attrs = {csv_sha256, schema_version, dt_s}`. It is importable from the repo root as `importlib.import_module("workspace.13_LAG.lag_data.dataset")` (verified: shape `(2124, 64)`). Experiments 14-17 instead prepend `workspace/13_LAG` to `sys.path` and use `from lag_data.dataset import ...`. Helpers: `history_steps()`, `select_records()`, `index_records()`.
- **Prompt renderings already in use**:
  1. 14/15/16 text blocks (`_render_block`): `Step 0 (time_s=0.0):` then `  state A0100: [10 floats]; valid=1; entity_type="aircraft"; team="A"; alive=1; launcher_id=null; event="update"` and `  action A0100: [2,3,0,1]; status=recorded`. Floats are unrounded.
  2. 17 JSON lines, one per step: `{"step","time_s","relative_step","entities":[{"object_id","entity_type","team","launcher_id","alive","state_valid","acmi_event","action_status","state":[10],"action":[3 ids],"missiles_remaining"}]}`. Masking rules: `state_valid=0` gives all-null state, `shoot_action`/probabilities/health/controller/filename are excluded, and effects are excluded. A 61-step window is about 130-190 KB of prompt (about 48.7k prompt tokens observed for gpt-5.6-sol).
- Input-leak rules shared by 14 and 17: exclude `p_*`, `missile_launched`, `health`, `controller`, and the win/loss filename. Use a neutral episode alias.

### 3.3 Video

| file set | codec | fps | resolution | frame to step mapping |
|---|---|---|---|---|
| `data/raw/.../recordings/<episode>.mp4` (original, tracked in git) | H.264 `avc1`, no audio | 10 | 960 x {940, 674, 742} | 4x speed. Frame `i` shows sim time `0.4*i` s = **step `2*i`**. Media time = sim time / 4. |
| `data/viz/action_videos/<episode>_with_actions.mp4` (derived) | H.264 | 20 | 1920x1080 | Frame `f` = **step `f`** (each source frame repeated twice). The left half is the recording; the right half is A-team action plots. |

- These numbers were measured by parsing MP4 boxes in pure Python (`mvhd/tkhd/mdhd/stts`), because `ffprobe` is not available. They are consistent with `viewer/prepare.py` (asserts `r_frame_rate == '10/1'`) and `render_action_videos.py` (asserts 10 fps source, 20 fps output, `n*2` frames).
- Canonical mapping from the viewer (`viewer/model.js:videoFrame`): `frame = floor((step*0.2 + 1e-8)/0.4)`, capped at the last frame; `mediaTime = (frame*0.4 + 0.2)/4`. Odd steps show the previous frame. The final frame indices (170/147/248 give steps 340/294/496) match the last ACMI steps for the two `ours_*` episodes and are within one step of it for MHI.
- **Rendered content** (from `workspace/13_LAG/viewer/screenshots/match_inspector.png`): a 2D top-down plot (North/East km) with aircraft trails and ID labels, an altitude panel below, per-team header text (speed, altitude), and an on-screen **event log that appears to list fire / HIT / SHOTDOWN events, including attacker-to-target pairs** (for example "A0200→B0200 HIT"). The video therefore discloses hit outcomes and missile targets, which the numeric trajectory inputs deliberately withhold (17 tells the model that targets and hits are not provided). The derived action videos additionally disclose the A-team actions.

### 3.4 Timing semantics relevant to start/change/request points

- `time_s = step * 0.2`, step 0-based. `s_t` precedes `a_t`, which acts over `[t, t+0.2)`.
- Launch convention: the CSV `missile_launched` is at step t; the missile first appears in ACMI at t+1. The 17 annotation comparison treats a one-step offset as not penalizable.
- The video plays at 4x, so "seconds" is ambiguous: sim seconds and media seconds differ by 4x. Store timestamps canonically as simulation **steps**, and derive sim seconds and video media seconds (and frame index) from them.

---

## 4. Translator / Annotation Data Found

**No Translator code, no Translator outputs, and no annotation dataset with start/change/request timestamps or scoring key points exist** in mtd-rl-xavier, mtd-rl-lag, mtd-rl-llm-as-a-policy, or work/. The searched terms were translator, 要求点, 変化点, 開始点, 採点の要点, request point, change point, and intent translation. What exists:

1. **Role assignment only.** `/home/user151005/workspace/.claude/docs/DESIGN.md` (lines 15 and 102) says that "`mtd-rl-xavier` prepares the LLM as a Translator code" (user assignment 2026-11-02). No implementation exists.
2. **Closest Translator-like outputs**: `workspace/17_commentary_LAG/results/tactical_batch_v1/`. Five MHI windows (centers 45, 85, 145, 245, 310; +/-30 steps), 3 models (gpt-5.6-sol/terra/luna), and JSON per-aircraft "most plausible intention" narratives. Files:
   - `evaluation/responses.json` (raw choices, usage, `intention_format` validity);
   - `evaluation/most_plausible_intentions.csv` (columns 試合, step数, モデル, 機体, Most plausible intention, 日本語訳);
   - `windows/center_*.json` (the exact visible input);
   - `manifest.json` (plan, `prompt_version: lag-tactical-v1`, `request_options: {reasoning_effort: medium}`).

   There is no instruction text in this input. It is retrospective intent reconstruction, not an instruction-conditioned translation.
3. **Human annotation, referenced but not stored**: `MHI_annotated.xlsx` (at `/Users/cls-lab/Downloads/`, not in any repo). It has 7 scenes with a start STEP (46, 49, 83, 95, 143, 243, 311), aircraft, action label (e.g. hit-and-away, attack, evasion, cover, decoy, pursuit start), a detailed explanation, and an interpretation. It is paraphrased in `workspace/17_commentary_LAG/results/tactical_batch_v1/evaluation/model_comparison_20261019/annotated_reference_comparison.txt`, together with a **0/1/2 rubric** (2 = core tactical purpose and main action flow reproduced; 1 = partial; 0 = core role missing or a different purpose) and per-model totals. The Google Sheets annotation URL (`commentary_experiment/config.py: REFERENCE_URL`) has never been retrieved (`contents_verified: false`). These scenes carry only a start step. There is no change point or request point.
4. **Human rubric template**: `workspace/17_commentary_LAG/results/*/evaluation/rubric_template.json`. The fields are `supported_claims`, `contradicted_claims`, `unverifiable_claims`, `important_facts_coverage`, `intention_score` (0/1/2 planned), `reviewer`, and `notes`. All rows are null.
5. **LLM-as-a-Judge precedent (no code)**: `workspace/07_view_rawdata/reasoning_score_analysis/Prompt.md`. A Japanese 1-4 rubric on reasoning quality, run interactively (`reasoning_for_scoring_scored_astra_high.csv`, judge appears to be gpt-6-astra high). Scoring precedents: "do not judge correctness; ignore length/fluency/confidence"; rubric levels with explicit counter-examples.
6. **Client instruction list (confidential)**: `/home/user151005/workspace/work/evaluation-data-handoff/output/20261102-second-round-paste/source-text/client-instructions-second-paste.txt`. 43 directives in 4 classes (strategic/tactical x single-aircraft/whole-team), dated 2026-10-22, transcribed and not verified against the original pptx, with the assumed situations omitted. `source-documents.csv` marks it **"顧客資料。別のサービスやモデルへ送らない"** (customer material; do not send to other services or models). Its text is intentionally not reproduced here.
7. **Draft data contract** (not implemented): `.../20261102-second-round-paste/data-contract-draft.md` and `implementation-plan.md`. The annotation record proposes `start_step`, `end_step`, `object_ids`, `claim_type` (`observed_fact|author_intent|interpretation`), `visibility` (`model_input|scorer_only|internal`), `criterion_id/owner/version`, `review_status`, `handling_class`, and `allowed_destinations`. It splits export into `model_input.jsonl` and `scorer_reference.jsonl`. It explicitly notes that Xavier 14-17 already send record-derived inputs to OpenAI. The planned home is `mtd-rl-lag/scripts/eval/evaluation_data/`, which does not exist yet (mtd-rl-lag is checked out on branch `evaluation-data-setup` with no such directory).
8. **Unrelated "translator"**: `mtd-rl-lag` branch `origin/wang/talar`, `talar/translator.py`. This is an MLP/VAE mapping a BERT sentence feature to a task code (TALAR), not an LLM producing natural-language intents.
9. **Instruction onsets in LAG**: `mtd-rl-lag/scripts/eval/instruction_eval/` (`onsets_s` = instruction onset times in policy-evaluation manifests). This is the policy-side instruction-following harness, not annotation.
10. `mtd-rl-llm-as-a-policy`: one commit, README title only.

---

## 5. Available Video Libraries

| Tool | Status in `.venv` (Python 3.12.16) / host |
|---|---|
| `cv2` (OpenCV) | missing |
| `imageio`, `imageio_ffmpeg` | missing |
| `av` (PyAV) | missing |
| `moviepy`, `decord` | missing |
| `PIL` (Pillow) | 13.0.0 (transitive via matplotlib). Cannot decode H.264. |
| `matplotlib` | 3.11.2 (writer side only) |
| `ffmpeg` / `ffprobe` binaries | **not on PATH**; `module avail ffmpeg` shows nothing |
| `torch` | missing |

Consequences:
- No frame can be decoded in this environment today. `workspace/13_LAG/viewer/prepare.py` and `render_action_videos.py` also cannot run here.
- A decoder dependency must be added through uv, for example `av` (PyAV, bundles FFmpeg libs), `imageio-ffmpeg` (bundles a static ffmpeg binary), or `opencv-python-headless`. This is a dependency decision for the user. Add it as an optional extra (pattern: `[project.optional-dependencies]` `data`/`notebook`/`estimation`).
- Video metadata (fps, frame count, size) can be read without any dependency by parsing MP4 boxes. The scanner's throwaway parser is not in the repo.

---

## 6. Recommended Integration Points

1. **Package shape**: `evaluation/__init__.py` plus modules with one responsibility each:
   - `config.py`: strict YAML into frozen dataclasses, mirroring `weightreading/config.py`.
   - `inputs/` renderers per input kind.
   - `backend.py`: reuse or adapt `weightreading.backends.OpenAIResponsesBackend` (Responses API, `store=False`, `max_retries=0`).
   - `plan.py` / `journal.py`: port the 17 `execution.py` plan hash and write-before-send journal.
   - `cli.py` / `__main__.py`.

   Use `mtd_rl_xavier.paths` for `ROOT/CONFIGS/OUTPUTS`. Do not derive the root from `__file__`, and do not edit `paths.py` unless a constant such as `EVALUATION` is truly needed (that edit touches the shared library).
2. **Config** (`evaluation/config/*.yaml`):
   - Explicit `model: gpt-6.1-sol`, `reasoning_effort`, `max_output_tokens`, and a `pricing` table. There is no code default.
   - An `inputs:` mapping with exactly six known keys (`instruction`, `video`, `timestamps`, `trajectory`, `translated_intent`, `scoring_key_points`), each `{enabled: bool, renderer: <choice>, ...renderer params}`.
   - Unknown keys are rejected and enum choices are validated. Changing the YAML must change the plan hash.
3. **Trajectory**: load through `workspace.13_LAG.lag_data.dataset.load_episode` (hash-verified). It must render **exactly what the Translator saw**, so the renderer choice should name an existing format (`lag-context-v3` 14-style text blocks or the 17 window JSON lines) or a new versioned one. The two existing renderers live in experiment code. The repo rule is that code needed by a second consumer is promoted to `src/mtd_rl_xavier/` with tests, so either promote a renderer or copy it with a version tag and a hash in provenance.
4. **Video**: decide between native video input (unverified for the OpenAI API; see Open Questions) and sampled frames sent as `input_image` (base64 PNG/JPEG). Use `data/raw/.../recordings/*.mp4` (10 fps, frame = step//2), not the action videos, which leak actions. Frame selection should be expressed in steps (for example around start/change/request points), mapped with the `viewer/model.js` formula, and recorded in provenance (video SHA-256 from `data/raw/manifest.json`, frame indices, and SHA-256 of each image sent).
5. **Timestamps**: store as steps and render as `step`, `sim time s`, and optionally `video time s` (sim/4) or a frame index, so the judge can align the timestamps with the frames and the trajectory.
6. **Output**: request a JSON score object. Two options exist: Responses structured outputs (`text.format` with `json_schema`; not yet used in the repo, so it needs a fake-client test), or the 17 pattern of strict post-hoc parsing with validity recorded and no repair. Keep the raw response, usage, and `response.id`.
7. **Safety and provenance**:
   - Preview by default (write the exact `input` items, image hashes, and an offline token and cost estimate). Paid calls require `--execute --confirm-paid-api`, and one flag alone is an error. The key is env-only and read after the gate. `client_factory` injection for tests.
   - Outputs go to a new directory per run (UTC stamp or plan-id-based). The run refuses to overwrite and records SHA-256 of the config, code, inputs, and prompts.
   - Decide whether results live in `outputs/evaluation/` (JSON gitignored) or in a git-tracked results directory.
8. **Tests and tooling**:
   - Put tests under `tests/evaluation/` (picked up by `poe test` with no change), or add `evaluation/tests` to `testpaths`.
   - Extend `poe lint` to `ruff check src tests evaluation`.
   - Fakes: copy the `ScriptedClient`/`RecordingFactory` pattern.
   - Add a leak test in the style of `test_all_samples_have_only_allowed_input_and_no_target_leak` (14), so scorer-only fields (key points, translated intent when disabled, and so on) never appear in disabled inputs.

---

## 7. Open Questions

1. **Confidentiality of instruction text.** The client instruction list is marked "do not send to other services or models". The judge would send instruction text (and possibly key points) to OpenAI `gpt-6.1-sol`. Does the user have clearance? Should config carry a per-input `handling_class`/`allowed_destinations` check (as in the data-contract draft) that refuses a paid run otherwise?
2. **`gpt-6.1-sol` specifics**: the exact API model ID, support for Responses vs Chat Completions, valid `reasoning.effort` values, structured-output support, pricing, and **whether native video input is accepted**. The repo has no record of any of these. A quick search found no official confirmation of video input for the OpenAI API: an openai-node issue says video is not among the supported `input_file` types and the cookbook recommends frame extraction ([openai/openai-node#1778](https://github.com/openai/openai-node/issues/1778)). Gateways such as [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/inputs-and-tools/video-input) accept `input_video`, which does not imply the direct API does. This must be verified against official docs before the video renderer is designed.
3. **Video information leak**: the recordings appear to show HIT/SHOTDOWN events and attacker-to-target pairs. Is that acceptable for the judge (it may even be desired as ground evidence), given that the Translator did not see it? Should the Translator also see video?
4. **Data for start/change/request points, key points, and translated intents does not exist yet.** Who produces them, in what file format, and where are they stored (in this repo, or in `mtd-rl-lag` `evaluation_data/` per the draft contract)? Is `instruction_links.csv` / `annotations.jsonl` (draft) the intended schema?
5. **Definitions**: what exactly are 開始点 (instruction issue / segment start?), 変化点 (behavior change after the instruction?), and 要求点 (the time the instruction demands an effect?). Are they per aircraft or per team? Are they single steps or ranges? Is the unit sim steps or video seconds (4x)?
6. **Translator input and output format**: the Translator does not exist yet. Which trajectory renderer and window will it use, and what is its output shape (free text, or JSON per aircraft like the 17 tactical variant)? The judge's trajectory renderer must match it byte-for-byte to be fair.
7. **Score definition**: the scale (0/1/2 as in the 17 annotation comparison, 1-4 as in 07, or 0-10), a single score vs. per-key-point scores, and whether rationale or evidence is required. Repeats and aggregation per window, aircraft, or instruction.
8. **Decoder dependency**: which library to add (`av`, `imageio-ffmpeg`, or `opencv-python-headless`), and as a core or optional dependency. Note that the README says pins use ranges in `pyproject.toml` while `uv.lock` pins exact versions.
9. **Results location and git policy**: `outputs/evaluation/` (gitignored JSON) vs. a tracked results directory (as in `workspace/*/results/`).
10. **Only 3 episodes exist** (one MHI, two ours), and only the MHI episode has human annotation (7 scenes). Is more LAG data coming from `mtd-rl-lag`? Will the new data keep the 64-column schema and the 10 fps / 4x recording convention?
