## Feature Brief: LLM as a Translator (two-layer) + LLM-as-a-judge evaluation for LAG

Rewritten 2026-11-04 after the requirement interview; scope widened from "judge only" to
"Translator + reusable LLM parts + judge". The authoritative decision list is
`.claude/docs/research/translator-requirements.md` (D1-D33); cite decision ids from there.
Facts: `feature-llm-judge-intent-translation-eval-codebase.md` (repo scan),
`llm-judge-intent-translation-eval.md` (gpt-6.1-sol API), `translator-contract-facts.md`
(instruction data, local serving), `openjev-vllm.md` (System One models; section
"Clef-Flash on vLLM"). All paths are under `.claude/docs/research/`.

### Current State
- Architecture: no Translator exists anywhere (only a role assignment). `evaluation/` is empty.
  Reusable library code lives in `src/mtd_rl_xavier/` (tests in `tests/`); experiments in
  `workspace/NN_name/`; root-level dirs are importable via pytest `pythonpath = ["."]` and `python -m`.
- Relevant files:
  - OpenAI Responses template: `src/weightreading/backends.py` (`reasoning.effort`, `store=False`,
    `max_retries=0`, lazy env-only key after the paid gate, injectable `client_factory`).
  - Paid gate: `src/weightreading/cli.py` (`--execute` + `--confirm-paid-api`, preview otherwise).
  - Strict config: `src/weightreading/config.py` + `configs/weightreading/run.yaml` (YAML -> frozen
    dataclasses, unknown keys rejected, no default model, pricing table). No pydantic in the repo.
  - OpenAI-compatible `base_url` chat example: `mentalmodeling/llm_x/backends.py`
    (`OpenAIChatBackend(compatible=True, endpoint=...)`; vendored, read-only).
    `src/mtd_rl_xavier/experiment/common.py::RecordingOpenAIBackend` rejects non-OpenAI URLs (do not reuse
    for local servers).
  - LAG episodes: `workspace/13_LAG/lag_data/dataset.py::load_episode` (hash-verified, 64 columns, one row
    per step per object, 0.2 s/step, 3 no-instruction 2v2 episodes of 293-495 steps).
  - 17-format JSON-lines rendering: `workspace/17_commentary_LAG/commentary_experiment/{dataset.py,prompts.py}`
    (61-step all-object window ~48.7k tokens).
  - Judge video: `workspace/13_LAG/data/viz/action_videos/*.mp4` (20 fps, 1920x1080, frame = step).
  - Test fakes: `tests/weightreading/test_wr_cli_gate.py` (`ScriptedClient`, `RecordingFactory`).
- Patterns: frozen dataclasses, strict YAML, preview-by-default paid calls, immutable results in new
  directories, ruff (py310, line 120), tests under `tests/<area>/`.
- Gaps: no structured outputs, image input, httpx-based local clients, video decoding (no decoder
  installed), or vLLM anywhere in the repo; `src/` must not import from `workspace/`.

### Feature Goal
Build, as separate small parts, (1) an offline two-layer "LLM as a Translator" that turns a LAG
trajectory plus the instructions in effect into natural-language intent translations at the start
point, at change points and at request points, and (2) an LLM-as-a-judge (`gpt-6.1-sol`) that scores
each translation event against key points, with a YAML config deciding which inputs the judge sees and
how they are rendered, so judge-visible information matches what a human rater sees.

### Scope
- Include (parts, D2/D25):
  1. LLM call parts in `src/mtd_rl_xavier/llm/`, each independent:
     - OpenAI API part: Responses API; text and image inputs; strict `json_schema` structured output;
       `reasoning.effort`; never `temperature`; `store=False`; usage capture; injectable client; key read
       from `OPENAI_API_KEY` only after the paid gate.
     - Local vLLM chat part: OpenAI-compatible `chat.completions` client at a configurable `base_url`
       (lets any layer use a local generative model, D7).
     - System One part: `POST /v1/systemone` client (httpx; TCP or Unix socket; Bearer key from env);
       typed `noul` / `choice` / `score` questions; parses only fields both vLLM and `wev serve` return;
       budgets the state and treats a full token budget as an error, because vLLM silently truncates the
       state from the end.
     - Local vLLM server part: launcher that builds the `vllm serve` command (pinned revision,
       served-model-name, max-model-len, dtype, `--uds` or `--host 127.0.0.1`, API key via env var never
       argv), waits on a health check with a timeout, and tears down; plus an sbatch template for
       `151-partition` with `--gres=gpu:1`. Default model: `Cloudflare/clef-flash` @
       `fb4417d31c1d0f9e7e77f1141b40aaede7491e00`, vLLM >= 0.33.0 in a separate env (D26).
  2. LAG data parts in `src/mtd_rl_xavier/lag/`: an episode-loader interface with a 13_LAG adapter
     (promote the loader into `src/` rather than importing `workspace/`; swappable later, D12), and
     trajectory renderers: 17-format JSON lines (default, D22) and a compact detector variant (short
     window, target and main opponents, key fields, D27) with a token counter.
  3. Translator parts in `src/mtd_rl_xavier/translator/`:
     - Scenario schema: episode reference; instruction schedule (issue step, target = aircraft or team,
       text; first at step 0); request steps; optional external change points with a `source` label.
     - Detector layer interface with two implementations: System One (default clef-flash, one `noul`
       question such as "the motion is NOT explained by the previous translation", threshold in config)
       and external change points (D6).
     - Translation layer interface with an implementation over any LLM part (default OpenAI
       `gpt-6.1-terra`, D18); one output per instruction target (D17) with step, trigger
       (`start` / `change` / `instruction` / `request`), target, intent, evidence; output language in config.
     - `Translator` pipeline (D7/D9/D10/D24): translate at step 0; every 5 steps (D28) ask the detector
       with the compact window and the previous translation; re-translate on a detected or external
       change, on a new instruction, and on each request step; translation window = last N = 60 steps
       (D32) plus the previous translation; online by default, retrospective +M steps as a config
       option (D5/D33). Writes an event log that records exactly what each layer saw.
     - Strict YAML config (models per layer are swappable) and a CLI: preview by default; paid OpenAI
       calls only with `--execute --confirm-paid-api`; results in new directories, never overwritten.
  4. Judge in `evaluation/` with configs in `evaluation/config/` (D15):
     - One judge call per translation event (D20); a sample is built from a Translator event log plus a
       key-points file (supplied later by the user).
     - Six inputs, each with `include` and rendering options: instructions in effect; video frames from
       `action_videos` (D30; sampling, max frames, resize, `detail`, timestamp label before each frame;
       frame/step mapping per video); start/change/request timestamps with sources; trajectory (default:
       exactly what the translation layer saw); translated intent; key points.
     - Output (D29): per key point 0/1/2 with rationale + overall 1-5 with rationale, via strict JSON schema.
     - Frame extraction through PyAV (`av`) as an optional extra (D31); frame source pluggable so tests
       need no decoder. Preview summarises frames and estimated tokens/cost instead of embedding base64.
     - CLI with the same paid gate and immutable results.
  5. Tests for every part with fakes (no network, no GPU, no paid calls), example configs, an example
     scenario and key-points file with placeholder text, and a README per new area.
- Exclude:
  - Executing any paid API call, downloading weights, creating the vLLM env, or submitting SLURM jobs
    (run-time actions needing user approval).
  - Authoring the real key points; building a labelled change-point set to calibrate the detector threshold.
  - Video-LLM (e.g. Gemini) timestamping; instruction-conditioned recordings (later swap-in).
  - Committing customer instruction text to git (the user allows sending it to OpenAI, D11, but real
    scenarios with customer text stay in a gitignored location).

### Complexity Classification (from Codex)
- Classification: COMPLEX (Codex scope consult 2026-11-04, `.claude/logs/codex/20261103T220250Z-translator-scope.md`)
- Estimated files: 43 new + 3 modified overall; PHASE-1 MINIMAL 38 new + 3 modified (after consolidation; scope consult first estimated 55 new + 6 modified)
- Estimated LOC: ~4.5k production + ~2.3k tests overall; PHASE-1 ~3.8k + ~2.0k (Codex arch/plan consults; scope consult first estimated 4.0-5.8k + 2.6-4.0k)
- Implementation route: /team-execute (COMPLEX), phases P0 -> P1a||P1b -> P2 -> P3 -> P4, then LATER L1-L4 (`.claude/docs/plans/feature-llm-judge-intent-translation-eval-plan.md`)

### Integration Points
- `pyproject.toml`: add `av` as an optional extra and ensure `httpx` is a declared dependency if the new
  clients import it directly; extend `poe lint` to the new paths; `uv.lock` updated via uv.
- `src/weightreading/*`: patterns to mirror (do not couple the new parts to weightreading).
- `workspace/13_LAG` and `workspace/17_commentary_LAG`: loader and renderer logic promoted into `src/`;
  the workspace experiments stay unchanged (their saved results and provenance hashes must not change).
- `scripts/`: new sbatch template for the vLLM server (not executed).
- `.gitignore`: location for real scenarios and run outputs that contain customer text.

### Risks
- Detector out of domain: clef-flash was trained on business/tool/NLI decisions; LAG accuracy and
  calibration are unknown. Mitigation: threshold in config, probabilities logged, calibration run later.
- Silent state truncation in vLLM System One: mitigation is client-side budgeting plus a usage check.
- Shared-node exposure: `vllm serve` binds all interfaces by default. Mitigation: `--uds` in a 0700 dir or
  `127.0.0.1`, API key from env, client in the same SLURM job, no request-body logging.
- Cost: translation windows (~48.7k tokens for 60 steps) times events per episode, plus judge frames.
  Mitigation: preview with token/cost estimates; per-run caps.
- Customer text: the handoff CSV still labels it "do not send"; the user overrode this (D11). Keep it out
  of git and out of logs committed to the repo.
- Layering: promoting 13_LAG/17 logic into `src/` must reproduce their behaviour without editing the
  workspace experiments.
- Scope size: implement in dependency order (LLM parts and LAG parts first, then Translator, then judge).

### Success Criteria
- Each part has unit tests with fakes; `uv run pytest tests/llm tests/lag tests/translator tests/evaluation`
  (exact paths per plan) passes with no network or GPU.
- Translator pipeline on a scripted scenario with fake layers emits exactly the expected events (start,
  detected change on a tick, external change, instruction change, request points) and respects the
  online window bounds and the retrospective option.
- System One client: request shape, budget check, and error mapping verified with `httpx.MockTransport`.
- vLLM launcher: built command never contains the API key and binds as configured.
- Judge: request content matches the YAML config exactly (excluded inputs absent); output parsing,
  refusal, incomplete status, and schema violations covered.
- CLIs make no network call without the paid flags; one flag alone is an error.
- `uv run ruff check` passes on all new paths; `uv run poe test` shows no failures beyond the known
  ones listed in README.md.
