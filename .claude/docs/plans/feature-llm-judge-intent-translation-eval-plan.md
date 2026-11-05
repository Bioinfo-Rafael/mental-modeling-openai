# Plan: LLM as a Translator (two-layer) + LLM-as-a-judge for LAG

> **Historical plan.** Implemented and merged in PR #3 / #5; kept for the reasoning behind the design. Where it differs from the code or from `.claude/docs/research/translator-requirements.md` (e.g. decisions that superseded parts of this plan), the code and the decision log win.

Date: 2026-11-04. Status: design + plan for user approval (nothing implemented).
Mode: /feature MODE=existing. Requirements: `.claude/docs/research/archive/feature-llm-judge-intent-translation-eval-brief.md`.
Decisions D1-D33 (fixed): `.claude/docs/research/translator-requirements.md`.
Codex consults (read-only, `gpt-5.6-sol`), all under `.claude/logs/codex/`:
scope `20261103T220250Z-translator-scope.md`, architecture `20261103T221121Z-translator-arch.md`,
plan `20261103T223227Z-translator-plan.md`, validation: see "Validation Verdict".
Overriding goal from the user: accurate and simple (正確にかつシンプルに).

## Complexity

- Classification: COMPLEX (5+ files). Route: `/team-execute`, dependency-ordered phases.
- Scope consult estimate before consolidation: 55 new + 6 modified files, 6.6k-9.8k LOC.
- After consolidation (lead instruction: minimise modules, keep only the required separations):
  - PHASE-1 MINIMAL: 38 new + 3 modified files, ~3.8k production LOC, ~2.0k test LOC.
  - Overall (PHASE-1 + LATER): 43 new + 3 modified files, ~4.5k production LOC, ~2.3k test LOC.
  - (Codex counted 36 / 41 new; the two short READMEs for `llm/` and `lag/` were added after
    validation because the brief asks for a README per new area.)
  - Markdown and `uv.lock` churn excluded; no Python module expected above ~350 lines.
  - The validation revisions (endpoint checks, concrete-adapter E2E test, AST import test, cap
    tests) add no files; roughly a few dozen production LOC and 100-200 test LOC (estimate).
- Required separations kept: four LLM parts as separate modules with no cross-imports (D2);
  detector layer, translation layer and pipeline as separate modules; judge in `evaluation/` (D25).
  Everything else is one module per concern area (one config, one schema, one rendering module).

## Architecture Overview

One authoritative contract connects the two halves: the Translator event log. The judge consumes
it instead of rebuilding inputs, so the judge's default trajectory is byte-identical to what the
translation layer saw.

```text
13_LAG CSV + manifest
  -> mtd_rl_xavier.lag.dataset            (promoted loader; no workspace import)
  -> mtd_rl_xavier.lag.rendering          (lag17-jsonl for translation; compact for detector)
  -> translator.detector  (DetectionModel Protocol)   translator.translation (IntentModel Protocol)
  -> translator.pipeline  (chronological per-target state machine)
  -> outputs/translator/<run>/{run.json, events.jsonl, requests.jsonl, responses.jsonl}
  -> evaluation.rendering (six configurable inputs) -> evaluation.judge (JudgeModel Protocol)
  -> preview, or one gpt-6.1-sol call per translation event
```

- LLM parts (`src/mtd_rl_xavier/llm/`), independent, no imports among them (D2):
  `openai_responses.py` (paid Responses API), `system_one.py` (`/v1/systemone`),
  `vllm_chat.py` (local OpenAI-compatible chat), `vllm_server.py` (launcher only).
- Protocols live on the consumer side (`translator/detector.py`, `translator/translation.py`,
  `evaluation/judge.py`). Concrete clients are wired only in the composition roots
  `translator/cli.py` and `evaluation/cli.py`. No plugin registry, no shared LLM base class.
- Defaults: detector `Cloudflare/clef-flash` @ `fb4417d31c1d0f9e7e77f1141b40aaede7491e00` served
  as `clef-flash` by vLLM >= 0.33.0 (D26, supersedes D8/D19); translator `gpt-6.1-terra` (D18);
  judge exactly `gpt-6.1-sol` (D8).
- Commands (no `__main__.py` needed): `uv run python -m mtd_rl_xavier.translator.cli ...`,
  `uv run python -m evaluation.cli ...`, `uv run python -m mtd_rl_xavier.llm.vllm_server ...` (LATER).
- Gating (D14): every CLI previews by default and constructs no client and reads no credential.
  `--confirm-paid-api` alone is an argparse error (exit 2). Any config with an OpenAI layer needs
  both `--execute --confirm-paid-api`; one alone is exit 2. A local-only Translator config
  (System One + vLLM chat) runs with `--execute` alone: free, but explicit so server traffic is
  intentional. The judge always needs both flags. `OPENAI_API_KEY` is read only after the paired
  gate; the local key variable only after `--execute`. The local `/v1/models` preflight runs before
  the paid client is built, so a missing detector cannot follow a paid step-0 call.
- PHASE-1 MINIMAL delivers: promoted loader and both renderers; OpenAI and System One clients;
  online two-layer Translator with fake or real layers; exact-input event log; text-only judge
  request construction and preview (zero paid calls); strict configs, immutable outputs, provenance,
  offline tests. LATER adds: local vLLM chat; vLLM launcher + sbatch; retrospective `+M`;
  PyAV/Pillow frames and text+video judging; an exact Qwen tokenizer only if the byte bound proves
  too strict.

## Module Structure

Phase column: `P0` = pre-implementation integration step; `P1` = PHASE-1 MINIMAL (built in
P1a/P1b/P2/P3, see Implementation Phases); `LATER` = after PHASE-1 works.

| Path | Responsibility | ~LOC | Phase |
|---|---|---:|---|
| `src/mtd_rl_xavier/{llm,lag,translator}/__init__.py` | Package markers, no eager imports | 5 each | P0 |
| `src/mtd_rl_xavier/llm/openai_responses.py` | Responses API request/result types, text+image input, strict `json_schema`, usage/refusal/incomplete, lazy env-only client factory | 180 | P1 |
| `src/mtd_rl_xavier/llm/system_one.py` | Typed questions/answers, TCP/UDS httpx, fixed question set, byte budget, status mapping | 310 | P1 |
| `src/mtd_rl_xavier/llm/vllm_chat.py` | Local OpenAI-compatible chat client, explicit base URL, optional UDS | 150 | LATER |
| `src/mtd_rl_xavier/llm/vllm_server.py` | Safe argv builder, `/health` polling, subprocess context manager, terminate/wait/kill | 250 | LATER |
| `src/mtd_rl_xavier/llm/README.md` | The four LLM parts, gating, local-server security | md | P1 |
| `src/mtd_rl_xavier/lag/README.md` | Loader, renderers, promotion/parity note | md | P1 |
| `src/mtd_rl_xavier/lag/dataset.py` | Promoted 13_LAG definitions, `load_episode`, helpers, `last_complete_step` | 160 | P1 |
| `src/mtd_rl_xavier/lag/rendering.py` | Generalised 17-format JSONL renderer + compact detector renderer | 280 | P1 |
| `src/mtd_rl_xavier/translator/schema.py` | Scenario, target, trigger, state, event-record, provenance dataclasses | 230 | P1 |
| `src/mtd_rl_xavier/translator/config.py` | Strict run-config and scenario YAML loading | 300 | P1 |
| `src/mtd_rl_xavier/translator/detector.py` | `DetectionModel` Protocol, `DetectorLayer`, threshold, exact detector input | 150 | P1 |
| `src/mtd_rl_xavier/translator/translation.py` | `IntentModel` Protocol, `TranslationLayer`, prompt + output schema, strict parsing | 170 | P1 |
| `src/mtd_rl_xavier/translator/pipeline.py` | Chronological per-target state machine, collisions, windows, caps, events | 330 | P1 (retrospective LATER) |
| `src/mtd_rl_xavier/translator/cli.py` | Composition root, gate, preview, artifacts, provenance | 280 | P1 |
| `src/mtd_rl_xavier/translator/README.md` | CLI, event contract, local-server needs, sensitive-data rules | md | P1 |
| `configs/translator/run.yaml` | Full default run config | 75 | P1 |
| `configs/translator/scenario.example.yaml` | Placeholder-only 13_LAG scenario | 45 | P1 |
| `evaluation/__init__.py` | Judge package marker | 5 | P0 |
| `evaluation/configuration.py` | Strict judge YAML (named so it does not collide with `evaluation/config/`) | 240 | P1 |
| `evaluation/schemas.py` | Key points, judge result, strict output-schema builder and parser | 260 | P1 |
| `evaluation/rendering.py` | Six input renderers, request + preview builder; later `FrameSource`/PyAV | 210 (+140 LATER) | P1 / LATER |
| `evaluation/judge.py` | `JudgeModel` Protocol, one call per event, refusal/incomplete/schema checks | 220 | P1 |
| `evaluation/cli.py` | Composition root, paid gate, event/key-point matching, artifacts | 260 | P1 |
| `evaluation/config/judge-text.yaml` | Text-only judge config | 80 | P1 |
| `evaluation/config/judge-text-video.yaml` | Text+video judge config | 90 | LATER |
| `evaluation/examples/key-points.example.yaml` | Placeholder per-event key points | 25 | P1 |
| `evaluation/README.md` | Judge inputs, preview/execute, video leakage, optional setup | md | P1 |
| `scripts/translator_vllm.sbatch` | One-GPU same-job server + client template (not executed) | 60 | LATER |
| `tests/llm/{llm_fakes.py,test_llm_clients.py}` | Fakes; OpenAI + System One contracts | 310 | P1 |
| `tests/llm/test_llm_local_runtime.py` | Local chat + launcher | 240 | LATER |
| `tests/lag/test_lag_contract.py` | Byte parity with workspace 13/17 code, end step, post-death rows | 280 | P1 |
| `tests/translator/{tr_fakes.py,test_tr_contracts.py,test_tr_layers.py,test_tr_pipeline_cli.py}` | Fakes; config/scenario; layers; pipeline + CLI | 710 | P1 |
| `tests/evaluation/{eval_fakes.py,test_eval_contracts.py,test_eval_render_judge.py,test_eval_cli.py}` | Fakes; config/schema; rendering/judge; CLI + E2E | 670 (+120 LATER) | P1 |
| `pyproject.toml`, `uv.lock`, `.gitignore` | Package list, deps, lint scope, ignores | small | modify |

`src/mtd_rl_xavier/paths.py` and root `README.md` need no change (composition roots use
`paths.ROOT` / `paths.OUTPUTS`). The workspace experiments 13/17 stay byte-identical.

## Public Interfaces

```python
# llm/openai_responses.py
@dataclass(frozen=True)
class OpenAIRequest:
    model: str; instructions: str; input_items: tuple[dict[str, Any], ...]
    schema_name: str; schema: dict[str, Any]; reasoning_effort: str; max_output_tokens: int
@dataclass(frozen=True)
class LLMUsage: input_tokens: int; cached_input_tokens: int; output_tokens: int; reasoning_tokens: int
@dataclass(frozen=True)
class OpenAIResult:
    response_id: str | None; model: str; status: str; output_text: str
    output_items: tuple[dict[str, Any], ...]; refusal: str | None; incomplete_reason: str | None
    usage: LLMUsage; raw: dict[str, Any]
class OpenAIResponsesClient:
    def __init__(self, client: Any) -> None: ...
    def create(self, request: OpenAIRequest) -> OpenAIResult: ...
    # sends model, instructions, input, text.format{json_schema, strict: True}, reasoning.effort,
    # max_output_tokens, store=False. Never temperature / top_p / top_logprobs.
def make_openai_client() -> Any: ...   # reads OPENAI_API_KEY here only; max_retries=0

# llm/system_one.py
@dataclass(frozen=True)
class NoulQuestion: question_id: str; instructions: str; true_criterion: str; false_criterion: str
@dataclass(frozen=True)
class ChoiceQuestion: question_id: str; instructions: str; criteria: tuple[tuple[str, str | None], ...]
@dataclass(frozen=True)
class ScoreQuestion: question_id: str; instructions: str; levels: tuple[str, ...]
SystemOneQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion
# NoulAnswer(question_id, p_yes)   # only `noul`: OpenJev `wev serve` returns no probabilities for noul
# ChoiceAnswer(question_id, choice, probabilities); ScoreAnswer(question_id, score, probabilities)
# SystemOneUsage(input_tokens, output_tokens). Ignored: confidence, legend, diagnostics, joint, id, latency.
@dataclass(frozen=True)
class SystemOneResult: model: str; answers: tuple[SystemOneAnswer, ...]; usage: SystemOneUsage; raw: dict[str, Any]
FIXED_FRAMING_RESERVE_TOKENS: int   # named internal constant; not reducible from config
def conservative_input_token_bound(*, state: str, questions: tuple[SystemOneQuestion, ...]) -> int:
    ...  # UTF-8 bytes of state + questions + FIXED_FRAMING_RESERVE_TOKENS
class SystemOneClient:
    def __init__(self, *, base_url: str | None, uds_path: Path | None, model: str,
                 questions: tuple[SystemOneQuestion, ...], input_token_budget: int,
                 timeout_s: float, api_key_env: str,
                 transport: httpx.BaseTransport | None = None) -> None: ...
    # endpoint validated before any transport exists: exactly one of loopback TCP / UDS;
    # UDS parent owned by the user with mode 0700; TCP host literally 127.0.0.1 (or ::1);
    # reject userinfo, remote hosts, 0.0.0.0; the Bearer key env var is required
    def health(self) -> None: ...                    # authenticated GET /v1/models
    def decide(self, state: str) -> SystemOneResult: ...

# llm/vllm_chat.py (LATER)
class VLLMChatClient:
    def __init__(self, client: Any) -> None: ...
    def complete(self, request: VLLMChatRequest) -> VLLMChatResult: ...
def make_vllm_chat_client(*, base_url: str, api_key_env: str, uds_path: Path | None) -> Any: ...

# llm/vllm_server.py (LATER)
@dataclass(frozen=True)
class VLLMServerConfig:
    executable: Path; model: str; revision: str; served_model_name: str; max_model_len: int
    dtype: str; uds_path: Path | None; host: str | None; port: int | None
    startup_timeout_s: float; shutdown_timeout_s: float
def build_vllm_argv(config: VLLMServerConfig) -> tuple[str, ...]: ...
@contextmanager
def running_vllm_server(config: VLLMServerConfig, *, popen_factory=subprocess.Popen) -> Iterator[ServerEndpoint]: ...

# lag/dataset.py
def load_episode(path: Path, *, verify_hash: bool = True) -> pd.DataFrame: ...
def last_complete_step(frame: pd.DataFrame) -> int: ...   # last step with all 4 aircraft has_csv_record == 1
def history_steps(query_step: int, *, length: int | None = None, stride: int = 1,
                  offsets: Sequence[int] | None = None) -> list[int]: ...
def select_records(frame, *, steps, object_ids, columns) -> pd.DataFrame: ...
def index_records(frame) -> dict[tuple[int, str], dict[str, object]]: ...

# lag/rendering.py
@dataclass(frozen=True)
class RenderedTrajectory:
    renderer: str; renderer_version: str; start_step: int; end_step: int; anchor_step: int
    object_ids: tuple[str, ...]; fields: tuple[str, ...]; text: str; sha256: str
def render_lag17_jsonl(frame, *, start_step: int, end_step: int, anchor_step: int,
                       episode_alias: str) -> RenderedTrajectory: ...
def render_detector_compact(frame, *, start_step: int, end_step: int, anchor_step: int,
                            object_ids: tuple[str, ...], fields: tuple[str, ...],
                            float_precision: int) -> RenderedTrajectory: ...

# translator/schema.py (all frozen)
TargetRef(kind: Literal["aircraft", "team"], target_id: str)
EpisodeRef(episode_id, alias, csv_path: Path, video_path: Path | None)
Instruction(instruction_id, issue_step: int, target: TargetRef, text)
RequestPoint(request_id, step: int, targets: tuple[TargetRef, ...], source)
ExternalChangePoint(change_id, step: int, target: TargetRef, source)
Scenario(schema_version, scenario_id, episode, instructions, requests, external_change_points)
TriggerReason(kind: Literal["start", "instruction", "change", "request"], source, reference_id, probability)
IntentTranslation(intent: str, evidence: tuple[str, ...])
TargetState(target, current_instruction_id, previous_event_id, previous_translation, last_translation_step, active)

# translator/detector.py
class DetectionModel(Protocol):
    def decide(self, state: str) -> DetectionModelReply: ...     # model, probability, usage, raw
class DetectorLayer:
    def __init__(self, model: DetectionModel, *, threshold: float) -> None: ...
    def detect(self, context: DetectionContext) -> DetectionResult: ...  # changed, probability, threshold, exact_input

# translator/translation.py
class IntentModel(Protocol):
    def translate(self, request: TranslationModelRequest) -> TranslationModelReply: ...
# TranslationModelRequest(target, instructions_in_effect, primary_instruction_id, trajectory,
#                         previous_translation, output_language)
class TranslationLayer:
    def __init__(self, model: IntentModel, *, output_language: str) -> None: ...
    def translate(self, request: TranslationModelRequest) -> TranslationModelReply: ...

# translator/pipeline.py
class Translator:
    def __init__(self, *, detector: DetectorLayer, translation: TranslationLayer,
                 detector_renderer: Callable[..., RenderedTrajectory],
                 translation_renderer: Callable[..., RenderedTrajectory]) -> None: ...
    def run(self, *, frame: pd.DataFrame, scenario: Scenario, config: TranslatorConfig,
            sink: EventSink) -> TranslatorSummary: ...

# evaluation/schemas.py
KeyPoint(key_point_id, text); EventKeyPoints(event_id, key_points)
CriterionResult(key_point_id, rationale, score)   # 0/1/2
JudgeResult(event_id, key_points, overall_rationale, overall_score, usage, response_id)  # 1..5
def build_judge_output_schema(key_points: tuple[KeyPoint, ...]) -> dict[str, Any]: ...
def parse_judge_output(*, event_id: str, output_text: str, key_points: tuple[KeyPoint, ...],
                       usage: dict[str, int], response_id: str | None) -> JudgeResult: ...

# evaluation/rendering.py
class FrameSource(Protocol):
    def read_frames(self, *, path: Path, frame_indices: tuple[int, ...], max_width: int,
                    max_height: int, jpeg_quality: int) -> tuple[EncodedFrame, ...]: ...
def map_step_to_frame(step: int, *, step_offset: int, frame_offset: int, steps_per_frame: int,
                      frame_count: int) -> int: ...
def build_judge_request(*, event: dict, event_log: tuple[dict, ...], key_points: EventKeyPoints,
                        config: JudgeConfig, frame_source: FrameSource | None
                        ) -> tuple[JudgeRequest, JudgePreview]: ...

# evaluation/judge.py
class JudgeModel(Protocol):
    def score(self, request: JudgeRequest) -> JudgeModelResponse: ...
def judge_one(*, request: JudgeRequest, key_points: EventKeyPoints, model: JudgeModel) -> JudgeResult: ...
```

## Translator Pipeline Algorithm

Unit of translation: one exact target, `aircraft:<id>` or `team:<id>` (D17). Each target has its
own immutable `TargetState`; aircraft and team targets coexist. A later instruction for the same
target replaces its current instruction (D21). All instructions issued at a step are applied before
any target is processed, so every target translated at that step sees the same
`instructions_in_effect`; the exact-target instruction is `primary_instruction_id`, the others are
context (D3).

```text
frame      = load_episode(scenario.episode.csv_path)            # hash-verified
last_step  = last_complete_step(frame)  # 340 / 293 / 495; the ACMI-only final step is excluded
N, W, T    = translation.window.past_steps (60, D32), detector.window.past_steps, tick_interval_steps (5, D28)
M          = future_steps if visibility.mode == retrospective else 0      # online default (D33, D5)
validate (preflight, before any client exists):
    >= 1 instruction at step 0 (D16); ids unique; all steps in [0, last_step];
    at most one instruction per exact target per step;
    every request / external change references a target that has an instruction issued at or
    before its step (D17: translation units are instruction targets)

for step in 0 .. last_step:
    for instruction issued at step (sorted by instruction_id):              # D21, D24
        active_instructions[instruction.target] = instruction
        create or replace states[instruction.target] (previous translation kept)
    for target that becomes dead at this step (aircraft: first alive != 1; team: all aircraft dead):
        emit target_closed; mark inactive (never reactivates)

    for target in canonical (kind, target_id) order:
        reasons = []
        if step == 0 and instruction for target issued at 0:   reasons += start        # D16
        if step > 0 and instruction for target issued at step: reasons += instruction  # D24
        reasons += external change points for target at step  (source label kept)      # D6
        reasons += request points selecting target at step                             # D4, D10
        if target inactive:
            if reasons: emit trigger_suppressed(reason=target_dead, reasons)
            continue
        is_tick = step > 0 and step % T == 0                    # fixed episode grid (D28)
        if is_tick and state.previous_translation is not None:  # every tick, even if reasons exist (D9)
            view = render_detector_compact([max(0, step-W+1), min(last_step, step+M)],
                                           target + main opponents, configured fields)  # D27
            state_text = target-specific previous translation + view.text
            stop_if_cap_reached("detector")                     # send nothing beyond the cap
            detection = detector.detect(...)                    # byte bound checked before send
            emit detector_call (exact text, probability, threshold)
            if detection.probability >= threshold: reasons += change(source=detector)    # D6
        if not reasons: continue
        view = render_lag17_jsonl([max(0, step-N+1), min(last_step, step+M)], anchor=step,
                                  episode_alias)                                        # D22, D32
        stop_if_cap_reached("translation")
        reply = translation.translate(target, instructions_in_effect, primary_instruction_id,
                                      view, state.previous_translation, output_language) # D10, D23
        emit translation_event(all reasons, primary trigger, exact input text, output, usage)
        states[target] = state with reply as previous translation
```

Rules:
- At most one translation per target per step. Primary trigger precedence:
  `start` > `instruction` > external `change` > detector `change` > `request`. Every coincident
  reason is kept in `trigger_reasons`, so a request coinciding with an instruction change is not lost.
- Every scheduled tick of a live target that already has a translation calls the detector (D9/D28),
  also when a start/instruction/external/request trigger exists at that step; its result is logged
  and merged into the same single translation. Ticks stay on the fixed grid: a request at step 3
  does not move the ticks at 5, 10, 15.
- A target's first translation is at step 0 (trigger `start`) when it is instructed at step 0. A
  target first instructed later is translated at its issue step with trigger `instruction` and no
  previous translation (D24); see Open Questions (this is the one point where the validator read
  D16/D21 more strictly).
- Windows clip to step 0 at the start and to `last_step` at the end. The previous translation is
  always target-specific.
- Death: rows of downed aircraft continue (`alive == 0`, verified on all three episodes), so
  windows after a death render normally; only the dead exact target stops. Requests, changes or
  instructions for a dead target become `trigger_suppressed` records; no model call, no judge call.
- Retrospective mode (LATER) lets both layers see through `min(last_step, step + M)`.
- Caps: preview reports mandatory translations, scheduled detector calls, the detector-positive
  upper bound, the caps and the headroom; an upper bound above a cap is a warning, not a refusal.
  Immediately before each detector or translation call, if the completed counter equals its cap,
  nothing is sent and the run stops with `status: cap_reached`; `run.json` records the unsent
  operation (kind, target, step) and the completed counts. No fifth event-record type.

## Config YAML Schemas

All mappings use exact key sets; unknown, missing or wrong-type keys fail; booleans never count as
integers; there is no default model in code. Relative output roots resolve from `paths.ROOT`;
scenario paths resolve relative to the scenario file.

### Translator run config (`configs/translator/run.yaml`)

```yaml
schema_version: 1

pipeline:
  tick_interval_steps: 5            # D28
  visibility:
    mode: online                    # online | retrospective (D5, D33)
    future_steps: 0                 # 0 for online; > 0 for retrospective (LATER)

limits:
  max_detector_calls: 500           # per run; never exceeded (rule in Validation Verdict)
  max_translation_calls: 250

detector:
  backend: system_one
  model: Cloudflare/clef-flash
  revision: fb4417d31c1d0f9e7e77f1141b40aaede7491e00
  served_model_name: clef-flash
  connection:
    base_url: http://127.0.0.1:8000
    uds_path: null                  # preferred on shared nodes: socket in a 0700 directory
    api_key_env: VLLM_API_KEY       # variable NAME only; the value never enters YAML or argv
  timeout_s: 30
  threshold: 0.65                   # provisional until a labelled LAG check
  input_token_budget: 16384         # MUST equal the serving cap: min(16384, --max-model-len) for
                                    # vLLM + clef-flash (4096 for OpenJev wev serve); never larger.
                                    # With the LATER launcher, equality is asserted mechanically.
  question:
    id: unexplained_change
    type: noul
    instructions: >-
      The motion in the trajectory window is NOT explained by the previous
      natural-language intent translation.
    criteria:
      "true": There is a material motion change requiring a new intent explanation.
      "false": The previous intent still explains the observed motion.
  window:
    past_steps: 10                  # D27 short window
    entity_scope: target_and_opponents
    fields: [alive, state_valid, lon_deg, lat_deg, alt_m, yaw_deg, tas_mps, vn_mps, ve_mps, vu_mps]
    float_precision: 3

translation:
  backend: openai_responses         # openai_responses | vllm_chat (LATER)
  connection: null                  # required mapping for vllm_chat: {base_url, uds_path, api_key_env}
  model: gpt-6.1-terra              # D18
  reasoning_effort: medium          # null for vllm_chat unless the local model supports it
  output_language: English          # D23
  max_output_tokens: 4000
  timeout_s: 1800
  window:
    past_steps: 60                  # D32
    renderer: lag17-jsonl           # D22

output:
  root: outputs/translator

pricing:                            # USD per 1M tokens, preview only
  gpt-6.1-terra: {input: 1.00, cached_input: null, output: 5.00}   # null = priced at input rate
```

### Judge text-only config (`evaluation/config/judge-text.yaml`)

```yaml
schema_version: 1
model: gpt-6.1-sol                  # the only accepted value (D8)
reasoning_effort: medium
max_output_tokens: 25000

limits:
  max_judged_events: 250

inputs:                             # exactly these six blocks (D15)
  instructions:
    include: true
    renderer: chronological
    show_ids: true
    show_targets: true
    show_issue_steps: true
  video_frames:
    include: false
    renderer: sampled-jpeg
    source: scenario_action_video
    mapping: {step_offset: 0, frame_offset: 0, steps_per_frame: 1}
    sampling: {strategy: event_window, before_steps: 10, after_steps: 10, stride_steps: 5, max_frames: 8}
    resize: {max_width: 960, max_height: 540, jpeg_quality: 85}
    detail: high
  timestamps:
    include: true
    renderer: chronological
    scope: through_event
    show_sim_seconds: true
    show_sources: true
  trajectory:
    include: true
    renderer: translation_input_verbatim   # exactly what the translation layer saw
    show_window_metadata: true
  translated_intent:
    include: true
    renderer: intent-and-evidence
  key_points:
    include: true
    renderer: numbered-with-fixed-rubric   # D29 anchors fixed in code

output:
  root: outputs/evaluation

pricing:
  gpt-6.1-sol:
    input: 2.00
    cached_input: 0.10
    output: 10.00
    long_context_threshold_tokens: 272000
    long_input_multiplier: 2.0
    long_output_multiplier: 1.5
```

### Judge text+video config (`evaluation/config/judge-text-video.yaml`, LATER)

```yaml
schema_version: 1
model: gpt-6.1-sol
reasoning_effort: medium
max_output_tokens: 25000

limits:
  max_judged_events: 250

inputs:
  instructions:
    include: true
    renderer: chronological
    show_ids: true
    show_targets: true
    show_issue_steps: true
  video_frames:
    include: true
    renderer: sampled-jpeg
    source: scenario_action_video          # workspace/13_LAG/data/viz/action_videos (D30)
    mapping: {step_offset: 0, frame_offset: 0, steps_per_frame: 1}   # 20 fps, frame = step
    sampling: {strategy: event_window, before_steps: 20, after_steps: 20, stride_steps: 5, max_frames: 12}
    resize: {max_width: 960, max_height: 540, jpeg_quality: 85}
    detail: high
  timestamps:
    include: true
    renderer: chronological
    scope: through_event
    show_sim_seconds: true
    show_sources: true
  trajectory:
    include: true
    renderer: translation_input_verbatim
    show_window_metadata: true
  translated_intent:
    include: true
    renderer: intent-and-evidence
  key_points:
    include: true
    renderer: numbered-with-fixed-rubric

output:
  root: outputs/evaluation

pricing:
  gpt-6.1-sol:
    input: 2.00
    cached_input: 0.10
    output: 10.00
    long_context_threshold_tokens: 272000
    long_input_multiplier: 2.0
    long_output_multiplier: 1.5
```

Each frame is preceded by an `input_text` label (frame index, step, sim seconds). Another video
needs its own explicit mapping (the original 10 fps recordings would use `steps_per_frame: 2`).

## Scenario & Event Log Schemas

### Scenario (committed example: placeholder text only; real ones under gitignored `private/translator/`)

Exact top-level keys; at least one instruction at step 0 (D16); unique ids; steps within
`[0, last_step]`; two instructions for the same exact target at the same step are invalid; every
request / external change references a target already instructed at or before its step.

```yaml
schema_version: 1
scenario_id: placeholder-episode-01
episode:
  episode_id: mhi_v1.3_ic100601_win
  alias: episode_01                       # neutral alias; win/loss id never reaches a model
  csv_path: ../../workspace/13_LAG/data/processed/mhi_v1.3_ic100601_win.csv
  video_path: ../../workspace/13_LAG/data/viz/action_videos/mhi_v1.3_ic100601_win_with_actions.mp4
instructions:
  - instruction_id: placeholder-aircraft-001
    issue_step: 0
    target: {kind: aircraft, id: A0100}
    text: "[PLACEHOLDER] Perform the assigned single-aircraft task."
  - instruction_id: placeholder-team-001
    issue_step: 0
    target: {kind: team, id: A}
    text: "[PLACEHOLDER] Perform the assigned team task."
  - instruction_id: placeholder-team-002
    issue_step: 20
    target: {kind: team, id: A}
    text: "[PLACEHOLDER] Switch to the revised team task."
requests:
  - request_id: placeholder-request-001
    step: 10
    targets: [{kind: aircraft, id: A0100}]
    source: placeholder-human
external_change_points:
  - change_id: placeholder-change-001
    step: 15
    target: {kind: aircraft, id: A0100}
    source: placeholder-annotation
```

### Event log (`events.jsonl`)

Record types: `detector_call`, `translation_event`, `target_closed`, `trigger_suppressed`. Every
record has `schema_version`, `sequence`, `record_type`, `scenario_id`, `step`, `target`. Call
records contain the exact rendered text the model saw, not only a hash.
`event_id = <scenario_id>:<kind>:<target_id>:<step:06d>` (unique: one translation per target per step).

```json
{
  "schema_version": 1, "sequence": 4, "record_type": "translation_event",
  "event_id": "placeholder-episode-01:aircraft:A0100:000010",
  "scenario_id": "placeholder-episode-01",
  "episode": {"episode_id": "mhi_v1.3_ic100601_win", "alias": "episode_01", "csv_sha256": "..."},
  "step": 10, "sim_time_s": 2.0,
  "target": {"kind": "aircraft", "id": "A0100"},
  "primary_trigger": "request",
  "trigger_reasons": [{"kind": "request", "source": "placeholder-human",
                       "reference_id": "placeholder-request-001", "probability": null}],
  "input": {
    "instructions_in_effect": [
      {"instruction_id": "placeholder-aircraft-001", "issue_step": 0,
       "target": {"kind": "aircraft", "id": "A0100"},
       "text": "[PLACEHOLDER] Perform the assigned single-aircraft task."},
      {"instruction_id": "placeholder-team-001", "issue_step": 0,
       "target": {"kind": "team", "id": "A"},
       "text": "[PLACEHOLDER] Perform the assigned team task."}],
    "primary_instruction_id": "placeholder-aircraft-001",
    "trajectory": {"renderer": "lag17-jsonl", "renderer_version": "1", "start_step": 0,
      "end_step": 10, "anchor_step": 10, "object_ids": ["A0100", "A0200", "B0100", "B0200"],
      "fields": [], "text": "{\"step\":0,...}\n...", "sha256": "..."},
    "previous_translation": {"event_id": "placeholder-episode-01:aircraft:A0100:000000",
      "intent": "[PLACEHOLDER] Initial inferred intent.",
      "evidence": ["[PLACEHOLDER] Initial state evidence."]},
    "output_language": "English"
  },
  "output": {"intent": "[PLACEHOLDER] Updated inferred intent.",
             "evidence": ["[PLACEHOLDER] Trajectory evidence."]},
  "model": {"backend": "openai_responses", "model": "gpt-6.1-terra", "revision": null},
  "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
}
```

```json
{
  "schema_version": 1, "sequence": 7, "record_type": "detector_call",
  "call_id": "placeholder-detector-call-001", "scenario_id": "placeholder-episode-01",
  "step": 15, "target": {"kind": "team", "id": "A"},
  "input": {"trajectory": {"renderer": "lag-detector-compact", "text": "[PLACEHOLDER COMPACT TRAJECTORY]",
              "sha256": "..."},
            "previous_translation": "[PLACEHOLDER] Initial inferred intent.",
            "state_text": "[PLACEHOLDER exact state string sent]",
            "question": {"id": "unexplained_change", "type": "noul",
                         "instructions": "...", "criteria": {"true": "...", "false": "..."}},
            "question_sha256": "..."},
  "output": {"probability": 0.72, "threshold": 0.65, "changed": true},
  "raw_response": {"model": "clef-flash", "answers": {"unexplained_change": {"type": "noul", "noul": 0.72}},
                   "usage": {"input_tokens": 0, "output_tokens": 0}},
  "model": {"backend": "system_one", "model": "clef-flash",
            "revision": "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"},
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

`run.json`: git commit + dirty flag, SHA-256 of every relevant `.py`, config and scenario hashes,
episode CSV + manifest hashes, renderer/prompt/question versions and hashes, model ids and
revisions, preview/execute mode, start/end/status/error, counts of planned/attempted/completed/
uncertain calls. `requests.jsonl` is written before each send (write-before-send);
`responses.jsonl` keeps raw responses.

### Key-points file (supplied later by the user; committed example is placeholder-only)

Exactly one entry for every `translation_event` in the complete event log (D20): a bijection.
Missing, duplicate, unknown or extra event ids are preflight errors; event filtering is not
supported. Key-point ids are unique per event. Scoring anchors are fixed by D29 and not repeated
per file. If the complete event set exceeds `max_judged_events`, preflight fails (judge calls are
exactly countable).

```yaml
schema_version: 1
events:
  - event_id: placeholder-episode-01:aircraft:A0100:000010
    key_points:
      - key_point_id: kp-001
        text: "[PLACEHOLDER] The core purpose that the translation should reproduce."
      - key_point_id: kp-002
        text: "[PLACEHOLDER] The main action or motion that the translation should mention."
```

## Judge Output Schema

Built per event from its key-point ids and sent as
`text.format = {type: json_schema, name: judge_verdict, strict: true, schema: ...}`.
Example for `kp-001`, `kp-002` (rationale is ordered before score so grounds come first):

```json
{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "key_points": {
      "type": "array", "minItems": 2, "maxItems": 2,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "key_point_id": {"type": "string", "enum": ["kp-001", "kp-002"]},
          "rationale": {"type": "string", "minLength": 1},
          "score": {"type": "integer", "enum": [0, 1, 2]}
        },
        "required": ["key_point_id", "rationale", "score"]
      }
    },
    "overall": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "rationale": {"type": "string", "minLength": 1},
        "score": {"type": "integer", "minimum": 1, "maximum": 5}
      },
      "required": ["rationale", "score"]
    }
  },
  "required": ["key_points", "overall"]
}
```

Post-parse checks (always, even with strict output): status `completed`; no `refusal` content
item; exactly one JSON object; exact key sets; `bool` rejected as integer; every requested
key-point id exactly once (no duplicate, unknown or missing); rationales non-blank after strip;
scores 0/1/2 and 1-5. No repair, retry, coercion or clamping; the raw response is kept even when
validation fails. Anchors (D29): 2 = core purpose and main action reproduced, 1 = partial,
0 = missing or a different purpose.

## Data Flow

1. Translator preview: parse run config + scenario -> load and hash episode -> validate targets,
   steps, fields, pricing -> classify layers (OpenAI / local) -> enumerate mandatory translations
   and the maximum detector ticks -> render representative windows for size estimates -> new
   directory (`mkdir(exist_ok=False)`) -> `run.json` + `preview.json` (calls, bytes/tokens/cost,
   cap headroom, hashes, `api_requests_made: 0`). No client, no credential.
2. Translator execute: gate -> validate -> local `/v1/models` preflight -> new run directory +
   provenance -> build clients in `translator/cli.py` -> pipeline -> write-before-send request
   record -> raw response -> parsed result + event record -> final counts/status.
3. Judge preview: load judge config, complete event log and key-points file -> bijection between
   ALL translation events and key-point entries (no filtering; count <= `max_judged_events`) ->
   one request per event -> only included blocks ->
   trajectory copied verbatim from `event.input.trajectory.text` -> frames (LATER) summarised by
   index, step, size, bytes, SHA-256, detail (never base64) -> text tokens `ceil(chars/4)` and
   image tokens `ceil(w/32)*ceil(h/32)*1.2` (both labelled approximate; the image multiplier is
   unverified for `gpt-6.1-sol`) -> long-context warning above 272k -> `api_requests_made: 0`.
4. Judge execute: same validation -> both flags -> fail early if video is on and PyAV/Pillow are
   missing -> new run directory -> per event: write request + hash, one Responses call, persist raw
   response, parse strictly, write result -> final provenance and status.

## Error Handling

- Exceptions stay local to their area: `OpenAIClientError`, `OpenAIRefusalError`,
  `OpenAIIncompleteError`; `SystemOneBudgetError` (pre-send bound or `usage.input_tokens >= budget`),
  `SystemOneRequestError` (400/422), `SystemOneAuthError` (401), `SystemOneModelError` (404),
  `SystemOneServerError` (500 and other 5xx), `SystemOneUnsupportedError` (501),
  `SystemOneTransportError` (connect failure, timeout), `SystemOneProtocolError` (missing common
  fields, unexpected status); every other 4xx maps to `SystemOneRequestError`; error messages
  never include response bodies or the key; `LagDataError`, `RenderError`; `TranslatorConfigError`, `ScenarioError`,
  `PipelineError`; `JudgeConfigError`, `KeyPointsError`, `FrameError`, `JudgeSchemaError`.
- System One parsing reads only the fields both vLLM and `wev serve` return: `model`, `answers`,
  `usage`; per answer `type` plus `noul` (noul), `choice` + `probabilities` (choice), `score` +
  `probabilities` (score). Confidence is not trusted. The question set is fixed for the client's
  lifetime (clef-flash scores questions jointly).
- Token budget: no Qwen tokenizer is installed, so the client uses a conservative bound (UTF-8 byte
  count of state + questions + the internal fixed-framing reserve; byte-level BPE never yields more
  tokens than bytes) and rejects a bound `>= budget` before sending; after the response,
  `usage.input_tokens >= budget` is an error because vLLM silently drops the END of an over-long
  state (the newest steps). The configured budget must equal the server's real cap, otherwise the
  post-hoc check cannot see truncation.
- No automatic retry of any external request (SDK `max_retries=0`). A timeout after
  write-before-send is recorded as `uncertain` because a resend could bill twice.
- CLI exit codes: 0 preview or completed run; 1 deterministic config/data/server/protocol/refusal/
  schema failure; 2 argparse or gate misuse; 3 uncertain external call.
- After the run directory exists, failures still leave `run.json` (`failed` / `uncertain`), the
  write-before-send record, any raw response, a redacted exception class/message and the counts.
  Console output carries paths, counts, hashes and status only.

## Security

- Local server binding: `--uds <path>` in a directory owned by the user with mode 0700 (preferred),
  or `--host 127.0.0.1` plus Bearer key. Never 0.0.0.0. Because PHASE-1 can talk to an existing
  server, the clients themselves (not only the LATER launcher) reject URL userinfo, remote hosts,
  0.0.0.0, TCP without a Bearer-key variable, UDS parents not owned by the user or not mode 0700,
  and configs that set both TCP and UDS. Loopback alone is not isolation on
  a shared node, so the key is always required. Server and client run in the same SLURM job.
- Secrets: `OPENAI_API_KEY` is the only OpenAI credential source, read after the gate. The local
  key comes from an env variable whose NAME is configured; the value never enters YAML, argv
  (`VLLM_API_KEY` is inherited by the subprocess, never `--api-key`), provenance, errors or logs.
  `--enable-log-requests` is never set; the launcher accepts no passthrough flags. `/health` is
  unauthenticated by vLLM design; readiness uses authenticated `/v1/models`.
- Customer text (D11: may be sent to OpenAI): real scenarios live under gitignored
  `private/translator/`, real key points under `private/evaluation/`. The whole
  `outputs/translator/` and `outputs/evaluation/` trees are gitignored (today only
  `outputs/**/*.{csv,json,jsonl}` are), because event logs, prompts and frames contain exact
  model-visible text. Committed examples use `[PLACEHOLDER]` text only. User-chosen output roots
  go through `derived_path()`.
- Video (D30): action videos show A-team actions and visible hit/shot-down outcomes. This is the
  same video the human rater sees; every video-enabled run records source and mapping.

## Implementation Phases

Order: P0 -> (P1a || P1b) -> P2 -> P3 -> P4 = PHASE-1 MINIMAL; then L1, L2 (retrospective) and L3
run in parallel, except that L2's `vllm_chat` wiring waits for L1's `vllm_chat.py` (L1 -> L2 wiring);
then L4.
Every step is test-first; every phase's acceptance runs offline (no network, GPU, paid call, SLURM).

| Phase | Teammate | Owned files | Depends on | Acceptance commands |
|---|---|---|---|---|
| P0 shared integration (blocks all) | `integrator` (sole owner of shared files) | `src/mtd_rl_xavier/{llm,lag,translator}/__init__.py`, `evaluation/__init__.py`, `pyproject.toml`, `uv.lock`, `.gitignore` | user approval of dependencies | `uv run pytest tests/test_mentalmodeling_layout.py -q`; `uv run poe test`; `uv run poe lint`; `uv lock --check`; `uv build`; `git check-ignore` on sample paths under the 4 ignored trees |
| P1a LLM foundation | `llm-core` | `src/mtd_rl_xavier/llm/{openai_responses,system_one}.py`, `llm/README.md`; `tests/llm/{llm_fakes,test_llm_clients}.py` | P0 | `uv run pytest tests/llm -q`; `uv run ruff check src/mtd_rl_xavier/llm tests/llm` |
| P1b LAG foundation (parallel with P1a) | `lag-core` | `src/mtd_rl_xavier/lag/{dataset,rendering}.py`, `lag/README.md`; `tests/lag/test_lag_contract.py` | P0 | `uv run pytest tests/lag -q`; `uv run ruff check src/mtd_rl_xavier/lag tests/lag`; `git diff --exit-code -- workspace/13_LAG workspace/17_commentary_LAG` |
| P2 Translator | `translator-core` | `src/mtd_rl_xavier/translator/{schema,config,detector,translation,pipeline,cli}.py`, `translator/README.md`; `configs/translator/**`; `tests/translator/**` | P1a + P1b | `uv run pytest tests/llm tests/lag tests/translator -q`; `uv run ruff check src/mtd_rl_xavier/translator tests/translator`; `uv run python -m mtd_rl_xavier.translator.cli --help` |
| P3 Judge | `judge-core` | `evaluation/{configuration,schemas,rendering,judge,cli}.py`, `evaluation/README.md`, `evaluation/config/judge-text.yaml`, `evaluation/examples/key-points.example.yaml`; `tests/evaluation/**` | P2 event contract | `uv run pytest tests/evaluation -q`; `uv run ruff check evaluation tests/evaluation`; `uv run python -m evaluation.cli --help` |
| P4 PHASE-1 verification (no edits) | lead | none | P3 | `uv run pytest tests/llm tests/lag tests/translator tests/evaluation`; `uv run ruff check src tests evaluation`; `uv run poe test`; `uv lock --check`; `uv build`; `git status --porcelain -- workspace/13_LAG workspace/17_commentary_LAG` prints nothing |
| L1 local runtime | `llm-runtime` | `src/mtd_rl_xavier/llm/{vllm_chat,vllm_server}.py`; `scripts/translator_vllm.sbatch`; `tests/llm/test_llm_local_runtime.py` | P4 | `uv run pytest tests/llm -q`; `uv run ruff check src/mtd_rl_xavier/llm tests/llm`; `bash -n scripts/translator_vllm.sbatch` |
| L2 retrospective + `vllm_chat` wiring | `translator-core` | its P2 files (config, pipeline, cli, run.yaml, tests) | P4; `vllm_chat` wiring after L1's `vllm_chat.py` | `uv run pytest tests/translator -q`; `uv run ruff check src/mtd_rl_xavier/translator tests/translator` |
| L3 video judge | `judge-core` | `evaluation/config/judge-text-video.yaml`; its P3 rendering/test files | P4; video extra declared (in P0 if approved then; otherwise an integrator-owned step L3-P0 edits `pyproject.toml`/`uv.lock` only, before L3 starts) | `uv run pytest tests/evaluation -q`; `uv run ruff check evaluation tests/evaluation` |
| L4 final verification (no edits) | lead | none | L1 + L2 + L3 | P4 commands again, plus diff review for weakened tests, placeholders, out-of-scope edits |

Step detail (from the plan consult, revised):
1. P0: create the four import-free markers; append `mtd_rl_xavier.llm`, `mtd_rl_xavier.lag`,
   `mtd_rl_xavier.translator` to the committed explicit package list (commit `b526120`); after
   approval `uv add` direct `httpx` and `pandas` (and the optional `video` extra); change `poe lint`
   to `ruff check src tests evaluation`; ignore `/private/translator/`, `/private/evaluation/`,
   `/outputs/translator/`, `/outputs/evaluation/`. No teammate starts before the P0 gate passes.
2. P1a-S1 fakes; P1a-S2 `openai_responses.py`; P1a-S3 `system_one.py` (typed questions, endpoint
   validation, byte bound, status/transport mapping, noul-only parsing); an AST test proves no
   module in `src/mtd_rl_xavier/llm/` imports a sibling (covers the LATER modules automatically).
3. P1b-S1 `dataset.py` (promoted loader + `last_complete_step` = 340/293/495); P1b-S2
   `rendering.py` (17-format byte parity with workspace 17 for centre +/-30 windows; compact
   renderer; post-death rows render).
4. P2-S1 `schema.py`; P2-S2 `config.py` + example YAMLs (caps, placeholder text); P2-S3
   `detector.py`; P2-S4 `translation.py`; P2-S5 `pipeline.py` (ticks on every grid step, collision
   precedence, target-init rule, death, runtime caps); P2-S6 `cli.py` + README (gate, preview,
   write-before-send, provenance, local preflight before paid client).
5. P3-S1 `configuration.py` + `judge-text.yaml`; P3-S2 `schemas.py` + key-points example; P3-S3
   `rendering.py` text blocks + preview; P3-S4 `judge.py`; P3-S5 `cli.py` + README + two E2E tests:
   (a) real 13_LAG episode -> fake layers -> event log -> judge preview, `api_requests_made == 0`;
   (b) real 13_LAG episode -> production `SystemOneClient` over `httpx.MockTransport` + production
   OpenAI translation adapter over a scripted client -> event log -> judge preview; no real network,
   paid API, GPU, decoder or judge-client construction.

## File Changes

| File | New/modify | Phase | Description |
|---|---|---|---|
| `src/mtd_rl_xavier/llm/__init__.py` | new | P0 | Import-free marker |
| `src/mtd_rl_xavier/lag/__init__.py` | new | P0 | Import-free marker |
| `src/mtd_rl_xavier/translator/__init__.py` | new | P0 | Import-free marker |
| `evaluation/__init__.py` | new | P0 | Judge package marker |
| `pyproject.toml` | modify | P0 | Append 3 packages; direct `httpx`, `pandas`; `video` extra; `poe lint` covers `evaluation` |
| `uv.lock` | modify | P0 | Regenerated by `uv` only |
| `.gitignore` | modify | P0 | Whole-tree ignores for private scenarios/key points and translator/judge outputs |
| `src/mtd_rl_xavier/llm/openai_responses.py` | new | P1a | Strict Responses API client |
| `src/mtd_rl_xavier/llm/system_one.py` | new | P1a | `/v1/systemone` client, endpoint checks, budget |
| `src/mtd_rl_xavier/llm/README.md` | new | P1a | LLM parts, gating, local-server security |
| `src/mtd_rl_xavier/lag/README.md` | new | P1b | Loader, renderers, parity note |
| `src/mtd_rl_xavier/lag/dataset.py` | new | P1b | Promoted 13_LAG loader, helpers, `last_complete_step` |
| `src/mtd_rl_xavier/lag/rendering.py` | new | P1b | 17-format JSONL + compact renderers |
| `src/mtd_rl_xavier/translator/schema.py` | new | P2 | Scenario/state/event/provenance dataclasses |
| `src/mtd_rl_xavier/translator/config.py` | new | P2 (L2 ext.) | Strict run config + scenario loader |
| `src/mtd_rl_xavier/translator/detector.py` | new | P2 | Detector Protocol + layer |
| `src/mtd_rl_xavier/translator/translation.py` | new | P2 | Translation Protocol + layer, prompt, output schema |
| `src/mtd_rl_xavier/translator/pipeline.py` | new | P2 (L2 ext.) | Chronological state machine |
| `src/mtd_rl_xavier/translator/cli.py` | new | P2 (L2 ext.) | Composition root, gate, artifacts |
| `src/mtd_rl_xavier/translator/README.md` | new | P2 | Usage, event contract, safety |
| `configs/translator/run.yaml` | new | P2 | Default run config |
| `configs/translator/scenario.example.yaml` | new | P2 | Placeholder scenario |
| `evaluation/configuration.py` | new | P3 | Strict judge config |
| `evaluation/schemas.py` | new | P3 | Key points, result, output schema + parser |
| `evaluation/rendering.py` | new | P3 (L3 ext.) | Six inputs, request/preview; later frames |
| `evaluation/judge.py` | new | P3 | One call per event |
| `evaluation/cli.py` | new | P3 | Gate, bijection check, artifacts |
| `evaluation/README.md` | new | P3 | Usage, inputs, video leakage |
| `evaluation/config/judge-text.yaml` | new | P3 | Text-only config |
| `evaluation/examples/key-points.example.yaml` | new | P3 | Placeholder key points |
| `tests/llm/llm_fakes.py`, `tests/llm/test_llm_clients.py` | new | P1a | Fakes; client contracts; AST import test |
| `tests/lag/test_lag_contract.py` | new | P1b | Parity, end step, post-death |
| `tests/translator/{tr_fakes,test_tr_contracts,test_tr_layers,test_tr_pipeline_cli}.py` | new | P2 | Fakes; contracts; layers; pipeline + CLI |
| `tests/evaluation/{eval_fakes,test_eval_contracts,test_eval_render_judge,test_eval_cli}.py` | new | P3 | Fakes; contracts; rendering/judge; CLI + 2 E2E |
| `src/mtd_rl_xavier/llm/vllm_chat.py` | new | L1 | Local chat client |
| `src/mtd_rl_xavier/llm/vllm_server.py` | new | L1 | Launcher |
| `scripts/translator_vllm.sbatch` | new | L1 | One-GPU same-job template (not executed) |
| `tests/llm/test_llm_local_runtime.py` | new | L1 | Local chat + launcher tests |
| `evaluation/config/judge-text-video.yaml` | new | L3 | Text+video config |

Unchanged: `src/mtd_rl_xavier/paths.py`, root `README.md`, all of `workspace/`, `src/weightreading/`.

## Test Plan

All tests use fakes (`ScriptedClient`/`RecordingFactory` style, `httpx.MockTransport`, fake
`FrameSource`, fake subprocess); each covers happy path, boundaries, errors and edge cases.

| Area | Key tests |
|---|---|
| P0 | Package-list equality test passes; `uv build` wheel contains the 3 new packages; `git check-ignore` covers `.txt`, `.md`, `.yaml`, `.jpg`, `.png`, `.json`, `.jsonl` under the 4 ignored trees |
| `openai_responses` | Exact request body: strict `text.format`, `store=False`, `reasoning.effort`, no temperature/top_p/top_logprobs; text+image items; usage; refusal; incomplete; lazy `OPENAI_API_KEY`; `max_retries=0` |
| `system_one` | noul/choice/score serialisation; OpenJev-style noul without probabilities parses; vLLM-only fields ignored; fixed question-set hash; TCP and UDS via MockTransport; endpoint rejects userinfo, remote host, 0.0.0.0, TCP without key var, insecure UDS parent, TCP+UDS together; bound `>= budget` rejected before transport; `usage.input_tokens >= budget` rejected; 400/401/404/422/500/501/other 4xx/5xx, connect error, timeout; no body or key in messages |
| LLM independence | AST scan (absolute and relative imports): no module in `src/mtd_rl_xavier/llm/` imports a sibling |
| Launcher (LATER) | argv pins revision/name/max-model-len/dtype; key never in argv; 0.0.0.0 and passthrough flags rejected; health timeout; terminate -> wait -> kill; client budget != `min(16384, max_model_len)` rejected |
| `lag` | Constants, dtypes, attrs and helper results equal workspace 13; 17-format JSONL bytes equal workspace 17 for centre +/-30; `last_complete_step` = 340/293/495; post-death `alive == 0` rows render; incomplete step raises; workspace files byte-identical |
| Translator contracts | Exact-key YAML; no hidden model; positive caps; bool rejected; scenario path resolution; >= 1 instruction at step 0; request/change on a not-yet-instructed target fails; duplicate ids |
| Translator layers | Detector record keeps the exact state string, full question object and raw response; detector input = compact window + target-specific previous translation only; threshold at, below, above; translation input = all instructions in effect + primary id + exact 17 text + previous translation + output language; strict intent/evidence parsing |
| Pipeline | Step-0 start per target; ticks every 5 steps on the grid; tick + instruction / request / external change -> one detector call and one aggregated translation; precedence; multiple targets; exact active-instruction set in every event; target first instructed mid-episode (kept distinct from a replacement instruction); window clipping at 0 and `last_step`; death closure (B0200 after 136, A0200 after 210) with suppressed triggers; caps at cap-1, cap, cap+1 (`cap_reached`, nothing sent, only the four record types present); an upper bound above the cap only warns |
| Translator CLI | No flags -> preview, zero factory calls, no credential read; one flag -> exit 2; OpenAI layer needs both; local-only `--execute` works; local preflight before paid client; new directory, refuses existing; provenance; write-before-send; `uncertain` on timeout |
| Judge contracts | Model must be `gpt-6.1-sol`; exactly six blocks; schema for 1 and several key points; parser rejects refusal, incomplete, invalid JSON, duplicate/unknown/missing ids, bool scores, out-of-range scores, blank rationale |
| Judge rendering | Every include/exclude combination; excluded text absent from the serialised request; trajectory verbatim from the event; deterministic request hash; no base64 in preview; later: frame mapping, clipping, sampling cap, each image item IMMEDIATELY preceded by its configured step/time label |
| Judge CLI | Bijection key points <-> all events; `max_judged_events` preflight; paired paid gate; one call per event; raw response kept before parsing; the two E2E tests in P3-S5 |

## Dependencies To Approve

| Dependency | Where | Why | Status today |
|---|---|---|---|
| `httpx>=0.28,<1` | core | `system_one.py` and `vllm_chat.py` import it directly; openai 3.x no longer installs it | installed 0.28.1 (transitive only) |
| `pandas>=2,<4` | core (moved from the `data` extra) | the promoted loader and renderers need it on a supported path | installed 3.0.6 (via `data` extra) |
| `av` (PyAV) + `Pillow` | new optional extra `video` | frame decoding, resize, JPEG (D31) | `av` not installed (download needed); Pillow 13.0.0 transitive; version bounds to be approved |

Not added: vLLM (separate env only, D26), `tokenizers`/`transformers` (byte bound instead),
pydantic (repo convention). `uv lock` may need package-index access.

## Run-Time Actions Needing Approval

| Action | First needed | Notes |
|---|---|---|
| `uv add` of the dependencies above + `uv lock` | P0 (blocks implementation) | Changes `pyproject.toml` / `uv.lock` |
| Real Translator run with an OpenAI layer | after P2 | `--execute --confirm-paid-api`; preview first; caps apply |
| Real judge run | after P3 | Always both flags; one `gpt-6.1-sol` call per event |
| Create the separate vLLM env (vLLM >= 0.33.0, ~10 GB) | after L1 | Never in the project `.venv` |
| Download clef-flash @ `fb4417d...` (~19 GB) to `/lustre/user151005/hf` | after L1 | `HF_HUB_OFFLINE=1` when serving |
| One short `--gres=gpu:1` probe job on `151-partition` | after L1 | Checks GPU/driver, memory, latency |
| Detector threshold check on a small labelled LAG set | after the probe | 0.65 is provisional |

No approval is needed for tests, fake runs, previews, lint, or static sbatch checks.

## Risks

- Detector out of domain: clef-flash was trained on business/tool/NLI decisions; LAG accuracy is
  unknown. Threshold in config, every tick's probability logged, calibration later.
- Silent truncation: mitigated by the byte bound + post-hoc usage check, but only if
  `input_token_budget` equals the real server cap; the byte bound is ~3-4x conservative and may
  reject long compact windows (mitigation: shorter window/fields; exact tokenizer LATER).
- Cost: a 61-step 17-format window is ~48.7k tokens, so a translation costs roughly $0.05 input at
  `gpt-6.1-terra` list price, and up to ~$12 input at the 250-call cap; a text judge call is ~$0.10
  input at `gpt-6.1-sol` (estimates from list prices; image-token multiplier for `gpt-6.1-sol`
  unverified). Mitigation: preview, caps, no retries.
- Customer text: allowed to OpenAI (D11) but must stay out of git; whole-tree ignores land in P0,
  before any run can write outputs. Event logs are sensitive by design.
- Action video (D30) shows A-team actions and hit/shot-down outcomes the Translator never saw;
  intentional (same as the human rater), recorded in provenance.
- Concurrent work: the user committed three weightreading changes while this plan was made
  (`b526120`, `5403a16`, `b6aeace`) and has uncommitted weightreading edits; P0 must start from the
  then-current HEAD and must not touch weightreading files.
- Packaging: the committed package-equality test fails if a new package is not listed; P0 lists
  them before any module is written.
- Runtime unverified: GPU model/driver, vLLM 0.33 install, clef-flash memory/latency, compute-node
  internet access; parity of vLLM probabilities with the reference implementation.
- Name shadowing: root `evaluation` package could be shadowed by an `evaluation.py` on `sys.path[0]`;
  run it only as `python -m evaluation.cli` from the repo root.

## Validation Verdict

**PASS** after one revision round (Codex re-validation,
`.claude/logs/codex/20261103T231047Z-translator-revalidate.md`).

- Round 1 (`.claude/logs/codex/20261103T224752Z-translator-validate.md`): NEEDS_REVISION with
  7 BLOCKING and 3 MAJOR items. Applied in this document:
  1. Detector runs on every scheduled tick, also when another trigger exists (D9/D28).
  2. Target initialisation: >= 1 instruction at step 0; a target first instructed later is
     translated at its issue step; requests/changes need an already-instructed target. The
     validator first asked for "every target starts at step 0"; on re-validation it accepted this
     reading as non-blocking, pending the user (Open Question 2).
  3. Judge key points are a bijection with ALL translation events (D20); no filtering.
  4. System One `noul` parsing needs only `noul` (OpenJev returns no probabilities).
  5. `last_step` = last step with all four aircraft CSV records (340/293/495).
  6. Caps stop at runtime with `cap_reached`; the upper bound only warns.
  7. P0 integration (package list, dependencies, lint scope, whole-tree ignores) precedes all work,
     because the committed package-equality test (commit `b526120`) would otherwise fail.
  8. Endpoint security enforced in `SystemOneClient` itself.
  9. Internal non-reducible framing reserve; transport and every status family mapped.
  10. Concrete-adapter offline E2E test (MockTransport + scripted OpenAI client).
- Round 2: PASS, all ten items RESOLVED. Its MAJOR/MINOR recommendations were applied afterwards
  (not re-validated again, documentation-level only): full `instructions_in_effect` in the example;
  detector record keeps exact state string, full question and raw response (`SystemOneResult.raw`);
  explicit L1 -> L2 dependency for `vllm_chat` wiring; launcher/client budget equality check;
  "immediately preceded" image-label test; README per new area; `stop_on_target_death` option
  removed (death policy is Open Question 9); named integrator step if the video extra is deferred.
- Caller-verified facts used by the plan: episode end steps 340/293/495 (ACMI-only 341/294/496
  excluded); downed aircraft keep `alive == 0` rows; the package-equality test exists and is
  committed; `httpx` is not a declared dependency; `pandas` is only in the `data` extra; no Qwen
  tokenizer is installed.
- Not run: no implementation exists, so no test, lint, build or lock check has been executed
  against this design. All Codex consults ran with `--sandbox read-only` and executed read
  commands only.

## Open Questions For The User

None blocks the design; items 1-2 block the start of implementation (P0).

1. Dependencies: approve `httpx` and `pandas` as core dependencies, and the optional `video` extra
   (`av`, `Pillow`) with version bounds (P0).
2. Mid-episode new targets: may a target (aircraft or team) receive its FIRST instruction after
   step 0? Plan default: yes; it is translated at its issue step with trigger `instruction` and no
   previous translation (D24). The validator read D16/D21 as "every target starts at step 0";
   choosing that instead only adds one preflight rule.
3. Judging all events: the key-points file must cover every translation event in a run (D20,
   one-to-one; no partial judging). Acceptable, or should a subset be judgeable?
4. Provisional caps per run: 500 detector calls, 250 translation calls, 250 judged events.
5. Detector settings: threshold 0.65 (provisional, uncalibrated), compact window 10 steps (2 s),
   fields `alive, state_valid, lon/lat/alt, yaw, tas, vn/ve/vu`, scope target + opponents.
6. First real scenario: provisional instructions (text, targets, issue steps), request steps,
   external change points, and output language (default English, D23).
7. Judge video window: frames from 10-20 steps before to 10-20 steps after the event (the human
   rater sees the whole video; the online Translator sees nothing after the event). Keep?
8. Retrospective mode (LATER): both layers see `+M` steps; choose `M` when enabling it.
9. Dead targets (not fixed by D1-D33): once a target aircraft is down (a team: all its aircraft),
   the plan stops translating it and logs later requests/changes for it as `trigger_suppressed`
   (no model call, nothing to judge). Keep this policy?

## Amendments At User Approval (2026-11-04, override the sections above)

- Open Questions 2, 9: accepted as planned (D35 mid-episode first instruction allowed; D37 dead-target policy).
- Open Question 3 CHANGED (D36): partial judging is allowed. Replace "key points are a bijection with
  ALL translation events" everywhere (Judge CLI, Test Plan, Validation Verdict item 3) with: every
  key-point entry must map to exactly one existing event (unknown id, duplicate id -> error before any
  call); events without key points are written to the results as `not_judged` with no API call;
  `max_judged_events` counts only events that have key points. Tests: subset judged, empty overlap
  (zero calls, all `not_judged`), unknown id, duplicate id.
- Open Question 1: dependencies approved (D34), including the `video` extra in P0.
- Open Questions 4-8: provisional values kept (caps 500/250/250; threshold 0.65, 10-step window and
  listed fields; output language English; judge video window 10-20 steps; `M` chosen when enabled).
