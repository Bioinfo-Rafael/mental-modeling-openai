# OpenJev and vLLM: What It Is and How to Serve It Locally

> **Decision note (2026-11-04):** On 2026-11-04 the user switched the default change-point detector
> from OpenJev-4B to Cloudflare's **clef-flash** (`Cloudflare/clef-flash`, 9B, served natively by
> vLLM >= 0.33.0 on `POST /v1/systemone`). The OpenJev findings below are kept as background and
> as a fallback option. The authoritative serving facts for the default detector are in the final
> section, [Clef-Flash on vLLM](#clef-flash-on-vllm), which has its own Sources and Unverified Items.

Research date: 2026-11-04. Scope: decide the "local vLLM runner" part and the "OpenJev change-point
detector" part of the LAG "LLM as a Translator". Read-only research; nothing was installed or
downloaded (beyond small text files fetched into a scratchpad), no SLURM jobs were submitted.

## Verdict

**vLLM does not serve OpenJev (alanhuangyoo/OpenJev, `alanhuangya/OpenJev-4B`), even with flags.**
The user is right that vLLM has a "System One" mechanism: since **v0.32.0** (2026-10-18) `vllm serve`
exposes **`POST /v1/systemone`**, the same TypeSafe/Jev request shape that OpenJev speaks. But that
endpoint dispatches on a fixed architecture registry (`READ_STRATEGIES`). OpenJev's checkpoint is a
headless `Qwen3_5TextModel` backbone plus a separate pointer head (`head.safetensors`), and it is
not in that registry in v0.32.0, v0.33.0 (latest, 2026-11-02) or `main`. OpenJev ships its own
server instead: **`wev serve`** (FastAPI + transformers, `POST /v1/systemone`, port 8009). Serve
OpenJev with `wev serve` in its own env and call it over HTTP. Because vLLM's `/v1/systemone` takes
the same request shape, one project client can later point at a vLLM-served decision model by
changing only `base_url`.

## What OpenJev Is

- An open, local **"System-One decision model"**: typed questions about a free-form state go in,
  and a calibrated probability for every declared option comes out, in one forward pass with **no
  text generation**. It is a drop-in local replacement for TypeSafe AI's hosted Jev API
  (`https://api.typesafe.ai/v1/systemone`).
- Built for two jobs. The first is **browser-agent steps** ("which operation?", "which element?")
  as the System One of the jev-ultrafast agent. The second is **general typed decisions**
  (triage, routing, policy checks, agent monitoring).
- Python package `wev-ai` (import `wev`; the paper's earlier name is "wev"). `openjev-ai` is an alias
  that depends on `wev-ai>=0.2.1`. CLI: `wev serve | export | evaluate | train`.
- Authors: Jun Huang, Xin Ren (UESTC). Paper: Zenodo DOI 10.5281/zenodo.22941164,
  "wev: Distilling LLM Browser Agents into Open, Local System-One Decision Models" (ICML-style
  preprint, not peer-review-confirmed).
- License: **Apache-2.0** for code and weights. The README adds a caveat: some training data is
  research-only (tasksource-jev), and teacher episodes are qwen3-max / GLM-5.3-Flash outputs. The
  authors say to "treat the models as research artifacts" before any commercial use.
- Activity: GitHub repo created 2026-09-22, last push **2026-10-09**, 70 stars, no issues. PyPI
  `wev-ai` 0.2.2 was released 2026-10-06. The HF `OpenJev-4B` main branch was last modified
  2026-10-09.
- **Name collision warning.** "OpenJev" is a generic name with dozens of unrelated HF repos. The
  biggest is the org `openjev/` (`openjev/openjev` 27B and `openjev/OpenJev-Flash-9B`, both
  `Qwen3_5ForConditionalGeneration`, generative, served with vLLM plus a shim). That org has **no
  4B**. "OpenJev 4B" from alanhuangyoo/OpenJev is `alanhuangya/OpenJev-4B`; the HF username is
  `alanhuangya` and the GitHub username is `alanhuangyoo`.

## Model Ids & Architecture

| HF id | Release | Backbone (base) | HF `architectures` | Layers | Weights (bf16) | Head |
|---|---|---|---|---|---|---|
| `alanhuangya/OpenJev-4B` (main, tag `v0.2`) | v0.2, the "4B" default | `Qwen/Qwen3.5-4B-Base`, continued from `jaredpalmer/kev-4b` | `Qwen3_5TextModel` (`qwen3_5_text`) | 32 (hybrid: 3x Gated DeltaNet linear-attention + 1x gated full attention, repeated) | `model.safetensors` 8.41 GB | pointer, fp32, 5.2 MB |
| `alanhuangya/OpenJev-4B@v0.1` | v0.1 | `Qwen/Qwen3-4B-Base` | `Qwen3Model` | 36 | ~8 GB | pointer |
| `alanhuangya/OpenJev-8B` | v0.1 | `Qwen/Qwen3-8B-Base` | `Qwen3Model` | 36 | 15.2 GB (4 shards) | pointer, 8.4 MB |
| `alanhuangya/OpenJev-1.7B` | v0.1 | `Qwen/Qwen3-1.7B-Base` | `Qwen3Model` | 28 | 3.44 GB | pointer, 4.2 MB |

HF refs for OpenJev-4B: `main` = `368713247ad2a50d56542b4d812823626eb504e7`; tags `v0.1`
(`77489397…`), `v0.2` (`ad82e751…`), `v2` (`1ab3cc22…`). Files: `config.json`, `model.safetensors`,
`head.safetensors`, `wev.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
(the inherited Qwen3.5 chat template, which OpenJev does not use).

Architecture (from `wev/model.py`, adapted from jaredpalmer/kev):

- A causal-LM backbone with its **vocabulary head removed**. LoRA r=16 on attention, MLP and (for
  Qwen3.5) Gated DeltaNet projections is merged into the export, so **the model cannot generate
  text**.
- Five rarely used Qwen special tokens are reused as delimiters (no new embeddings):
  `<|fim_prefix|>`=state, `<|fim_middle|>`=q, `<|box_start|>`=opt, `<|box_end|>`=/opt,
  `<|fim_suffix|>`=decide. Caller text is escaped so it can never produce these tokens.
- Packed sequence: `<state> … <q> instr <opt> o1 </opt> <opt> o2 </opt> … <decide>` per question.
  Each question attends only to the state and itself, and its positions restart right after the
  state. Answers are therefore independent of question order, and many questions share one state
  encoding.
- **PointerHead**: `k(h[</opt>]) · q(h[<decide>]) / sqrt(256)`, two `Linear(hidden→256)` layers,
  softmax over options. v0.2 divides logits by a calibration **temperature of 1.5599** from
  `wev.json`, fitted on dev rows, which lowered ECE from 0.062 to 0.0135.
- v0.2 is hybrid (recurrent Gated DeltaNet layers cannot take the block-causal mask), so it runs
  each question as its own causal row "state + branch". When the state is long, it runs the state
  once and copies the KV/recurrent cache to every question (`forward_prefix`). Inference is
  serialised by a lock, because the Triton autotuner for the DeltaNet kernels is not thread-safe.
- Limits (`wev.json`): `max_state` 4096 tokens and `max_branch` 8192 tokens per question (trained
  at 2048). Choice questions allow 1..255 options.

## Input/Output Format

- **Modalities: text only (English).** The state is any JSON value (string, object, list,
  number), flattened by `wev.api.render()` into indented `key: value` text. There are **no
  images, video, or numeric time-series encoders**. A trajectory has to be rendered as text.
- No chat template, no system prompt, no reasoning tokens. The prompt format is fixed by
  `encode()` and cannot be changed by the caller beyond `instructions` and `criteria` text.
- Request (`POST /v1/systemone`, pydantic `SystemOneRequest`):

```json
{"model": "optional-string",
 "state": "<str | object | list>",
 "questions": {
   "changed": {"type": "noul", "instructions": "…",
               "criteria": {"true": "desc of yes", "false": "desc of no"}},
   "kind":    {"type": "choice", "instructions": "…",
               "criteria": {"opt_a": "desc", "opt_b": "desc"}},
   "level":   {"type": "score", "instructions": "…", "criteria": ["low", "mid", "high"]}}}
```

- Response (`wev serve`), probabilities rounded to 4 digits:

```json
{"model": "...",
 "answers": {
   "changed": {"type": "noul", "noul": 0.02},
   "kind":    {"type": "choice", "choice": "opt_b", "confidence": 0.61,
               "probabilities": {"opt_a": 0.2, "opt_b": 0.8}},
   "level":   {"type": "score", "score": 1.3, "legend": {"0": "low", "1": "mid", "2": "high"},
               "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}, "confidence": 0.7}},
 "usage": {"input_tokens": 512, "output_tokens": 40},
 "latency_ms": 41}
```

  `noul` is the probability of yes. `choice.confidence = (max p − 1/K)/(1 − 1/K)`. `score` is the
  expected 0-indexed level. `output_tokens` is a billing-style count; nothing is generated.
- In-process API: `wev.load(path_or_repo, device=None, dtype=None, revision=None)` returns an
  object whose `.predict(state, questions)` returns the same dict.
- Over-long states: automatic shrinking only applies to jev-ultrafast-shaped dict states (keys
  `page.text`, `page.url`, `elements`, `recent_actions`). **Any other state over 4096 tokens fails
  with `ContextTooLong`, which the server returns as HTTP 422.** Our client must budget the
  trajectory rendering to stay under 4096 Qwen3.5 tokens itself.
- Server endpoints: `POST /v1/systemone`, `GET /v1/models` (use it as the health check). Invalid
  requests return 422 with the reason. **There is no authentication.** The server binds
  `127.0.0.1:8009` by default.

## System 1 Meaning

- In this ecosystem, "System One" (often "System-One") is **a model class and an API**, not a vLLM
  scheduling mode. The OpenJev paper says: "System-One decision models answer typed questions about a
  free-form state with a probability for every declared option, in one forward pass and without
  generating text." TypeSafe AI's hosted **Jev** introduced it (`typesafe2026jev`, training method
  unpublished). The wire format is "the TypeSafe `/v1/systemone` request shape".
- In jev-ultrafast, the System One makes fast per-step decisions, and a separate text LLM writes
  any free-text values. That maps directly onto our design: OpenJev is the fast detector ("did the
  motion change beyond what the last translation explains?"), and the gpt translator is the slow
  writer.
- The name presumably echoes Kahneman's fast/slow System 1/2. The paper never says so (no
  "Kahneman" or "System Two" in `main.tex`), so treat that as interpretation.
- vLLM uses the same meaning. Its docs section is titled "TypeSafe System One Endpoint", and the
  model page calls these "Structured Decisions Models (aka JEV Models)". vLLM has **no fast/slow
  dual-system scheduling feature**.

## vLLM Support

**What vLLM has (since v0.32.0):** `POST /v1/systemone` (PR #59299, merged 2026-10-08, first
released in v0.32.0 on 2026-10-18). v0.33.0 (2026-11-02) also adds `POST /v1/decisions`, an MVP of
OpenAI's Decisions API.

- It is enabled by default on supported architectures. Generative ones need
  `--logprobs-mode raw_logprobs` or `processed_logprobs`. The docs also recommend
  `--enable-prefix-caching` so the state is prefilled once.
- Mechanism for generative models (`NextTokenStrategy`): one read per question. The prompt is
  chat-templated state + question + options labelled A–Z / yes,no / 0–9, then one next-token read
  takes a softmax over the label-token logprobs.
- Mechanism for head-based models: a model class with a custom pooler (Clef-Flash: a
  `Qwen3_5ForConditionalGeneration` subclass + `JointSchemaHead` loaded from a secondary weights
  file, `tok_pooling_type="ALL"`), or a classification model detected from `decision_config.json`
  (Decision 3.0).
- Limits: 64 questions/request, 26 options/choice, 10 levels/score (Decision 3.0: 255 options).
  `extra="forbid"`: unknown fields return 400.
- `READ_STRATEGIES` in v0.33.0: `ClefFlashForJointSchema`, `ClefOmniForJointSchema`,
  `D3ForSequenceClassification`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`,
  `Gemma4UnifiedForConditionalGeneration`, `Lfm2VlForConditionalGeneration`, `Qwen3ForCausalLM`,
  `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`,
  `DiffusionGemmaForBlockDiffusion`. v0.32.0 had only 6 of these.
  `select_read_strategy()` raises, and the endpoint returns 501, for anything else.
- Response superset: vLLM adds `id`, `object: "structured_decision"`, `created`, and per-question
  `diagnostics.label_mass` / `argmax_is_label`. Its `noul` answers also carry `probabilities` and
  `confidence`. The wev response has `latency_ms` and no diagnostics.

**OpenJev on vLLM:**

| Route | Status |
|---|---|
| Native `vllm serve alanhuangya/OpenJev-4B` | **Not supported.** `Qwen3_5TextModel` is not a registered vLLM architecture (registry has `Qwen3_5ForCausalLM` and `Qwen3_5ForConditionalGeneration`), and the pointer head + `wev.json` format is unknown to vLLM. No vLLM issue or PR mentions alanhuangyoo's OpenJev, wev, or kev. |
| `--trust-remote-code` | Not applicable. The repo ships no `auto_map` or modeling code. |
| `--hf-overrides` to a Qwen3.5 generative class + `NextTokenStrategy` | **Invalid.** It would read label tokens through the tied embedding matrix of a backbone whose LoRA was trained for the pointer head, ignore `head.safetensors`, and use a different prompt. The output would be meaningless. |
| vLLM pooling fallback: `--runner pooling --convert embed --pooler-config.task token_embed --hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}'`, then apply the pointer head client-side on the hidden states at `</opt>` / `<decide>` positions, one "state + branch" row per question | **Theoretically possible, unverified, not recommended.** vLLM docs say any architecture converts to `token_embed` to extract last hidden states, and the pooling adapter loads both `*ForCausalLM` and `*Model` checkpoints. But it returns L×2560 floats per row over HTTP, needs normalisation disabled, and needs the delimiter-token encoding re-implemented exactly. Hybrid-model pooling and prefix caching are untested for this checkpoint. |
| Out-of-tree vLLM plugin (model class like `clef_flash.py` + a `ReadStrategy` registered in `READ_STRATEGIES`) | Feasible engineering project (days), unverified. Only worth it for high throughput. |
| OpenJev's own vLLM integration | **None.** The repo has no vllm/sglang/onnx/gguf mention. Its server is `wev serve` (FastAPI/uvicorn + HF transformers). |

**Exact flags if a vLLM-native decision model is used instead** (alternatives, not OpenJev):
`vllm serve Qwen/Qwen3.5-4B --logprobs-mode raw_logprobs --enable-prefix-caching` (send
`"chat_template_kwargs": {"enable_thinking": false}` per request; thinking is off by default on
this endpoint). `vllm serve vllm-sr/d3-mini` (a 4B Decision 3.0 classifier with no extra flags;
prefix caching and chunked prefill are forced off). Neither is OpenJev, and their accuracy on our
task is unknown.

## Serving Recommendation

1. **Serve OpenJev-4B with `wev serve` in a dedicated env, not with vLLM:**

```bash
# separate env (never the project .venv); uv only, per dev-environment rule
uv venv ~/envs/openjev --python 3.12
uv pip install --python ~/envs/openjev/bin/python "wev-ai[serve,fast]==0.2.2"   # fast = flash-linear-attention for Qwen3.5 on CUDA
# pin weights: pre-download a fixed revision to a local dir (wev serve has no --revision flag)
#   e.g. huggingface_hub.snapshot_download("alanhuangya/OpenJev-4B", revision="<sha>", local_dir=...)
~/envs/openjev/bin/wev serve --model /path/to/OpenJev-4B --host 127.0.0.1 --port 8009 --dtype bf16
curl -s localhost:8009/v1/models    # health check
```

   Resources: **one GPU** is enough. Weights are 8.4 GB bf16; the authors run it on an RTX 5090
   (32 GB) and on a 16 GB Apple M5 laptop. The ~10–12 GB peak at a 4k-token state is my estimate,
   not measured. Reported latency on an RTX 5090: ~40 ms per general decision. Requests are
   serialised by a lock, so the ceiling is about 25 req/s. That is ample for change-point checks.
2. **Project code talks HTTP, not in-process.** Add a small typed `SystemOneClient(base_url)` that
   POSTs `/v1/systemone` with `httpx`, which is already in the project `.venv`. Do not use the
   OpenAI SDK, because this is not chat completions. Parse only the fields both servers return:
   `answers[q].type`, `noul`, `choice`, `probabilities` (choice/score), and `score`. Treat
   `latency_ms`, `diagnostics` and `confidence` as optional. Then the same client works against
   `wev serve` (OpenJev) and `vllm serve` (Qwen3.5 / d3 / Clef via `/v1/systemone`). Map HTTP 422
   (wev: too long or invalid) and 400/501 (vLLM) to distinct errors.
3. **Make the "local runner" a generic local-model-server launcher**, not vLLM-specific. Its config
   holds `command` (`wev serve …` or `vllm serve …`), `env path`, `host/port`, `health URL`
   (`/v1/models` on both), and `startup timeout`. The detector uses `/v1/systemone` from whichever
   server runs. A future local translator would use vLLM's OpenAI-compatible
   `/v1/chat/completions`, where the existing `OpenAIChatBackend(compatible=True, endpoint=...)`
   in `mentalmodeling/llm_x/backends.py` already fits.
4. **Separate environments.** The project `.venv` (Python 3.12.16) has no torch or transformers;
   keep it that way.
   - `wev-ai` 0.2.2 needs `torch>=2.8`, `transformers>=5.17,<6`, and `peft`.
   - vLLM 0.33.0 pins `torch==2.14.0` and `transformers>=5.16.1,<5.21.0`, and requires
     Python >=3.11,<3.15.
   - The two are co-installable on paper (transformers 5.17–5.20), but vLLM's exact torch/CUDA pin
     and size argue for a dedicated `~/envs/vllm` if vLLM is used at all.
5. **Security.** `wev serve` has no auth. Bind `127.0.0.1` and run the client in the same SLURM
   job, or use an SSH tunnel. Do not bind `0.0.0.0` on the shared `151-partition` nodes. vLLM's
   `--api-key` exists, but `/v1/systemone` auth coverage was not verified.
6. **Detector prompt design constraints:**
   - The state must fit in **≤ 4096 tokens**.
   - Inputs are text-only, so numeric trajectory windows must be rendered compactly (rounded
     values, downsampled steps).
   - Use a `noul` question (e.g. "The motion since the last translation is NOT explained by it")
     and threshold `p(yes)`, optionally adding a `choice` for the change type.
   - LAG air-combat kinematics are **far out of OpenJev's training domain**: browser steps,
     classification, QA and ops decisions. Its calibration (ECE 0.01–0.04) was measured in-domain,
     so zero-shot accuracy and calibration on our data are unknown. Plan a small labelled check
     before trusting thresholds.

## Environment Check (this machine)

Checked 2026-11-04 on login node `srpv01541`, read-only:

| Check | Result |
|---|---|
| `curl -sI https://huggingface.co` | HTTP/2 200 |
| `curl -sI https://github.com` | HTTP/2 200 (also `github.com/alanhuangyoo/OpenJev` 200) |
| HF API / raw.githubusercontent / PyPI JSON | reachable (used for this research) |
| `uv` | `/home/user151005/.local/bin/uv`, version 0.13.7 |
| `mtd-rl-xavier/.venv` Python | 3.12.16; has `openai 2.54.0`, `httpx 0.28.1`, `numpy 2.5.4`; **no torch, transformers, vllm, wev** |
| `mtd-rl-xavier/pyproject.toml` | `requires-python >=3.10`, `openai>=1,<3` |
| GPU on login node | none (per `translator-contract-facts.md`; not re-checked) |

Not checked: internet/HF access from compute nodes `srdgx*`, GPU model, driver and CUDA version
there, and free disk for the HF cache. No `~/.cache/huggingface` exists, so set `HF_HOME` or use
`local_dir` on a large filesystem.

## Sources

All accessed 2026-11-04.

- OpenJev repo and API metadata: https://github.com/alanhuangyoo/OpenJev ,
  https://api.github.com/repos/alanhuangyoo/OpenJev
- README / code read: https://raw.githubusercontent.com/alanhuangyoo/OpenJev/main/README.md ,
  `…/main/pyproject.toml`, `…/main/wev/{model,inference,api,serve,export,prompts,cli}.py`,
  `…/main/results/exports/*/wev.json`, `…/main/results/exports/wev-4b/MODEL_CARD.md`
- Paper source: https://raw.githubusercontent.com/alanhuangyoo/OpenJev/main/paper/icml/main.tex ;
  record https://zenodo.org/records/22941164 (not opened)
- HF model pages/API: https://huggingface.co/alanhuangya/OpenJev-4B (+ `config.json`, `wev.json`,
  `README.md`, `tokenizer_config.json`, `/api/models/alanhuangya/OpenJev-4B/refs`),
  https://huggingface.co/alanhuangya/OpenJev-8B , https://huggingface.co/alanhuangya/OpenJev-1.7B
- PyPI: https://pypi.org/project/wev-ai/ , https://pypi.org/project/openjev-ai/ ,
  https://pypi.org/project/vllm/ (0.33.0), https://pypi.org/project/flash-linear-attention/
- vLLM docs: https://docs.vllm.ai/en/latest/models/structured_decisions_models/ ; source
  https://github.com/vllm-project/vllm/blob/main/docs/models/structured_decisions_models.md
- vLLM code: `vllm/entrypoints/openai/decisions/strategies.py` (main, v0.33.0),
  `vllm/entrypoints/generate/structured_decisions/{protocol,strategies}.py` (main, v0.32.0),
  `vllm/model_executor/models/{registry,clef_flash,d3,adapters}.py`,
  `docs/models/pooling_models/token_embed.md` (v0.33.0) on https://github.com/vllm-project/vllm
- vLLM releases: https://github.com/vllm-project/vllm/releases (v0.31.0 2026-10-05, v0.32.0
  2026-10-18, v0.33.0 2026-11-02); PR https://github.com/vllm-project/vllm/pull/59299 (endpoint),
  https://github.com/vllm-project/vllm/pull/62841 (hybrid prefix-cache fix; benchmarks the unrelated
  `openjev/openjev` 27B)
- Ecosystem list: https://github.com/andyrewlee/awesome-system-one
- Name-collision check: https://huggingface.co/api/models?search=openjev ,
  https://huggingface.co/openjev/openjev

## Unverified Items

- **Actual runtime of OpenJev-4B on the `srdgx*` GPUs:** peak memory, latency, and whether
  `flash-linear-attention` 0.6.0 builds and works with that CUDA/driver. Without `fast`, Qwen3.5
  DeltaNet falls back to slower torch code of unknown magnitude.
- Whether compute nodes can reach huggingface.co. If not, pre-download on the login node, which
  this task forbade.
- The vLLM pooling-fallback route (`--convert embed` + `token_embed` + client-side pointer head) for a
  hybrid `Qwen3_5TextModel` checkpoint. Not tested; may fail at weight loading, normalisation, or
  hybrid prefix caching.
- Whether vLLM's `--api-key` protects `/v1/systemone`.
- OpenJev accuracy and calibration on LAG change-point questions. It is entirely out of domain;
  no evidence either way.
- Paper claims (benchmarks, ECE, latency) are the authors' own and were not reproduced. The Zenodo
  PDF was not opened; `main.tex` was read instead.
- Which HF revision to pin: `main` (`36871324…`, 2026-10-09) vs tag `v0.2` (`ad82e751…`). The
  diff between them was not inspected; it is likely card/README renames only.
- Whether the `GEMMA-VLLM` jobs of another group member (see `translator-contract-facts.md`)
  indicate a working vLLM build recipe for these nodes.

## Clef-Flash on vLLM

Researched 2026-11-04 after the user chose clef-flash as the default detector. This was read-only:
vLLM v0.33.0 source, the HF API, and the model card. Nothing was installed, no weights were
downloaded, and no SLURM job was submitted.

### Verdict

**Yes, clef-flash runs on stock vLLM today, from v0.33.0 (2026-11-02).** v0.32.0 does not
have it. Nothing extra is needed: no `--trust-remote-code`, no `--runner`/`--convert`, no
`--logprobs-mode`. vLLM sees `joint_head_config.json` in the repo and rewrites the architecture
`Qwen3_5ForConditionalGeneration` to `ClefFlashForJointSchema`, a pooling model with
`task=token_classify`. It then loads `joint_head.safetensors` as a second weight source and
serves `POST /v1/systemone` with one pooled forward pass per request.

```bash
# official minimal form (docs): vllm serve Cloudflare/clef-flash --max-model-len 16384
# recommended form for this cluster (pinned, local-only, keyed):
HF_HOME=/lustre/user151005/hf \
VLLM_API_KEY="$(cat ~/.config/vllm/api_key)" \
~/envs/vllm-0.33.0/bin/vllm serve Cloudflare/clef-flash \
  --revision fb4417d31c1d0f9e7e77f1141b40aaede7491e00 \
  --served-model-name clef-flash \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --host 127.0.0.1 --port 8000          # or: --uds /tmp/$USER-vllm/clef.sock (dir mode 0700)
# health: GET /health (unauthenticated) ; GET /v1/models (needs Bearer key when a key is set)
```

`--max-model-len 16384` matches Clef-Flash's own prompt cap
(`ClefFlashStrategy.MAX_LENGTH = 16384`). Without it, vLLM takes 262144 from
`max_position_embeddings` and sizes the KV cache for that.

### Model Facts

| Item | Value |
|---|---|
| HF id | `Cloudflare/clef-flash` (public, **not gated**, Apache-2.0) |
| Pinned revision | `fb4417d31c1d0f9e7e77f1141b40aaede7491e00` (main, last modified 2026-10-30; **no tags**, so pin the sha) |
| Base | `Qwen/Qwen3.5-9B` (post-trained), hybrid Gated DeltaNet + full attention, 32 text layers, hidden 4096, plus the 27-layer vision encoder |
| Head | joint schema head (`joint_head_config.json`: width 1024, 2 routing layers, 4 layers, 16 heads); `joint_head.safetensors` 0.24 GB |
| Download size | 19.08 GB total (4 bf16 shards ~18.8 GB + head + tokenizer) |
| GPU memory | weights ~19 GB in bf16. The card was "tested on a single H200". vLLM takes `--gpu-memory-utilization` 0.9 of the GPU by default. Estimate: >= 40 GB GPU comfortable, 24 GB tight (not measured). |
| Inputs | text or JSON state; images and video as URL or data-URL lists in `images` / `videos`; **no audio** (400) |
| Language | not stated on the card; benchmarks are English. Assume English. |
| Siblings | `Cloudflare/clef` (27B on Qwen3.8-27B, 55 GB), `Cloudflare/clef-omni` (Qwen3-Omni-30B-A3B, 70.8 GB). There is no smaller Clef. |
| Reported quality | Card (Cloudflare's own Decision Index run): median latency 38.8 ms, p95 122 ms. Third-party typed-decisions eval (2026-10-16): accuracy 0.71, ECE 0.0164, Brier 0.1105. |
| Other servers | SGLang is documented on the card (`lmsysorg/sglang:dev-clef`). The card does not mention vLLM; vLLM documents it on its own side. |

### `/v1/systemone` for Clef-Flash (v0.33.0 `ClefFlashStrategy`)

- **Prompt (fixed).** It mirrors the checkpoint's `encode_record`:
  - System prompt: "Read the complete state and schema. Decide every field jointly…"
  - Then `STATE:` + state, then `SCHEMA FIELDS:` with one `FIELD n / ID / TYPE / INSTRUCTION /
    ALLOWED OPTIONS` block per question.
  - Then an empty `<think></think>` and `JOINT SCHEMA DECISIONS:`.
  - **All questions are scored jointly in one pass**, so one question's answer can depend on the
    other questions in the same request. OpenJev scores each question independently.
- **Question types:** `choice`, `noul`, `score`. Limits:
  - 64 questions per request.
  - 26 options per choice. Open PR #62471 would raise this, but it is not merged.
  - 10 levels per score.
  - At least 1 choice option and at least 2 score levels; no duplicate option names.
  - Choice options are shown to the model sorted by name. Answers are mapped back to request order.
- **Question fields:** only `type`, `instructions`, `criteria`; anything else returns 400.
  `instructions` is cast with `str()` (send strings), and an empty one falls back to the question id.
  Choice `criteria` maps option → description string or `null`. `noul` `criteria` is optional
  `{true, false}`. Score `criteria` is a list of level names.
- **Top-level fields:** `model` (optional; must equal the served name or the server returns
  **404**), `state`, `questions`, `images`, `videos`, `cache_salt`, `priority`, `request_id`.
  - **Top-level `instructions` is rejected with 400** for Clef ("fixed prompt").
  - `chat_template_kwargs` has no effect.
  - `joint` is DiffusionGemma-only.
  - Unknown fields return 400 (`extra="forbid"`).
- **State rendering:** a string goes in verbatim. Any JSON value is re-serialized as **compact JSON
  with sorted keys**, so the key order we send is not preserved.
- **Length:** the token budget is `min(16384, max_model_len)`.
  - If the schema alone (system prompt + questions + suffix) exceeds the budget, the server
    returns **400**.
  - Otherwise **the state is truncated silently, keeping the beginning and dropping the end**
    (`state_ids[: max_length - fixed]`, the same as the reference code).
  - For a time-ordered trajectory, overflow would drop the **most recent** steps without any error.
- **Response:** `{id, object: "structured_decision", created, model, answers, usage, diagnostics,
  joint: null}`.
  - `choice`: `{choice, probabilities{name: p}, confidence}`.
  - `noul`: `{noul: p_yes, probabilities{yes, no}, confidence}`.
  - `score`: `{score: expected index, legend, probabilities{"0".."n"}, confidence}`.
  - For Clef, `label_mass` is always 1, so `confidence` = top probability. Probabilities are full
    floats, not rounded.
  - `usage.input_tokens` = prompt length (counted once); `output_tokens` = 0.
- **Determinism:** probabilities can vary slightly with batching (docs). Set `VLLM_BATCH_INVARIANT=1`
  for repeatable reads, at a throughput cost. vLLM always disables prefix caching for hybrid
  pooling models (`ModelConfig.is_prefix_caching_supported`), so `--enable-prefix-caching` has no
  effect. Chunked prefill stays on.
- **Status codes:**

  | Code | When |
  |---|---|
  | 200 | ok |
  | 400 | `StructuredDecisionError`: invalid question, limits, unknown field, top-level `instructions`, audio, schema too long |
  | 401 | missing or wrong Bearer token when a key is set |
  | 404 | model name mismatch |
  | 501 | no decision strategy, i.e. the wrong model is loaded |
  | 500 | engine or scoring failure |

  Error body: `{"error": {"message", "type", "code", ...}}`.

### Differences From OpenJev That the Client Must Handle

| Aspect | OpenJev `wev serve` | vLLM + clef-flash |
|---|---|---|
| Over-long state | 422 `ContextTooLong` (non-browser states) | **silent head-keep truncation**; detect via `usage.input_tokens >= budget` and treat as an error |
| Validation errors | 422, body `{"detail": [...]}` | 400 (404 model, 401 auth, 501), body `{"error": {...}}` |
| Unknown fields | ignored (pydantic default) | 400 |
| Top-level `instructions` | not a field | 400 for Clef |
| Question coupling | independent per question | joint across all questions in the request |
| `noul` answer | `{noul}` only | `{noul, probabilities, confidence}` |
| `choice.confidence` | `(max p − 1/K)/(1 − 1/K)` | `max p` (label_mass = 1) |
| Rounding | 4 digits | none |
| Extra response fields | `latency_ms` | `id`, `object`, `created`, `diagnostics`, `joint` |
| Option limits | 255 options per choice | 26 options per choice, 10 score levels, 64 questions |
| JSON state rendering | indented `key: value` text, order kept | compact JSON, keys sorted |
| Concurrency | serialized by a lock | continuous batching |
| Auth | none | Bearer key on `/v1/*` when `--api-key` / `VLLM_API_KEY` is set |

Client rules that follow from the table:

- Parse only `answers[q].type`, `choice`, `noul`, `score`, and `probabilities`.
- Compute our own confidence if needed, rather than trusting the server's field.
- Keep the question set fixed across calls, because of joint scoring.
- Budget the state well under 16k tokens, or put the newest data first so that truncation drops
  the oldest data.

### Environment (this cluster)

- **vLLM 0.33.0 requirements:** Python >=3.11,<3.15; `torch==2.14.0`; `transformers>=5.16.1,<5.21.0`;
  `flashinfer-python==0.7.2.post1`; GPU compute capability >= 7.5.
  - The prebuilt binaries are CUDA 12.9 (a cu130 variant also exists). The docs install with
    `uv pip install vllm --torch-backend=auto` followed by `vllm download-kernels`.
  - `--torch-backend=auto` inspects the local CUDA driver, but **the login node has no GPU or
    driver**. Either install on a GPU node or pin `--torch-backend=cu129` explicitly.
  - CUDA 12.9 wheels need a recent NVIDIA driver, typically R575+ (CUDA minor-version
    compatibility may allow older R5xx drivers with limitations; unverified here).
- **SLURM, read-only:** `sinfo -o '%P %N %G %f'` shows partition `151-partition` with nodes
  `srdgx[00043,00051-00053]`, `gpu:8`, feature `location=local`. **The GPU model is not exposed.**
  `scontrol show node srdgx00043` shows 224 CPUs, ~2.0 TB RAM, kernel `5.15.0-1053-nvidia`
  (Ubuntu/DGX OS), and 6 of 8 GPUs allocated at check time. My inference that these are DGX H100
  80 GB nodes (224 threads, 2 TB) is **unverified**. The driver version is unknown.
- **Login-node checks:**
  - huggingface.co and github.com return HTTP 200, and no proxy variables are set (direct egress).
  - **Compute-node egress could not be checked** without a job.
  - Safe plan: pre-download the pinned snapshot on the login node into `HF_HOME`, then serve with
    `HF_HUB_OFFLINE=1`.
- **Storage:**
  - `/home` is a 100 GB group NFS share with 86 GB free.
  - `/lustre/tenant151` has 10 TB with 9.4 TB free and group-writable per-user directories;
    `/lustre/user151005` does not exist yet.
  - Recommendation: `HF_HOME=/lustre/user151005/hf` (19 GB of weights).
  - Recommendation: the vLLM env at `~/envs/vllm-0.33.0`, about 10 GB of torch/CUDA wheels (an
    estimate), or on Lustre if `/home` quota is tight. This is separate from the project `.venv`.
- **Separate env, yes.** The project `.venv` has no torch. vLLM pins an exact torch, and the
  project only needs `httpx` (already installed, 0.28.1) to call the server.
- **SLURM job shape:** the existing template `scripts/weightreading_run.sbatch` has **no
  `--gres`**. The vLLM job needs `--gres=gpu:1` on `-p 151-partition`, a startup health wait on
  `/health`, and teardown at job end.

### Security (shared SLURM node)

- **The default bind is all interfaces.** In `vllm serve`, `--host` defaults to `None`, which binds
  `("", port)`, i.e. 0.0.0.0. Always pass `--host 127.0.0.1`, or better `--uds <path>` (Unix
  socket; host and port are ignored) inside a mode-0700 directory. `httpx` supports this with
  `httpx.HTTPTransport(uds=...)`.
- **`--api-key` exists** (repeatable; or `VLLM_API_KEY`; the CLI value takes precedence). It is
  enforced only on paths starting with `/v1`, `/v2`, `/inference`, `/cohere`, so it does cover
  `/v1/systemone` and `/v1/models`.
  - **`/pooling`, `/classify`, `/invocations`, `/health` and `/metrics` stay unauthenticated**,
    as the vLLM CLI help itself warns.
  - Loopback on a shared node is reachable by every user on that node, so a key alone is not
    enough.
- **Recommendation:**
  1. Use `--uds` in a 0700 directory, or else `--host 127.0.0.1`, **plus** a random
     `VLLM_API_KEY` passed through the environment (never on the command line, where it is
     visible in `ps`/`/proc`).
  2. Keep the key in a 0600 file.
  3. Run the client in the same job.
  4. Never bind 0.0.0.0. Never log request bodies (`--enable-log-requests` off).

### Sources (Clef-Flash, accessed 2026-11-04)

- HF model and API: https://huggingface.co/Cloudflare/clef-flash ,
  https://huggingface.co/api/models/Cloudflare/clef-flash?blobs=true (`/refs`; files
  `README.md`, `config.json`, `joint_head_config.json`, `joint_schema_model.py`,
  `.eval_results/Cloudflare__clef-flash.yaml` at revision `fb4417d3…`)
- Siblings: https://huggingface.co/Cloudflare/clef , https://huggingface.co/Cloudflare/clef-omni
- vLLM docs: https://docs.vllm.ai/en/latest/models/structured_decisions_models/ (Clef-Flash section:
  `vllm serve Cloudflare/clef-flash --max-model-len 16384`)
- vLLM v0.33.0 source (tarball https://codeload.github.com/vllm-project/vllm/tar.gz/refs/tags/v0.33.0):
  `vllm/transformers_utils/model_arch_config_convertor.py` (`Qwen3_5ModelArchConfigConvertor`),
  `vllm/model_executor/models/{clef_flash,clef_joint_schema,registry}.py`,
  `vllm/entrypoints/openai/decisions/{strategies,question_types,state}.py`,
  `vllm/entrypoints/generate/structured_decisions/{api_router,protocol,serving}.py`,
  `vllm/config/model.py` (`is_prefix_caching_supported`, `is_chunked_prefill_supported`),
  `vllm/entrypoints/serve/middleware/authenticate.py` (`GUARDED_PREFIX`),
  `vllm/entrypoints/common/cli_args.py` (`api_key`, `host`, `uds`),
  `vllm/entrypoints/launchers/launcher.py` (`("", port)` bind),
  `requirements/{cuda,common}.txt`, `pyproject.toml`,
  `docs/getting_started/installation/gpu.cuda.inc.md`, `tests/models/registry.py`
- v0.32.0 comparison: https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/model_executor/models/registry.py
  (no `ClefFlash`)
- Open PRs: https://github.com/vllm-project/vllm/pull/62249 , https://github.com/vllm-project/vllm/pull/62471
- PyPI: https://pypi.org/project/vllm/ (0.33.0, uploaded 2026-11-02)
- Cluster: `sinfo -o '%P %N %G %f'`, `scontrol show node srdgx00043`, `df -h /home /lustre`
  (login node `srpv01541`)

### Unverified Items (Clef-Flash)

- **GPU model, driver version and CUDA support on `srdgx*`.** This decides whether the CUDA 12.9
  `torch==2.14.0` wheels run, and how much memory is left after 19 GB of weights.
- **Compute-node access to huggingface.co**, and whether flashinfer's JIT/kernel download works
  there; pre-run `vllm download-kernels` if not.
- **Real peak GPU memory and latency** of clef-flash under vLLM on these GPUs. The card's
  38.8 ms median comes from Cloudflare's own infrastructure.
- **Whether `--limit-mm-per-prompt '{"image":0,"video":0}'` is safe** for text-only serving, to
  skip vision-encoder memory profiling. Untested with `ClefFlashStrategy`.
- **Whether vLLM's probabilities match the reference `joint_schema_model.systemone()`** exactly.
  The docs claim this for d3 and Liquid but not explicitly for Clef-Flash. A parity check on a few
  requests is advisable.
- **Clef-Flash accuracy and calibration on LAG change-point questions.** Its training domain
  (business, tool and NLI decisions) is far from air-combat kinematics, so a small labelled check
  is needed before fixing thresholds.
- **Whether open PRs #62249 / #62471 land in a later release and change limits or behaviour.**
  The Clef strategy code on `main` is currently identical to v0.33.0.
