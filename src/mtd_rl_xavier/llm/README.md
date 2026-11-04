# `mtd_rl_xavier.llm`: LLM call parts

Independent clients for the two layers of the Translator and for the judge. Each part is one
module. No part imports another (`tests/llm/test_llm_independence.py` checks this with an AST scan).
Protocols live on the consumer side (`translator/`, `evaluation/`), and only the composition roots
(`translator/cli.py`, `evaluation/cli.py`) build concrete clients.

| Module | Part | Phase |
|---|---|---|
| `openai_responses.py` | Paid OpenAI Responses API: strict JSON-schema output, text + image input | PHASE-1 |
| `system_one.py` | Local `POST /v1/systemone` (vLLM `clef-flash`, OpenJev `wev serve`) | PHASE-1 |
| `vllm_chat.py` | Local OpenAI-compatible chat client | LATER |
| `vllm_server.py` | vLLM launcher (argv builder, health wait, teardown) | LATER |

## `openai_responses.py`

- `OpenAIRequest`: model, instructions, `input_items` (built with `input_text(...)` and
  `input_image(url, detail=...)`), schema name and schema, `reasoning_effort`
  (`low|medium|high|xhigh|max`), `max_output_tokens` (covers reasoning plus visible output).
  The request is validated when it is constructed.
- `build_responses_body(request)` returns the exact keyword arguments sent: `model`,
  `instructions`, one user message, `text.format` = `json_schema` with `strict: true`,
  `reasoning.effort`, `max_output_tokens` and `store: false`. It never sends `temperature`,
  `top_p` or `top_logprobs`. Previews can show this body without a client.
- `OpenAIResponsesClient(client).create(request)` returns an `OpenAIResult`: text, output items,
  usage (input, cached, output, reasoning tokens) and the full `raw` response. It raises:
  - `OpenAIRefusalError` or `OpenAIIncompleteError`. Both keep `.result`, including `raw`.
  - `OpenAIClientError` for every other failure. `.uncertain` is True when the server may have
    run and billed the call without us knowing the outcome: connection failure, timeout, HTTP
    5xx, an unknown status, or a response that cannot be parsed (kept on `.raw`). Treat such a
    call as `uncertain`, and do not resend it.
  - Messages carry only the error class and HTTP status. The SDK error code is on `.error_code`;
    the SDK message (it can echo request text or a masked key) is never copied.
- `make_openai_client()` is the only reader of `OPENAI_API_KEY`, and the only place that imports
  the `openai` SDK. It sets `max_retries=0`. Call it after the paid-call gate.
- The endpoint is pinned: it passes `base_url=API_BASE_URL` (`https://api.openai.com/v1`), so an
  `OPENAI_BASE_URL` exported for a local vLLM server cannot receive the key or the prompts. It
  refuses to build a client while `OPENAI_CUSTOM_HEADERS` is set (the SDK would add those headers
  to every request) or `OPENAI_LOG` is set (the SDK would log request bodies); the error names the
  variable, never its value. `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` are still honoured by the SDK: they only
  select the organisation or project on api.openai.com. Proxy variables (`HTTPS_PROXY`, ...) are
  honoured by httpx; TLS to api.openai.com is still verified. `PROVENANCE_ENV` lists these
  honoured variables (plus `SSL_CERT_FILE` / `SSL_CERT_DIR`) and `set_provenance_env_names()`
  returns the names that are set, for run records (names only: proxy URLs can hold credentials).

## `system_one.py`

- Questions: `NoulQuestion` (yes/no), `ChoiceQuestion` (1-26 options) and `ScoreQuestion`
  (2-10 levels). A request carries at most 64 questions.
- The question set is fixed when the client is built, because clef-flash scores all questions
  jointly. `question_set_sha256` hashes the exact questions JSON that every request sends.
- `SystemOneClient.decide(state)` returns a `SystemOneResult` with typed answers, usage and the
  `raw` response. `health()` sends an authenticated `GET /v1/models` and checks that the model is
  listed. Use `health()` as the readiness check, because vLLM's `/health` is unauthenticated.
- Only the fields that vLLM and `wev serve` both return are read. A noul answer needs only
  `noul`. Fields such as `confidence`, `legend` and `diagnostics` are ignored.
- Token budget:
  - Before sending, the client computes a conservative upper bound: UTF-8 bytes of the state plus
    the questions JSON, plus `FIXED_FRAMING_RESERVE_TOKENS` and
    `PER_QUESTION_FRAMING_RESERVE_TOKENS` per question. Config cannot reduce either reserve.
  - A bound `>= input_token_budget` is rejected before anything is sent.
  - A response with `usage.input_tokens >= input_token_budget` raises `SystemOneBudgetError`,
    because vLLM silently drops the end (the newest steps) of an over-long state.
  - Set the budget to the server's real cap, `min(16384, max_model_len)`.
- Errors:

  | Cause | Error |
  |---|---|
  | HTTP 400, 422 and every other 4xx | `SystemOneRequestError` |
  | HTTP 401 | `SystemOneAuthError` |
  | HTTP 404 | `SystemOneModelError` |
  | HTTP 501 | `SystemOneUnsupportedError` |
  | HTTP 500 and every other 5xx | `SystemOneServerError` |
  | Connection failure or timeout (`.timed_out` tells them apart) | `SystemOneTransportError` |
  | Any other status, or a malformed 200 body | `SystemOneProtocolError` |
  | Invalid configuration, before any transport exists | `SystemOneConfigError` |
  | Budget exceeded, before or after the call | `SystemOneBudgetError` |

  Nothing is retried. Messages carry only the path and the HTTP status. When a response was
  received, its parsed body is kept on `.raw` and never copied into the message.

## Gating

- Every CLI previews by default. A preview builds no client and reads no credential.
- `OPENAI_API_KEY` is read only by `make_openai_client()`, after `--execute --confirm-paid-api`.
- The local server key variable is read when `SystemOneClient` is built, after `--execute`.
- Tests use fakes only (`tests/llm/llm_fakes.py`): a scripted Responses client and
  `httpx.MockTransport`. They make no network or paid call and use no real key.

## Local-server security (shared SLURM node)

- Endpoint: prefer `--uds <path>` in a directory that you own with mode `0700`. The alternative is
  `--host 127.0.0.1` plus a Bearer key. Never bind `0.0.0.0`; `vllm serve` binds all interfaces by
  default.
- `SystemOneClient` checks the endpoint itself, before any transport exists. The pure checks are
  public, `checked_tcp_url(base_url)` and `checked_env_name(name)`, so a caller's config loader can
  reject exactly what the client rejects (their messages never echo the URL or the name). It
  accepts exactly one of two endpoints:
  - TCP: the host must be literally `127.0.0.1` or `::1`. The URL must be the server root, with no
    userinfo, path, query or whitespace. A key variable is required.
  - UDS: the path must be absolute and canonical, with no symlink or `..` component. Its parent
    must be a directory owned by the current user with mode exactly `0700`. Every directory above
    it must be owned by root or the current user, and must not be writable by other users unless
    the sticky bit is set (as on `/tmp`). So no other user can swap a path component between
    the check, which runs once at construction, and the connection.
- Key: `api_key_env` holds the NAME of an environment variable, and the value is read from the
  environment. The NAME must be a POSIX variable name, `[A-Z_][A-Z0-9_]{0,63}`, so a pasted key
  (lowercase hex, base64, `sk-...`) is rejected. The value never enters YAML, argv, provenance,
  error messages or logs. For the
  server, pass `VLLM_API_KEY` through the environment, never with `--api-key`, and keep it in a
  `0600` file. The key is optional over UDS and sent when configured.
- Loopback alone is not isolation on a shared node, because every user on the node can reach it.
  Run the server and the client in the same SLURM job.
- Do not log request bodies: never set `--enable-log-requests`. Client error messages never
  contain the state, a response body or the key. The client ignores proxy and `.netrc` settings
  from the environment and never follows redirects.
