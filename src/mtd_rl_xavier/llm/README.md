# `mtd_rl_xavier.llm`: LLM call parts

Independent clients for the two layers of the Translator and for the judge. Each part is one
module. No part imports another (`tests/llm/test_llm_independence.py` checks this with an AST scan).
The endpoint rules the local parts share live outside this package, in
`mtd_rl_xavier/local_endpoint.py`, and each part re-raises their fixed messages as its own
configuration error. Protocols live on the consumer side (`translator/`, `evaluation/`), and only
the composition roots (`translator/cli.py`, `evaluation/cli.py`, `evaluation/assess_cli.py`,
`weightreading/cli.py`, `mtd_rl_xavier/intent_compare/method_a_cli.py`) build concrete clients.

| Module | Part | Phase |
|---|---|---|
| `openai_responses.py` | Paid OpenAI Responses API: strict JSON-schema output, text + image input | PHASE-1 |
| `retry.py` | Narrow, logged retry (HTTP 5xx and non-timeout connection errors) for the intent comparison (methods A and B, weightreading) | - |
| `system_one.py` | Local `POST /v1/systemone` (vLLM `clef-flash`, OpenJev `wev serve`) | PHASE-1 |
| `vllm_server.py` | `vllm serve` launcher: allowlisted argv, health wait, teardown, CLI | L1 |

The SLURM template `scripts/translator_vllm.sbatch` runs the launcher and the Translator CLI in one
one-GPU job. Nothing in L1 has been run against a real server: see
[Unverified vLLM facts](#unverified-vllm-facts) and [Still needs approval](#still-needs-approval-d49).

## `openai_responses.py`

- `OpenAIRequest`: model, instructions, `input_items` (built with `input_text(...)` and
  `input_image(url, detail=...)`), schema name and schema, `reasoning_effort`
  (`REASONING_EFFORTS` = `low|medium|high|xhigh|max`, the repo's one allowlist: the gpt-6.1-sol
  subset of the openai SDK `ReasoningEffort` type), `max_output_tokens` (covers reasoning plus visible output).
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
- `make_openai_client()` is the library's one OpenAI client factory: the Translator, the judge,
  weightreading and both methods of the intent comparison build their client here, and it is their
  only reader of `OPENAI_API_KEY` (legacy chat-completions code in `experiment/`, `tools/` and `lag/`
  still reads the key itself). It sets `max_retries=0`. Call it after the paid-call gate.
- The endpoint is pinned: it passes `base_url=API_BASE_URL` (`https://api.openai.com/v1`), so an
  `OPENAI_BASE_URL` exported for a local vLLM server cannot receive the key or the prompts. It
  refuses to build a client while `OPENAI_CUSTOM_HEADERS` is set (the SDK would add those headers
  to every request) or `OPENAI_LOG` is set (the SDK would log request bodies); the error names the
  variable, never its value. `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID` are still honoured by the SDK: they only
  select the organisation or project on api.openai.com. Proxy variables (`HTTPS_PROXY`, ...) are
  honoured by httpx; TLS to api.openai.com is still verified. `PROVENANCE_ENV` lists these
  honoured variables (plus `SSL_CERT_FILE` / `SSL_CERT_DIR`) and `set_provenance_env_names()`
  returns the names that are set, for run records (names only: proxy URLs can hold credentials).

## `retry.py`

- `call_with_retry(call, policy)` retries only HTTP 5xx answers and connection errors that are not
  timeouts; timeouts, 4xx (429 included) and non-`openai` exceptions are re-raised at once. At most
  `MAX_ALLOWED_RETRIES` (2) retries, backoff 10 s then 30 s by default; every failed attempt goes to
  `log` (stderr by default: attempt, error class, status, request id; never the message or body).
- `RetryingClient(sdk_client, policy)` wraps the client from `make_openai_client()` so that
  `responses.create` goes through `call_with_retry` with the identical keyword arguments. Method A of
  the intent comparison puts it under `OpenAIResponsesClient`; weightreading's backend calls
  `call_with_retry` directly. The Translator and judge CLIs do not use it and keep treating a 5xx as
  "outcome uncertain, not retried".

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

## `vllm_server.py`

- `VLLMServerConfig` has one typed field per allowed flag, and `build_vllm_argv(config)` emits
  exactly:

  ```
  <env>/bin/vllm serve <model> --revision <sha> --tokenizer-revision <sha>
      --served-model-name <name> --max-model-len <n> --dtype bfloat16
      (--uds <path> | --host 127.0.0.1 --port <1024-65535>)
  ```

  There is no passthrough. `--api-key`, `--enable-log-requests`, `--trust-remote-code` and
  `0.0.0.0` can never appear. Values are pattern-checked (repo id, 40-hex sha, name), so none can
  start with `-`. The CLI rejects unknown and abbreviated flags with exit 2, and its error messages
  never echo a value: a mistaken `--api-key <key>` does not reach the job log.
- The key never appears in argv. It is read from the variable named by `api_key_env` and handed to
  the child only as `VLLM_API_KEY` in its environment.
- Child environment:
  - removed:
    - every inherited `VLLM_*`, `OPENAI_*`, `HF_*`, `HUGGING_FACE_*`, `HUGGINGFACE_*` and
      `TRANSFORMERS_*` variable;
    - the `api_key_env` variable itself;
    - every name containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD` or `CREDENTIAL`;
    - `PYTHONPATH`, `PYTHONHOME` and `VIRTUAL_ENV`.

    This drops the paid key, a second copy of the server key, other credentials, the HF token, and
    hidden vLLM debug or logging switches. It is a denylist, not an allowlist: what vLLM needs from
    the environment is unverified, so a strict allowlist waits for the probe.
  - set: `HF_HOME` (from the config), `HF_HUB_OFFLINE=1`, `VLLM_API_KEY`, and the usage-statistics
    opt-outs `VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1` (unverified names);
  - the separate env is activated for the server only: `<env>/bin` first on `PATH`, and
    `VIRTUAL_ENV` set to that env.
- Budget: `client_input_token_budget` must equal `min(serving_cap_tokens, max_model_len)`
  (clef-flash: `min(16384, max_model_len)`), otherwise the launcher refuses. That equality is
  what lets `SystemOneClient` detect vLLM's silent truncation.
- Socket (`mtd_rl_xavier.local_endpoint`):
  - The parent directory is created with mode `0700` if it is missing. An existing parent that is
    not owned by you with mode exactly `0700` is refused and never repaired.
  - The path must fit in an AF_UNIX address (107 bytes).
  - An exclusive `flock` on `<socket>.lock` is held for the server's whole lifetime. A second
    launcher on the same path stops at once, instead of unlinking a live socket.
  - An existing socket that still accepts connections is refused. One that refuses connections is
    stale and is removed. Anything else at the path is refused. After the server stops, our socket
    is removed and the lock is released.
- `preflight(config)` runs the read-only checks: the key, the executable, the HF cache, the socket
  path length, and an existing socket directory.
- Lifecycle of `running_vllm_server(config, ...)`:
  1. Preflight, create the socket directory and take the lock, then start the server in its own
     session.
  2. Poll the authenticated `GET /v1/models` every 2 s until the served name is listed.
  3. Fail fast if the child exits, the key is rejected (401: another server may own the endpoint),
     another model is listed, or `startup_timeout_s` passes.
  4. On exit, SIGTERM the process group and wait up to `shutdown_timeout_s` until the child has
     exited AND the group is empty (`killpg(pgid, 0)`). Then SIGKILL the group and wait up to 30 s.

  The group is swept even if the child already exited, so leftover workers cannot keep the GPU.
  `/health` is not used because vLLM leaves it unauthenticated.
- CLI: `python -m mtd_rl_xavier.llm.vllm_server <flags> -- <client command>` runs the client
  without a shell, in the same environment and in its own process group, once the server is ready.
  - It exits with the client's code, where a death by signal N is reported as `128 + N`, or 1 on a
    launcher error.
  - The first SIGTERM (`scancel`, the time limit) or Ctrl-C unwinds through teardown; later ones
    are ignored so they cannot cut the teardown short.
  - The client's group first gets SIGINT, so a Python CLI can record its own interrupted status,
    and SIGKILL after 10 s.
  - `--dry-run` runs the preflight and prints the argv and the environment variable NAMES only.
- The subprocess factory, clock, sleep, `killpg` and HTTP transport are injectable. The tests
  need no vLLM, GPU or network.

## SLURM template `scripts/translator_vllm.sbatch`

- Directives: `-p 151-partition`, `--gres=gpu:1`, `--time=00:30:00`. The log goes to
  `outputs/translator/slurm/`, which is gitignored.
- Placeholders you set: `VLLM_ENV` (the separate env directory), `HF_CACHE_DIR` and `SCENARIO`.
  Optional: `CONFIG`, `MAX_MODEL_LEN`, `STARTUP_TIMEOUT_S`, `SHUTDOWN_TIMEOUT_S` and `UV`. No
  personal path is hardcoded.
- The model, revision, served name, socket path and budget are read from the Translator config
  (`detector.*`), so the server and the client cannot drift. The job requires a UDS endpoint and
  `api_key_env: VLLM_API_KEY`. A failing config reader fails the job.
- The job creates the socket directory (default `/tmp/$USER-vllm`) with mode `0700` if it is
  missing; a concurrent job may create it first.
- It generates a fresh per-job `VLLM_API_KEY` with Python `secrets`, and never prints it.
- It runs the launcher `--dry-run` (preflight and argv), then `exec`s the launcher with the
  project interpreter directly, so SIGTERM reaches it with no `uv` process in between. The client
  is one of:
  - default: a GPU probe (the detector client's authenticated `health()`), with no paid call;
  - `TRANSLATOR_CONFIRM_PAID=yes`: the Translator CLI with `--execute --confirm-paid-api` (needs
    `OPENAI_API_KEY`).
- Two of your jobs on one node with the same config cannot share the socket: the second one stops
  at the launcher's lock. A per-job socket path would need a Translator config placeholder (not in
  L1).
- It passes `bash -n`, and `tests/llm/test_llm_vllm_sbatch.py` runs it once offline: a stub `UV`
  and a fake `vllm` record what the server would get, and the key is in its environment only. It
  has never been submitted.

## Unverified vLLM facts

`.claude/docs/research/openjev-vllm.md` (section "Clef-Flash on vLLM") verified these against
vLLM 0.33.0: `vllm serve <model>`, `--revision`, `--served-model-name`, `--max-model-len`,
`--dtype bfloat16`, `--host`/`--port`, `--uds`, `VLLM_API_KEY`, the `/v1` key guard
(`/v1/models` and `/v1/systemone` need the key; `/health` does not), request logging off unless
`--enable-log-requests`, the budget `min(16384, max_model_len)`, `HF_HOME` and
`HF_HUB_OFFLINE=1`. Everything below is NOT verified and must be checked during the approved probe:

| Item | Where | Failure mode if wrong |
|---|---|---|
| `--tokenizer-revision <sha>` exists in vLLM 0.33.0 | `build_vllm_argv` | argparse error: the server exits, and the launcher reports "exited before it was ready" |
| `vllm serve --uds` cannot bind over an existing socket file (so the launcher removes a stale one) | `_acquire_socket` | none expected; removal is limited to a refusing socket you own |
| `/v1/models` refuses connections, or returns 5xx, until the model is loaded | `_probe_ready` | none expected; any other 4xx stops the start-up |
| Every vLLM worker process stays in the server's process group | `stop_server` | a worker outside the group survives until SLURM ends the job |
| `VLLM_NO_USAGE_STATS` / `DO_NOT_TRACK` are the usage-statistics opt-outs | child env | usage statistics may be sent if the compute node has egress |
| The denylisted variables include nothing vLLM needs | child env | the server exits during start-up |
| `HF_HUB_OFFLINE=1` alone resolves the pinned sha from `HF_HOME` (no `TRANSFORMERS_OFFLINE`) | child env | the server exits during start-up |
| `/tmp` on the compute node is not a symlink (the socket path must be canonical) | `check_uds_parent` | the launcher refuses the socket path before start |

## Still needs approval (D49)

None of these ran:

- Creating the separate vLLM env (vLLM >= 0.33.0, about 10 GB), outside the project `.venv`.
- Downloading clef-flash at `fb4417d31c1d0f9e7e77f1141b40aaede7491e00` (about 19 GB) into the HF
  cache, for example on `/lustre`.
- One short `--gres=gpu:1` probe job on `151-partition` (`scripts/translator_vllm.sbatch` without
  `TRANSLATOR_CONFIRM_PAID`), which checks the GPU, driver, memory, start-up time and the
  unverified items above.
- Any paid Translator run (`TRANSLATOR_CONFIRM_PAID=yes`).

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
- The same endpoint rules (`mtd_rl_xavier.local_endpoint`) apply to `SystemOneClient` and the
  launcher, so both accept and reject the same endpoints.
