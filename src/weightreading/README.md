# weightreading: method B of the intent comparison

Method B translates an RL policy's behaviour into intent from its weights and activations. For each
(step, aircraft) of the shared `configs/intent_compare/compare.yaml` (parsed by
`mtd_rl_xavier.intent_compare`, the same parser method A uses), a frontier LLM runs a code-execution
session in a sandbox with the raw weights preloaded as `W` and, read-only as `A`, that aircraft's
validated replay activations over the window `[max(0, t-59), t]` (`x`, `h_prev`, `base.a1/a2`,
`gru.r/z/n/h_new`, `gru_norm.y`, `act_mlp.a1/a2`, `heads.logits_k/probs_k`, `action`, `step`). It
sees no world trajectory. The model replies with one Python block per turn; no tools are passed.

The episode is replayed free-running from `h = 0` on the recorded observations and refused unless it
matches the recording (logits, `h'`, `h_prev` vs `rnn_in` <= 1e-4, all actions). The final answer is
one ```json `{"intent", "evidence"}` block (the Translator schema); a JSON attempt ends the session
even when invalid. Each aircraft's latest valid translation is chained into its next session.

The weights-only explainer (the former `run` and `evaluate` commands) was removed; it is in git history.

## Commands (from the repository root)

```bash
.venv/bin/python -m weightreading sandbox-selftest   # fail-closed sandbox preflight
.venv/bin/python -m weightreading intent-preview     # local: messages, inputs, cost bound; no API call
.venv/bin/python -m weightreading intent-run --run-id <id> [--resume] --execute --confirm-paid-api  # PAID
```

On Slurm: `slurm/translator_method_b.sbatch` (see `slurm/README.md`). `intent-run` without both paid
flags writes the same local preview as `intent-preview`; one flag alone is an argparse error. The API
key is read from the environment only after both flags and the sandbox preflight pass. The SDK client
is the shared `mtd_rl_xavier.llm.openai_responses.make_openai_client` (endpoint pinned to
`https://api.openai.com/v1`, `OPENAI_LOG` / `OPENAI_CUSTOM_HEADERS` refused, `max_retries=0`).

## Inputs and outputs

| Path | Content |
|---|---|
| `configs/intent_compare/compare*.yaml` | steps, targets, window, model, effort, instruction, `rollout_dir` |
| `configs/weightreading/intent_b.yaml` | limits, sandbox limits and prices (model and effort must equal compare.yaml) |
| `configs/weightreading/lag_mappo_2v2_33.spec.json` | 2v2 MAPPO actor (MHI v1.3): 33-D obs, MLP -> GRU -> LN -> MLP -> heads [3, 5, 3, 2] |
| `<rollout_dir>/{manifest.json,episode_seed<seed>.npz}` | recorded rollout (`slurm/translator_record_rollouts.sbatch`); the manifest names the actor `.pt` in the sibling mtd-rl-lag checkout, read in place and never copied |
| `outputs/weightreading/intent/previews/<UTC>_seed<seed>/` | `preview.json`, per session `message.txt` and `inputs/` |
| `outputs/weightreading/intent/runs/<run_id>/` | `run.json`, per session `message.txt`, `inputs/`, `work/`, `transcript.jsonl`, `tool_calls.jsonl`, `state.json`, `final_answer.md`, `audit.json`, `session.json` |
| `outputs/intent_compare/<run_id>/cues_B.jsonl` | the shared cue contract (exclusive) |

Specs carry `provenance` and `notes` that never reach the prompt; the random `network_alias` is the
only name the model sees.

## Resume

`intent-run --resume` continues an interrupted run after comparing `RESUME_KEYS` of its `run.json`
(`inputs`, prompt version, model, effort, limits, steps, targets, window). `inputs` holds the
compare, config and spec **paths as strings** plus their SHA-256, so moving or editing
`compare*.yaml`, `intent_b.yaml` or the spec, or resuming from another checkout, refuses the resume.
`scripts/check_method_b_resume.py` checks this offline. Completed runs are unaffected.

## Transient API errors

`mtd_rl_xavier/llm/retry.py` (shared with method A) retries a request only when it almost certainly
was not processed (so not billed): an HTTP 5xx answer, or a connection error that is not a timeout.
4xx answers (including 429), bad requests, auth errors and non-`openai` exceptions are never
retried. Default: at most 2 retries, waiting 10 s then 30 s. Each failed attempt is printed to
stderr (attempt, error class, status code, request id; never the key, message or request body) and
recorded in the reply's `transcript.jsonl` line. An optional `retry` block in the config overrides
the defaults; the shipped `intent_b.yaml` has none on purpose, because editing it would block
resuming an interrupted run.

## Safety properties

- **Checkpoint reader** (`torch_reader.py`): stdlib-only allow-list unpickler, no torch. It accepts
  only `OrderedDict`, `_rebuild_tensor_v2`, `_rebuild_parameter` and `torch.*Storage`, and rejects
  every other global, compressed or unknown zip members, bad offsets, strides or sizes, and any
  dtype other than float32.
- **Sandbox** (`sandbox.py`): `unshare` user, mount, pid and net namespaces; a tmpfs root holding
  only `/usr`, the interpreter, numpy, `/work` and the read-only inputs, entered via `pivot_root`;
  `env -i`, `python -I`, RLIMIT_AS/FSIZE/NPROC/CPU, a timeout, a `/work` size cap and an output cap.
  There is no aggregate memory cgroup limit on these nodes, so run paid sessions inside a Slurm
  job. The preflight (`preflight.py`) must pass every check, or no request is sent.
- **Audit** (`audit.py`): flags nonlinearities, LayerNorm-like expressions, recurrence loops and
  torch imports; `audit.json` records `mode: intent-b`.
- **Reference forward** (`reference.py`): numpy float64, mirrors mtd-rl-lag (Linear -> ReLU ->
  LayerNorm, GRU gates `[r; z; n]`). Torch parity: `slurm/translator_method_b_parity.sbatch`.
