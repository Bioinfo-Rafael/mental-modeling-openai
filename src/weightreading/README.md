# weightreading

A frontier LLM reads an RL policy's raw weights - never running the network - and explains what
it computes. Protocol: blog "Exp 3" style code-execution loop. The model gets the architecture as
exact equations, a tensor table and the input/output semantics (or only the I/O layout in `blind`
mode), and replies with one Python block per turn (several blocks are merged into one run). The code runs in a sandbox with the
weights preloaded as `W`. No tools are passed to the model; forward passes are forbidden by the
prompt and flagged by an audit.

Design: `.claude/docs/plans/weightreading-design.md` (workspace root).

## Commands (from the repository root)

```bash
M=/home/user151005/workspace/mtd-rl-lag/envs/JSBSim/model
uv run python -m weightreading inspect $M/baseline_model.pt                  # 22 tensors, 137,753 params
uv run python -m weightreading inspect $M/baseline_model.pt --tensor-hashes  # for the torch parity check
uv run python -m weightreading validate-spec --checkpoint $M/baseline_model.pt \
  --spec configs/weightreading/lag_baseline_12.spec.json
uv run python -m weightreading preview --checkpoint $M/baseline_model.pt \
  --spec configs/weightreading/lag_baseline_12.spec.json [--task-mode blind]  # local, no API call
uv run python -m weightreading sandbox-selftest                              # fail-closed preflight
uv run python -m weightreading run ... --execute --confirm-paid-api          # PAID; needs OPENAI_API_KEY
uv run python -m weightreading audit outputs/weightreading/runs/<run>        # forward-pass flags
```

### 2v2 MAPPO high-level actors (MHI v1.3)

`configs/weightreading/lag_mappo_2v2_33.spec.json` describes the delivered actors
(`mtd-rl-lag/models/{v1.3,v1.3-1}/files/actor_latest.pt`, 30 tensors, 155,917 params): 33-D
`mhi_multiplecombat_shoot_33_v1` observation, MLP -> GRU -> LN -> ACT MLP -> 4 heads
(altitude/heading/speed command, shoot). The checkpoints are client files: read them in place,
never copy them into this repository.

```bash
A=/home/user151005/workspace/mtd-rl-lag/models/v1.3/files/actor_latest.pt
uv run python -m weightreading validate-spec --checkpoint $A --spec configs/weightreading/lag_mappo_2v2_33.spec.json
unshare -rn .venv/bin/python -m weightreading preview --checkpoint $A \
  --spec configs/weightreading/lag_mappo_2v2_33.spec.json                    # provably offline
mkdir -p outputs/weightreading/slurm
SPEC=lag_mappo_2v2_33 CHECKPOINT=$A sbatch --export=ALL scripts/weightreading_run.sbatch   # self-test + preview
SPEC=lag_mappo_2v2_33 CHECKPOINT=$A WEIGHTREADING_CONFIRM_PAID=yes OPENAI_API_KEY=... \
  sbatch --export=ALL scripts/weightreading_run.sbatch                       # PAID
```

The spec has one alias (`net-6a3a494d`), and the bundle directory is named after it, so after a v1.3
run a v1.3-1 run with the same spec refuses with "existing bundle ... differs" (before any API call).
For v1.3-1, copy the spec under a new name with a fresh `network_alias` (`net-` + 8 hex digits), e.g.
`lag_mappo_2v2_33_v1_3_1.spec.json`, and pass `SPEC=lag_mappo_2v2_33_v1_3_1`.

Heads use `bin_value.values` (one listed value per bin, strictly monotonic) with optional `labels`
and `unit`; `blind` mode shows the values but withholds labels and unit. Tests that read the actors
skip unless they exist under `WEIGHTREADING_MHI_MODEL_DIR` (default `mtd-rl-lag/models`).

`run` without both paid flags writes the same local preview as `preview`; one flag alone is an
argparse error. The API key is read from the environment only after both flags and the sandbox
preflight pass. The SDK client uses `max_retries=0`. Paid runs belong on compute nodes (see
`scripts/weightreading_run.sbatch`); the login node is for tests and previews.

## Files

| Path | Content |
|---|---|
| `configs/weightreading/run.yaml` | model id (no default in code), reasoning effort, `task_mode`, limits, sandbox limits, prices |
| `configs/weightreading/lag_baseline_12.spec.json` | `baseline_model.pt`: 12-D obs, MLP -> GRU -> LN -> 4 heads |
| `configs/weightreading/lag_dodge_21.spec.json` | `dodge_missile_model.pt`: 21-D obs, ... -> LN -> MLP -> 4 heads |
| `configs/weightreading/lag_mappo_2v2_33.spec.json` | 2v2 MAPPO `actor_latest.pt` (MHI v1.3, v1.3-1): 33-D obs, ... -> LN -> MLP -> heads [3, 5, 3, 2] |
| `outputs/weightreading/bundles/<alias>/` | `weights/<key>.npy` + `weights/index.json` (mounted read-only), `manifest.json` |
| `outputs/weightreading/previews/<UTC>_<alias>_<mode>/` | `message.txt`, `preview.json` (offline token/cost bound) |
| `outputs/weightreading/runs/<UTC>_<alias>_<mode>/` | `message.txt`, `run.json` (provenance, preflight), `transcript.jsonl`, `tool_calls.jsonl`, `state.json`, `final_answer.md`, `audit.json`, `work/` |

Specs carry `provenance` and `notes` that never reach the prompt; the random `network_alias` is
the only name the model, the bundle and the run directory use.

## Safety properties

- **Checkpoint reader** (`torch_reader.py`): stdlib-only allow-list unpickler, no torch. It accepts
  only `OrderedDict`, `_rebuild_tensor_v2`, `_rebuild_parameter` and `torch.*Storage`. It rejects
  every other global, BUILD on allow-listed objects, compressed or unknown zip members, more than
  one prefix, bad offsets, strides or sizes, and any dtype other than float32. Tests:
  `tests/weightreading/test_wr_torch_reader.py`.
- **Sandbox** (`sandbox.py`): `unshare` user, mount, pid and net namespaces. A new tmpfs root holds
  only `/usr`, the interpreter, numpy, `/work` and the read-only `/work/weights`, entered via
  `pivot_root`. Code runs in a nested, unmapped user namespace, so it cannot remount. It also gets
  `env -i`, `python -I`, RLIMIT_AS/FSIZE/NPROC/CPU, a timeout that SIGKILLs the PID-namespace init, a
  `/work` size cap and an output cap. There is no aggregate memory/CPU cgroup limit (cgroups are not
  delegated on these nodes), so run paid sessions inside a Slurm job. The preflight
  (`preflight.py`) must pass every check, or no request is sent. There is no unsandboxed fallback.
- **Audit** (`audit.py`): flags nonlinearities, LayerNorm-like expressions, recurrence loops and
  torch imports. `human_verdict` stays null until a reviewer sets it.
