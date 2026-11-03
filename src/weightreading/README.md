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

The bundle, preview and run directories are keyed by the spec alias (`net-6a3a494d`) plus the first 12
hex digits of the checkpoint sha256 (`net-6a3a494d-<sha12>`), so one spec can be used with several
checkpoints: after a v1.3 run, the v1.3-1 run with the same spec writes its own bundle and run
directory. Re-bundling the same checkpoint and spec reuses the bundle; a bundle at the same key whose
bytes differ is still refused ("existing bundle ... differs"). Only the alias reaches the prompt; the
checkpoint hash lives in the directory names and in `run.json` (`inputs.checkpoint_sha256`, full).
Directories from the older alias-only layout are left untouched and ignored.

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
| `outputs/weightreading/bundles/<alias>-<sha12>/` | `weights/<key>.npy` + `weights/index.json` (mounted read-only), `manifest.json` |
| `outputs/weightreading/previews/<UTC>_<alias>-<sha12>_<mode>/` | `message.txt`, `preview.json` (offline token/cost bound) |
| `outputs/weightreading/runs/<UTC>_<alias>-<sha12>_<mode>/` | `message.txt`, `run.json` (provenance, preflight), `transcript.jsonl`, `tool_calls.jsonl`, `state.json`, `final_answer.md`, `audit.json`, `work/` |

Specs carry `provenance` and `notes` that never reach the prompt; the random `network_alias` is
the only name the model sees.

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

## Evaluation (offline, after a run)

`evaluate` scores a run's `claims` against the policy's measured behaviour. It never touches the
run directory, makes no API call and writes `evaluation.json` + `evaluation.md` into a NEW
directory `outputs/weightreading/evaluations/<run name>/` (it refuses if that directory exists).

```bash
uv run python -m weightreading evaluate --run-dir outputs/weightreading/runs/<run> \
  --checkpoint <the .pt the run read> --spec configs/weightreading/<spec>.spec.json [--config eval.yaml]
```

- **Reference forward** (`reference.py`): numpy float64, mirrors mtd-rl-lag (`mlp.py`, `gru.py`,
  `act.py`, `baseline_actor.py`): Linear -> ReLU -> LayerNorm(eps 1e-5, biased variance), GRU gates
  `[r; z; n]` with `b_hn` inside `r * (...)`, carried state = pre-LayerNorm `h'`. Torch parity:
  `sbatch scripts/weightreading_forward_parity.sbatch` (throwaway CPU torch via `uv run --isolated`;
  reports under `outputs/weightreading/parity/forward-<job>/`).
- **Base states** (`eval_ranges.py`, `sampling.py`): no recorded observations exist, so N states
  (default 512, seed 0) are sampled from the spec's harness-only `evaluation` section (uniform
  `range`, discrete `values`, sin/cos `angles` sampled as one angle, `presence` groups that are
  all-zero when absent). Inputs are independent: results describe a synthetic state distribution.
- **Effect** (`effects.py`): per state, the clamped central difference of the head's expected
  value `E[v] = sum_k p_k v_k` (shoot: p(yes)), step 5% of the input range, scaled to "change over
  the whole range / head value span". sin input = increase the angle; cos input = move |angle|
  towards 0; discrete = lowest vs highest value; presence members only where present. Conditions:
  `h0` (primary), `warm` (30 steps of the repeated base observation, then one perturbed step),
  `sustained` (perturbed observation for all 31 steps). Secondary: argmax chosen-value flips.
- **Truth and scores** (`scoring.py`, `bootstrap.py`): '+'/'-' if >= 70% of states agree in sign
  and |mean effect| >= 0.02, else 'none'; accuracy overall / signed / by confidence, outcomes
  (wrong_sign, overclaim, missed), Brier, recall of signed truths over ALL pairs, top-3 coverage per
  head, a 4x4 threshold grid and a state-level bootstrap (95% CI, per-claim label stability).
  Settings (`--config` YAML): `samples, seed, warmup_steps, step_fraction, agree, min_effect,
  top_k, angle_mode (angle|raw), bootstrap_resamples`.
