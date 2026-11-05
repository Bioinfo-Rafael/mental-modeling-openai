# Translator Contract Facts (LAG, LLM as a Translator)

Collected 2026-11-04 by read-only inspection (git `ls-tree` / `show` / `grep`, filesystem listing,
`sinfo` / `scontrol` / `squeue`). No file other than this one was written, no `.env` was read, no API
call was made, no job was submitted, no test was run.

Confidentiality: customer instruction texts (work/evaluation-data-handoff, and the deleted
`docs/instructions.md` in mtd-rl-lag history) are NOT quoted here. Only structure, counts,
categories and paths are reported. The internal English command templates in
`talar/commands.py` are also not quoted.

Abbreviations: IC = initial-condition seed; TALAR = the "task language" pipeline on the `wang/*`
branches (hindsight command labeler + BERT translator, text -> code direction); MHI = the customer;
P/G/T/R/S/C/O = instruction "families" on `origin/manabe/dev` (Preference, Goal, Target, Role,
Spatial, Constraint/shoot, Order).

## Instruction-Conditioned Episodes

**Bottom line: no instruction-conditioned episode, trajectory or video exists on this machine.**
Every recorded output of an instruction-/command-conditioned policy is referenced by code at a path
on another machine.

| Source | What was recorded | Where (per code/docs) | Available here |
|---|---|---|---|
| `origin/wang/talar`, `origin/wang/pref12` `scripts/eval/cmd_analysis/cmd_traj.py` | 4v4, team A pinned to one of 11 commands for the whole episode, deterministic; one JSON per episode `{red_ids, blue_ids, launches[{uid,parent,target,step,dist_km,hit}], frames[{t, ac{id:[n,e,u,alive]}, missiles{uid:[n,e,u]}}], result{...follow, n_win...}}`; 6 commands x 8 seeds x 2 models | `/home/user/mtdrl_work/eval/cmd_qual/traj/` (machine "106") | No (`/home/user` does not exist) |
| `origin/wang/pref12` `scripts/eval/prefcmd_v2/pref_traj.py`, `run_eval.sh` | 4v4 weight-conditioned policy `wang_4v4_far_prefcmd_v2`: strategic-weight sweep (5 settings x 64 eps), maneuver-pinned deterministic trajectories (9 maneuvers x 4), strategic trajectories (4 x 4); same JSON frame format | `/home/user/mtdrl_work/eval/prefcmd_v2/` | No |
| `origin/wang/pref12` `scripts/talar/cmd_eval.py --dump-hist` | One `.npz` per episode of the labeler history (pos/vel/alt/alive/chased/crash + missile events) and closed command windows `{team, cmd, t0, feasible, labels[11], deltas}` | `/home/user/mtdrl_work/eval/prefcmd_v2/hist_{stochastic,deterministic}` | No |
| `origin/wang/*` `scripts/talar/collect_states.py`, `build_windows.py` | Raw 4v4 MORL episodes `ep_XXXXX.npz` (pos, vel, rpy, alt, alive, missiles_left, obs[T+1,8,60], act[T,8,4], w[2,3], missile events) and `windows_v1.npz` (`s0`,`s1` of shape (N,8,9), `labels` (N,11), `episode`, `pre_death`) | `/home/user/mtdrl_work/talar_data/` | No |
| `origin/manabe/dev` Goal / Role / Preference / Order-swap evaluations | 2v2 vs baseline, per-step measurement CSV `instruction-conditioned-measurement-v1/v2` (agent_id x timestep, raw/accepted action, probabilities, reward components, condition weights) | `C:\Users\keisu\LAG\...` (Colab / Windows archive) | No |
| `scripts/eval/instruction_eval/` (sawada, okumura, local `evaluation-data-setup`) | Offline harness only; it reads `evaluation.json` + `telemetry.csv` produced by `evaluate_mhi_team.py`; no outputs are committed (`.gitignore` excludes `results`, `runs`, `*.acmi`) | n/a | No outputs |
| `mtd-rl-xavier/workspace/13_LAG` | 3 episodes, NOT instruction-conditioned (see "Relation To 13_LAG Data") | local | Yes |

Filesystem search (`evaluation.json`, `telemetry.csv`, `*.npz`, `windows_v1*`, `*cmd_stats*`,
`conditions.json`, `*.acmi`, `*.mp4`) under `/home/user151005` found only the 13_LAG files, Xavier
experiment `conditions.json` analysis files (14/15/16), and one WEZ table npz in mtd-rl-lag.

Customer instruction material (structure only):

- `work/evaluation-data-handoff/output/20261102-second-round-paste/source-catalog.csv`: 43 rows,
  one per instruction (`instruction_id`, `source_item_label`, `original_text_ref`,
  `original_text_sha256`, `verification_status=transcribed` for all). Categories:
  A strategic / single aircraft 14, B strategic / whole side 8, C tactical / single aircraft 14,
  D tactical / whole side 7. Assumed situations were omitted from the paste. Text lives in
  `source-text/client-instructions-second-paste.txt` (56 lines; not read in full, not quoted).
  `source-documents.csv` marks all three source documents as customer material, not to be sent to
  other services or models; the originals (`指示内容案_R1_マージ.pptx`, first-round list of
  2026-10-15) are not provided.
- `mtd-rl-lag` git history `adb95c9:docs/instructions.md` (on `origin/okumura/dev` and local
  `evaluation-data-setup`; deleted at tip by `4b66c46`): a 44-row table with columns level
  (strategic/tactical), target (single aircraft / whole side), task type (attack, defense-evasion,
  lure, shield), instruction text, assumed situation, verdict, and the weight mapping. Verdicts:
  7 fully expressible by the 4 reward weights, 26 partially, 11 not. Largest cells: tactical/single/
  defense-evasion (5), tactical/single/attack (6), strategic/single/attack (7).
  Security note: this customer-derived text is in pushed git history (already flagged as a risk in
  `work/evaluation-data-handoff/output/20261102-second-round-paste/implementation-plan.md` §13).

## Instruction Representation & Onsets

There is no single notion of "instruction". Four incompatible code-level representations exist:

| Where | Instruction at code level | Who sees it | When it enters | Per episode |
|---|---|---|---|---|
| `scripts/eval/instruction_eval/` (`io.py`, `tactics.py`, `README.md`) | A free-text string in a manifest condition (`id`, `model`, `instruction`, `role` in baseline/instruction/placebo/surrogate, `baseline`, `target_reward`, `root`, optional `onsets_s`, `pattern`, `expected_seeds`). The policy never receives it; a condition is just a group of runs | Only the offline report | `onsets_s`: list of issue times in **seconds**, one list per condition, applied to every match of that condition; each onset opens a 5 s window `[onset, onset+5 s]` (`rules.window_s`); windows needing missing telemetry are dropped. Default (no `onsets_s`): back-to-back 5 s windows from the first recorded time ("re-issued every 5 s") | Many windows; one instruction string |
| `origin/wang/talar`, `pref12` `envs/JSBSim/tasks/multiplecombat_cmd_task.py`, `talar/commands.py` | Integer command id 0..10 (`hold, engage, disengage, climb, descend, flank_left, flank_right, spread, regroup, attack, evade`), appended as an 11-dim one-hot to the 57-dim obs; per **team** (all 4 members share it) | Policy | Issued at reset and whenever its duration expires; duration ~ U[H, 3H] steps, H = 25 steps = 5 s (dt 0.2 s); event switch to `evade` with prob 0.5 when a member becomes newly targeted; eval hook `command_override` pins a command (re-issued) | Many (every 5-15 s) in training; one pinned in eval |
| `origin/wang/pref12` `multiplecombat_prefcmd_task.py`, config `HierarchySelfplay_far_prefcmd.yaml` | Weight vector: 4 strategic weights (survival, kill, lure, shield; simplex, per team per episode) + 8 maneuver weights in [0,1] (per team per segment); obs 69 = 57 + 4 + 8. Docstring: "so that an instruction translated by an LLM into weights can drive the policy" ("approach 1", 2026-10-15 MHI meeting) | Policy | Strategic: episode start. Maneuver: segments of `pref_segment_steps: [50, 75]` = 10-15 s; 0/1/2 active maneuvers with p = 0.25/0.5/0.25 | One strategic + many maneuver segments |
| `origin/manabe/dev` Goal / Role / Preference specs (`docs/*_condition_spec.md`) | 2-dim one-hot or 2 numeric weights concatenated by the runner (actor 33 -> 35, critic 132 -> 134), 2v2 vs baseline | Policy | Episode start, fixed for the episode | One |
| `origin/manabe/dev` `docs/order_role_swap_spec.md` | Mid-episode swap of the Role one-hot (KEEP vs SWAP), no retraining | Policy | Swap at step 100 (20 s); 40 s rejected (both egos rarely alive); RNN kept (main) or zeroed (auxiliary) | Two (before/after) |

Category vocabularies that already exist (useful for the Translator output space or a rule baseline):

- 11 team maneuver/tactic labels, defined twice with different thresholds: `talar/commands.py`
  (`THRESH`, `THRESH_V2`, judge v1/v2 in `talar/labeler.py`) and
  `scripts/eval/instruction_eval/tactics.py` (Japanese label keys, 5 s windows, provisional
  thresholds in `default_config.json`: 1000 m range, 500 m altitude, 15 deg turn, 500 m spread).
- 4 strategic objectives (survival, kill, lure, shield) in prefcmd; instruction_eval uses a
  different provisional 4 (kill, survival, support, altitude) and states they are not the
  customer's final definitions.
- `talar/commands.py` has 10 train + 6 held-out English imperative templates per command (110/66)
  and `make_sentence()` composing 1-2 clauses: an internal, non-customer synthetic instruction
  source.

## Relation To 13_LAG Data

- `mtd-rl-xavier/workspace/13_LAG/data/raw/20261005_LAG_2v2_ACMI/`: 3 episodes
  (`mhi_v1.3_ic100601_win`, `ours_seed100100_ic100601_loss`, `ours_seed100100_ic100606_win`), each
  ACMI + `actions/*_actions.csv` + 4x-speed MP4; plus `data/viz/action_videos/*_with_actions.mp4`.
  2v2: A0100/A0200 learned (MHI v1.3 or in-house seed 100100), B0100/B0200 fixed rule (pursue);
  deterministic; dt 0.2 s; 200 s cap; 293-495 steps (58.6-99.0 s).
- `data/processed/*.csv`: 64-column, one row per (step, object), step 0-based, `time_s = step*0.2`,
  degrees; loaded by `workspace/13_LAG/lag_data/dataset.py::load_episode`.
- **No instruction, condition vector, command, or onset is recorded in 13_LAG.** It is the
  "no-instruction baseline" kind of episode.
- Shared ground with mtd-rl-lag: same 2v2 MHI-integration world and IC seed family (1006xx) as
  `instruction_eval` (`expected_seeds` example 100600/100601) and `manabe/dev` evaluation seeds
  100600-100604. The generator of `*_actions.csv` is unknown (handoff D7).
- Format gaps: LAG `telemetry.csv` is long-form `step,metric,value`, step 1-based, radians,
  `blood`/`ammo` names; TALAR/prefcmd use 4v4 NEU arrays/JSON; no converter to the 64-column
  format exists for either. Name/unit mapping is drafted in
  `work/evaluation-data-handoff/output/20261102-second-round-paste/data-contract-draft.md` §6.
- The 4v4 TALAR/prefcmd episodes (22.2 km "far" scenario, self-play) are a different scenario and
  team size from 13_LAG.

## Time-Point Concepts Found

No code anywhere implements change-point detection or a "request point". The terms 開始点 /
変化点 / 要求点 appear only in Xavier's judge planning notes. Closest existing concepts:

| Concept | Where | Meaning |
|---|---|---|
| Issue time `onsets_s` (発令時刻) | `instruction_eval/README.md`, `tactics.py::windows` | Instruction issue time in seconds; window start for following judgement; no end/cancel time |
| Command issue `t0` / `seg_t0`, `cmd_end`, interrupted window | `multiplecombat_cmd_task.py` | Command start step, scheduled end, preemption by the event-driven evade switch (a closed window is scored at `t0 + W`) |
| Swap time | `order_role_swap_spec.md` | Mid-episode instruction change at step 100; O-L1/O-L2/O-L3 = probability change / executed action change / approach to the new target after the swap |
| Family O columns | `instruction_evaluation_readiness.md` (manabe/dev) | "instruction ID start, end, cancel, residual window" listed as needed but not implemented |
| Human-picked centre steps | `mtd-rl-xavier/workspace/17_commentary_LAG/README.md` | User picked steps "where motion changed most" (45, 145, 245, 315 in the shared run; repo fixes 45, 85, 145, 245, 310) with a ±30-step window; LLM explains intent per aircraft (`A0100..B0200` JSON) |
| Judge timestamps | `mtd-rl-xavier/.claude/docs/research/feature-llm-judge-intent-translation-eval-brief.md` | Start/change/request points are judge inputs, stored in simulation steps (0.2 s) with a `source` label (human / video_llm / translator); how to detect the change point is explicitly open |

## vLLM / Local Serving Found

- **No vLLM (or any local serving) setup exists in mtd-rl-xavier, mtd-rl-lag (all branches),
  mtd-rl-llm-as-a-policy, or work/.** Matches for `vllm`/`tgi` were base64 noise in notebooks/HTML;
  "OpenAI" in LAG is the Gym/Baselines docstring.
- `vllm` is not installed: `mtd-rl-xavier/.venv` (Python 3.12.16) has `openai 2.54.0`, `httpx`;
  `mtd-rl-lag/.venv` has `torch 2.4.1`. No `vllm`, `ollama`, `docker`, `apptainer`, `singularity`
  binaries; no `~/.cache/huggingface`. `uv` is at `~/.local/bin/uv`.
- Login node: `nvidia-smi` missing (no GPU here). Modules: `nvhpc/24.5`, `nvhpc-hpcx-cuda12/24.5`
  (no vLLM module).
- SLURM: partition `151-partition` (AllowGroups=group151, MaxTime unlimited), nodes
  `srdgx00043, srdgx00051-00053`, each `gpu:8` (GPU model not exposed), 224 CPUs, ~2 TB RAM. At
  check time two nodes were idle and two partly allocated.
- Another group member's running jobs are named `GEMMA-VLLM` (2 GPUs each). This shows vLLM runs on
  this cluster, but that setup is not in our repos and was not inspected.
- Existing sbatch template to copy: `mtd-rl-xavier/scripts/weightreading_run.sbatch`
  (`-p 151-partition -n 1`, no `--gres`, uv-based, paid-call gate via env var). Untracked file.

## Reusable LLM-Call Code

| Code | What it is | Coupled to |
|---|---|---|
| `mtd-rl-xavier/src/weightreading/backends.py` (`OpenAIResponsesBackend`, `make_openai_client`) | Stateless multi-turn Responses API backend, `store=False`, `max_retries=0`, usage accounting, request record without credentials | OpenAI Responses API and reasoning models (`reasoning.encrypted_content`); no `base_url`; key from `OPENAI_API_KEY`. Untracked (not committed) |
| `mtd-rl-xavier/src/weightreading/cli.py`, `commands.py`, `config.py` (`client_factory` injection) | Paid-call gate (`--execute --confirm-paid-api`), strict frozen-dataclass YAML config with pricing, fake-client tests (`tests/weightreading/test_wr_cli_gate.py`) | The weightreading session/sandbox flow; pattern is reusable, code is not generic |
| `mtd-rl-xavier/mentalmodeling/llm_x/backends.py` (`ChatBackend`, `OpenAIChatBackend(compatible=True, endpoint=...)`, `OllamaChatBackend`, `SequenceBackend`) | The only OpenAI-compatible `base_url` chat-completions backend in the repos (directly usable against a vLLM server), plus an offline fixture backend | Vendored upstream LLM-X code (ruff-exempt style); single-turn system+user only; temperature 0; in-backend retries with sleep |
| `mtd-rl-xavier/src/mtd_rl_xavier/experiment/common.py` (`RecordingOpenAIBackend`) | Journal/replay wrapper with request verification | Hard-rejects any base URL other than `https://api.openai.com/v1` (SafetyStop), so it blocks vLLM by design; tied to the LLM-X experiment plans |
| `mtd-rl-xavier/workspace/17_commentary_LAG/commentary_experiment/execution.py` (also 14's `lag_experiment/execution.py`) | Plan -> journal executor (`responses.jsonl`, run lock, resume, `client_factory` for tests), `chat.completions.create` | Experiment plan/sample format; `OpenAI()` built without `base_url`; key may be prompted via `getpass` |
| `mtd-rl-xavier/workspace/17_commentary_LAG/commentary_experiment/batch.py`, `workspace/14_mental-modeling_LAG/batch_run.py` | OpenAI Batch API submission/collection on `/v1/chat/completions` | OpenAI Batch API (not offered by `vllm serve`) |
| Trajectory renderers | 14 `_render_block` text blocks (`workspace/14_mental-modeling_LAG/lag_experiment/prompts.py`), 17 JSON lines (`workspace/17_commentary_LAG/commentary_experiment/{dataset,prompts}.py`), 13 `load_episode` | 64-column 13_LAG format only |

## Unverified Items

- Existence and contents of the machine-106 outputs (`/home/user/mtdrl_work/...`) and the
  Windows/Colab archives (`C:\Users\keisu\LAG\...`); whether they can be copied here.
- GPU model and driver/CUDA version on `srdgx*` nodes; internet / Hugging Face access from compute
  nodes; whether `vllm` installs into the Xavier uv env (Python 3.12).
- Whether the `GEMMA-VLLM` jobs expose an endpoint our jobs may use (not inspected; other user).
- Whether vLLM's `/v1/responses` support is sufficient for `OpenAIResponsesBackend`; the
  chat-completions path is the safe assumption.
- Whether the 43 pasted instructions are the same set as the 44 rows of the deleted
  `docs/instructions.md` (handoff D13 is still open).
- Labels from `tactics.py` and `talar/labeler.py` are program annotations, not reference values.
- No tests in any repo were run.
