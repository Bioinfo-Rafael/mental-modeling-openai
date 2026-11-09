# Migration from `rafael/mental-modeling-openai`

`main` holds a snapshot of `origin/rafael/mental-modeling-openai` at `a232376`, reorganised into
the current layout (ported in `9657427` on `develop`, merged into `main` by PR #1, `31ca410`).
`main` is the source of truth; `origin/develop` has not moved since that merge and is stale.
Rafael's commit history stays on the legacy branch. Use the original branch to resume saved runs
(see "Compatibility with saved results" below).

The legacy branch kept moving after the snapshot: it added experiments `18_desicion_LAG`,
`19_RealTime_LAG`, `19_RealTime_Lag_v2` and `20_foresight_LAG` and changed `13_LAG` and
`17_commentary_LAG`. None of that is ported. Numbers 18–20 are reserved for it, so new experiments
on `main` start at `21_` (see [workspace/README.md](../workspace/README.md)).

## Path mapping

| Old (rafael branch) | Now |
| --- | --- |
| `upstream/LLM-Xavier/` | `src/mentalmodeling/` (byte-identical; `mentalmodeling/` until it moved under `src/`) |
| `dataset_adapters/` | `src/mtd_rl_xavier/datasets/` |
| `preprocessing/` | `src/mtd_rl_xavier/preprocessing/` |
| `tools/<name>.py` | `src/mtd_rl_xavier/tools/<name>.py`, run as `python -m mtd_rl_xavier_workspace.tools.<name>` |
| `experiments/{common,analysis,split_execution,joint_questions}.py`, `experiments/common_analysis/` | `src/mtd_rl_xavier/experiment/` |
| `experiments/NN_name/` | `workspace/NN_name/`, imported as `workspace.NN_name` |
| `experiments/README.md`, `experiments/REVIEW.md` | `workspace/EXPERIMENTS.md`, `workspace/REVIEW.md` |
| `notebooks/` | `workspace/notebooks/` |
| `README.md` | `docs/mental-modeling-openai.md` |
| `configs/`, `data/`, `outputs/`, `tests/` | unchanged |

Experiments sit at the same depth as before (`workspace/NN` vs `experiments/NN`), so their
`Path(__file__).parents[k]` root computations are unchanged. `paths.LEGACY_PREFIXES` encodes this
table.

## Deliberately left unchanged

- Everything under `results/`, saved analysis outputs, HTML viewers, `data/`, `outputs/`. These
  still contain old-layout paths.
- `workspace/13_LAG/lag_data/prepare.py`: its hash is recorded in
  `13_LAG/data/processed/manifest.json`, so its docstring still says `python experiments/13_LAG/run.py`.
- `configs/preprocessing/llmx_original.yaml` keeps `implementation: upstream/LLM-Xavier`, a label
  checked by `preprocessing/prompting.py` and part of the config hash.
- `preprocessing/llmx_original.py` keeps the name `WORKSPACE_ROOT` (= repository root) because
  tools import it.

## Compatibility with saved results

1. **Recorded paths.** Readers call `paths.resolve_recorded(root, relative)`, which uses the path
   as recorded when it exists and otherwise maps the old prefix. It is applied where code reads
   `upstream_output`, `source_records`, `source_manifest`, `source_results` and semantics-file keys.
2. **Semantics hash.** `experiment/common.py` hashes the prompt/scoring files under their
   repo-relative paths, so moving them changed the hash. `RESTRUCTURED_SEMANTICS` maps the hash of
   the branch head (`2048668f…`) to the current one. Evidence: `make_plan` in the new layout
   regenerates identical query identities and prompt hashes for 03_1 (960), 05 (120) and 06 (960).
   Manifests with older hashes (03, 04: `57369ab9…`) were already rejected on the branch head.
   Moving `mentalmodeling/` to `src/mentalmodeling/` later changed only the path keys (every file
   byte-identical), so the hash of that layout (`830a5841…`) is mapped to the current one too.
3. **Not supported here:** commands that re-validate a saved run against the code or paths it was
   planned with: `--resume`, LAG batch `status`/resume (14–17), and re-planning 09–12 over their
   saved dry runs. Their recorded source hashes, input paths or query identities include files
   moved or edited by the restructure (15 and 17 already failed this check on the branch head).
   They stop with an explicit error before any API request. Every ported run is complete; use the
   original branch if one ever needs this.
4. READMEs and reports generated inside `results/` still link to old locations (for example
   `../../common.py`); they are part of the immutable results.

## Verification performed

- Every test group was run before (original branch, with `data/llmx_data`) and after. Pass/fail
  counts match exactly; failing tests are the same set (external datasets not downloaded,
  `test_refactor_reproduces_all_previous_numeric_tables`, one test each in 16 and 17).
- `03_1/analysis/analyze.py` and `06/analysis/compare_models.py` were run with the original code
  and with the new layout. All tables are byte-identical apart from path prefixes, the merge
  snapshot id, timing metadata and two newly protected source files.
