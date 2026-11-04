# mtd-rl-xavier

Experiments on how well language models build a *mental model* of a reinforcement-learning
agent from its state/action/reward history, built on
[LLM-Xavier](https://github.com/LukasWill/LLM-Xavier)
([TMLR 2024](https://openreview.net/forum?id=JN7iNWaPTe)).
The code was ported from the `rafael/mental-modeling-openai` branch; see
[docs/migration-from-rafael-branch.md](docs/migration-from-rafael-branch.md).

## Layout

```
src/mentalmodeling/    Prior work: vendored LLM-Xavier (`llm_x` package, own distribution
                       `llm-x-evaluator` with its LICENSE/NOTICE). Treat as read-only.
src/mtd_rl_xavier/     Shared library: tested, reused by more than one experiment.
  paths.py             Repository layout (ROOT, WORKSPACE, DATA, ...) and old-layout path resolution.
  datasets/            Raw dataset readers (llmx_data NPZ, external candidate datasets).
  preprocessing/       Raw data -> prompts (original LLM-Xavier semantics, external configs).
  experiment/          Planning, guarded OpenAI execution, split batches, Joint-question analysis.
  tools/               CLI utilities: inventories, token counting, schema docs.
  lag/                 LAG episode loader and trajectory rendering (promoted from workspace/13_LAG, 17).
  llm/                 Independent LLM call parts: OpenAI Responses, System One, local vLLM chat/server.
  translator/          Two-layer LLM Translator (change-point detector + translation model) and its CLI.
  local_endpoint.py    Endpoint rules shared by the local-server LLM parts.
src/weightreading/     LLM reads an RL policy's raw weights (no forward pass); also method B of
                       intent_compare (`intent-preview` / `intent-run`). See its README.md.
evaluation/            LLM-as-a-judge and the 3-point x 4-aspect assessment of Translator runs
                       (`python -m evaluation.cli`, `python -m evaluation.assess_cli`); config/, examples/.
scripts/               Slurm jobs and helpers: weightreading rollouts/parity, Translator on vLLM,
                       intent_compare (method A runner, ACMI replay/convert, clips, HTML viewer).
                       check.sh / install.sh / update.sh belong to the Claude Code Orchestra template.
tests/                 Tests for src/, evaluation/ and scripts/ (src/mentalmodeling/tests is collected too).
workspace/             Exploratory experiments, one directory per experiment (NN_name/).
  13_LAG/data/         LAG 2v2 match data (ACMI, recordings, action CSVs, incl. MHI v1.3), tracked in git;
                       read by 14-17, the Translator and judge configs.
  notebooks/           Ad-hoc data and token inspection notebooks.
configs/               Dataset sources (pinned revisions), preprocessing, weightreading, translator,
                       intent_compare configs.
data/                  Dataset schema docs. Downloaded data is gitignored.
outputs/               Generated output. Tracked: token_counts/REPORT.md, token_estimation/ (figures),
                       notebook_checks/ (executed inspection notebooks). Local only (gitignored):
                       dataset_inventory/ and other *.csv/*.json/*.jsonl, weightreading/, translator/,
                       evaluation/, intent_compare/, smoke/.
private/               Gitignored. Real Translator scenarios and evaluation key points/ratings
                       (customer text); never commit. Committed examples live in configs/ and evaluation/examples/.
docs/                  Background notes, including Rafael's original README.
```

Where new code goes:

- One-off or still-changing experiment code goes in `workspace/NN_name/` (see
  [workspace/README.md](workspace/README.md)).
- Once a second experiment needs it, promote it to `src/mtd_rl_xavier/` and add tests in `tests/`.
- Translator, LLM-client and LAG-loader code goes in `src/mtd_rl_xavier/{translator,llm,lag}/`
  (see their README.md files); the judge and assessment go in `evaluation/`
  ([evaluation/README.md](evaluation/README.md)).
- Weight-reading code (LLM explains a policy from its weights) goes in `src/weightreading/`,
  with tests in `tests/weightreading/`, specs and run config in `configs/weightreading/`, and
  outputs in `outputs/weightreading/`. See [src/weightreading/README.md](src/weightreading/README.md).
- `src/mentalmodeling/` changes only to track upstream LLM-Xavier. Its files are hashed into
  experiment provenance (see below), so an edit there invalidates comparisons with saved results.
  It is not part of the `mtd-rl-xavier` wheel. Import it only as `llm_x`: `mentalmodeling` is
  deliberately not an importable name (`pyproject.toml` maps only `mtd_rl_xavier` and
  `weightreading`, so `src/` is not put on `sys.path`), and a test checks that.
  A new subpackage under those two needs an entry in `[tool.setuptools] packages`.

## Setup

```bash
uv sync --all-extras          # Python 3.12 venv in .venv; installs src/mentalmodeling/ as editable `llm_x`
uv run hf download lerrhoo/llmx_data --repo-type dataset \
  --revision dc2b798f72bc02f7285949ccfcdcb42e5ff326fd --local-dir data/llmx_data
uv run python -m mtd_rl_xavier.tools.inventory_llmx_data     # outputs/dataset_inventory/ (dataset registry)
uv run python -m mtd_rl_xavier.tools.inventory_candidates    # same, for external datasets (works when absent)
```

Most tests and every experiment need `data/llmx_data` (172 NPZ, ~56 MB); the dataset registry
used by `tools/` reads the two generated inventories. External candidate datasets
(`data/candidate_datasets/*`) are optional; their sources are pinned in
[configs/sources.json](configs/sources.json).

Paid API calls read `OPENAI_API_KEY` (or a hidden prompt, depending on the experiment).

## Commands

```bash
uv run poe test                                   # tests/ + src/mentalmodeling/tests
uv run poe test-workspace                         # every workspace experiment, one pytest run each
uv run poe lint                                   # ruff check src tests evaluation
uv run python -m weightreading --help             # weight reading: inspect/preview/run (paid only with flags)
uv run pytest tests/test_token_tools.py::test_prompt_generation_reuses_upstream   # one test
uv run pytest workspace/14_mental-modeling_LAG    # one experiment's tests
uv run python -m mtd_rl_xavier.tools.show_data --dataset MountainCar-v0 --episode 0 --index 25
uv run python workspace/06_new_models_n10/analysis/analyze.py
node --test workspace/13_LAG/viewer/test_model.mjs
```

Known failures, identical on the original branch: tests in `tests/` that read the external
candidate datasets (not downloaded by default), `test_refactor_reproduces_all_previous_numeric_tables`,
and one test each in `16_dynamics_LAG` and `17_commentary_LAG` whose expected values predate
their saved inputs.

## Conventions

- **Paths**: import locations from `mtd_rl_xavier.paths`; do not compute the repo root from
  `__file__` in library code.
- **Results are immutable.** Results already tracked in git (`workspace/*/results/`, viewers and
  analysis outputs) stay as they are; history is not rewritten. Analyses write to new directories
  and refuse to overwrite.
- **New large generated artifacts do not go into plain git**: images, videos, API request/response
  journals, self-contained HTML viewers and dry runs go to Git LFS or external storage. Neither is
  set up yet (no `.gitattributes`; storage location TBD), so until it is, keep such files local
  and commit only code, READMEs, manifests and small summaries. Real match data under
  `workspace/13_LAG/data/` is tracked on purpose.
- **Provenance**: plans record SHA-256 of the prompt/scoring code (`src/mentalmodeling/llm_x/*.py`,
  `preprocessing/llmx_original.py`, `datasets/llmx.py`) and refuse to reuse, resume or merge
  results produced under different code. Editing those files is a semantic change. The hash is
  keyed by path, so layout moves are listed in `experiment.common.RESTRUCTURED_SEMANTICS`.
- **Saved results use old-layout paths** (`experiments/...`, `upstream/LLM-Xavier/...`,
  `mentalmodeling/...`).
  Read them through `paths.resolve_recorded(root, relative)`; never rewrite result files.
- **Paid API safety**: most entry points default to a local preview. Check the README before
  running anything that can call OpenAI:

  | Entry point | Sends paid requests when |
  | --- | --- |
  | `workspace/01`–`06`, `08` `run.py` | `--execute --confirm-paid-api` (01 refuses paid runs) |
  | `workspace/09`–`12` `run.py` | run **without arguments**; `--dry-run` only previews |
  | `workspace/14`–`17` `run.py` | `--execute` alone (OpenAI Batch; no `--confirm-paid-api`) |
  | `workspace/14_mental-modeling_LAG/batch_run.py` | the `run`, `submit`, `complete-prefix`, `rolling` commands (no flag) |
  | `python -m mtd_rl_xavier.translator.cli` | `--execute`; a config with a paid (OpenAI) layer also needs `--confirm-paid-api` |
  | `python -m evaluation.cli`, `python -m evaluation.assess_cli judge` | `--execute --confirm-paid-api` |
  | `python -m weightreading run` / `intent-run` | `--execute --confirm-paid-api` |
  | `scripts/intent_compare_run_a.py` | `--execute --confirm-paid-api` (both or neither) |
  | `python -m mtd_rl_xavier.tools.probe_exp06_query` | `--execute --confirm-paid-api` |

  The `scripts/*.sbatch` wrappers add `--execute --confirm-paid-api` themselves when
  `WEIGHTREADING_CONFIRM_PAID=yes`, `INTENT_COMPARE_CONFIRM_PAID=yes` or
  `TRANSLATOR_CONFIRM_PAID=yes` is exported; read the script header before submitting.
