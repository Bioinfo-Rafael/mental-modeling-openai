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
src/weightreading/     LLM reads an RL policy's raw weights (no forward pass); see its README.md.
tests/                 Tests for src/ (src/mentalmodeling/tests is collected too).
workspace/             Exploratory experiments, one directory per experiment (NN_name/).
  notebooks/           Ad-hoc data and token inspection notebooks.
configs/               Dataset sources (pinned revisions) and preprocessing configs.
data/                  Dataset schema docs. Downloaded data is gitignored.
outputs/               Generated inventories and reports from src/mtd_rl_xavier/tools.
docs/                  Background notes, including Rafael's original README.
```

Where new code goes:

- One-off or still-changing experiment code goes in `workspace/NN_name/` (see
  [workspace/README.md](workspace/README.md)).
- Once a second experiment needs it, promote it to `src/mtd_rl_xavier/` and add tests in `tests/`.
- Weight-reading code (LLM explains a policy from its weights) goes in `src/weightreading/`,
  with tests in `tests/weightreading/`, specs and run config in `configs/weightreading/`, and
  outputs in `outputs/weightreading/`. See [src/weightreading/README.md](src/weightreading/README.md).
- `src/mentalmodeling/` changes only to track upstream LLM-Xavier. Its files are hashed into
  experiment provenance (see below), so an edit there invalidates comparisons with saved results.
  It is not part of the `mtd-rl-xavier` wheel. Import it only as `llm_x`: the editable install
  also exposes `mentalmodeling.llm_x`, a second module identity that a test rejects.

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
uv run poe lint                                   # ruff on src/ and tests/
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
- **Results are immutable and tracked in git** (`workspace/*/results/`). Analyses write to new
  directories and refuse to overwrite.
- **Provenance**: plans record SHA-256 of the prompt/scoring code (`src/mentalmodeling/llm_x/*.py`,
  `preprocessing/llmx_original.py`, `datasets/llmx.py`) and refuse to reuse, resume or merge
  results produced under different code. Editing those files is a semantic change. The hash is
  keyed by path, so layout moves are listed in `experiment.common.RESTRUCTURED_SEMANTICS`.
- **Saved results use old-layout paths** (`experiments/...`, `upstream/LLM-Xavier/...`,
  `mentalmodeling/...`).
  Read them through `paths.resolve_recorded(root, relative)`; never rewrite result files.
- **Paid API safety**: most entry points default to a local preview, but `workspace/09`–`12`
  send requests when run without arguments. Check the experiment README before running `run.py`.
