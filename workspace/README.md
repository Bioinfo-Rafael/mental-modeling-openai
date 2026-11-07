# workspace

Exploratory experiments. Code here may be messy and experiment-specific; anything a second
experiment needs belongs in `src/mtd_rl_xavier/` with tests.

> **Paid API:** `09_ablation_reward`, `10_ICRL`, `11_ICRL_n10` and `12_fewshots_episodes` send
> requests when `run.py` is run **without arguments**; use `--dry-run` to inspect the plan.
> Experiments 01–06 and 08 need `--execute --confirm-paid-api`. `14`–`17` send OpenAI Batch
> requests with `--execute` alone (no `--confirm-paid-api`), and `14_mental-modeling_LAG/batch_run.py`
> submits on `run`, `submit`, `complete-prefix` and `rolling`. Check each README first.

## Experiment directory

```
NN_short_name/
  README.md        Purpose, conditions, commands, findings
  run.py           Entry point; edit the experiment grid here
  results/         Saved plans, API journals, scores. Committed, never rewritten
  analysis/        Scripts that read results/ and write derived tables/figures
  tests/           Offline tests (fixtures or saved data, never the API)
```

- Number new experiments from `21_...` upward. Numbers `18`–`20` are reserved: the legacy
  branch `rafael/mental-modeling-openai` already uses them (`18_desicion_LAG`, `19_RealTime_LAG`,
  `19_RealTime_Lag_v2`, `20_foresight_LAG`) and has not been ported.
- Entry points put the repository root on `sys.path` so sibling experiments can be imported:
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  from mtd_rl_xavier_workspace.experiment import common          # shared library
  runner = import_module("workspace.10_ICRL.runner")   # another experiment (digit-leading name)
  ```
- Experiment-local packages (e.g. `13_LAG/lag_data`, `14_.../lag_experiment`) are added to
  `sys.path` by the entry point and imported by their bare name.
- Tests: `uv run pytest workspace/NN_short_name`, or `uv run poe test-workspace` for all.

## Experiments

01–17 were ported from `rafael/mental-modeling-openai` at `a232376`; later work on that branch
(experiments 18–20 and changes to 13 and 17) is not on `main`. Rafael's detailed index
(conditions, raw-data locations, commands, slides) is [EXPERIMENTS.md](EXPERIMENTS.md); the static
review of 01–06 is [REVIEW.md](REVIEW.md).

| Dir | Topic |
| --- | --- |
| [00](00_state_action_inspection/) | State/action inspection of the LLM-Xavier episodes |
| [01](01_prompt_preview/README.md)–[06](06_new_models_n10/README.md) | Reproduction on MountainCar/Pendulum: GPT-3.5 pilot and H sweep, Joint questions, sample size, three newer models |
| [07](07_view_rawdata/README.md) | Raw response viewer and LLM-as-judge reasoning scores |
| [08](08_fewshot/README.md) | Next-action few-shot prompting |
| [09](09_ablation_reward/README.md)–[12](12_fewshots_episodes/README.md) | Reward ablation, in-context RL prompting (ICR-FQI), few-shot vs. reference episodes |
| [13](13_LAG/README.md) | LAG 2v2 air-combat data: ACMI + action CSV integration and viewer |
| [14](14_mental-modeling_LAG/README.md)–[17](17_commentary_LAG/README.md) | LAG mental modelling, trajectory prediction, dynamics prior, commentary |
| [notebooks](notebooks/) | Dataset and token-count inspection |
