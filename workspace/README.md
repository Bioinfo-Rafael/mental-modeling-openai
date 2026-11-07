# workspace

`workspace/` は、このリポジトリで自由に実験してよい唯一の場所です。
ここにある実験、旧LLM-Xavierのコード、結果は、core（Translator・手法A/B・Judge）から独立した研究作業で、coreのAPIの一部ではありません。
coreはこのディレクトリをimportも読み取りもしません。逆に、ここからcoreをimportすることも禁止です。

## 独立性と保証の範囲

- 外への依存は1つだけです。14〜17の実験が、リポジトリ直下の `../data/mhi/processed/`（MHI試合のCSVとmanifest）を読み取り専用で使います。
- coreのテスト・Slurm job・READMEは、ここの動作を保証しません。ここが壊れても、coreの検証は止まりません。
- coreの変更（`mtd_rl_xavier` など）がここに伝わることはありません。必要なコードはこのプロジェクト内にコピーして使います。`src/mtd_rl_xavier_workspace/lag/dataset.py` は、分離時点のcoreの読込処理を固定したコピーです。

## 構成

| パス | 内容 |
|---|---|
| `00_*`〜`17_*`、`notebooks/` | 実験。場所と結果のパスは移行前のまま |
| `src/mtd_rl_xavier_workspace/` | 実験用ライブラリ（`datasets`、`preprocessing`、`experiment`、`tools`、`lag`、`paths`） |
| `src/mentalmodeling/` | vendoredのLLM-Xavier（配布名 `llm-x-evaluator`、import名 `llm_x`） |
| `configs/`、`data/`、`docs/`、`outputs/` | 旧実験の設定、データの説明、文書、追跡済みの出力 |
| `tests/` | ライブラリのテスト |
| `slurm/validate.sbatch` | このプロジェクトの検証job |

`paths.ROOT` はこの `workspace/` ディレクトリです。
実験の `run.py` は従来どおりリポジトリのrootを `sys.path` に入れ、`workspace.<実験名>` として兄弟の実験をimportします。

## セットアップとテスト

独自の `pyproject.toml`、`uv.lock`、`.venv` を持ちます。coreの `.venv` とは別物です。
重い処理はlogin nodeで動かさず、リポジトリのrootからSlurm jobで実行します。

```bash
sbatch workspace/slurm/validate.sbatch sync          # workspace/.venv を作る
sbatch workspace/slurm/validate.sbatch library       # tests/ と src/mentalmodeling/tests
sbatch workspace/slurm/validate.sbatch experiments   # 実験ごとに1回ずつpytest
sbatch workspace/slurm/validate.sbatch static        # coreをimportしていないかの確認
```

対話的に使うときは `workspace/` に移動して `uv run --project . <コマンド>` とします（例: `uv run --project . poe test`）。
`data/llmx_data/` は手元にないため、それを読むテスト（08〜12とライブラリの一部）は失敗します。分離前から同じ状態です。16と17にも、既知の失敗が1件ずつあります。

## 実験の一覧

| Dir | 内容 |
| --- | --- |
| [00](00_state_action_inspection/) | LLM-Xavierエピソードの状態・行動の確認 |
| [01](01_prompt_preview/README.md)〜[06](06_new_models_n10/README.md) | MountainCar/Pendulumでの再現（GPT-3.5のpilotとH sweep、Joint質問、サンプル数、新しい3モデル） |
| [07](07_view_rawdata/README.md) | 生回答のviewerと推論スコア |
| [08](08_fewshot/README.md) | 次行動のfew-shot prompting |
| [09](09_ablation_reward/README.md)〜[12](12_fewshots_episodes/README.md) | reward ablation、in-context RL、few-shotと参照エピソード |
| [13](13_LAG/README.md) | LAG 2v2試合データの統合とviewer（統合CSVは `../data/mhi/` に移動済み） |
| [14](14_mental-modeling_LAG/README.md)〜[17](17_commentary_LAG/README.md) | LAGのmental modelling、軌道予測、dynamics prior、解説 |
| [notebooks](notebooks/) | データとtoken数の確認 |

詳しい経緯は [EXPERIMENTS.md](EXPERIMENTS.md)、01〜06のレビューは [REVIEW.md](REVIEW.md)、移植の記録は [docs/](docs/) にあります。
新しい実験は `21_...` から番号を振ります。18〜20は旧ブランチ `rafael/mental-modeling-openai` が使っています。

## 結果は書き換えない

- gitで追跡している結果（`*/results/`、viewer、図）は変更しません。新しい結果は新しいディレクトリに出します。
- 01〜06の結果は、プロンプトと採点のコード12ファイルのSHA-256（semantics hash）を記録しています。
  現在の値は `ffe1d9f1…` です。過去の配置の3つの値（`2048668f…`、`830a5841…`、`27e601e9…`）は別名として受け付けるので、03_1・05・06の再利用、resume、mergeは引き続き使えます。
- この12ファイル（`src/mentalmodeling/llm_x/*.py`、`preprocessing/llmx_original.py`、`datasets/llmx.py`）を編集すると、hashが変わってこれらの互換性が切れます。
- 14〜17の `--execute`／`--resume` は、分離前から記録済みの `source_sha256` と合わず動きません（既知の問題）。

## 課金に注意

旧実験の実行口は、coreより課金の安全策が弱いものがあります。各READMEを読んでから実行してください。

- `09`〜`12` の `run.py` は、**引数なし**で実行するとAPIに送信します（`--dry-run` で確認だけできます）。
- `01`〜`06`、`08` は `--execute --confirm-paid-api` が必要です（01は有料実行を拒否します）。
- `14`〜`17` は `--execute` だけでOpenAI Batchに送信します。`14_mental-modeling_LAG/batch_run.py` は `run`、`submit`、`complete-prefix`、`rolling` で送信します。

検証用のjob（`slurm/validate.sbatch`）は、これらの実行口を呼ばず、`OPENAI_API_KEY` を消してから動きます。
有料で実行するときのキーは、coreと同じくリポジトリ直下の `.env` から、実行直前にだけ読み込みます。

## LLM-Xavierのデータ

`data/llmx_data/` は手元でダウンロードして置きます（gitignore済み、手順は [data/README.md](data/README.md)）。
外部の候補データセットの説明は `data/candidate_datasets/` にあります。
