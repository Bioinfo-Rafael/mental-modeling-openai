# Exp.5 — Sol / Terra / Luna single-query pilot

新3モデルについて、Pendulum・H20・4 metricを各1 queryだけ評価します。**3 models×4 metrics＝12 query**です。公式prompt・送信引数・parserで動作するかを確認し、Exp.6で再利用する先頭応答を取得します。

## 1. 実行方法

既存のMental Modeling / LLM-Xavier用環境を使います。[01の環境説明](../01_prompt_preview/README.md)と同じ `.venv`（Notebookでは `Python (mental-modeling)`）で、新規作成は不要です。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/05_new_models_single/run.py
```

フラグなしはdry-run。API key不要・API 0で、`results/dry_run/manifest.json/csv` に予定12件を保存します。`--dry-run` という引数はありません。

model・task・prompt・件数を確認し、`OPENAI_API_KEY` を環境変数に設定した後、明示的に有料実行する場合だけ次を使います。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/05_new_models_single/run.py --execute --confirm-paid-api
```

全件成功・既定 `--retries 0` なら12 API試行です。両フラグが必要で、一方だけでは拒否します。retryを明示的に増やすと同じqueryの送信試行が増える可能性があります。

アカウントからのモデル利用可否や、公式backendが渡す引数の互換性は、このREADMEの作成では実測していません。エラー時に別モデル・別API・別引数へ自動変更はしません。05がその確認用pilotです。

既存結果の上書き・自動resumeはありません。再実行前に結果を手動退避し、再課金を確認してください。このREADME作成では実験・dry-run・API送信を行っていません。

## 2. 入力元

公式rawのPendulum episodesを使用します。

```text
data/llmx_data/offline_data/<dataset>/raw_transitions/Pendulum-v1/episodes/*.npz
```

[`dataset_adapters/llmx.py`](../../dataset_adapters/llmx.py) の `discover_episodes()` で探し、path順に並べ、各metricで最初の有効queryを選びます。promptは [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) の `build_prompt_queries()` 経由で公式関数が生成します。

同じmetric/Hでは、3モデルは同じsourceとpromptを使います。ただしquery identityにはmodel IDも含むため、モデル間で `query_id` は異なります。rawの取得・書換えはしません。再現性の記録には関連ソースhash、`configs/sources.json`、Git revisionも読みます。

01〜04の結果は入力にしません。03/04の完了を必須条件とするコードもありません。

## 3. 出力先とファイルの見方

出力先は `experiments/05_new_models_single/results/` です。

```text
results/
├── dry_run/                  # 計画確認時のmanifest.json/csv
├── .started.json
├── manifest.json
├── manifest.csv
├── requests.jsonl
├── responses.jsonl
├── records.jsonl
├── run.log
├── summary.json
├── summary.csv
└── runs/
    └── sol__Pendulum-v1__next-action__H20/  # 12条件のうち1つ
        └── episode_000/
            ├── run.json
            ├── config.effective.json
            ├── metrics.json
            └── predictions.jsonl
```

生成結果はGitのignore対象です。1ファイルにつき1モデルという構成ではなく、全モデルを同じJSONLへ保存し、`model`・`condition_id`・`query_id` で識別します。

| ファイル | 主な内容 |
| --- | --- |
| `manifest.json/csv` | 12 queryの計画。source/prompt SHA、H、model、条件別件数。JSONにはprompt全文も保持 |
| `requests.jsonl` | 実際にAPIへ渡すkwargsと試行番号 |
| `responses.jsonl` | 全SDK response、assistant text、usage、UTC時刻、時間、例外 |
| `records.jsonl` | 公式scoreとraw応答・送信内容・usage・query時間を対応付けた記録。Exp.6の再利用入力 |
| `summary.json/csv` | 全体の完了状態・試行数と条件別のAccuracy・時間など。正常完了時、CSVは12行 |
| `runs/.../` | 公式CLIのepisode別標準出力 |
| `.started.json` / `run.log` | 開始記録 / 公式CLIの出力・エラー |

### pilotで確認する項目

| 項目 | 確認内容 |
| --- | --- |
| `summary.status` | `complete` か。不完全な05を06は利用しない |
| `successful_queries` | 採点済み12件が揃ったか。正解12件という意味ではない |
| queryの `status` | `match` / `mismatch` / `ignored` を区別。`ignored` はparserで扱えなかった応答 |
| `assistant_text` / `raw_response` | 実際の応答文と全SDK response |
| `prediction` / `ground_truth` | 公式parserの予測値と公式正解値 |
| `input_tokens` / `output_tokens` / `total_tokens` | API responseのusage。tiktokenの概算ではない。欠損はnull |
| `api_attempts` / `retry_attempts` | 実試行数と追加retry数。N1とは別に数える |
| `source_experiment` | この実験で新規取得した記録は `05_new_models_single` |

`ignored` も採点済みとして数えるため、**完了しただけでparserとの相性が良いとは限りません**。06に進む前に応答とparse率を確認してください。現行の06は、整合する `ignored` recordも再利用対象として許可します。

`Accuracy` は公式 `legacy_compatible_match_rate`。actionは全query数、stateはparsed数が分母で、N1でstate応答がignoredならAccuracyはnullです。parsed/all/element Accuracyも別に保存します。

APIエラー時は以降の送信を止め、`summary.status` は未完了となります。03と同じprefix採点回収の仕組みを共有しますが、各条件N1のため、その条件で失敗したqueryの代替取得はしません。既に完了した別条件の結果は保持します。ログの詳細は [Exp.2](../02_gpt35_single/README.md) を参照してください。

## 4. `run.py` の中身

出典：[`experiments/05_new_models_single/run.py`](run.py)。

```python
EXPERIMENT = common.Experiment(
    name="05_new_models_single", models=common.NEW_MODELS,
    tasks=("Pendulum-v1",), metrics=common.METRICS, histories=(20,), n=1,
)
```

[`experiments/common.py`](../common.py) の `Experiment` dataclassを使って条件オブジェクトを作っています。ここは型の定義ではありません。`("Pendulum-v1",)` と `(20,)` は1要素tupleです。

同じ `run.py` の次の部分で、スクリプト起動時に実行関数へ渡します。

```python
if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
```

`common.run` の定義は `experiments/common.py` の `run(spec, argv=None)`。渡したインスタンスを `spec` で受け取り、dry-runか有料実行かを分岐します。06を自動で起動することはありません。先頭のpath設定とpycache抑止は01と同じです。

## 5. 各設定がどこで使われるか

| 設定 | 値 | 使用箇所・意味 |
| --- | --- | --- |
| `name` | `05_new_models_single` | `common.py:run()` でresultsの場所を決定 |
| `models` | `("sol", "terra", "luna")` | `common.py:NEW_MODELS`。`make_plan()` で3モデルを展開 |
| `tasks` | `("Pendulum-v1",)` | `common.py:make_plan()` でPendulumだけを選択 |
| `metrics` | 4 family | `common.py:METRICS` と `QUESTIONS` から質問を選択 |
| `histories` | `(20,)` | `common.py:history_size()` で公式パラメータ19に変換 |
| `n` | `1` | 各model×metric条件から先頭1件 |
| `preview` | `False`（既定） | 有料実行には二重フラグが必要 |
| `preview_from` / `reuse_from` | 両方 `None` | `common.py:check_inputs()` は過去結果を読まない |

出典：[`experiments/common.py`](../common.py) のmodule定数 `NEW_MODELS`。

```python
NEW_MODELS = ("sol", "terra", "luna")
```

同じファイルの `MODELS` は `sol → gpt-5.6-sol`、`terra → gpt-5.6-terra`、`luna → gpt-5.6-luna` と解決します。API model IDは既存token estimatorと同じです。

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
question = QUESTIONS[task][metric]
upstream_h = history_size(H)
```

| Pendulumのmetric | 公式question_name |
| --- | --- |
| `next-action` | `next_action_prediction_continuous_bins` |
| `last-action` | `last_action_prediction_continuous_bins` |
| `next-state` | `next_state_prediction` |
| `last-state` | `last_state_prediction` |

表の出典は `experiments/common.py` の `QUESTIONS` です。H20は実際の履歴20 recordsを意味します。`make_plan()` が範囲差とStep行数を検査します。promptの書式・質問文・action bin範囲・scoringは公式実装を利用します。

### 応答の記録と公式評価

出典：[`experiments/common.py`](../common.py) の `RecordingSession.backend_class()` が定義する `complete()` 内。

```python
text = super().complete(system_prompt=system_prompt, user_prompt=user_prompt)
```

公式 [`llm_x/backends.py`](../../upstream/LLM-Xavier/llm_x/backends.py) の `OpenAIChatBackend.complete()` を利用し、その内部のSDK `create()` をwrapして記録します。記録用に独自送信・parserを作り直したものではありません。採点は公式 [`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) のままです。

## 6. 呼び出し順

```text
05_new_models_single/run.py
└─ common.run(EXPERIMENT)
   ├─ make_plan() → 3 models × Pendulum × 4 metrics × H20 × N1
   │  └─ discovery → build_prompt_queries() → 公式prompt関数
   ├─ check_inputs() → 過去実験への依存なし
   ├─ フラグなし → dry_run/manifest保存 → 終了（API 0）
   └─ 二重フラグあり → execute_plan() → invoke_cli()
      └─ 公式cli.main() → _evaluate() → EvaluationConfig → _backend()
         └─ evaluate_episode() → Recording backend → super().complete()
            → 実request/response記録 → 公式採点 → 標準出力保存
      → collect_scores() / summarize_scored() → records/summary保存
```

公式CLIは [`llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py)。応答を確認した後、同じqueryを再送せずに [Exp.6](../06_new_models_n10/README.md) の先頭1件として再利用します。
