# Exp.5 — Joint prompt、Pendulum H=20、N=10

03_1と同じJoint questionで、Sol / Terra / Lunaの応答を確認する。
各モデルはPendulum × 4 Metric × 10 query = **40件**。3モデルで120件。
directory名の `single` は互換性のため残すが、現在はN=1ではなくN=10。
この120件すべてを06で再利用し、同じqueryを再送しない。

## 1. 実行方法

[01と同じ環境](../01_prompt_preview/README.md)を使用する。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/05_new_models_single/run.py
```

フラグなしはdry-run、API 0。計画は `results/dry_run/manifest.json/csv` に保存する。
予定はlogical=120、新規API=120、reuse=0。API keyは不要。

確認後、`OPENAI_API_KEY` を環境変数に設定し、**ユーザーが有料実行するときだけ**：

```bash
python experiments/05_new_models_single/run.py --execute --confirm-paid-api
```

既定retry=0で、新規送信は120件。2つのフラグが揃わないと送信しない。
### 完了後の費用・時間見積もり

05完了後のtoken・金額・時間の集計と06追加分（×7）の概算は、[解析README](analysis/README.md#05の実測から06のtoken金額時間を概算する)を参照。
`python experiments/05_new_models_single/analysis/estimate_exp06.py`で、`analysis/cost_estimates/<日時>_<識別子>/RESULTS.md`と`estimate.json`を新規保存する。API再送信はしない。

### APIの生成設定（05・06共通）

sol / terra / lunaには、`model`・`messages`に加えて **`reasoning_effort="medium"`だけ**を指定する。`temperature`はキー自体を送らない（`0`や`null`を送るのではない）。`top_p`・出力token上限・その他の生成パラメータも指定せず、API既定値を使用する。Chat CompletionsとモデルIDは変更しない。通信timeout・再試行・課金確認などの安全設定は従来どおり。

研究上の意図は、単なる数値補完ではなく、agent historyからreasoningしてmental modelを構築できるかを評価すること。ユーザーが示した元論文のCoT方針（全モデルにCoTを促し、答えの前に説明を求める）に合わせ、03_1の説明を求めるJoint questionを維持し、05・06ではreasoning effortをmediumに固定する。ただしAPIのreasoning effortと、プロンプトで説明を求めることは別の設定であり、内部推論の全文が出力されることや元論文と同一の推論過程になることを保証するものではない。

実装は[`../common.py`](../common.py)の`api_request_options()`と`RecordingSession.backend_class()`内の`recording_create()`。upstreamが渡す`temperature=0`を新モデルの送信直前に除去し、上記設定に変更してから記録・送信する。GPT-3.5の既存実験は`temperature=0`のままで、reasoning effortを追加しない。

設定は`manifest.json`の`api_request_options_by_model`、実際の送信内容は`requests.jsonl`の`kwargs`と`records.jsonl`の`request`に残る。06の再利用でも同じ設定であることを照合し、旧temperature設定や異なるreasoning effortの結果は拒否する。

Chat Completionsでのパラメータ名は[OpenAI公式資料](https://developers.openai.com/api/docs/guides/latest-model)で確認。今回の検証はモックによるオフラインテストのみで、実APIの再送信は行っていない。

以前の`temperature=0`で失敗した`results/`は自動削除・上書きしない。既存結果がある状態では再実行を停止するため、再実行前に失敗した`results/`全体を別名へ退避して保管する必要がある（05の`--resume`は未対応）。
エラー時にモデルや引数を自動変更しない。

既存の実行結果があれば上書きせず停止する。05/06の `--resume` は未対応。
失敗後にresultsを削除して再実行すると重複課金の可能性があるため、ログを残してから対応する。

## 2. 入力元とquestion

rawはrepository rootの `data/llmx_data/offline_data/.../raw_transitions/Pendulum-v1/episodes/*.npz`。
`dataset_adapters/llmx.py:discover_episodes()` → `preprocessing/llmx_original.py:build_prompt_queries()` を使い、path順のepisodeから各条件の先頭10有効queryを選ぶ。
同じpromptを10回繰り返すのではない。H=20は実際に20 timestep、episode境界は跨がない。

questionの対応は [../joint_questions.py](../joint_questions.py) の `JOINT_QUESTIONS` を03_1/05/06で共有する。

| Metric | feedbackのquestion名 | 取得する値 |
|---|---|---|
| next-action | next_action_prediction_continuous_joint | raw torque、action bin、説明 |
| last-action | last_action_prediction_continuous_joint | raw torque、action bin、説明 |
| next-state | next_state_prediction_more_options_joint | DEC/INC/UNCH、absolute state、forward delta、説明 |
| last-state | last_state_prediction_more_options_joint | DEC/INC/UNCH、absolute state、forward delta、説明 |

質問文の本体は [feedback.py](../../upstream/LLM-Xavier/llm_x/feedback.py)。今回、文章の再実装・変更はしていない。
03_1の回答を新モデルの回答として流用することもない。

## 3. 出力先・見るべきファイル

`experiments/05_new_models_single/results/` に保存する。全モデルを同じJSONLに保存し、`model_alias / condition_id / query_id` で区別する。

| ファイル | 生成する関数（experiments/common.py） | 内容 |
|---|---|---|
| manifest.json/csv | make_plan() → save_manifest() | 12条件・120 query、prompt全文とSHA、episode/index |
| requests.jsonl / responses.jsonl | RecordingSession.backend_class() 内 recording_create() | 送信引数、全raw response、usage、時間、例外 |
| records.jsonl | collect_scores() | queryと回答・旧scoreを結合。正常完了時120行 |
| summary.json/csv | execute_plan() | 完了状態、試行数、条件別の旧scorer集計。CSVは12条件 |
| runs/.../predictions.jsonl 等 | invoke_cli() → 公式CLI | 公式CLIの標準出力 |
| run.log / .started.json | invoke_cli() / _run() | 実行ログ・上書き防止の開始記録 |

`successful_queries=120` は応答取得・採点記録完了の件数で、正解120件という意味ではない。
**Jointの正式な評価にはresultsの旧prediction/Accuracyを使わず、次の共通解析CSVを使う。**
特に旧scorerはPendulumのtorqueをbinと誤認し得る。

## 4. 05の応答をオフライン評価する

05がcompleteになった後：

```bash
python experiments/05_new_models_single/analysis/analyze.py
```

APIは呼ばず、[共通解析](../common_analysis/README.md)で各モデルのN=10のCSVを作る。
出力は `analysis/sol/tables/`、`analysis/terra/tables/`、`analysis/luna/tables/`。
`parsed_records.csv`、`parse_summary.csv`、`accuracy_metrics.csv`、`continuous_metrics.csv`、`diagnostics.csv` 等を確認する。

05はPendulum・H20のみなので、存在しない他task/Hを補って比較図は作らない。
全Task/Hの図は06完了後の共通解析で生成する。05は同じ評価式のpilot表を出す。

再生成は `--overwrite-derived` を明示する。別保存先は自分のanalysis以下の `--output-dir` で指定できる。
rawやresultsは上書きしない。

## 5. run.pyと呼出順

```python
EXPERIMENT = common.Experiment(
    name="05_new_models_single", models=common.NEW_MODELS,
    tasks=("Pendulum-v1",), metrics=common.METRICS, histories=(20,), n=10,
    questions=JOINT_QUESTIONS,
)
```

[run.py](run.py)で条件オブジェクトを作り、`common.run(EXPERIMENT)` へ渡す。
`Experiment` と `run / make_plan / execute_plan` は [common.py](../common.py)。

```text
run.py → common.run() → make_plan()
  → build_prompt_queries() → 既存feedback question
  → dry-run、または二重フラグ確認 → execute_plan() → invoke_cli()
  → 公式backend + 送受信ログ → collect_scores()

analysis/analyze.py → common_analysis.runner.main()
  → load_and_parse() → aggregate_all() → CSV（N=10）
```

## 6. 06へ進む条件

05の `summary.status=complete` と120件のrecordsが必要。
正解だけを選ばず、整合する `match / mismatch / ignored` をすべて再利用する。
旧N=1や非Jointの05結果は、現在の06とidentityが違うため拒否される。

[06の手順](../06_new_models_n10/README.md)でdry-runの `prerequisites: ready` を確認してから進む。
今回の変更時にはdry-runとオフラインテストのみ実施し、有料API送信は行っていない。
