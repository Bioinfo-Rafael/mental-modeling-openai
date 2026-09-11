# Exp.6 — Joint prompt、全条件N=10、Exp.5を重複送信しない

03_1と同じquestionを使い、Sol / Terra / Lunaについて
2 Task × 4 Metric × H={5,10,20,30} × 10 queryを揃える。
各モデル320 queryのうち、05の40 queryを再利用し、**06の新規送信は280件/model**。

| 対象 | 05の新規送信 | 06の新規送信 | 06に含める再利用 | 最終ユニークquery |
|---|---:|---:|---:|---:|
| 1モデル | 40 | 280 | 40 | 320 |
| 3モデル合計 | 120 | 840 | 120 | 960 |

retryなしの件数。統合後の960 recordsに05の120 recordsが含まれるため、05をさらに足して集計しない。

## 1. 実行方法

05を今回のJoint・N=10設定で完了させる。[05の手順](../05_new_models_single/README.md)。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/06_new_models_n10/run.py
```

フラグなしはAPI 0のdry-run。`results/dry_run/manifest.json/csv` に計画を保存する。
06未着手なら新規840 query（280/model）を計画する。05の120件は送信batchへ入れず、最後に統合する。
05の完了・設定一致が必須で、不足していればdry-runでもエラーになる。

`OPENAI_API_KEY` を環境変数に設定し、ユーザーが有料実行するときだけ：

```bash
python experiments/06_new_models_n10/run.py --execute --confirm-paid-api
```

05が未完了・不足・旧N1・非Joint・prompt不一致なら、最初の新規送信より前に止まる。
05の不足を新規APIで補うfallbackはない。
既定retry=0。タイムアウトのサーバー到達や課金はローカルログだけでは断定できないため、明示retryを増やすと同一queryの再送が起き得る。
モデルID・API引数・question・queryの選択順は変更しない。

既存resultsの上書き・自動再送はしない。`--resume` は非対応。結果を削除して再実行しないこと。
今回の分割機能の検証はモックによるオフラインテストのみ。有料APIは呼んでいない。

### まずPendulum・H5だけ実行する

```bash
# 予定確認：3モデル × 4 Metric × 10 query = 120件（40/model）
python experiments/06_new_models_n10/run.py --task Pendulum-v1 --history 5

# 実際に送信する場合のみ
python experiments/06_new_models_n10/run.py --task Pendulum-v1 --history 5 --execute --confirm-paid-api
```

`--task`はMountainCar-v0 / Pendulum-v1、`--history`は5 / 10 / 20 / 30。いずれも複数値を指定できる。省略した軸は全てを対象とする。モデルはsol/terra/luna、Metricは全4種類、N=10で固定する。

指定範囲に1条件でも完了済みのものがあれば、**「実行済みです」エラーで指定全体を拒否し、1件も送らない**。例えばH5完了後の`--history 5 10`も拒否する。明示指定に05取得済みのPendulum/H20が含まれる場合も同じ。query IDだけでなくmodel/task/metric/H単位の重複を検査し、別の指定方法でも重複を許さない。

### 完了済みを除く残りをまとめて実行する

```bash
python experiments/06_new_models_n10/run.py --remaining
python experiments/06_new_models_n10/run.py --remaining --execute --confirm-paid-api
```

**`--remaining`を明示したときだけ、完了済み条件を除外する。** 05と06のPendulum/H5が完了した状態なら、残り240件/model、全720件を新規送信する。`--remaining --task Pendulum-v1`のように範囲も限定できる。対象が全て完了していればエラーで終了する。

### 途中失敗の扱い（再開機能は実装しない）

今回のsol/MountainCar/next-action/H5の2件目（index=6）の500エラーを単独確認するには、repository rootで`python tools/probe_exp06_query.py --execute --confirm-paid-api`を実行する。
`612bf8080963456b9286c5481885a700`の保存済みrequestをそのまま1回だけ送信し、SDK自動retryも0にする。両フラグを外すと検証のみ。
`outputs/exp06_single_query_probe/<日時>_<識別子>/`に`request.json`と`result.json`を新規保存し、成功時の全response/usage、失敗時のHTTPコード・error body・request ID、経過時間を記録する。認証ヘッダーは保存しない。
実験の予約・結果には触れず、成功しても06の正式結果へは自動統合しない。再実行ごとに新たな有料試行になる。request IDとretry設定は[OpenAI SDK公式仕様](https://developers.openai.com/api/reference/python#request-ids)に従う。

送信前にbatchの全対象条件をmanifestへ予約する。そのbatchが失敗・中断した場合、未送信だった条件も含めて予約を残し、同じ範囲の再指定や`--remaining`での自動再送を禁止する。エラーは「予約済み/未完了の条件です」。別の未予約条件を明示して実行することはできるが、全体統合は未完了batchがある限り停止する。復旧はログ確認のうえ別途対応する。API結果が不明なrequestを安易に再送しないための制約。

### 全条件が揃ったら統合（APIなし）

```bash
python experiments/06_new_models_n10/run.py --merge
```

05の120件と、全batchの重複しない840件を検証し、960件を`results/merged/<識別子>/`へ新規保存する。足りない条件・重複・未完了・prompt/設定不一致があれば統合しない。`--merge`に有料実行フラグは付けない。再実行時も別snapshotを作り、既存snapshotを上書きしない。

統合後は通常の`analysis/analyze.py`を使う。分割結果がある場合、解析コマンドは同じ統合処理を先に自動実行するため、手動の`--merge`を省略してもよい。全条件が揃うまで全体の評価は生成しない。

## 2. 入力と重複防止

入力は公式raw episodesと、05の `results/manifest.json / summary.json / records.jsonl`。
rawの書換えや再取得はしない。各条件はpath順episodeの先頭10有効query。

[run.py](run.py)の設定：

```python
EXPERIMENT = common.Experiment(
    name="06_new_models_n10", models=common.NEW_MODELS,
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=10,
    reuse_from="05_new_models_single",
    reuse_n=10,
    questions=JOINT_QUESTIONS,
)
```

[common.py](../common.py) の `make_plan()` は、Pendulum・H20・ordinal<reuse_nの全queryを必須再利用に指定する。
今回はその4条件×10件を丸ごと再利用する。残り28条件×10件のみを新規送信する。

`check_inputs()` は、再利用対象の集合全120件について以下を送信前に照合する：

- model、task、metric、question、H、episode path/hash、query index、system/user prompt SHAで構成するquery ID。
- 05の全query集合との完全一致、重複・不足の不在、履歴範囲・ordinal。
- 05の完了状態と120件のrecord。
- assistant_textとraw_responseの一致、保存requestのmodel/messages/reasoning_effortの一致とtemperatureが存在しないこと。その他の生成パラメータが追加されていた場合も拒否する。
- upstream/preprocessingのsemantics hash。

### 実行前の費用・時間見積もり

05の完了後、`python experiments/05_new_models_single/analysis/estimate_exp06.py`で06の追加280件/modelのtoken・金額・時間を概算できる。05の40件/modelの実測を単純に7倍する。再利用する40件分は追加料金に含めない。05＋06累計（8倍）も別表で出力する。
出力は05の`analysis/cost_estimates/<日時>_<識別子>/RESULTS.md`と`estimate.json`。詳しい入出力・前提は[05解析README](../05_new_models_single/analysis/README.md)を参照。06の実験コード・結果には触れず、APIも呼ばない。

### APIの生成設定

05と同じく、sol / terra / lunaは`reasoning_effort="medium"`のみ明示し、`temperature`は送らない。`model`・`messages`以外の他の生成パラメータは未指定（API既定値）。通信・再試行・課金確認の安全設定は変更しない。

単なる数値補完ではなくagent historyからreasoningしてmental modelを作れるかを評価するため、03_1の説明を求めるJoint questionを維持する。プロンプトによるCoTの促しとAPIのreasoning effortは別の設定である。研究上の意図・実装箇所・記録先は[05のAPI設定](../05_new_models_single/README.md#apiの生成設定0506共通)を参照。05の旧設定の結果を混ぜず、再利用前に実際のrequest全体を照合する。

分割方式では05の応答を新規送信batchへ含めず、最後のオフライン統合時に保存済み応答・公式scoreをコピーする。
05不足時に新規送信へ切り替えない。
API成功を意味する `ignored` recordも内容が整合していれば再利用し、正解だけを選ばない。

## 3. Joint questionと評価の分離

対応表は [joint_questions.py](../joint_questions.py) の `JOINT_QUESTIONS` を03_1/05/06で共有。
question本体は既存 [feedback.py](../../upstream/LLM-Xavier/llm_x/feedback.py) にある。

| 対象 | 呼ぶquestion |
|---|---|
| MountainCar Next/Last Action | next_action_prediction / last_action_prediction |
| Pendulum Next/Last Action | next_action_prediction_continuous_joint / last_action_prediction_continuous_joint |
| 両Task Next/Last State | next_state_prediction_more_options_joint / last_state_prediction_more_options_joint |

**実行時の旧scorer出力はログとして残すが、Jointの正式な評価値には使わない。**
03_1から移した [common_analysis](../common_analysis/README.md) がassistant_textを再parseする。
raw torqueとbin、directionとabsolute stateとdeltaを独立に評価する。

## 4. 実行結果の出力

`experiments/06_new_models_n10/results/`の構造：

```text
results/
  .execution.lock                 # 全分割実行・統合で共通の排他ロック
  dry_run/                        # 計画のみ。条件予約には数えない
  batches/<識別子>/               # 各回の新規送信結果。上書きしない
  merged/<識別子>/                # 完成した全960件の派生snapshot
```

各`batches/<識別子>/`の内容：

| ファイル | 内容・生成元 |
|---|---|
| manifest.json/csv | 今回選んだ条件・queryのみ。`make_plan → select_plan → save_manifest` |
| requests.jsonl / responses.jsonl | 新規API試行の送受信ログ。`RecordingSession` |
| records.jsonl | 今回新規送信したqueryのみ。Pendulum/H5なら120行。`collect_scores` |
| summary.json/csv | 完了状態・実API試行数・旧score等。`execute_plan` |
| runs/<condition>/episode_000/... | 公式CLIの設定・旧prediction/metrics。`invoke_cli` |
| run.log / .started.json | ログ・上書き防止記録 |

統合snapshotには`manifest.json/csv`、`records.jsonl`（960行）、`summary.json`、`runs/source_*/...`を保存する。元の送受信ログは各batchに残す。コピーしたrunsとrecordsを既存の共通評価コードへ渡す。

統合後の05由来recordは `reused=true`、`api_request_made=false`、`api_attempts=0`。
`source_experiment=05_new_models_single` と `source_record` で出典を追える。
全recordの`source_record`・`source_upstream_output`から元ファイルを追える。統合manifestの`merge_source_sha256`で元manifest/summary/recordsのhashを記録する。

`raw_response / usage` は05の元応答を保持する。
`query_elapsed_seconds / request_elapsed_seconds`も05の実測時間を保持する。統合はSDK再生ではなくファイルコピーなので、架空の`replay_elapsed_seconds`は生成しない。
したがってrecordsのtoken/time全件合計は「05を含む最終dataset」の量であり、「06の追加課金・実行wall time」ではない。
追加通信は各batchのrequests/responsesの`api_request_made=true`とsummaryの試行数で確認する。
統合summaryは実行ログではない。`merge_api_calls=0`、`source_api_attempts`は06各batchの試行数合計、`source_06_experiment_seconds`は各batchの実験処理時間合計（05や各回の間の待ち時間を含まない）。

### コードの呼び出し順

`run.py` → [`split_execution.run()`](../split_execution.py) → 共通ロック → `common.make_plan()` / `common.check_inputs()` → `scan_batches()` → `select_plan()` → 新batchの予約・保存 → 既存の`common.execute_plan()`。
prompt生成・API送信・公式scorerは既存common/upstreamを再利用し、再実装していない。

統合は`merge_results()`が全条件・各recordと公式scoreを検証し、元runsをコピーしてmanifest順に960件を並べる。共通解析runnerは分割結果があるときだけこの関数を呼び、その後従来の`load_and_parse()` / 評価関数を使う。旧方式のroot直下の結果がある場合、分割実行との混在は禁止し、旧解析経路は維持する。

## 5. N=10だけを共通解析する

06完了後：

```bash
python experiments/06_new_models_n10/analysis/analyze.py
```

APIなし。05を含む06の960 recordsを一度だけ読み、05のraw responseとの一致と再送の不在も照合する。
**05のrecordsをさらに追加結合しない。**

```text
analysis/
  analyze.py
  sol/
    cross_task_10/     # Accuracy、confusion、state連続指標
    pendulum_10/       # torque、bin MAE、delta theta
    diagnostics_10/    # parse、token、元の推論時間
    tables/           # 12種類のCSV
    query_subsets.json
    validation.json
    analysis_metadata.json
  terra/              # 同じ構造
  luna/               # 同じ構造
```

モデルごとに22図×PNG/SVG、3モデルで66図（132画像ファイル）。
N20/N30 directoryは生成しない。各条件の10件すべてを使い、parse失敗の補充抽出はしない。
model/model_aliasを表にも残し、モデルをpoolしない。

既定seed=42、bootstrap=1,000。解析再実行は `--overwrite-derived` を明示する。
新しい保存先にしたい場合は `--output-dir experiments/06_new_models_n10/analysis/<名前>`。
これらは解析派生ファイルだけの指定で、実験resultsは変更しない。

計算式、欠損、分母、正規化、図・CSV一覧は [共通解析README](../common_analysis/README.md) と
[03_1の定義](../03_1_gpt35_history_n30_joint/analysis/README.md) を参照。

## 6. 呼出順と検証

```text
run.py → common.run() → make_plan() → check_inputs()
  → dry-run、または execute_plan()
    → 必須cached条件：保存responseをreplay（新規API 0）
    → 残りの条件：公式backendで送信・ログ記録

analysis/analyze.py → common_analysis.runner.main()
  → load_and_parse() → モデル別に分割 → select_subsets(sizes=(10,))
  → aggregate_all() → make_figures(sizes=(10,)) → CSV/PNG/SVG/監査JSON
```

[tests/test_joint_modern_experiments.py](../../tests/test_joint_modern_experiments.py) で、
40+280=320/model、再利用120件の全一致、不足・重複・旧question時の送信前停止、cached条件のclient未生成、
元推論時間の保持、03_1の全既存数値との一致、N10だけの図出力をオフライン検証する。
