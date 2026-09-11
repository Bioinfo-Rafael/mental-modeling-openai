# 段階的な Mental Modeling 実験

## 実験の意図・作業メモ

以下は各実験で何を確認したいかのメモです。「4通り」は `next-action`、`last-action`、`next-state`、`last-state` を指します。送信件数はretryなしの場合です。

1. **[01：送信前のprompt確認](01_prompt_preview/README.md)**
   request送信直前のsystem/user promptを `.txt` などへ出力し、APIには送らない。MountainCar・Pendulumの2タスクについて、行動予測と状態予測の4通り全てを作る。合計 **8件**。現在のpilot履歴長はH=5。

2. **[02：GPT-3.5に各1回送るpilot](02_gpt35_single/README.md)**
   01で確認した8 queryをchatGPT3.5（aliasは `3.5`）へそれぞれ1回送信し、返ってきた値を全てログに保存して確認する。「1回」は実験全体で1回ではなく、**8条件それぞれ1回、計8件**。実装ではrequest・全response・usage・時間・scoreをJSONLなどに保存する。

3. **[03：GPT-3.5で履歴長を比較](03_gpt35_history_n30/README.md)**
   chatGPT3.5のみで、H=5/10/20/30 × MountainCar/Pendulum × 4通り、計32条件を評価する。1条件につき30 queryを使い、条件ごとの平均的な性能などを集計する。概算では約900回、正確には **32×30＝960件**。30回は同じpromptの反復送信ではなく、各条件の先頭30有効queryを使う。Accuracyの分母は公式実装に従う。

4. **[04：N30→N20→N10で統計を比較](04_sample_size_analysis/README.md)**
   03の同じ30件について、全30件・先頭20件・先頭10件へと使用件数を減らし、平均と分散を比較する。追加API送信はしない。論文Fig.3を参考に、横軸をH、縦軸をAccuracy・処理時間・token消費として比較する。**希望する配置は、縦方向にN=30/20/10を並べる形**。
   現行実装はN=30/20/10を同じパネル内の系列として描き、task×metricでパネル分割しているため、この縦方向の配置は未反映。ここでは希望をメモとして残し、描画コードは変更していない。

5. **[05：Sol・Terra・Lunaのpilot](05_new_models_single/README.md)**
   Sol・Terra・Lunaについて、H=20、Pendulum、行動予測と状態予測の4通りを**各10 query**送信する。questionは03_1と同じJoint版。返ってきた値は全てログへ保存する。**40件/model、3モデルで120件**。この全120件を06で再利用する。

6. **[06：新3モデルで全条件N10](06_new_models_n10/README.md)**
   Sol・Terra・Lunaについて、03_1と同じJoint question・全32条件を各N=10で揃える。05のPendulum/H20の4条件は10件ずつ丸ごと再利用し、残り28条件だけ新規送信する。**280件/model、3モデルで840件の追加送信**。05+06の新規送信合計・最終ユニーク件数は **320件/model、計960件**。評価は[common_analysis](common_analysis/README.md)を共有し、06はN=10だけをモデル別に生成する。

   `--task Pendulum-v1 --history 5`で120件だけ先に実行できる。実行済み条件を含む指定はエラー。残りの完了済み除外は`--remaining`で明示する。各回を`results/batches/<識別子>/`へ保存し、`--merge`または解析コマンドで全960件を統合する。途中失敗からの再開は非対応。[分割の手順](06_new_models_n10/README.md#まずpendulumh5だけ実行する)を参照。

## 実装・実行状況について

初回実装時は静的レビューのみを行い、実験・API呼び出しは実行していません。その後、ユーザーの指示でコードとExp.1の既存preview結果をcommit・pushしています。今回のメモ追記では実験・API送信を実行していません。

01でprompt確認 → 02でGPT-3.5 pilot → 03でH sweep → 04でNを再解析。
新モデルは05のpilotを確認してから06へ進みます。

| Experiment / 目的 | 入力 | logical query数 | 新規API件数（retryなし） | 出力 |
| --- | --- | ---: | ---: | --- |
| 01_prompt_preview：prompt全文確認 | 公式raw episodes | 8 | 0 | `01_prompt_preview/results/` |
| 02_gpt35_single：GPT-3.5 pilot | raw＋01のmanifest | 8 | 8 | `02_gpt35_single/results/` |
| 03_gpt35_history_n30：GPT-3.5 H sweep | 公式raw episodes | 32条件×30＝960 | 960 | `03_gpt35_history_n30/results/` |
| 04_sample_size_analysis：N30/20/10の先頭subset比較 | 03のrecords＋manifest/summary | 新規queryなし | 0 | `04_sample_size_analysis/results/` |
| 05_new_models_single：Joint N10 pilot | Pendulum、H20、4 metrics | 40/model＝120 | 40/model＝120 | `05_new_models_single/results/` |
| 06_new_models_n10：Joint全条件N10 | raw＋05の一致する120 records | 320/model＝960 | 280/model＝840 | `06_new_models_n10/results/` |

## コマンド（ユーザーが後日実行するとき）

全実験で、[01の環境説明](01_prompt_preview/README.md)と同じ作成済みのLLM-Xavier用 `.venv` を使います。Notebookの `Python (mental-modeling)` もこの環境です。新規作成は不要です。

実行は「repositoryへ移動 → 環境を有効化 → 対象のrun.py」の順です。例えば01の場合：

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/01_prompt_preview/run.py
```

02〜06では最後の行を下表の対象コマンドに置き換えます。各実験のREADMEにも、`cd` と `source` を含めた実行例を記載しています。基本依存は既存projectのもの、04の図には既存の `notebook` extraにあるmatplotlibが必要です。
API keyは `OPENAI_API_KEY` 環境変数のみから読みます。キーをコマンド引数・ファイル・ログに入れないでください。

| Experiment | dry-run / APIなし | execute |
| --- | --- | --- |
| 01 | `python experiments/01_prompt_preview/run.py`（preview保存） | 同左。paidフラグは拒否 |
| 02 | `python experiments/02_gpt35_single/run.py` | `python experiments/02_gpt35_single/run.py --execute --confirm-paid-api` |
| 03 | `python experiments/03_gpt35_history_n30/run.py` | `python experiments/03_gpt35_history_n30/run.py --execute --confirm-paid-api` |
| 04 | `python experiments/04_sample_size_analysis/run.py --dry-run`（03が必要） | `python experiments/04_sample_size_analysis/run.py`（APIなし） |
| 05 | `python experiments/05_new_models_single/run.py` | `python experiments/05_new_models_single/run.py --execute --confirm-paid-api` |
| 06 | `python experiments/06_new_models_n10/run.py` | `python experiments/06_new_models_n10/run.py --execute --confirm-paid-api` |

## 条件・安全性

- 条件は各 `run.py`、共通のpilot履歴長は `common.PILOT_H=5`。`H` は実際の履歴step数で、変換は `common.history_size()` の1か所だけ。履歴範囲は半開区間 `[history_start, history_end)` です。
- episodeをpath順に並べ、upstreamで有効なqueryの先頭N件を採用します。足りなければ次のepisodeへ進み、境界は跨ぎません。01/02は同一8件。06は05とのidentity・model・prompt SHA・API引数・source hash一致を全件確認し、不足/不一致なら送信前に停止します。
- 有料実験はフラグなしではmanifestのみ。dry-runは `results/dry_run/` に保存し、実行済みの `results/manifest.json` を上書きしません。表示件数はplanから計算します。
- `--execute --confirm-paid-api` の両方が必要。既存結果の上書き・自動再送はしません。Exp.3は明示的な `--resume` で保存済み応答を再利用できます。まず `python experiments/03_gpt35_history_n30/run.py --resume` で計画を確認してください。タイムアウト等の結果不明queryの再送には、重複課金の可能性を確認したうえで `--retry-uncertain` も必要です。[再開手順と保存先](03_gpt35_history_n30/README.md#途中から再開する方法)を参照してください。
- SDK内部retryは0。upstream retryも既定0ですが、ユーザーが `--retries N` で明示できます。retry込み上限もplanに表示します。`api_attempts` はSDK create試行数であり、通信失敗がサーバーに到達/課金されたかまでは保証できません。
- APIはChat Completionsを維持します。05・06のsol/terra/lunaは`temperature`を送らず、`reasoning_effort="medium"`のみ明示し、他の生成設定はAPI既定値です。agent historyからのreasoningを評価する意図で、03_1の説明を求めるJoint questionを維持します（[設定詳細](05_new_models_single/README.md)）。GPT-3.5の既存実験は`temperature=0`のままです。モデルIDは既存token estimatorと同じで、エラー時にモデル/API/引数を自動変更しません。通信・再試行の安全設定は維持し、実APIでの再検証はしていません。
- `data/`・`upstream/`・既存Notebookは変更しません。結果は各experimentの `results/` のみ。ユーザー指定でExp.1の開始記録・manifest・prompt txtはGitで共有し、他の実験結果は `.gitkeep` のみを含めます。

## 保存内容・再利用

`manifest.json/csv` はquery identity・source SHA・prompt SHA・H・条件別予定件数、JSONにはprompt全文も入ります。
01は8つの `.txt`、有料実験は `requests.jsonl`（各試行の実kwargs）、`responses.jsonl`（全SDK response/usage/時刻/例外）、`records.jsonl`（query＋score＋raw response＋時間）を保存します。`query_id` と `attempt_id` で対応します。秘密情報はredactし、HTTP headerやclient/environmentそのものは保存しません。

`runs/<condition>/episode_<番号>/` にupstreamの `run.json`、`config.effective.json`、`metrics.json`、`predictions.jsonl` をそのまま保存します。upstreamの `raw_responses_stored=false` はその標準出力についての値で、追加JSONLにはraw responseを保存します。
`summary.json/csv` にAccuracy・parse率・全件/parsed/elementの精度と経過時間、JSONに実験全体の試行数・成功/失敗数を記録します。成功は採点済み（`ignored`も含む）であり、正解数とは別です。

06以外はAPIエラー時に以降の送信を止めます。先に取得できたprefixのresponseはAPIなしで同じCLIへreplayし、`episode_<番号>_partial/` に標準スコアを回収します。強制終了・安全性違反・ディスク障害では回収を保証できませんが、送受信は都度flush/fsyncします。未完了結果を04/06が完了結果として使うことはありません。

06の新規batchは既定で**terra → luna → sol**の順（`--model-order`で変更可能）。APIエラーをquery・request・例外/request IDとともに保存して次へ進みます。全試行後、一部失敗なら`complete_with_errors`とし、失敗行も統合に残します。ログ保存失敗や安全性違反では停止します。過去の途中停止batchは自動再開・削除しません。[06の詳細](06_new_models_n10/README.md#モデル順とapiエラー後の継続)を参照。
06の統合snapshotでは05のreuseを`reused=true`、`api_request_made=false`、`source_experiment`、`source_record`で示し、新規試行数は0。元response/usageと元query時間を残します。分割方式の統合はファイルコピーであり、SDK replayは行いません。

04は `statistics.csv/json` と `figures/`（Accuracy、time、input/output/total tokensを別図）を生成します。primary Accuracyはupstreamの `legacy_compatible_match_rate`。actionは全件、stateはparsed件数が分母です。全件のcorrect=1/0統計（ignored=0）とは区別します。分散/標準偏差はddof=0/1を両方保存し、time/token図は母標準偏差、Accuracy図は分母混同を避けerror barなし。欠損usageは0で埋めません。

実装の呼び出し経路と13項目の静的レビューは [REVIEW.md](REVIEW.md) を参照。
SDK retryの根拠は [公式OpenAI Python SDK仕様](https://developers.openai.com/api/reference/python#retries)、図の構成は [論文Fig.3](https://arxiv.org/html/2406.18505v1#S3.F3) を参考にしています。これは指定のtask・model・sampling条件での段階的再現であり、論文数値の完全一致を保証するものではありません。
