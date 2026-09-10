# 段階的な Mental Modeling 実験

**今回の納品は実装・静的レビューのみ。実験01〜06（dry-run含む）・API呼び出し・commit・pushは未実行です。**

01でprompt確認 → 02でGPT-3.5 pilot → 03でH sweep → 04でNを再解析。
新モデルは05のpilotを確認してから06へ進みます。

| Experiment / 目的 | 入力 | logical query数 | 新規API件数（retryなし） | 出力 |
| --- | --- | ---: | ---: | --- |
| 01_prompt_preview：prompt全文確認 | 公式raw episodes | 8 | 0 | `01_prompt_preview/results/` |
| 02_gpt35_single：GPT-3.5 pilot | raw＋01のmanifest | 8 | 8 | `02_gpt35_single/results/` |
| 03_gpt35_history_n30：GPT-3.5 H sweep | 公式raw episodes | 32条件×30＝960 | 960 | `03_gpt35_history_n30/results/` |
| 04_sample_size_analysis：N30/20/10の先頭subset比較 | 03のrecords＋manifest/summary | 新規queryなし | 0 | `04_sample_size_analysis/results/` |
| 05_new_models_single：Sol/Terra/Luna pilot | Pendulum、H20、4 metrics | 12 | 12 | `05_new_models_single/results/` |
| 06_new_models_n10：新モデルの全条件N10 | raw＋05の一致する12 records | 320/model＝960 | 316/model＝948 | `06_new_models_n10/results/` |

## コマンド（ユーザーが後日実行するとき）

repository直下で `.venv` を有効化して使用します。基本依存は既存projectのもの、04の図には既存の `notebook` extraにあるmatplotlibが必要です。
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
- `--execute --confirm-paid-api` の両方が必要。既存結果があれば停止し、自動resume/上書きはしません。再実験する場合は結果を手動で退避し、再課金の可能性を確認してください。
- SDK内部retryは0。upstream retryも既定0ですが、ユーザーが `--retries N` で明示できます。retry込み上限もplanに表示します。`api_attempts` はSDK create試行数であり、通信失敗がサーバーに到達/課金されたかまでは保証できません。
- APIはupstreamのChat Completions・`temperature=0`を維持します。モデルIDは既存token estimatorと同じです。アカウントの利用可否・モデルの引数互換性は未検証で、エラー時にモデル/API/引数を自動変更しません。
- `data/`・`upstream/`・既存Notebookは変更しません。結果は各experimentの `results/` のみ。ユーザー指定でExp.1の開始記録・manifest・prompt txtはGitで共有し、他の実験結果は `.gitkeep` のみを含めます。

## 保存内容・再利用

`manifest.json/csv` はquery identity・source SHA・prompt SHA・H・条件別予定件数、JSONにはprompt全文も入ります。
01は8つの `.txt`、有料実験は `requests.jsonl`（各試行の実kwargs）、`responses.jsonl`（全SDK response/usage/時刻/例外）、`records.jsonl`（query＋score＋raw response＋時間）を保存します。`query_id` と `attempt_id` で対応します。秘密情報はredactし、HTTP headerやclient/environmentそのものは保存しません。

`runs/<condition>/episode_<番号>/` にupstreamの `run.json`、`config.effective.json`、`metrics.json`、`predictions.jsonl` をそのまま保存します。upstreamの `raw_responses_stored=false` はその標準出力についての値で、追加JSONLにはraw responseを保存します。
`summary.json/csv` にAccuracy・parse率・全件/parsed/elementの精度と経過時間、JSONに実験全体の試行数・成功/失敗数を記録します。成功は採点済み（`ignored`も含む）であり、正解数とは別です。

APIエラー時は以降の送信を止めます。先に取得できたprefixのresponseはAPIなしで同じCLIへreplayし、`episode_<番号>_partial/` に標準スコアを回収します。強制終了・安全性違反・ディスク障害では回収を保証できませんが、送受信は都度flush/fsyncします。未完了結果を04/06が完了結果として使うことはありません。
06のreuseは `reused=true`、`api_request_made=false`、`source_experiment`、`source_record` で示し、新規試行数は0。元response/usageと元query時間を残し、今回のreplay時間と区別します。

04は `statistics.csv/json` と `figures/`（Accuracy、time、input/output/total tokensを別図）を生成します。primary Accuracyはupstreamの `legacy_compatible_match_rate`。actionは全件、stateはparsed件数が分母です。全件のcorrect=1/0統計（ignored=0）とは区別します。分散/標準偏差はddof=0/1を両方保存し、time/token図は母標準偏差、Accuracy図は分母混同を避けerror barなし。欠損usageは0で埋めません。

実装の呼び出し経路と13項目の静的レビューは [REVIEW.md](REVIEW.md) を参照。
SDK retryの根拠は [公式OpenAI Python SDK仕様](https://developers.openai.com/api/reference/python#retries)、図の構成は [論文Fig.3](https://arxiv.org/html/2406.18505v1#S3.F3) を参考にしています。これは指定のtask・model・sampling条件での段階的再現であり、論文数値の完全一致を保証するものではありません。
