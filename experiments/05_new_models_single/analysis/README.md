# Exp.05の解析

05完了後のtoken・費用・時間と06追加分の概算は[RESULT.md](RESULT.md)にまとめた。冒頭に06の全モデル合計を記載している。

05が完了してから、repository rootで実行する。

```bash
source .venv/bin/activate
python experiments/05_new_models_single/analysis/analyze.py
```

03_1と同じ[共通評価コード](../../common_analysis/README.md)を呼び、N=10のpilot CSVを`sol/tables/`、`terra/tables/`、`luna/tables/`へ保存する。
Pendulum・H20しか取得しないため比較図は06で生成する。APIを呼ばず、元resultsやrawを変更しない。
使用手順・入出力は[実験README](../README.md)を参照。

## 05の実測から06のtoken・金額・時間を概算する

**05の完了後**に実行する。`analyze.py`の事前実行は不要。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/05_new_models_single/analysis/estimate_exp06.py
```

このスクリプトは標準ライブラリのみ使用し、APIも実験runnerも呼ばない。05の`results/`を読み取り専用で扱う。
実行中・未完了なら出力せず停止する。120件すべてが揃った、retryなし・再利用なしの05だけを対象とし、usage欠損を0埋めしない。

### 入力と出力

入力は`../results/manifest.json`、`records.jsonl`、`summary.json`と、[estimate_prices.json](estimate_prices.json)。`--results PATH`、`--prices PATH`で別の保存元・価格設定を指定できる。

出力はこのanalysisディレクトリの下の`cost_estimates/<UTC日時>_<識別子>/`。毎回新規ディレクトリなので、既存レポートも実験結果も上書きしない。

- `RESULTS.md`：日本語の表。モデル別と全モデル合計の05実測、06追加分の見積もり、05＋06累計、1 query平均、料金内訳、時間の定義、単価と出典。
- `estimate.json`：丸め前の値、使用単価、入力ファイルと単価ファイルのSHA256、cache/reasoning token内訳（欠損はnull）。

### 計算方法

| 対象 | 1モデルの件数 | 05に対する倍率 |
| --- | ---: | ---: |
| 05 実測 | 40 | 1 |
| 06で追加送信する分 | 280 | 7 |
| 05＋06の累計 | 320 | 8 |

input/output tokenと時間をモデル別に合計して倍率を掛ける。**06で再利用する40件分を追加料金に二重計上しない。** 3モデルの追加送信合計は840件。

input料金 = input tokens / 1,000,000 × input単価。output料金も同様。05についても料金はusageからの試算であり、請求明細の実額ではない。

OpenAI Docsで確認した通常text単価（USD / 1M tokens、2026-09-11時点）：[sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) input $4 / output $20、[terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra) $2 / $12、[luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) $0.20 / $1.20。価格は自動取得せずJSONに固定し、確認日・出典とともにレポートへコピーする。将来の価格変更時は設定を更新して再集計する。

通常入力単価で単純計算するため、cache割引・cache write料金・サービスtier差・税等は含めない。長いcontext用の価格には対応せず、1 queryのinputが272,000 tokensを超えた場合は停止する。reasoning tokensは`completion_tokens`の内訳なので、outputにもう一度加算しない。

### どの値をどこから読むか

| 値 | ファイルとキー | 意味 |
| --- | --- | --- |
| input / output | records.jsonl → raw_response.usage.prompt_tokens / completion_tokens | APIから返った実測usage。tiktoken推定ではない |
| API時間 | records.jsonl → request_elapsed_seconds | SDK通信を含む時間。サーバー内部の推論時間だけではない |
| query時間 | records.jsonl → query_elapsed_seconds | backendの1 query処理範囲 |
| モデル別の条件時間 | summary.json → conditions[].condition_elapsed_seconds | 当該モデルの4条件の合計。準備・評価・保存も含む |
| 実験処理全体 | summary.json → experiment_elapsed_seconds | 全モデルの実行範囲。起動・事前計画を含まない |

これらの時間は重なった計測範囲なので足さない。全体時間も7倍するが、固定処理まで単純比例させる概算である。
05はPendulum/H20のみ、06はMountainCarや他のHも含むため、token長・出力長・混雑・rate limit・再利用処理の時間差は補正していない。信頼区間・上限額・確定請求額ではない。

### コードの関数

[estimate_exp06.py](estimate_exp06.py)の`main()` → `load_completed()` → `build_estimate()`（内部で`validate_run()`）→ `markdown()`の順で処理する。
`load_completed()`は完了確認と読み取り、`validate_run()`は件数・条件・送信設定・usageを検証、`build_estimate()`は集計と1/7/8倍、`markdown()`は表の整形を担当する。最後に`main()`が新規出力ディレクトリへ保存する。
