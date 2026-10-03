# Exp.06の費用・時間見積もりとExp.05の実測結果

## 結論：06の追加実行は約263万tokens・$13.21・4時間21分

**06で新規送信する840件（280件/model）の見積もりは、input 1,632,855 tokens、output 999,243 tokens、合計2,632,098 tokens。input料金 $3.374567、output料金 $9.836761、合計 $13.211328。3モデルを逐次実行する時間は約261.42分（4時間21分25秒）です。**

05の実測40件/modelを単純に7倍した概算です。05から再利用する120件（40件/model）は追加送信・追加料金に含めていません。料金は通常入力単価での試算であり、cache割引・cache write料金・税等は反映していません。確定請求額・所要時間の上限ではありません。

### 06の追加分：モデル別

| モデル | 新規送信件数 | input tokens | output tokens | 合計tokens | input料金 USD | output料金 USD | 合計料金 USD | 時間の目安 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sol | 280 | 544,285 | 320,061 | 864,346 | 2.177140 | 6.401220 | 8.578360 | 103.19分 |
| terra | 280 | 544,285 | 242,641 | 786,926 | 1.088570 | 2.911692 | 4.000262 | 57.52分 |
| luna | 280 | 544,285 | 436,541 | 980,826 | 0.108857 | 0.523849 | 0.632706 | 100.71分 |
| 全モデル合計 | 840 | 1,632,855 | 999,243 | 2,632,098 | 3.374567 | 9.836761 | **13.211328** | **261.42分** |

モデル別時間は当該モデルの05の条件時間合計×7、冒頭の全体時間は05の`experiment_elapsed_seconds`×7です。全体には条件外の処理時間もわずかに含まれるため、丸め前の合計は一致しません。

## 05の実測：120件すべて完了

完了状態・件数・モデル/条件・送信設定・usageを検証したうえで、2026-09-11にオフライン集計しました。新規API送信120件、retry 0件、実行失敗0件、再利用0件です。ここでの完了は応答取得・処理の完了を意味し、予測が全件正解したという意味ではありません。

| モデル | 件数 | input tokens | output tokens | 合計tokens | 通常単価による料金概算 USD | 条件時間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sol | 40 | 77,755 | 45,723 | 123,478 | 1.225480 | 14.74分 |
| terra | 40 | 77,755 | 34,663 | 112,418 | 0.571466 | 8.22分 |
| luna | 40 | 77,755 | 62,363 | 140,118 | 0.090387 | 14.39分 |
| 全モデル合計 | 120 | 233,265 | 142,749 | 376,014 | 1.887333 | 37.35分 |

実験処理全体は2,240.775秒（約37分21秒）。token数と時間は実測、料金は単価から計算した概算です。

## 計算方法と05＋06の累計

```text
06追加分の件数/model = 320 − 40 = 280
倍率 = 280 / 40 = 7

06 input tokens  = 233,265 × 7 = 1,632,855
06 output tokens = 142,749 × 7 =   999,243
06実験処理時間    = 05の実験処理全体 × 7 ≒ 15,685.424秒

モデル別input料金  = input tokens  / 1,000,000 × そのモデルのinput単価
モデル別output料金 = output tokens / 1,000,000 × そのモデルのoutput単価
全モデル料金      = モデル別input料金とoutput料金の合計
```

05＋06累計は05の8倍です。960件（320件/model）でinput 1,866,120 tokens、output 1,141,992 tokens、合計3,008,112 tokens、料金は約$15.10、時間は約298.77分（4時間58分46秒）。**これは既に完了した05も含む値であり、これから06に必要な追加分ではありません。**

## 単価・時間・tokenの前提

単価は[estimate_prices.json](estimate_prices.json)の2026-09-11確認値です。USD / 1M tokensで、[sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)はinput $4 / output $20、[terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)は$2 / $12、[luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)は$0.20 / $1.20。請求明細から取得した金額ではありません。

- token数は`records.jsonl`の`raw_response.usage.prompt_tokens` / `completion_tokens`。tiktokenによる推定ではありません。reasoning tokenはoutputの内訳であり、二重に加算しません。
- 05は全件Pendulum・H20です。06では別task・別Hも含みますが、ご指定どおり条件差を補正せず7倍しています。出力長、混雑、rate limit等で実際の値は変わります。
- 時間は逐次実行前提です。全体時間の範囲は`common.py:execute_plan()`で、起動・事前計画は含みません。固定処理も単純比例させた値で、再利用応答を再生する処理時間は別途見積もっていません。
- API時間・query時間・条件時間・実験全体時間は重複する計測範囲なので、足し合わせません。詳細レポートには各範囲の時間を別々に掲載しています。

## 根拠ファイルと再現方法

- 入力：[summary.json](../results/summary.json)、[manifest.json](../results/manifest.json)、[records.jsonl](../results/records.jsonl)。元ファイルは変更していません。
- 自動生成した全表：[RESULTS.md](cost_estimates/20260911T040931Z_8b86ae80/RESULTS.md)。05実測、06追加分、05＋06累計、1 query平均、時間の内訳を収録。
- 丸め前の計算値・入力SHA256・単価：[estimate.json](cost_estimates/20260911T040931Z_8b86ae80/estimate.json)。本ファイルはこの保存済み見積もりの要約です。
- 計算コード：[estimate_exp06.py](estimate_exp06.py)の`build_estimate()`。読み取りと検証は`load_completed()` / `validate_run()`、自動レポートの整形は`markdown()`、保存は`main()`。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/05_new_models_single/analysis/estimate_exp06.py
```

再実行時は新しい`cost_estimates/<日時>_<識別子>/`へ保存されます。この`RESULT.md`は今回の集計の要約として固定され、自動更新・上書きされません。集計ではAPIを呼んでいません。
