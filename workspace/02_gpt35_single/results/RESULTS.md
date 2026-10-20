# Exp.2 結果まとめ

既存ログを2026-09-10にread-onlyで集計したレポートです。実験の再実行・追加API送信はしていません。

## 結論

**8 queryすべての送信・応答取得・解析・採点が完了しました。実験処理時間は25.337秒、入力8,011 tokens、出力1,288 tokens、合計9,299 tokensです。完全一致は4/8件（50%）でした。**

各条件1件だけのpilotです。実行が成功したことと、予測が正解したことは別であり、この8件から一般的なモデル精度は判断できません。

## 1. 時間・tokenはどこにあるか

| 知りたい値 | 元ファイル | キー・参照先 |
| --- | --- | --- |
| 実験処理全体の時間 | [summary.json](summary.json) | `experiment_elapsed_seconds` |
| 各条件の時間 | [summary.json](summary.json) / [summary.csv](summary.csv) | `conditions[i].condition_elapsed_seconds` / 同名列 |
| 1 queryの処理時間 | [records.jsonl](records.jsonl) | 各行の `query_elapsed_seconds` |
| 1 queryのAPI試行時間の合計 | [records.jsonl](records.jsonl) | 各行の `request_elapsed_seconds` |
| 個別API試行の時間・時刻 | [responses.jsonl](responses.jsonl) | `request_elapsed_seconds`、`request_started_at_utc`、`response_received_at_utc` |
| 1 queryの入力token数 | [records.jsonl](records.jsonl) | `input_tokens`。元は `usage.prompt_tokens` |
| 1 queryの出力token数 | [records.jsonl](records.jsonl) | `output_tokens`。元は `usage.completion_tokens` |
| 1 queryの合計token数 | [records.jsonl](records.jsonl) | `total_tokens`。元は `usage.total_tokens` |
| usageの追加内訳 | [responses.jsonl](responses.jsonl) / [records.jsonl](records.jsonl) | `raw_response.usage`。同じusageを上位の `usage` にも保持 |
| 公式精度・解析成功数 | [summary.json](summary.json) | `conditions[i].metrics` |

`runs/` 内の公式4ファイルだけでは、tokenとAPI時間は確認できません。上位の追加ログ `records.jsonl` / `responses.jsonl` にあります。ファイル間は `query_id`、個別試行は `attempt_id` で対応します。

## 2. 実行条件・完了状態

| 項目 | 結果 |
| --- | --- |
| 実験 | `02_gpt35_single` |
| 指定model / alias | `gpt-3.5-turbo` / `3.5` |
| responseに記録されたmodel | 全8件 `gpt-3.5-turbo-0125` |
| タスク | MountainCar-v0 / Pendulum-v1 |
| metric | next-action / last-action / next-state / last-state |
| 実際の履歴長H | 全件5（公式 `history_size=4`） |
| N | 各条件1件、8条件 |
| 処理開始（JST） | 2026-09-10 14:38:49.240 |
| 処理終了（JST） | 2026-09-10 14:39:14.577 |
| 完了状態 | `complete` |
| 計画 / 処理query数 | 8 / 8 |
| API試行 / retry | 8 / 0 |
| 採点済み / 実行失敗 / 未着手 / 未採点 | 8 / 0 / 0 / 0 |
| 解析できた応答 / ignored | 8 / 0（parse率100%） |
| 応答のfinish_reason | 全8件 `stop` |

時間は `common.py:execute_plan()` の計測範囲であり、スクリプト起動からの全時間ではありません。事前のquery計画・01との照合・manifest保存などは、この25.337秒に含まれません。

## 3. 8 queryの時間・token・精度

元データ：[records.jsonl](records.jsonl)、[summary.json](summary.json)。秒数は小数第3位、要素精度は百分率で丸めています。

| task | metric | 条件時間（秒） | query時間（秒） | API時間（秒） | input tokens | output tokens | total tokens | 完全一致 | 要素精度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| MountainCar-v0 | next-action | 4.814 | 3.755 | 3.284 | 748 | 118 | 866 | 正解 | 対象外 |
| MountainCar-v0 | last-action | 2.670 | 2.653 | 2.649 | 736 | 129 | 865 | 正解 | 対象外 |
| MountainCar-v0 | next-state | 3.573 | 3.550 | 3.546 | 906 | 129 | 1,035 | 正解 | 100% |
| MountainCar-v0 | last-state | 3.655 | 3.627 | 3.622 | 912 | 195 | 1,107 | 不一致 | 50% |
| Pendulum-v1 | next-action | 2.691 | 2.665 | 2.660 | 1,237 | 129 | 1,366 | 正解 | 100% |
| Pendulum-v1 | last-action | 2.651 | 2.624 | 2.619 | 1,265 | 169 | 1,434 | 不一致 | 0% |
| Pendulum-v1 | next-state | 3.079 | 3.055 | 3.051 | 1,103 | 274 | 1,377 | 不一致 | 33.333% |
| Pendulum-v1 | last-state | 2.185 | 2.164 | 2.160 | 1,104 | 145 | 1,249 | 不一致 | 66.667% |

条件時間はepisode読込・backend準備・評価・保存などを含みます。query時間は `complete()` 内の処理、API時間はSDK createの所要時間です。API時間はサーバー内部の推論時間だけを意味せず、通信等も含みます。今回retryはありません。

合計は、API時間23.592秒、query時間24.093秒、条件時間25.317秒、実験処理全体25.337秒です。計測範囲が異なるため一致しません。重なった時間なので、これら4つを足して総所要時間にしてはいけません。

## 4. タスク別集計

各タスクの4 queryをまとめた記述統計です。4種類の異なるmetricを混ぜた集計であり、各metricの性能推定ではありません。

| task | query数 | 完全一致 | parse率 | input合計 | output合計 | token合計 | query時間合計（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MountainCar-v0 | 4 | 3/4（75%） | 100% | 3,302 | 571 | 3,873 | 13.584 |
| Pendulum-v1 | 4 | 1/4（25%） | 100% | 4,709 | 717 | 5,426 | 10.508 |
| 全体 | 8 | 4/8（50%） | 100% | 8,011 | 1,288 | 9,299 | 24.093 |

公式primary Accuracyは `legacy_compatible_match_rate` で、actionは全query数、stateはparsed件数が分母です。今回はignoredが0なので、全件ベースの完全一致率と一致します。上の全体50%は、この8件をまとめた追加集計です。

## 5. 全8 queryの分布

nullのない8観測を対象に計算しました。異なるタスク・metric間のばらつきであり、「同じ条件を8回反復したばらつき」ではありません。各条件N1のため、条件内の標本分散は算出できません。

| 項目 | 平均 | 中央値 | 最小 | 最大 | 母標準偏差（ddof=0） |
| --- | ---: | ---: | ---: | ---: | ---: |
| query時間（秒） | 3.012 | 2.860 | 2.164 | 3.755 | 0.541 |
| API時間（秒） | 2.949 | 2.856 | 2.160 | 3.622 | 0.479 |
| input tokens | 1,001.375 | 1,007.500 | 736 | 1,265 | 192.868 |
| output tokens | 161.000 | 137.000 | 118 | 274 | 48.946 |
| total tokens | 1,162.375 | 1,178.000 | 865 | 1,434 | 213.416 |

| 項目 | 母分散（ddof=0） | 標本分散（ddof=1） | 標本標準偏差（ddof=1） |
| --- | ---: | ---: | ---: |
| query時間 | 0.292535 | 0.334325 | 0.578209 |
| API時間 | 0.229774 | 0.262598 | 0.512444 |
| input tokens | 37,197.984375 | 42,511.982143 | 206.184340 |
| output tokens | 2,395.750000 | 2,738.000000 | 52.325902 |
| total tokens | 45,546.484375 | 52,053.125000 | 228.151540 |

母分散は偏差平方和÷8、標本分散は偏差平方和÷7、標準偏差はそれぞれの平方根です。時間の分散は秒²、tokenの分散はtokens²です。標準偏差を信頼区間としては扱いません。

正誤をmatch=1、mismatch=0として全8件をまとめると、平均0.5、母分散0.25、母標準偏差0.5、標本分散0.285714、標本標準偏差0.534522です。これも混合条件での記述統計です。

## 6. usageの補足・解釈上の注意

- token数はAPI responseに記録されたusageです。tiktokenや「本文＋9」の概算ではありません。
- 全件で `total_tokens = input_tokens + output_tokens`。欠損usageはありません。
- `cached_tokens`、`reasoning_tokens`、audio関連tokenなど、今回usageに記録されたそれらの内訳は全件0でした。`cache_write_tokens` は全件nullであり、0と置き換えていません。
- 今回はretry・再利用とも0なので、8件の最終responseの合計と、記録された全API試行のusage合計が一致します。一般にretry・再利用がある実験では区別が必要です。
- 金額は下記8節でExcel記載の単価から試算しています。APIのusageに請求金額が記録されているわけではありません。
- stateの要素精度は変化方向ラベルの精度です。Pendulum actionは連続制御値そのものではなく、公式のbin番号の評価です。
- 各条件はpath順の最初の有効queryを1つだけ使っています。全件の履歴範囲は `[0, 5)`、next-stateのquery indexは4、その他は5で、タスク内で履歴が重なります。独立な無作為sampleの評価とはみなしません。

## 7. Excelの予想input tokensとの比較

比較元：[token_estimation.xlsx](../../../notebooks/01data/token_estimation.xlsx) の `chatGPT3.5` シート、MountainCarは10行目、Pendulumは11行目です。2026-09-10時点の保存値と数式を読み取り、Excel自体は変更していません。

**履歴長の違いを補正した参考比較では、実測は予想とほぼ同じ規模でした。MountainCarは+0.84%、Pendulumは−0.70%です。** ただし、ExcelにH=5の見積もりはないため、同条件での厳密な誤差検証ではありません。

### 比較条件と計算方法

- Excelは `next-action` のみ、実際の履歴長H=1・10を計測しています。生成コードは [01_token_estimation.py](../../../notebooks/01_token_estimation.py) の `choose_windows()`、`estimate()`、`measure()` です。有効なwindowからseed=42で最大10,000件を非復元抽出した平均です。
- Exp.2はH=5、各タスク4種類のmetricを1件ずつ計測しています。ここではExcelとmetricを合わせ、`next-action` の2件だけを比較します。他の6件も含めた平均をExcelのnext-action平均と直接比較していません。
- Excelの固定部分＋質問はE列、H=1履歴はF列、H=10履歴はH列、H=10合計はI列（`=E+H`）です。固定部分にはAPI framingの概算9 tokensが含まれます。
- 下表のH=1合計は `E+F`、H=5参考値は `E+F+(H−F)×4/9` で計算しました。履歴tokenが履歴長に対して線形に増えると仮定した補間であり、H=5のpromptを再計測した値ではありません。固定部分はExcelのE列を据え置いています。
- Excelはtiktokenによる見積もり、実測はAPIの `usage.prompt_tokens` です。sample、質問中の数値、serializationのtoken境界、framing概算の違いによっても差が出ます。

| タスク（next-action） | Excel由来 H=1合計 | Excel H=10合計（I列） | H=5参考値（補間） | Exp.2 H=5実測 | 実測−H=5参考値 | 差率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MountainCar-v0 | 613.230 | 902.470 | 741.781 | 748 | +6.219 | +0.84% |
| Pendulum-v1 | 1,067.894 | 1,467.896 | 1,245.672 | 1,237 | −8.672 | −0.70% |

単位はtokens、差率は `(実測 / H=5参考値 − 1) × 100` です。元の入力値は `chatGPT3.5!E10:H11`、H=10合計は `I10:I11`。プロンプト全体を数えた `M10:M11` もI列と丸め誤差の範囲で一致し、`N10:N11` の分割集計との差は0です。

実測H=5がExcel H=10より小さいのは、履歴が半分だからです。「予想から大幅に下振れした」とは解釈しません。各タスク1件なので、この近さが全episode・全metricでも成り立つとはまだ確認できていません。

なお、Excelの「H=10 合計金額」は **input部分だけの金額** です。例えば `C10=I10/1000000*Model_Prices!$C$6` であり、output料金は含まれません。「100回分金額」もこの入力料金の100倍です。

## 8. Exp.2相当のinput/outputを240回送った場合

### 回数と単価の前提

ご指定の240回を、**2タスク × 4 metric × 各30回** として試算します。Exp.2の8件を30倍する、つまり同じ条件配分で1回平均input 1,001.375 / output 161 tokensを使う想定です。実際に240回送信した結果ではありません。

現在の [Exp.3 run.py](../../03_gpt35_history_n30/run.py) は、[common.py](../../common.py) の `H_VALUES=(5, 10, 20, 30)` を使用するため、**Hを1つに絞れば240回、デフォルトの全Hでは960回** です。この試算はH=5のpilotを基準とし、H=10・20・30の240回分やExp.3全体の見積もりにはそのまま適用できません。

料金はExcelの `Model_Prices` シートの記録を使用します（`G6` の確認日：2026-09-09）。現在の公式価格を再検証したものではありません。

| 区分 | 単価（USD / 1M tokens） | Excelの参照セル |
| --- | ---: | --- |
| GPT-3.5 input（通常入力） | 0.50 | `Model_Prices!C6` |
| GPT-3.5 output | 1.50 | `Model_Prices!F6` |

### 240回分のtoken数・料金

| 区分 | Exp.2実測8回 | 1回平均 | 240回想定（8回実測×30） | 240回の料金（USD） |
| --- | ---: | ---: | ---: | ---: |
| input | 8,011 | 1,001.375 | 240,330 | 0.120165 |
| output | 1,288 | 161.000 | 38,640 | 0.057960 |
| 合計 | 9,299 | 1,162.375 | 278,970 | **0.178125** |

```text
input tokens  = 8,011 × 30 = 240,330
output tokens = 1,288 × 30 = 38,640
total tokens  = 240,330 + 38,640 = 278,970

input料金  = 240,330 / 1,000,000 × $0.50 = $0.120165
output料金 =  38,640 / 1,000,000 × $1.50 = $0.057960
合計料金   = $0.120165 + $0.057960 = $0.178125（約18米セント）
```

タスク別に分けると、MountainCarの120回分はinput 99,060 / output 17,130 tokens、料金は$0.049530 + $0.025695 = $0.075225。Pendulumの120回分はinput 141,270 / output 21,510 tokens、料金は$0.070635 + $0.032265 = $0.102900です。両者の合計が上表と一致します。

参考として、既に実行したExp.2の8回自体は、同じ単価でinput $0.0040055、output $0.001932、合計$0.0059375相当です。

これは通常入力・追加retryなしのtoken従量料金の概算です。出力長は毎回変わり、Exp.3では使用するwindowも変わります。各条件1件のpilotからの外挿なので、上限額や確定請求額ではありません。税、為替換算、クレジット、契約上の調整は含めていません。

## 9. 240回分の所要時間の試算

**Exp.2と同程度の応答時間で逐次実行するなら、240回は約12〜13分が目安です。** 8節と同じくH=5の8件を30倍した参考値であり、240回の実測ではありません。

元データは [records.jsonl](records.jsonl) の `request_elapsed_seconds` / `query_elapsed_seconds` と、[summary.json](summary.json) の `experiment_elapsed_seconds` です。丸め前の値で計算しています。

| 計測範囲 | Exp.2実測8回（秒） | 1回あたり（秒） | 240回想定（秒） | 240回想定（分秒） |
| --- | ---: | ---: | ---: | ---: |
| API試行時間の合計（通信を含む） | 23.592 | 2.949 | 707.747 | 約11分48秒 |
| query処理時間の合計 | 24.093 | 3.012 | 722.779 | 約12分3秒 |
| 実験処理全体を単純比例 | 25.337 | 3.167 | 760.105 | 約12分40秒 |

```text
API試行時間 = 23.59156704286579 × (240 / 8)
            = 707.747秒 ≒ 11分48秒
実験処理全体 = 25.336825707927346 × (240 / 8)
             = 760.105秒 ≒ 12分40秒
```

これらは重なった計測範囲であり、足し合わせません。1回あたりの「実験処理全体」は全体時間÷8という配賦値で、個別APIリクエストの時間ではありません。API時間からinput処理とoutput生成の時間を別々に分離することも、このログだけではできません。

全体時間の30倍は、初期化・条件ごとの準備・保存なども30倍する簡易計算です。Exp.3は1条件で30件を扱うため、その固定処理が送信回数に比例するとは限りません。また、元の全体時間は `execute_plan()` の範囲であり、起動・事前のquery計画などは含まれません。実際のコマンド実行完了までの時間とは区別してください。

APIの混雑、通信、生成されるoutputの長さ、rate limit待ち、retryによって実際の所要時間は変わります。「12〜13分」は信頼区間でも上限でもありません。H=10・20・30で同じ速度になることは未確認です。デフォルトのExp.3全960回についても、Hごとの実測なしに所要時間を確定できません。

保存元の関数とデータ構造は [READMEの3.1〜3.4](../README.md) を参照してください。本ファイルは既存結果の派生レポートとして `results/` に置いています。
