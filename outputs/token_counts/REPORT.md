# 実施報告 — 2026-09-09

作業場所: /Users/cls-lab/Git/Matsuo/mental-modeling-openai

## 完了範囲

raw / dataset adapter / preprocessing / outputを分離。データ一覧25件（公式20＋candidate5）、公式rawは20 taskすべて読めます。original LLM-X prompt compatibleは11 task。外部4 adapterを実装し、BaiduはNOT_ACQUIREDのままです。

取得物34,227ファイルについて作業前後SHA256が一致しました。[検証結果](../dataset_inventory/raw_integrity_verification.json)。NPZ/CSV/取得物は未変更、derived dataをdata/へ保存していません。既存data/README・SCHEMAも不変です。今後生成する文書はoutputs/docs/です。

## 未対応公式9 task

| Task | data readable | original prompt | 理由 |
| --- | --- | --- | --- |
| BipedalWalker-v3 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| HalfCheetah-v4 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| MiniGrid-DoorKey-5x5-v0 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| MiniGrid-Empty-Random-5x5-v0 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| MiniGrid-Fetch-5x5-N2-v0 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| MiniGrid-GoToDoor-5x5-v0 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| MiniGrid-KeyCorridorS3R1-v0 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| Pusher-v4 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |
| Reacher-v4 | yes | no | 上流task class/descriptionとregistry entryが存在しない。安全なversion aliasなし。generic questionの有無だけではtask説明を補えないためskip。 |

全20に見せるためのtask説明・質問・benchmark追加は行いませんでした。

## 外部readerとcapability

| Dataset | reader | capability | 既定preprocessing |
| --- | --- | --- | --- |
| F16Capstone | 3 CSV全55列、numeric MAT、ULog全topic | state/controlあり、同一aligned streamで次state構成可能、rewardなし | CSVのみ7 state＋4 control＋timestamp |
| TrajAir | processed TXTの全7列、raw CSV/weather | trajectory/風あり、action/rewardなし | file×aircraft track、frame gap維持 |
| AirCombat-WEZ | 全12 CSV | static scenario/targetあり、action/reward/flight sequenceなし | Data/4 CSV、static |
| Calculated Moves | CSV/script/config/winner | rule weights/encounter fitness/outcomeあり。flight state/action/timestep rewardはない | agent/validation CSV、static |
| Baidu fighter jet | 未取得 | 未確認 | なし |

F16のopaque MATLAB objectはschemaを列挙しreader非対応として明示。モデル・画像・XLSX等のsupport物はinventory参照のみです。reader件数は異種streamやduplicate exportを含み、物理的な独立軌跡の数ではありません。

「このpreprocessingはraw datasetそのものではなく、LLM入力用に本projectで定義した変換である」。外部baselineの選択列はYAMLから変更可能です。存在しないaction/rewardの生成、欠損値の0埋め、normalizationはありません。Calculated Movesのfitnessはrewardへ改名しません。

## 実計測

GPT-4o / o200k_base。15 dataset、110,177 query/H/config行。公式59,272行、外部50,905行。公式H=1だけでは10,095 valid queriesを全件計測しました。Hが異なる同じqueryは別行であり、110,177件の独立した元recordを意味しません。

content文字列は実測tokenです。以下のinput tokensはcontent+9のAPI framing概算で、server usageの厳密値ではありません。OpenAI Docs確認に基づき両者を区別しました。

| Prompt mode | H | 計測queries | Input tokens |
| --- | --- | --- | --- |
| external_generic | 1 | 45,041 | 22,662,084 |
| external_static | 0 | 5,864 | 1,523,733 |
| original_llmx | 0 | 10,183 | 11,734,143 |
| original_llmx | 1 | 10,095 | 12,493,713 |
| original_llmx | 2 | 10,007 | 13,220,611 |
| original_llmx | 3 | 9,919 | 13,914,910 |
| original_llmx | 5 | 9,745 | 15,207,883 |
| original_llmx | 10 | 9,323 | 17,930,894 |

外部generic/staticの合計は全件scopeとsampleの計測分を足した値で、全candidate datasetの総token見積もりではありません。異なるHの総和を単一実験費用へ使わないでください。

| 外部dataset | H | 全reader records¹ | eligible query pool | 実計測 | scope | Input tokens |
| --- | --- | --- | --- | --- | --- | --- |
| f16capstone | 1 | 836572 | 43041 | 43041 | full_selected_scope | 22070288 |
| trajair | 1 | 12130491 | 2724711 | 2000 | deterministic_sample | 591796 |
| aircombat_wez | 0 | 4050 | 3864 | 3864 | full_selected_scope | 626500 |
| calculated_moves | 0 | 5298468 | 2804268 | 2000 | deterministic_sample | 897233 |

¹ 全reader件数はscope外のstream/table/script/configを含む件数です。F16 CSVのみ43,044 records、TrajAir対象trackは2,731,255 records、WEZ対象4表は3,864 records、Calculated Moves対象CSVは2,804,268 records。H制約でeligible数が減ります。

TrajAir/Calculated Movesの数百万query全件は大量のprompt構築・CSV書き出しが必要なため、seed=0で2,000件を非復元抽出しました。F16/WEZは設定で選んだscopeを全件計測。sampling母集団とsource_globsはCSV/summaryに保存済み。全件化するなら --full-count を明示できます。

## 最大・最小（今回計測した集合内）

| 種別 | Dataset | Episode | Sequence | Row | Query index | H | Input tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| min | calculated_moves | null | 21108 | 3430469 | 200 | 0 | 132 |
| max | MiniGrid-Unlock-v0 | 0 | 0 | null | 11 | 10 | 6726 |

同値が複数あれば最初の1件を代表表示します。最小はsample中の値であり、未計測部分を含めた外部dataset全体の最小ではありません。元filenameとSHA256、history範囲、前処理設定は[summary.json](summary.json)のextremaまたは[queries.csv](queries.csv)で追跡できます。

## 成果物・検証

- [token_inventory.csv](token_inventory.csv): mode/H/config/scopeごとの平均・母標準偏差・min/max/median/nearest-rank p95/total。
- [queries.csv](queries.csv): dataset/source/episode/sequence/row/query番号と実効前処理JSON＋SHA256、各token数。
- [summary_by_sequence.csv](summary_by_sequence.csv)、[summary_by_task.csv](summary_by_task.csv)、[summary_by_dataset.csv](summary_by_dataset.csv)。
- [datasets.csv](../dataset_inventory/datasets.csv): 25 datasetのraw目次。
- preprocessing_configs/ に10個の実効設定JSON。dataset別には70 scope集計。
- 48 tests passed（既存36＋新規12）。全公式172 NPZのraw値/shape/dtype、外部全列とfeature selection、F16 MAT/ULog、設定SHA、元prompt一致、write guard、static IDを検証。
- verify_token_inventory.py: 110,177行のID重複なし、70 scopeの件数・token総和一致、10設定SHA一致を確認。
- 最終全計測はsocket.connectを禁止して実行。OpenAI API request = 0。API key/.env未参照。paid pilot / accuracy evaluation / RL training / simulation未実行。
- upstream/LLM-Xavierのworktreeは変更なし。

## 次に5-query output-token pilotを行う場合（今回は実行していない）

以下は有料APIを呼ぶため、将来実行を決めた場合のみ使用してください。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python tools/pilot_output_tokens.py --task MountainCar-v0 --history-size 1 --model gpt-4o --sample-size 5 --seed 0 --max-output-tokens 256 --execute-paid-api
```
