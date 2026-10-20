# 概要

[Calculated Moves](https://data.4tu.nl/articles/_/12688547/1)、DOI `10.4121/uuid:3521e3e6-a05a-4b9c-9151-c269c15b7f30`。CC BY 4.0。元の4tu mirror URLは取得不能だったため、同名資料のDOIから公式archiveを取得済み。data.zipのMD5は `8c1c24e77633c4d6f5c7f0fa4cfb0b4e`（取得時一致）。全nested ZIP展開済み。

2026-09-09のdownload済み実ファイル全走査。元データは変更していない。

# ファイル構成

| format | file数 |
| --- | --- |
| .csv | 18001 |
| .txt | 12601 |
| .zip | 4 |

全filenameとbytesは `outputs/dataset_inventory/candidate_files.csv`（datasetでfilter）に記録。schema、file別missing/unique/type・shape、archive treeは `candidate_schema.json`。support file（code/model/media/PDF）はmetadata inventoryのみで、flight dataとして実行/deserializeしていない。上流READMEは原文保存し、このSCHEMA.mdを日本語の調査結果とする。

# データ量

| 量 | 値 |
| --- | --- |
| file_count | 30606 |
| disk_bytes_excluding_git_cache | 1870335497 |
| compressed_bytes_all_zip_including_nested | 433793427 |
| compressed_bytes_top_level | 211171930 |
| extracted_bytes_excluding_nested_zip | 1436542070 |
| format_counts | {'.zip': 4, '.txt': 12601, '.csv': 18001} |
| schema_group_count | 24 |
| errors | [] |

bytesはファイル内容のサイズ合計で、filesystem allocation（du）ではない。`.git`, `.cache` と自作wrapper文書を除外。top-level ZIP、nested ZIP、展開物を区別しており、ZIPの展開後サイズを現在のdisk bytesへ重複加算しない。Git取得でZIPが0なのは圧縮配布物を保存していない意味で、Git pack size=0の主張ではない。

# Schema

schema fingerprintは列名順序＋reader dtype＋delimiter/header等の構造からSHA256の先頭16桁を計算。CSVの行数・missing数をfingerprintへ混ぜない。MATはvariable shapeを含む。dtypeはCSVにnative保存された型ではなくreaderの推定結果。空fileは未知schemaとして独立。文字列/全欠損/数値型の差も分離する。

| schema group | file数 | 物理rows合計 | kind | 例filename |
| --- | --- | --- | --- | --- |
| calculated_moves:54db2f2e1f5fad48 | 1 | 0 | archive | data.zip |
| calculated_moves:139bd38ca2ffe9c0 | 4200 | 4200 | configuration | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/SimulationConfiguration.txt |
| calculated_moves:30b2208b8b9ec81b | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/blue1.csv |
| calculated_moves:7e7d0aa67210fb2c | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/blue2.csv |
| calculated_moves:3f92ad91a0146469 | 2000 | 512000 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/red.csv |
| calculated_moves:f3689bc98c00a34d | 8400 | 18799955 | encounter_script | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/scripts-blue1.txt |
| calculated_moves:26faa7c4e69c7116 | 4200 | 4200 | winner_sequence | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/mixed/20180108110111/winHistory.csv |
| calculated_moves:a2edcb5f265d0dd6 | 1350 | 218850 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/red/20180105140503/red.csv |
| calculated_moves:de595dc8fac46948 | 1450 | 248950 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/cent/red_range50/20180105160210/red.csv |
| calculated_moves:d3197f983726d785 | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/decent/mixed/20180110193737/blue1.csv |
| calculated_moves:838a78594f17cd23 | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/decent/mixed/20180110193737/blue2.csv |
| calculated_moves:438cea9540b6f9eb | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/tactic/mixed/20180108144437/blue1.csv |
| calculated_moves:0f69570f0b7d6fbe | 200 | 30200 | delimited | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination/Chapter 3 - Team coordination/tactic/mixed/20180108144437/blue2.csv |
| calculated_moves:add958930fe8bef2 | 3 | 0 | archive | extracted/Dataset Calculated Moves/Chapter 3 - Team coordination.zip |
| calculated_moves:95ecde5dd012244e | 2400 | 542400 | delimited | extracted/Dataset Calculated Moves/Chapter 4 - Rewards/Chapter 4 - Rewards/aareward/mixed/20150406180701/blue1.csv |
| calculated_moves:ba48bc10f27022fc | 2400 | 542400 | delimited | extracted/Dataset Calculated Moves/Chapter 4 - Rewards/Chapter 4 - Rewards/aareward/mixed/20150406180701/blue2.csv |
| calculated_moves:3b408eb351c116c4 | 600 | 120600 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v1-transfer-sources/mixed/20150406180701/blue1.csv |
| calculated_moves:7244204460a7337b | 600 | 120600 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v1-transfer-sources/mixed/20150406180701/blue2.csv |
| calculated_moves:5046ac051839df6b | 300 | 67800 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v2-no-transfer/lead-trail/20150527143440/red1.csv |
| calculated_moves:57367374e12e5568 | 300 | 67800 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v2-no-transfer/lead-trail/20150527143440/red2.csv |
| calculated_moves:f59fe7ed8fa2ef8c | 600 | 90600 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v2-transfer-with-mixed-as-source/lead-trail/20150406180701/blue1.csv |
| calculated_moves:8cc3e31a1487bf54 | 600 | 90600 | delimited | extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning/Chapter 5 - Transfer learning/2v2-transfer-with-mixed-as-source/lead-trail/20150406180701/blue2.csv |
| calculated_moves:dda4f0b05d8ff6bf | 1 | 468 | delimited | extracted/Dataset Calculated Moves/Chapter 7 - Validation/Validation.csv |
| calculated_moves:8372931050a8036b | 1 | 164 | unstructured_text | extracted/Dataset Calculated Moves/README.txt |

# 1 record / timestep の意味

`blue1.csv` 等の行はrule weightと `fitness`, `avgfitness` の推移。**飛行simulationのtimestep-level trajectoryではない**。runはtimestamp付きdirectory、encounterはscript内 `Encounter n` とwinHistoryの項目順で識別する。1runのagent CSVには `maxEncounters+1` 行があり、初期値と更新後の行を含む。最初のfitness=0行を「第1encounterの結果」と誤認しない。原READMEの「nth encounter開始時のweight」とreward更新の厳密なoffsetは生成コードなしでは一意に特定できない。

`scripts-blue*.txt` は `[weight] [priority] rule name` をEncounterブロックにまとめた構造化text。全行をparseして異常行数を記録。空scriptもあり、原READMEは固定script制御の場合があると説明する。`winHistory.csv` はheaderなし横1行のblue/red列（物理shape=[1,100/150/300]、論理的には同数のencounter outcomes）。`Validation.csv` は **semicolon区切り468回答×10列** のATACC質問票で、飛行軌跡ではない。`flight` は閲覧した映像番号、`tactical` は匿名回答者ID。

# trajectory構造

4,200 learning runs。Chapter3はcent/decent/tacticのcoordination、Chapter4はreward条件、Chapter5は2v1-transfer-sources / 2v2-no-transfer / 2v2-transfer-with-mixed-as-source。blue1/blue2=lead/wingman、redまたはred1/red2=敵側。filenameをagent ID、pathをrun/scenario IDとして利用できる。trajectory episodeとlearning run/encounterは別物。

# state候補

rule weight配列はencounter-level学習状態の候補だが、飛行状態の代用品にはならない。

| 調査field | 実ファイルでの結論 |
| --- | --- |
| scenario ID / episode ID | 条件path / timestamp付きrun directoryとEncounter番号。独立したflight episode ID列はなし |
| timestep / simulation time | 飛行timestep・simulation timeはなし。configのspeedはsimulation設定であり時刻ではない |
| aircraft / agent ID / team | blue1,blue2,red,red1,red2のfilenameとblue/red outcome |
| position x/y/z・latitude/longitude・altitude | なし |
| velocity・acceleration・heading・pitch・roll・attitude | なし |
| control input | なし |
| selected action / maneuver | encounter開始時の選択rule scriptあり。実行されたmaneuverの時刻列はなし |
| reward / cumulative reward | fitnessとavgfitnessあり。cumulative reward列はなし（avgfitnessは累積和ではない） |
| target aircraft | default-target等のrule名はあるが、対象機ID時系列はなし |
| weapon state / missile / fire event | rule名にfire/missileがある。weapon state値・実発射eventログはなし |
| hit / kill event / terminal state | winHistoryの勝利teamはあるが、hit/kill時刻・terminal state vectorはなし |
| 1v1 / 2v1 / 2v2 | 2v1・2v2条件pathあり。独立1v1条件pathなし |

# action候補

encounter単位の選択ruleとweight/priority。flight control inputやstep actionは存在しない。

# reward候補

`fitness`, `avgfitness` はencounter-level。具体的reward関数や更新時刻の完全復元は未確認。winHistoryは勝利teamの結果ラベル。

# LLM-Xavierとの互換性

**Cを維持**。飛行state/actionの時系列が存在せず、flight-level Mental Modeling historyを復元できない。encounter-level dynamic scriptingを別のtaskとして定義する研究には使えるが、同じ再評価実験ではない。

# 不明点

原READMEに列挙された `fitness.csv`, `fitness2.csv`, `scripts-red.txt` は実配布ファイルには存在しない。原READMEのValidationData.csvに対し実名はValidation.csv。ruleの厳密な動作定義はthesis参照が必要。飛行kinematicsをrule名から推測していない。

## run設定・script全件集計

| 設定 | run数 |
| --- | --- |
| {'maxEncounters': '150', 'scriptSize': '12', 'speed': '5.0'} | 200 |
| {'maxEncounters': '150', 'scriptSize': '6', 'speed': '5.0'} | 400 |
| {'maxEncounters': '300', 'scriptSize': '6', 'speed': '9.0'} | 1800 |
| {'maxEncounters': '100', 'scriptSize': '6', 'speed': '9.0'} | 1200 |
| {'maxEncounters': '150', 'scriptSize': '6', 'speed': '9.0'} | 600 |

scripts 8400 files、空script 200 files、parseできない非空行 0。

| basename | file数 |
| --- | --- |
| data.zip | 1 |
| SimulationConfiguration.txt | 4200 |
| blue1.csv | 4200 |
| blue2.csv | 4200 |
| red.csv | 3000 |
| scripts-blue1.txt | 4200 |
| scripts-blue2.txt | 4200 |
| winHistory.csv | 4200 |
| Chapter 3 - Team coordination.zip | 1 |
| Chapter 4 - Rewards.zip | 1 |
| red1.csv | 1200 |
| red2.csv | 1200 |
| Chapter 5 - Transfer learning.zip | 1 |
| Validation.csv | 1 |
| README.txt | 1 |

# Schema group別の全column / variable

同じschemaは全columnを一度掲載。少数groupはfilenameも記録し、多数groupの全対応はcandidate_files.csvで検索できる。missingはgroup内合計、uniqueのfile別値はJSON内 `unique`。空文字やNA/NaN等を欠損とし、weatherのみMも欠損扱い。XLSXは全OOXML cellを読み、formulaを実行せずcached typeとtext cellを調査する。

## calculated_moves:54db2f2e1f5fad48

1 files。

- `data.zip`

`data.zip`: 5内部files、uncompressed=222659364 bytes、formats={'.zip': 3, '.txt': 1, '.csv': 1}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

## calculated_moves:139bd38ca2ffe9c0

4200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| maxEncounters | int64 | 0 |
| scriptSize | int64 | 0 |
| speed | float64 | 0 |

## calculated_moves:30b2208b8b9ec81b

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| default-support | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| support | float64 | 0 |
| support-left | float64 | 0 |
| support-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR180 | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile+180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| wingmanEventRadar+90 | float64 | 0 |
| wingmanEventRadar-90 | float64 | 0 |
| wingmanEventRadarEI | float64 | 0 |
| wingmanEventRWR+90 | float64 | 0 |
| wingmanEventRWR-90 | float64 | 0 |
| wingmanEventRWREU | float64 | 0 |
| wingmanEventMissile+90 | float64 | 0 |
| wingmanEventMissile-90 | float64 | 0 |
| wingmanAskFireFrom50 | float64 | 0 |
| wingmanAskFireFrom60 | float64 | 0 |
| wingmanAskFireFrom70 | float64 | 0 |
| wingmanAskFireFrom80 | float64 | 0 |
| wingmanAskFireFrom90 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:7e7d0aa67210fb2c

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| eventNewRadarObservation | float64 | 0 |
| eventNewRadarWarningReceiverObservation | float64 | 0 |
| eventMissileFlyingAtMe | float64 | 0 |
| askFireFrom50 | float64 | 0 |
| doFireFrom50 | float64 | 0 |
| askFireFrom60 | float64 | 0 |
| doFireFrom60 | float64 | 0 |
| askFireFrom70 | float64 | 0 |
| doFireFrom70 | float64 | 0 |
| askFireFrom80 | float64 | 0 |
| doFireFrom80 | float64 | 0 |
| askFireFrom90 | float64 | 0 |
| doFireFrom90 | float64 | 0 |
| breakp90 | float64 | 0 |
| breakm90 | float64 | 0 |
| engageInformed | float64 | 0 |
| engageUninformed | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:3f92ad91a0146469

2000 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| initialCAP1 | float64 | 0 |
| flyToCAP2 | float64 | 0 |
| CAP2 | float64 | 0 |
| flyToCAP1 | float64 | 0 |
| CAP1 | float64 | 0 |
| engageRWR | float64 | 0 |
| engageRadar | float64 | 0 |
| search | float64 | 0 |
| lock | float64 | 0 |
| fire | float64 | 0 |
| support | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:f3689bc98c00a34d

8400 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| encounter | int64 | 0 |
| weight | float64 | 0 |
| priority | int64 | 0 |
| rule | str | 0 |

## calculated_moves:26faa7c4e69c7116

4200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| winner | str | 0 |

## calculated_moves:a2edcb5f265d0dd6

1350 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| initialCAP1 | float64 | 0 |
| flyToCAP2 | float64 | 0 |
| CAP2 | float64 | 0 |
| flyToCAP1 | float64 | 0 |
| CAP1 | float64 | 0 |
| engageRWR | float64 | 0 |
| engageRadar | float64 | 0 |
| search | float64 | 0 |
| lock | float64 | 0 |
| fire | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:de595dc8fac46948

1450 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| initialCAP1 | float64 | 0 |
| flyToCAP2 | float64 | 0 |
| CAP2 | float64 | 0 |
| flyToCAP1 | float64 | 0 |
| CAP1 | float64 | 0 |
| engageRWR | float64 | 0 |
| engageRadar | float64 | 0 |
| search | float64 | 0 |
| lock | float64 | 0 |
| fire | float64 | 0 |
| support | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:d3197f983726d785

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| default-support | float64 | 0 |
| support | float64 | 0 |
| support-left | float64 | 0 |
| support-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR180 | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evading->engage | float64 | 0 |
| m:engaging->engage | float64 | 0 |
| m:evadeRWR->evade180 | float64 | 0 |
| m:evadeRWR->evade+90 | float64 | 0 |
| m:evadeRWR->evade-90 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:838a78594f17cd23

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-formation | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| default-support | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| support | float64 | 0 |
| support-left | float64 | 0 |
| support-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR180 | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evading->engage | float64 | 0 |
| m:engaging->engage | float64 | 0 |
| m:evadeRWR->evade180 | float64 | 0 |
| m:evadeRWR->evade+90 | float64 | 0 |
| m:evadeRWR->evade-90 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:438cea9540b6f9eb

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| default-support | float64 | 0 |
| support | float64 | 0 |
| support-left | float64 | 0 |
| support-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR180 | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m1 | float64 | 0 |
| m2 | float64 | 0 |
| m3 | float64 | 0 |
| m4 | float64 | 0 |
| m5 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:0f69570f0b7d6fbe

200 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-formation | float64 | 0 |
| default-search | float64 | 0 |
| default-track | float64 | 0 |
| default-fire | float64 | 0 |
| default-support | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| support | float64 | 0 |
| support-left | float64 | 0 |
| support-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR180 | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m1 | float64 | 0 |
| m2 | float64 | 0 |
| m3 | float64 | 0 |
| m4 | float64 | 0 |
| m5 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:add958930fe8bef2

3 files。

- `extracted/Dataset Calculated Moves/Chapter 3 - Team coordination.zip`
- `extracted/Dataset Calculated Moves/Chapter 4 - Rewards.zip`
- `extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning.zip`

`extracted/Dataset Calculated Moves/Chapter 3 - Team coordination.zip`: 4200内部files、uncompressed=148989489 bytes、formats={'.csv': 2400, '.txt': 1800}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

`extracted/Dataset Calculated Moves/Chapter 4 - Rewards.zip`: 12600内部files、uncompressed=604527558 bytes、formats={'.csv': 7200, '.txt': 5400}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

`extracted/Dataset Calculated Moves/Chapter 5 - Transfer learning.zip`: 13800内部files、uncompressed=682987156 bytes、formats={'.csv': 8400, '.txt': 5400}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

## calculated_moves:95ecde5dd012244e

2400 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-lock | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade | float64 | 0 |
| m:evadingRWR->evade.1 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade | float64 | 0 |
| m:evadingMissile->evade.1 | float64 | 0 |
| m:engaging->engage | float64 | 0 |
| m:engaging->engage.1 | float64 | 0 |
| m:engaging->engage.2 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:ba48bc10f27022fc

2400 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-formation | float64 | 0 |
| default-alone | float64 | 0 |
| default-lock | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade | float64 | 0 |
| m:evadingRWR->evade.1 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade | float64 | 0 |
| m:evadingMissile->evade.1 | float64 | 0 |
| m:engaging->engage | float64 | 0 |
| m:engaging->engage.1 | float64 | 0 |
| m:engaging->engage.2 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:3b408eb351c116c4

600 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-lock | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade1 | float64 | 0 |
| m:evadingRWR->evade2 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade1 | float64 | 0 |
| m:evadingMissile->evade2 | float64 | 0 |
| m:engaging->engage1 | float64 | 0 |
| m:engaging->engage2 | float64 | 0 |
| m:engaging->engage3 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:7244204460a7337b

600 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-formation | float64 | 0 |
| default-alone | float64 | 0 |
| default-lock | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade1 | float64 | 0 |
| m:evadingRWR->evade2 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade1 | float64 | 0 |
| m:evadingMissile->evade2 | float64 | 0 |
| m:engaging->engage1 | float64 | 0 |
| m:engaging->engage2 | float64 | 0 |
| m:engaging->engage3 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:5046ac051839df6b

300 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| true | float64 | 0 |
| distract | float64 | 0 |
| formation | float64 | 0 |
| search | float64 | 0 |
| lock | float64 | 0 |
| fire | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:57367374e12e5568

300 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| true | float64 | 0 |
| formation | float64 | 0 |
| search | float64 | 0 |
| lock | float64 | 0 |
| fire | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:f59fe7ed8fa2ef8c

600 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-target | float64 | 0 |
| default-search | float64 | 0 |
| default-lock | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade1 | float64 | 0 |
| m:evadingRWR->evade2 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade1 | float64 | 0 |
| m:evadingMissile->evade2 | float64 | 0 |
| m:engaging->engage1 | float64 | 0 |
| m:engaging->engage2 | float64 | 0 |
| m:engaging->engage3 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:8cc3e31a1487bf54

600 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| default-formation | float64 | 0 |
| default-alone | float64 | 0 |
| default-search | float64 | 0 |
| default-lock | float64 | 0 |
| twoship-formation-left | float64 | 0 |
| twoship-formation-right | float64 | 0 |
| trail-formation | float64 | 0 |
| wall-formation-left | float64 | 0 |
| wall-formation-right | float64 | 0 |
| engageRWR | float64 | 0 |
| evadeRWR+90 | float64 | 0 |
| evadeRWR-90 | float64 | 0 |
| evadeMissile180 | float64 | 0 |
| evadeMissile+90 | float64 | 0 |
| evadeMissile-90 | float64 | 0 |
| fireFrom50 | float64 | 0 |
| fireFrom60 | float64 | 0 |
| fireFrom70 | float64 | 0 |
| fireFrom80 | float64 | 0 |
| fireFrom90 | float64 | 0 |
| fire2From50 | float64 | 0 |
| fire2From60 | float64 | 0 |
| fire2From70 | float64 | 0 |
| fire2From80 | float64 | 0 |
| fire2From90 | float64 | 0 |
| filler1 | float64 | 0 |
| filler2 | float64 | 0 |
| filler3 | float64 | 0 |
| filler4 | float64 | 0 |
| filler5 | float64 | 0 |
| filler6 | float64 | 0 |
| m:evadingRWR->engage | float64 | 0 |
| m:evadingRWR->evade1 | float64 | 0 |
| m:evadingRWR->evade2 | float64 | 0 |
| m:evadingMissile->engage | float64 | 0 |
| m:evadingMissile->evade1 | float64 | 0 |
| m:evadingMissile->evade2 | float64 | 0 |
| m:engaging->engage1 | float64 | 0 |
| m:engaging->engage2 | float64 | 0 |
| m:engaging->engage3 | float64 | 0 |
| fitness | float64 | 0 |
| avgfitness | float64 | 0 |

## calculated_moves:dda4f0b05d8ff6bf

1 files。

- `extracted/Dataset Calculated Moves/Chapter 7 - Validation/Validation.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| date | str | 0 |
| flight | int64 | 0 |
| tactical | str | 0 |
| status | str | 0 |
| models | str | 0 |
| redformation | str | 0 |
| question | int64 | 0 |
| rawvalue | float64 | 12 |
| invertQ4 | float64 | 12 |
| value | float64 | 12 |

## calculated_moves:8372931050a8036b

1 files。

- `extracted/Dataset Calculated Moves/README.txt`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |

# ファイル間対応の全件照合

`tools/audit_dataset_relationships.py` の実測。sourceコードは実行せず比較のみ。

```json
{
  "run_count": 4200,
  "agent_row_minus_maxEncounters": {
    "1": 13800
  },
  "winner_count_minus_maxEncounters": {
    "0": 4200
  },
  "script_count_minus_maxEncounters": {
    "0": 8200,
    "-150": 200
  },
  "exceptions": []
}
```

