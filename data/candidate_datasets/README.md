# 航空・空戦candidate datasetの全ファイル調査

2026-09-09、download済み全34,036ファイルを再帰scan。CSV/TXT/MAT/ULog/XLSXは内容を読み、code/model/media/PDF等はmetadataを列挙した。元ファイルは変更せず、simulation・学習・API・本番token countは実行していない。上流READMEは原文保存し、この文書と各SCHEMA.mdを日本語の調査結果とする。

## 互換性判定

A=episode内のstate/action/rewardがそのまま対応。B=逐次stateはあるが同期・不足field・episode定義等の変換が必要。C=flight-level sequential historyが存在しない。Bは元の操縦actionを復元できるという保証ではない。

| Dataset | 取得 | Size（保存bytes、git/cache除外） | Format | Records / Trajectories | State | Action | Reward | Sequential | Multi-agent | LLM-X互換性 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| [Calculated Moves](calculated_moves/SCHEMA.md) | 済、nested ZIP展開済 | 1,870,335,497 | CSV/TXT/ZIP | 4,200 learning runs、468質問票回答 | rule weightのみ、飛行状態なし | encounter単位rule script、操舵なし | fitness/avgfitness | encounter学習曲線のみ | blue/red、2v1/2v2 | C |
| [AirCombat-WEZ](aircombat_wez/SCHEMA.md) | 済 | 462,715 | 12 CSV＋code等 | Data 864＋各1,000×3関連case表、Paper_Results 8表 | 初期交戦条件 | なし | なし | なし | 射手/標的 | C |
| [Baidu Fighter jet](baidu_fighter_jet/SCHEMA.md) | NOT_ACQUIRED | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 未確認 | 論文記述のみ | 未確認 | 判定保留 |
| [F16Capstone](f16capstone/SCHEMA.md) | 済 | 256,752,038 | 3 CSV/8 MAT/3 ULog/5 XLSX等 | CSV 43,044行、numeric clipped MAT 4,992行、3 ULog sessions（重複合算しない） | 位置/速度/姿勢等 | RC/actuator指令 | なし | あり | 個別機記録 | B |
| [TrajAir](trajair/SCHEMA.md) | 済、全111日版＋weather | 2,464,731,087 | CSV/TXT/ZIP | processed 6,544 file×aircraft tracks、2,731,255行／raw 9,330,850行 | xyz、raw速度/heading、wind | なし | なし | あり、gapあり | 最大8機同時 | B |

Calculated Movesはrule weightの推移で、飛行timestepのposition/velocity/actionがなくCを維持。WEZも1行が独立caseまたはmodel性能集計でtrajectoryではない。F16Capstoneは同時刻state/controlを持つCSV・numeric MATからtransitionを作れるためB（rewardなし、ULog同期は別途）。TrajAirはstate trajectoryがあるがcontrol/rewardなしのB。Baiduは実ファイル未取得につきschemaもA/B/Cも推測しない。

## 実測サイズ・全ファイル数・schema群

| Dataset | 全file数 | top-level compressed bytes | nested ZIP含むcompressed bytes | ZIPを除く展開/checkout bytes | schema群（support含む） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calculated Moves | 30,606 | 211,171,930 | 433,793,427 | 1,436,542,070 | 24 |
| AirCombat-WEZ | 17 | 0 | 0 | 462,715 | 12 |
| F16Capstone | 210 | 0 | 0 | 256,752,038 | 33 |
| TrajAir | 3,203 | 316,594,152 | 316,594,152 | 2,148,136,935 | 13 |
| Baidu | 0 data files | 未確認 | 未確認 | 未確認 | 0（未取得） |

合計82群（support/archive/unstructured text含む）、構造化data/empty schemaのみ54群。Git取得の0 compressedは配布ZIPがないという意味。以前のF16「438 MiB checkout」は.git等を含むdu値と混在していたため、今回の実ファイルbytesへ置き換えた。Calculated Movesの30,605展開ファイル＋data.zip=30,606。自作文書はdata file数へ加算しない。

schema fingerprintはcolumn順・dtype・形式から作り、全fileをgroupへ割り当てた。代表例から同一性を推定していない。全columnは各SCHEMA.md、全filename対応は `outputs/dataset_inventory/candidate_files.csv`、column集計は `candidate_columns.csv`、完全schemaとfile別missing/uniqueは `candidate_schema.json`。

## 今回訂正・追加した点

- Calculated MovesのValidation.csvはsemicolon区切り質問票でtrajectoryではない。agent CSVの初期行とencounter結果を区別する。
- WEZはData 4 CSVに加えてPaper_Results 8 CSVを走査。
- F16の3 CSV（18,980 / 1,486 / 22,578行）を解析。MATLAB opaque objectのshape=[1]を1 timestepと解釈しない。
- TrajAirの空processedは7件、空rawは2件。rawは13列/14列混在。scene=3,088、file×aircraft track=6,544、1frame連続segment=8,000を区別。processed全行数は原READMEより1少ない。
- raw day directory=111、非空file=109、実Date値=112。timestampは重複・逆順があり、無条件に1Hzとはいえない。

## 出典・license

| Dataset | 出典 | 固定識別子 | license |
| --- | --- | --- | --- |
| Calculated Moves | [4TU](https://data.4tu.nl/articles/_/12688547/1) | DOI 10.4121/uuid:3521e3e6-a05a-4b9c-9151-c269c15b7f30 | CC BY 4.0 |
| AirCombat-WEZ | [GitHub](https://github.com/andrekuros/AirCombat-WEZ) | 5ae42f4e56362e4df2bf29c5c01cbfb50c5795ed | CC0 1.0 |
| F16Capstone | [GitHub](https://github.com/camdeno/F16Capstone) | acedfe4b2401b600a89f14f771dbcdf88c803b14 | MIT repository license |
| TrajAir | [KiltHub](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251) | DOI 10.1184/R1/14866251.v1 | CC BY 4.0 |
| Baidu Fighter jet | [論文](https://link.springer.com/article/10.1007/s42452-026-08398-3) | exact dataset ID未確定 | 論文CC0表記、file未確認 |

Calculated Movesの元mirrorは到達できず、DOIから同名の公式配布を確定済み。TrajAirの7day archivesは111日版の部分集合のため重複downloadしていない。Baiduは検索URLしか分からず、似た名前の別datasetで代用していない。取得時MD5は `configs/sources.json` と各SCHEMA.md。

## 再現

```bash
.venv/bin/python tools/inventory_candidates.py
.venv/bin/python tools/audit_dataset_relationships.py
.venv/bin/python tools/render_schema_docs.py
```

JSONには全filename、型/shape、missing/unique、sampling、ZIP全entry、エラー一覧、source size/mtime不変性を保存。「なし」は配布schemaにfieldがないこと、「未確認」は内部を読めない/確認不足、「一意に特定できない」は複数解釈や識別子不足を表す。
