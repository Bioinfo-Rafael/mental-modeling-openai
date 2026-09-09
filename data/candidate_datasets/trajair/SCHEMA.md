# 概要

[TrajAir公式配布](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251)、DOI `10.1184/R1/14866251.v1`。CC BY 4.0。111_days.zipとweather_data.zipを取得済み。MD5は順に `50dc9f4d271da435b6f1be2d13e70202`、`279aa9141d77ec9b86d952e9a1448570`（取得時一致）。7days1〜4は111日版に含まれる部分集合のため未取得。

2026-09-09のdownload済み実ファイル全走査。元データは変更していない。

# ファイル構成

| format | file数 |
| --- | --- |
| .csv | 112 |
| .txt | 3089 |
| .zip | 2 |

全filenameとbytesは `outputs/dataset_inventory/candidate_files.csv`（datasetでfilter）に記録。schema、file別missing/unique/type・shape、archive treeは `candidate_schema.json`。support file（code/model/media/PDF）はmetadata inventoryのみで、flight dataとして実行/deserializeしていない。上流READMEは原文保存し、このSCHEMA.mdを日本語の調査結果とする。

# データ量

| 量 | 値 |
| --- | --- |
| file_count | 3203 |
| disk_bytes_excluding_git_cache | 2464731087 |
| compressed_bytes_all_zip_including_nested | 316594152 |
| compressed_bytes_top_level | 316594152 |
| extracted_bytes_excluding_nested_zip | 2148136935 |
| format_counts | {'.zip': 2, '.txt': 3089, '.csv': 112} |
| schema_group_count | 13 |
| errors | [] |

bytesはファイル内容のサイズ合計で、filesystem allocation（du）ではない。`.git`, `.cache` と自作wrapper文書を除外。top-level ZIP、nested ZIP、展開物を区別しており、ZIPの展開後サイズを現在のdisk bytesへ重複加算しない。Git取得でZIPが0なのは圧縮配布物を保存していない意味で、Git pack size=0の主張ではない。

# Schema

schema fingerprintは列名順序＋reader dtype＋delimiter/header等の構造からSHA256の先頭16桁を計算。CSVの行数・missing数をfingerprintへ混ぜない。MATはvariable shapeを含む。dtypeはCSVにnative保存された型ではなくreaderの推定結果。空fileは未知schemaとして独立。文字列/全欠損/数値型の差も分離する。

| schema group | file数 | 物理rows合計 | kind | 例filename |
| --- | --- | --- | --- | --- |
| trajair:add958930fe8bef2 | 1 | 0 | archive | 111_days.zip |
| trajair:8372931050a8036b | 1 | 152 | unstructured_text | README.source.txt |
| trajair:8f208a575cf95ea4 | 3081 | 2731255 | delimited | extracted/111_days/processed_data/test/1003.txt |
| trajair:1e6671e1bae83f53 | 7 | 0 | empty_processed | extracted/111_days/processed_data/test/1022.txt |
| trajair:790f5c0a491498f4 | 43 | 4815072 | delimited | extracted/111_days/raw_data/01-19-21_adsb/1.csv |
| trajair:eaf7ffbd5c892c54 | 4 | 80232 | delimited | extracted/111_days/raw_data/01-25-21_adsb/1.csv |
| trajair:e629b77b087fdc4d | 2 | 25945 | delimited | extracted/111_days/raw_data/03-14-21_adsb/1.csv |
| trajair:f5506c55d0b90769 | 5 | 614213 | delimited | extracted/111_days/raw_data/03-15-21_adsb/1.csv |
| trajair:cec8d07221740474 | 34 | 3348346 | delimited | extracted/111_days/raw_data/09-12-20_adsb/1.csv |
| trajair:53fdbc7137fa40d0 | 21 | 447042 | delimited | extracted/111_days/raw_data/09-16-20_adsb/1.csv |
| trajair:a4b7a0f3aeb43262 | 2 | 0 | empty_csv | extracted/111_days/raw_data/12-01-20_adsb/1.csv |
| trajair:41791cc67755b558 | 1 | 68386 | delimited | extracted/weather_data/weather.csv |
| trajair:27128fdfd14a64be | 1 | 0 | archive | weather_data.zip |

# 1 record / timestep の意味

processed TXTはheaderなし、空白区切り7列。1行はscene内の1航空機の1frame観測。全列順は `frame_number, aircraft_id, x_km, y_km, z_km, wind_x_mps, wind_y_mps`。位置は空港基準座標(km)、風はm/s。単位と1Hzは同梱README.source.txtを根拠とする。rawはADS-BのID/Date/Timeと位置等、weatherは観測所とvalid時刻の気象観測。

# trajectory構造

`(train/testを含む相対filename, aircraft_id)` をtrackキー、`frame_number` を時刻順序とする。異なるsceneの同じIDを無条件に結合しない。同じscene・同じframeの異なるIDは同時航空機。欠測gapを分割しないtrack数と、delta_frame=1だけで連続とするsegment数を区別する。これは実flight離着陸境界を同定した数ではない。

```text
scene relative_path
  aircraft_id
    frame f:     x, y, z, wind_x, wind_y
    frame f + 1: x, y, z, wind_x, wind_y
    frame f + k: gap（k>1なら連続segmentの切れ目候補）
```

rawでは日別file＋ID＋Date/Timeで記録をまとめられるが、1日同一IDに複数flightが含まれ得る。rawとprocessedのtrack数が異なるのは定義/抽出が異なるため。raw timestampの順序差分も全ファイルで集計し、規則的1Hzと決めつけない。

# state候補

processedのxyz、wind xy。rawのLat/Lon、Altitude、Speed、Headingも候補。processedに速度/headingそのものの列はなし（位置差分から導く量は派生特徴と明記）。rawは13列と14列があり `AltisGNSS` の有無が異なる。flight/trajectory ID列はなし。weatherのwind contextはstation/validを基準に時刻・空間対応を定義する必要がある。位置と風の単位を混ぜない。

# action候補

action/controlは存在しない。位置差分や速度変化を作っても真の操縦指令ではなくderived motion proxy。

# reward候補

存在しない。研究目的に応じたreward定義が必要。

# LLM-Xavierとの互換性

**B**。sequentialなstate/next_stateは作れる。actionとrewardは未提供なので、公式NPZへ直結はできない。自然な用途は複数航空機trajectory prediction。実controlが必要な評価taskへの転用は制約がある。

# 不明点

同梱READMEの2,731,256行に対し実ファイルは2,731,255行（差1）。原因は一意に特定できない。元のREADMEで空ファイルを1件とした記述は誤りで、今回全走査ではprocessed7件とraw2件。収集日は2020-09-12〜2021-04-27と記載されるが、全期間が毎日連続ではない。raw day directory数・実Date値数・非空file数を区別する。licenseは公式配布のCC BY 4.0を採用し、別論文のCC0との記述に置き換えない。

## 全日・全sceneの実測集計

| 項目 | 値 |
| --- | --- |
| processed_scenes | 3088 |
| processed_rows | 2731255 |
| file_aircraft_tracks | 6544 |
| rows_per_track | {'count': 6544, 'min': 1, 'max': 5974, 'mean': 417.36781784841077, 'median': 254.0} |
| rows_per_scene | {'count': 3088, 'min': 0, 'max': 25386, 'mean': 884.4737694300518, 'median': 287.0} |
| aircraft_ids_count | 527 |
| one_frame_contiguous_segments | 8000 |
| multi_aircraft_scenes | 1107 |
| max_simultaneous | 8 |
| duplicate_frame_id_rows | 0 |
| raw_files | 111 |
| raw_rows | 9330850 |
| raw_aircraft_ids | 3994 |
| weather_files | 1 |
| weather_rows | 68386 |
| raw_file_aircraft_tracks | 10511 |
| raw_rows_per_file_aircraft_track | {'count': 10511, 'min': 1, 'max': 16869, 'mean': 887.7223860717344, 'median': 395} |

raw day directory数=111、raw非空file数=109、実Date値数=112。processed train=2,230、test=858。

`frame_delta_counts`: 異なるdelta=660、min=1.0、max=6239.0、頻度上位=[('1.0', 2723255), ('10.0', 13), ('15.0', 13), ('16.0', 13), ('19.0', 13), ('5.0', 12), ('12.0', 12), ('4.0', 12)]。完全な頻度はJSON参照。

`raw_time_delta_ms_counts`: 異なるdelta=48477、min=-50431.0、max=66578370.0、頻度上位=[('0', 3312575), ('1000', 25222), ('1006', 21512), ('1049', 21029), ('1021', 20608), ('1064', 19867), ('991', 19223), ('1019', 18389)]。完全な頻度はJSON参照。

空processed filename:

- `extracted/111_days/processed_data/test/1022.txt`
- `extracted/111_days/processed_data/train/172.txt`
- `extracted/111_days/processed_data/train/1897.txt`
- `extracted/111_days/processed_data/train/1903.txt`
- `extracted/111_days/processed_data/train/230.txt`
- `extracted/111_days/processed_data/train/758.txt`
- `extracted/111_days/processed_data/train/780.txt`

空raw filename: `12-01-20_adsb/1.csv`, `12-02-20_adsb/1.csv`。

# Schema group別の全column / variable

同じschemaは全columnを一度掲載。少数groupはfilenameも記録し、多数groupの全対応はcandidate_files.csvで検索できる。missingはgroup内合計、uniqueのfile別値はJSON内 `unique`。空文字やNA/NaN等を欠損とし、weatherのみMも欠損扱い。XLSXは全OOXML cellを読み、formulaを実行せずcached typeとtext cellを調査する。

## trajair:add958930fe8bef2

1 files。

- `111_days.zip`

`111_days.zip`: 3199内部files、uncompressed=2135991684 bytes、formats={'.csv': 111, '.txt': 3088}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

## trajair:8372931050a8036b

1 files。

- `README.source.txt`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |

## trajair:8f208a575cf95ea4

3081 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| frame_number | int64 | 0 |
| aircraft_id | int64 | 0 |
| x_km | float64 | 0 |
| y_km | float64 | 0 |
| z_km | float64 | 0 |
| wind_x_mps | float64 | 0 |
| wind_y_mps | float64 | 0 |

## trajair:1e6671e1bae83f53

7 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |

## trajair:790f5c0a491498f4

43 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | float64 | 25323 |
| Heading | float64 | 25323 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 20019 |
| AltisGNSS | bool | 0 |
| Metar | str | 0 |

## trajair:eaf7ffbd5c892c54

4 files。

- `extracted/111_days/raw_data/01-25-21_adsb/1.csv`
- `extracted/111_days/raw_data/01-31-21_adsb/1.csv`
- `extracted/111_days/raw_data/02-03-21_adsb/1.csv`
- `extracted/111_days/raw_data/04-10-21_adsb/1.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | float64 | 736 |
| Heading | float64 | 736 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 50 |
| AltisGNSS | str | 3 |
| Metar | str | 0 |

## trajair:e629b77b087fdc4d

2 files。

- `extracted/111_days/raw_data/03-14-21_adsb/1.csv`
- `extracted/111_days/raw_data/04-17-21_adsb/1.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | int64 | 0 |
| Heading | int64 | 0 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 220 |
| AltisGNSS | bool | 0 |
| Metar | str | 0 |

## trajair:f5506c55d0b90769

5 files。

- `extracted/111_days/raw_data/03-15-21_adsb/1.csv`
- `extracted/111_days/raw_data/03-19-21_adsb/1.csv`
- `extracted/111_days/raw_data/03-21-21_adsb/1.csv`
- `extracted/111_days/raw_data/04-22-21_adsb/1.csv`
- `extracted/111_days/raw_data/04-26-21_adsb/1.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | float64 | 3310 |
| Heading | float64 | 3310 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 6369 |
| AltisGNSS | str/mixed | 4 |
| Metar | str | 0 |

## trajair:cec8d07221740474

34 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | float64 | 19062 |
| Heading | float64 | 19062 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 12116 |
| Metar | str | 0 |

## trajair:53fdbc7137fa40d0

21 files。

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| ID | int64 | 0 |
| Time | str | 0 |
| Date | str | 0 |
| Altitude | int64 | 0 |
| Speed | int64 | 0 |
| Heading | int64 | 0 |
| Lat | float64 | 0 |
| Lon | float64 | 0 |
| Age | float64 | 0 |
| Range | float64 | 0 |
| Bearing | float64 | 0 |
| Tail | str | 1176 |
| Metar | str | 0 |

## trajair:a4b7a0f3aeb43262

2 files。

- `extracted/111_days/raw_data/12-01-20_adsb/1.csv`
- `extracted/111_days/raw_data/12-02-20_adsb/1.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |

## trajair:41791cc67755b558

1 files。

- `extracted/weather_data/weather.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| station | str | 0 |
| valid | str | 0 |
| tmpf | float64 | 60040 |
| dwpf | float64 | 60043 |
| relh | float64 | 60043 |
| drct | float64 | 1189 |
| sknt | float64 | 102 |
| p01i | str | 56883 |
| alti | float64 | 128 |
| mslp | float64 | 62211 |
| vsby | float64 | 3 |
| gust | float64 | 65689 |
| skyc1 | str | 14 |
| skyc2 | str | 52561 |
| skyc3 | str | 62916 |
| skyc4 | unknown(empty) | 68386 |
| skyl1 | float64 | 27386 |
| skyl2 | float64 | 52561 |
| skyl3 | float64 | 62916 |
| skyl4 | unknown(empty) | 68386 |
| wxcodes | str | 55133 |
| ice_accretion_1hr | unknown(empty) | 68386 |
| ice_accretion_3hr | unknown(empty) | 68386 |
| ice_accretion_6hr | unknown(empty) | 68386 |
| peak_wind_gust | float64 | 68275 |
| peak_wind_drct | float64 | 68275 |
| peak_wind_time | str | 68275 |
| feel | float64 | 63820 |
| metar | str | 0 |

## trajair:27128fdfd14a64be

1 files。

- `weather_data.zip`

`weather_data.zip`: 1内部files、uncompressed=12141297 bytes、formats={'.csv': 1}。全archive treeはJSONのarchive.entries（path/size/compressed/CRC）へ保存。

