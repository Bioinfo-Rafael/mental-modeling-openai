# 04_sample_size_analysis: resultsの構成とデータの読み方

03の既存応答を条件別に先頭N={30,20,10}件に切って再集計するAPI不要の実験。現時点で存在するのは.gitkeepのみで、統計・図・raw応答はここに保存されていない。

2026-09-16確認。既存ファイルとコードを読み取り、READMEだけを追加した。実験・解析は再実行していない。構成と件数は確認時点の保存内容。

## ディレクトリ構成

```text
results/
  .gitkeep
```

## 実在するファイルの説明

| ファイル名（実行世代・batch内でも同じ役割） | 内容 | 生成元 |
| --- | --- | --- |
| `.gitkeep` | 空ディレクトリをGitで保持するための管理用ファイル。実験データではない。 | リポジトリ管理用。run.pyの生成物ではない。 |

## コード上の出力予定（現在は未生成）

[run.py](../run.py) → [common.py](../../common.py) の `analysis_main()` → [analysis.py](../../analysis.py) の `main()`。`load_source()` は03の最新完了世代（現状 `resumes/0001/`）のmanifest、summary、recordsを読み、`calculate()` が条件内ordinal順の先頭30/20/10件を使う。新しいsampleやAPI応答は作らない。

| 予定パス | 保持するデータ | 生成元 |
| --- | --- | --- |
| `dry_run/manifest.json` | 入力のhash、sample_sizes、条件数、予定集計行数、API件数0 | `analysis.main(--dry-run)` |
| `.started.json` | 解析開始時刻 | `analysis.main()` |
| `manifest.json` | 解析計画・元データhash。API実験のquery本文入りmanifestとは別形式 | `analysis.main()` |
| `statistics.csv` | 32条件×3種類のN＝96行の集計。精度、時間・tokenの平均、分散、標準偏差、採用query_ids | `calculate()` → `main()` → common.write_csv() |
| `statistics.json` | 上記rows、集計定義、計画・出典情報 | `analysis.main()` |
| `figures/accuracy.png` | HとNによるAccuracy比較 | `analysis.figures()` |
| `figures/elapsed_time.png` | query時間の平均±母標準偏差 | `analysis.figures()` |
| `figures/input_tokens.png` / `output_tokens.png` / `total_tokens.png` | token数の平均±母標準偏差 | `analysis.figures()` |
| `cache/matplotlib/` | matplotlib設定・フォント等のキャッシュ。科学データではない。内部ファイルは実行環境依存 | `analysis.figures()` がMPLCONFIGDIRを指定し、matplotlibが生成 |
| `timing.json` | 解析開始・終了・経過時間、API件数0 | `analysis.main()` |

`runs/` はこの解析では作らない。元の1 sampleは03の完了版 `records.jsonl` にあり、ここには条件×Nの集計だけを作る設計。

## READMEと再実行時の既存出力チェック

現在の `analysis.main()` はresults直下で `.gitkeep` と `dry_run` 以外の既存項目があれば停止するため、今回追加した `README.md` も初回の本解析実行を止める対象になる。依頼どおりコード・既存データは変更していない。将来実行する際は、このREADMEを許容する方針を別途決める必要がある。
