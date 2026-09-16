# 01_prompt_preview: resultsの構成とデータの読み方

GPT-3.5用プロンプトのプレビュー。2タスク × 4 metric × H5 × N1＝8 query。APIを送信しないため、応答・採点結果はない。02が送信前にこのmanifestとqueryの同一性を照合する。

2026-09-16確認。既存ファイルとコードを読み取り、READMEだけを追加した。実験・解析は再実行していない。構成と件数は確認時点の保存内容。

## ディレクトリ構成

```text
results/
  .gitkeep
  .started.json
  3.5__MountainCar-v0__last-action__H5.txt
  3.5__MountainCar-v0__last-state__H5.txt
  3.5__MountainCar-v0__next-action__H5.txt
  3.5__MountainCar-v0__next-state__H5.txt
  3.5__Pendulum-v1__last-action__H5.txt
  3.5__Pendulum-v1__last-state__H5.txt
  3.5__Pendulum-v1__next-action__H5.txt
  3.5__Pendulum-v1__next-state__H5.txt
  manifest.csv
  manifest.json
```

`results/` がプレビューの保存先。子ディレクトリおよび `runs/` はない。8つのtxtがそれぞれ1 sampleの入力に当たる。生成経路は [run.py](../run.py) → `common.run()` → `_run()` のpreview分岐。

## 実在するファイルの説明

| ファイル名（実行世代・batch内でも同じ役割） | 内容 | 生成元 |
| --- | --- | --- |
| `.gitkeep` | 空ディレクトリをGitで保持するための管理用ファイル。実験データではない。 | リポジトリ管理用。run.pyの生成物ではない。 |
| `.started.json` | 実行開始のUTC時刻。二重実行・上書きを防ぐマーカー。完了を示すものではない。 | [common.py](../../common.py) の _run()、06のbatchは split_execution.run()。 |
| `3.5__MountainCar-v0__last-action__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__MountainCar-v0__last-state__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__MountainCar-v0__next-action__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__MountainCar-v0__next-state__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__Pendulum-v1__last-action__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__Pendulum-v1__last-state__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__Pendulum-v1__next-action__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `3.5__Pendulum-v1__next-state__H5.txt` | この条件のSYSTEM MESSAGEとUSER MESSAGE全文。API応答ではない。 | [common.py](../../common.py) の _run() のpreview分岐 → write_text()。 |
| `manifest.csv` | manifestのqueriesを1 query＝1行で保存。system_prompt / user_prompt本文を除く識別・条件・出典情報。JSON全体のCSV版ではない。 | [common.py](../../common.py) の save_manifest() → write_csv()。 |
| `manifest.json` | 実行計画。conditions、queries、モデル、H、Nに相当する件数、元episodeのパスとSHA-256、query_id、query_index、ordinal、プロンプト全文・hash、コード由来情報、予定API件数など。実績ではない。 | [common.py](../../common.py) の make_plan() → save_manifest()。06では split_execution.py が選択・統合情報を追加。 |

## 条件名とデータの単位

`<model_alias>__<task>__<metric>__H<履歴長>` が1条件。aliasの対応は `3.5=gpt-3.5-turbo`、`sol=gpt-5.6-sol`、`terra=gpt-5.6-terra`、`luna=gpt-5.6-luna`。metricは `next-action` / `last-action` / `next-state` / `last-state`。実際の質問はmanifestの `question_name` とプロンプト本文で確認する。

- Hはプロンプトに含める履歴ステップ数。upstreamの `history_size` はH−1なので、H5に対し4でも矛盾しない。
- Nは条件あたりのquery数。1 sampleを1 query（1回の質問対象）として読む。新しく独立したepisodeをN本実行したという意味ではない。入力episodeをパス順に探索し、有効queryの先頭N件を選ぶ。
- `episode_000/` はその条件で最初に処理した入力episodeの保存先。元のepisode名ではなく、複数queryを含められる。
- `query_000/` は06のAPIエラー継続方式で1 queryずつ呼び出す場合の条件内連番。元データの時刻indexではない。
- JSONLは1行がJSONオブジェクト。`ordinal` は条件内の0始まり連番、`query_index` は元episode上の対象index。`history_start` 以上 `history_end` 未満が履歴範囲。

元の状態・行動の軌跡はリポジトリの `data/llmx_data/` にある。ここでいうraw応答はモデル実行の出力であり、元の軌跡データ自体ではない。元ファイルは `manifest.json` / `records.jsonl` の `episode_path` と `episode_sha256` で追える。

## runs配下の4ファイルと、1 sampleの読み方

以下の4ファイルは [common.py](../../common.py) の `execute_plan()` → `invoke_cli()` がupstream CLIを呼び、[cli.py](../../../upstream/LLM-Xavier/llm_x/cli.py) の `_write_result()` が保存する。06のmerged内はこれらのコピーで、再実行の出力ではない。

| ファイル | 保持しているデータ | 内容の生成元 |
| --- | --- | --- |
| `run.json` | 保存時刻、Python/NumPy/llm_x version、backend/model、入力SHA-256、system_prompt、raw_responses_stored | `cli.py:_write_result()` |
| `config.effective.json` | 実際に使った評価設定。task_name、metric、question_name、history_size、max_queries、閾値、bin数、prompt保存設定、model、入力hash等 | `EvaluationConfig` を `cli.py:_write_result()` が辞書化・追加 |
| `metrics.json` | このleaf内queryだけの集計。query_count、parsed_count、ignored_count、exact_matches、parse_rate、完全一致率、要素平均正解率 | [evaluation.py](../../../upstream/LLM-Xavier/llm_x/evaluation.py) の `_summarize()` |
| `predictions.jsonl` | 1行＝1 queryの採点。index、履歴範囲、prompt、status、ground_truth、解釈できたprediction、vectorのelement_accuracy、解釈失敗時のparse_error | `evaluation.py:_score_response()` / `evaluate_episode()` |

`run.json` の `raw_responses_stored=false` はこの4ファイルに応答全文を保存しないという意味。本文・usage・通信時間を調べるときは上位の `responses.jsonl` / `records.jsonl` を読む。

1 sampleは `records.jsonl` の1行を起点にする。`query_id` でmanifest・requests・responsesを結び、`attempt_id` でAPI試行を結ぶ。採点ファイルは `upstream_output` が示すディレクトリにあり、`predictions.jsonl` の `index` とrecordの `query_index` を照合する（別条件・別episodeの同じindexを混同しない）。

`prediction` / `ground_truth` は採点用の値で、常に物理量の生値ではない。MountainCarのactionは離散行動、Pendulumのactionはupstreamのbin化した値、stateは各次元の増減方向。`status=match/mismatch` は解釈できた応答の一致/不一致、`ignored` は解釈不可、`failed` は実行失敗。Joint質問でも保存済みのupstream採点は同じコード経由であり、応答中の連続値そのものを評価した結果とは限らない。

`legacy_compatible_match_rate`（summaryのAccuracy）はactionでは全query、stateでは解釈できたqueryを分母にする。`query_elapsed_seconds` はquery処理、`request_elapsed_seconds` はAPI試行時間の合計、`input_tokens/output_tokens` は最終応答usage由来。再利用行は `reused`、`source_record`、`api_request_made` 等で由来を確認する。採点済み件数は正解件数と区別する。


## 生成コードの参照先

[実験のrun.py](../run.py)、[common.py](../../common.py)、[前処理llmx_original.py](../../../preprocessing/llmx_original.py)、[upstream CLI](../../../upstream/LLM-Xavier/llm_x/cli.py)、[upstream採点](../../../upstream/LLM-Xavier/llm_x/evaluation.py)。README自身は構成説明として追加したもので、run.pyの出力ではない。
