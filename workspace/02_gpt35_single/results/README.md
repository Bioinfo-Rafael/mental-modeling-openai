# 02_gpt35_single: resultsの構成とデータの読み方

GPT-3.5のpilot実行。2タスク × 4 metric × H5 × N1＝8 query。現存summaryはcomplete、API試行8件。

2026-09-16確認。既存ファイルとコードを読み取り、READMEだけを追加した。実験・解析は再実行していない。構成と件数は確認時点の保存内容。

## ディレクトリ構成

```text
results/
  .gitkeep
  .started.json
  RESULTS.md
  dry_run/
    manifest.csv
    manifest.json
  manifest.csv
  manifest.json
  records.jsonl
  requests.jsonl
  responses.jsonl
  run.log
  runs/
    <condition>/（mergedでは先にsource_XXX/）
      <episode_XXX または query_XXX>/
        run.json / config.effective.json / metrics.json / predictions.jsonl
  summary.csv
  summary.json
```

`results/` は本実行、`dry_run/` は送信前計画（manifest 2形式のみ）、`runs/` は条件別のupstream採点出力。8条件に各 `episode_000/` があり、それぞれ1 sample・4ファイルを保持する。生成経路は [run.py](../run.py) → `common.run()` → `execute_plan()`。

## 実在するファイルの説明

| ファイル名（実行世代・batch内でも同じ役割） | 内容 | 生成元 |
| --- | --- | --- |
| `.gitkeep` | 空ディレクトリをGitで保持するための管理用ファイル。実験データではない。 | リポジトリ管理用。run.pyの生成物ではない。 |
| `.started.json` | 実行開始のUTC時刻。二重実行・上書きを防ぐマーカー。完了を示すものではない。 | [common.py](../../common.py) の _run()、06のbatchは split_execution.run()。 |
| `RESULTS.md` | 既存ログを読んでまとめた日本語の結果レポート。raw dataではない。 | 本文に2026-09-10のread-only集計と記載。run.pyの自動生成対象ではなく、専用生成コード・作成者は確認できない。 |
| `manifest.csv` | manifestのqueriesを1 query＝1行で保存。system_prompt / user_prompt本文を除く識別・条件・出典情報。JSON全体のCSV版ではない。 | [common.py](../../common.py) の save_manifest() → write_csv()。 |
| `manifest.json` | 実行計画。conditions、queries、モデル、H、Nに相当する件数、元episodeのパスとSHA-256、query_id、query_index、ordinal、プロンプト全文・hash、コード由来情報、予定API件数など。実績ではない。 | [common.py](../../common.py) の make_plan() → save_manifest()。06では split_execution.py が選択・統合情報を追加。 |
| `records.jsonl` | 1 queryの条件・入力・応答・token数・時間・試行履歴・採点を結合したデータ。score、prediction、ground_truth、status、upstream_outputを参照できる。失敗行は例外情報を持ち、score等がないことがある。 | [common.py](../../common.py) の collect_scores()（成功・採点済み）、RecordingOpenAIBackend.complete()（失敗）。06のmergedは split_execution.merge_results() がコピー・由来情報を追加。 |
| `requests.jsonl` | API送信引数kwargs（model、messages、生成設定）、query_id、attempt_id、送信開始時刻など。通常1行＝1 API試行。再試行で同じquery_idが複数行になり得る。 | [common.py](../../common.py) の RecordingSession.backend_class() 内 recording_create()。送信直前に追記。 |
| `responses.jsonl` | 各試行の応答記録。raw_responseはSDK応答全体（choices、message、usage等）。assistant_text、usage、時刻、所要時間、例外も保持。API失敗時はraw_responseがnull。 | [common.py](../../common.py) の RecordingSession.backend_class() 内 recording_create() のfinallyで追記。 |
| `run.log` | upstream CLIのstdout/stderrと対象query_idsを追記した実行ログ。API応答本文の正本ではない。 | [common.py](../../common.py) の invoke_cli()。 |
| `summary.csv` | 条件ごとに1行。条件設定、件数、時間、metricsを平坦化した集計。summary.json全体をそのまま表にしたものではない。 | [common.py](../../common.py) の execute_plan() → write_csv()。 |
| `summary.json` | 実行状態、予定・処理・成功・失敗・再利用件数、API試行数、実行時間、条件別集計。successful_queriesは正解数ではなく採点処理が済んだ数（ignoredも含む）。 | [common.py](../../common.py) の execute_plan()。mergedは split_execution.merge_results() が統合状態を生成し、通常実行と項目が異なる。 |

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

## 代表sampleの具体的な位置

- `runs/3.5__MountainCar-v0__last-action__H5/episode_000/predictions.jsonl` の先頭行：`index=5`、履歴範囲 `[0, 5)`、`status=match`。同じleafの残り3ファイルがこの呼出しの設定・由来・集計。JSONL全体は1 sample。

## runsの実在条件とサンプル数

各行は実在する1条件。leaf数は保存ディレクトリ数、sample数は `predictions.jsonl` の行数合計。失敗して採点ファイルがないqueryは含めない。

| runsの位置 / 条件 | leafの形式 | leaf数 | sample数 |
| --- | --- | ---: | ---: |
| `runs/3.5__MountainCar-v0__last-action__H5` | episode_000 | 1 | 1 |
| `runs/3.5__MountainCar-v0__last-state__H5` | episode_000 | 1 | 1 |
| `runs/3.5__MountainCar-v0__next-action__H5` | episode_000 | 1 | 1 |
| `runs/3.5__MountainCar-v0__next-state__H5` | episode_000 | 1 | 1 |
| `runs/3.5__Pendulum-v1__last-action__H5` | episode_000 | 1 | 1 |
| `runs/3.5__Pendulum-v1__last-state__H5` | episode_000 | 1 | 1 |
| `runs/3.5__Pendulum-v1__next-action__H5` | episode_000 | 1 | 1 |
| `runs/3.5__Pendulum-v1__next-state__H5` | episode_000 | 1 | 1 |

## 生成コードの参照先

[実験のrun.py](../run.py)、[common.py](../../common.py)、[前処理llmx_original.py](../../../preprocessing/llmx_original.py)、[upstream CLI](../../../upstream/LLM-Xavier/llm_x/cli.py)、[upstream採点](../../../upstream/LLM-Xavier/llm_x/evaluation.py)。README自身は構成説明として追加したもので、run.pyの出力ではない。
