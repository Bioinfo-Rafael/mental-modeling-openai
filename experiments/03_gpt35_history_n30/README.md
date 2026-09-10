# Exp.3 — GPT-3.5 H sweep、各条件N=30

MountainCar/Pendulumの4 metricについて履歴長H=5/10/20/30を比較します。**32条件×30 query＝960 query**を計画します。フラグなしでは送信せず、計画だけ保存します。

取得済み結果のAction Matching Rate / State-change Accuracyを作図する手順は [analysis/README.md](analysis/README.md)、生成した図の解釈は [analysis/RESULT.md](analysis/RESULT.md) を参照してください。こちらの解析は固定seedでN10 ⊂ N20 ⊂ N30を抽出し、APIには送信しません。

## 1. 実行方法

既存のLLM-Xavier用環境（Notebookでは `Python (mental-modeling)`）を使います。[01の環境説明](../01_prompt_preview/README.md)と同じ `.venv` で、新規作成は不要です。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/03_gpt35_history_n30/run.py
```

上はdry-runで、API key不要・API 0です。`results/dry_run/manifest.json/csv` の件数・prompt・sourceを確認してください。`--dry-run` オプションではなく、フラグなしがdry-runです。

有料実行は、ユーザーが `OPENAI_API_KEY` を環境変数に設定し、費用と条件を確認した後に限ります。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/03_gpt35_history_n30/run.py --execute --confirm-paid-api
```

全件成功・既定 `--retries 0` なら960 API試行です。どちらか一方のフラグだけでは実行できません。retryを明示的に増やすとAPI試行数は増えますが、N30自体は変わりません。

先に [Exp.2のpilot](../02_gpt35_single/README.md) を確認する運用を想定しています。ただし、**現行コードはExp.2完了を必須条件として検査しません**。03だけでも二重フラグがあれば送信へ進みます。

既存結果がある場合、通常の実行コマンドは上書き・再送を拒否します。途中再開には以下の `--resume` を使用してください。結果の削除・手動退避は不要です。

### 途中から再開する方法

まず、追加送信なしで再開計画を確認します。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/03_gpt35_history_n30/run.py --resume
```

`results/dry_run/manifest.json` の `planned_api_requests`（これから送る件数）、`resume_cached_queries`（保存済み応答の再利用数）、`resume_uncertain_queries`（送信を試みたが応答が保存されていないquery）を確認します。

今回の停止状態では、960件中30件を再利用し、残り930件を送信する計画です。930件にはタイムアウトした1件の再送と、未送信929件が含まれます。タイムアウトはサーバー側で未処理だったことを保証せず、再送すると重複課金の可能性があります。その1件を再送してよい場合は、次のコマンドを実行します。

```bash
python experiments/03_gpt35_history_n30/run.py \
  --resume --retry-uncertain --execute --confirm-paid-api
```

`OPENAI_API_KEY` は初回と同様に環境変数へ設定してください。`--retry-uncertain` がない場合、結果不明の送信が1件でもあれば送信前に停止します。結果不明の送信がない場合は、このフラグは不要です。これは今回の再送への確認であり、新しいtimeoutを自動retryする指定ではありません。`--retries` は従来どおり既定0、API timeoutも既定60秒です。

再び停止しても同じ手順で再開できます。初回の結果は `results/` に残し、再開1回目は `results/resumes/0001/`、2回目は `0002/` のように保存します。**最新の結果は番号が最大のフォルダの `summary.json` / `records.jsonl` を確認してください。** 元の `results/summary.json` は初回停止時の記録のままです。フォルダを削除・改名しないでください。

再開の実装は [common.py](../common.py) の `prepare_resume()` → `execute_plan()` → `RecordingSession` です。

- 全世代のmanifest、raw hash、prompt、モデル、公式preprocessing/scoringのhash、保存requestのパラメータを照合し、不整合・壊れたログがあれば送信前に停止します。
- 保存済みresponseは公式CLIへローカルで返して再採点します。APIへ再送しません。採点前に中断しても、`responses.jsonl` に応答が保存済みなら回収します。
- 最新世代の `records.jsonl` は再利用分も含めた結果です。全件完了なら960件になります。Exp.4は `latest_results()` により最新世代を自動で読みます。最新が未完了なら分析を拒否します。
- `summary.json` のAPI試行数・全体時間、`requests.jsonl` / `responses.jsonl` の実送信は、その世代の分だけです。累積試行数・課金usageは初回と全世代の実送信ログを合算します。再利用行のusageを再課金として足してはいけません。今回、追加retryなしで完了した場合、累積試行は初回31＋再開930＝961回です。
- 再利用queryの `query_elapsed_seconds` / `request_elapsed_seconds` は元の実測を引き継ぎ、ローカル再利用の時間は `replay_elapsed_seconds` に分離します。採点前中断でquery時間が残っていない場合だけAPI試行時間で回収し、`query_time_recovered_from_request=true` と明記します。
- `.execution.lock` のOSロックで同じ実験の有料実行を同時起動できないようにします。通常の終了・プロセス停止でロックは解放されるので、ファイルを手動削除する必要はありません。

この変更の検証はローカルテストと再開dry-runのみで、追加API送信は行っていません。現在 `--resume` 対応はExp.3のみです。

## 2. 入力元

```text
data/llmx_data/offline_data/<dataset>/raw_transitions/<task>/episodes/*.npz
```

既存 [`dataset_adapters/llmx.py`](../../dataset_adapters/llmx.py) の `discover_episodes()` で探し、episode path順に並べます。各条件について [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) の `build_prompt_queries()` が生成する有効queryを先頭から30件取得します。

1 episodeで不足すれば次へ進みます。例えば20件＋10件でもN30です。raw episodeは丸ごと公式CLIへ渡し、`--max-queries` でそのepisodeから使うprefix件数を指定します。rawを短く切って上書きする処理ではありません。

ランダム抽出はしません。**Hを変えると公式の有効index範囲も変わるため、異なるHで同じtarget indexを固定して比較する設計ではありません。** 履歴がepisode境界を跨ぐこともありません。

01/02のmanifest・応答は読みません。したがって02と重なるqueryがあっても、03では新規送信します。Git revision、`configs/sources.json`、関連ソースhashは再現性の記録に使います。raw取得・raw書換えは行いません。

## 3. 出力先とファイルの見方

すべて `experiments/03_gpt35_history_n30/results/` へ保存します。

```text
results/
├── dry_run/                       # 計画確認時のmanifest.json/csv
├── .started.json
├── manifest.json
├── manifest.csv
├── requests.jsonl
├── responses.jsonl
├── records.jsonl
├── run.log
├── summary.json
├── summary.csv
└── runs/
    └── 3.5__Pendulum-v1__next-action__H20/  # 条件例
        ├── episode_000/
        │   ├── run.json
        │   ├── config.effective.json
        │   ├── metrics.json
        │   └── predictions.jsonl
        └── episode_001/           # 次のepisodeも必要な場合のみ
```

`episode_000` はその条件で使ったepisodeの順番であり、元NPZのファイル名・episode IDではありません。実sourceはmanifestやrecordsの `episode_path` で確認します。生成結果はGitのignore対象です。

| ファイル | 中身 |
| --- | --- |
| `manifest.json/csv` | 960件の選択計画。JSONは32条件の予定件数とprompt全文も保持 |
| `requests.jsonl` / `responses.jsonl` | API試行ごとの実kwargs / 全response・usage・時間・例外。`query_id` と `attempt_id` で対応 |
| `records.jsonl` | query-levelの公式scoreとraw応答・token・時間。正常完了時は960件。Exp.4の入力 |
| `summary.json` | 全体の完了状態、logical/API/retry/成功/失敗件数、実験時間、条件別集計 |
| `summary.csv` | 条件別集計。正常完了時は32行 |
| `runs/.../` | episode単位の公式CLI標準出力。条件全体のN30集計は上位summaryで確認 |
| `.started.json` / `run.log` | 実行開始記録 / 公式CLIの出力・エラー |

manifestは計画時の記録です。実API件数・完了状態は `summary.json` を見ます。`successful_queries` は採点済み件数であり、正解数ではありません。`ignored`も採点済みに含まれ、通信などの失敗と区別します。

### Accuracyの読み方

出典：[`experiments/common.py`](../common.py) の `summarize_scored()` と [`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) の `_summarize()`。

| 項目 | 定義 |
| --- | --- |
| `Accuracy` / `legacy_compatible_match_rate` | primary指標。actionは一致数÷全query数、stateは一致数÷parsed数 |
| `exact_match_rate_all_queries` | 一致数÷全query数 |
| `exact_match_rate_parsed_queries` | 一致数÷parsed数。parsedが0ならnull |
| `mean_element_accuracy_parsed_queries` | parsedのうち要素精度を持つvector応答の平均。なければnull |
| `parse_rate` | parsed数÷全query数 |
| `ignored_count` | 解析できなかった応答数 |

`input_tokens/output_tokens/total_tokens` は最終responseのusageで、tiktokenの見積もりではありません。欠損はnull。retryの過去usageはrawログを確認します。時間はquery、SDK create、condition、experimentで分けて記録します。

APIエラー時は後続送信を停止します。取得済みprefixはAPIなしのCLI replayで `episode_000_partial/` などへ採点を回収しますが、安全性違反・強制終了・ディスク障害時は保証しません。未完了の03はExp.4が拒否します。復旧のために勝手に追加送信する処理はありません。

## 4. `run.py` の中身

出典：[`experiments/03_gpt35_history_n30/run.py`](run.py)。

```python
EXPERIMENT = common.Experiment(
    name="03_gpt35_history_n30", models=("3.5",),
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=30,
)
```

`common.Experiment` は [`experiments/common.py`](../common.py) で定義したdataclassです。上のコードは実験条件のインスタンス生成で、API実行ではありません。

同じ `run.py` にある `common.run(EXPERIMENT)` が、起動時に `common.py` の `run(spec, argv=None)` を呼びます。`spec` がこの設定を受け取ります。先頭の `sys.path` 設定でrepositoryをimport可能にし、`dont_write_bytecode` で公式ソースへのpycache書込を防ぎます。他の実験を自動実行することはありません。

## 5. 各設定がどこで使われるか

| 設定 | 値 | 使用箇所・意味 |
| --- | --- | --- |
| `name` | `03_gpt35_history_n30` | `common.py:run()` で `experiments/<name>/results/` を決める |
| `models` | `("3.5",)` | `common.py:MODELS` から `gpt-3.5-turbo` に解決。aliasは文字列 |
| `tasks` | MountainCar-v0 / Pendulum-v1 | `common.py:TASKS`、`make_plan()` のtask別選択 |
| `metrics` | 4 family | `common.py:METRICS` と `QUESTIONS` による公式質問の指定 |
| `histories` | `(5, 10, 20, 30)` | `common.py:H_VALUES`。4つの履歴長条件を作る |
| `n` | `30` | `common.py:make_plan()` で条件ごとの選択数を30にする |
| `preview` | `False`（既定） | 二重フラグが揃った場合だけ有料実行 |
| `preview_from` / `reuse_from` | 両方 `None` | `common.py:check_inputs()` は過去結果を読まず空の辞書を返す |

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
for alias in spec.models:
    model = MODELS[alias]
    for task in spec.tasks:
        for metric in spec.metrics:
            for H in spec.histories:
                question = QUESTIONS[task][metric]
                upstream_h = history_size(H)
```

この直積が `1×2×4×4=32条件` です。`n=30` は各条件に適用され、retry数ではありません。`history_size()` は同じ `common.py` にあり、HをH−1に変換します。

| 実際の履歴H | 公式 `history_size` |
| --- | ---: |
| 5 | 4 |
| 10 | 9 |
| 20 | 19 |
| 30 | 29 |

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内（選択停止部分の抜粋）。

```python
if len(selected) == spec.n:
    break
```

選択queryは [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) の `build_prompt_queries()` から得ます。query index/history range/文面は公式関数を利用し、Pendulumのaction bin範囲などを独自に変更しません。

出典：[`experiments/common.py`](../common.py) の `summarize_scored()` 内。

```python
scores = [r["score"] for r in records if r["status"] in SCORED_STATUSES]
return _summarize(scores, metric) if scores else {}
```

複数episodeに分かれても、条件内の全scoreを公式 `_summarize()` へ渡します。episodeごとのAccuracyを単純平均する実装ではありません。

## 6. 呼び出し順

```text
03_gpt35_history_n30/run.py
└─ common.run(EXPERIMENT)
   ├─ make_plan() → 32条件、各条件の先頭30 queryを選ぶ
   │  └─ discover_episodes() → build_prompt_queries() → 公式prompt関数
   ├─ check_inputs() → 過去実験への依存なし
   ├─ フラグなし → dry_run/manifest保存 → 終了（API 0）
   └─ 二重フラグあり → execute_plan()
      ├─ 条件ごと・episodeごとに invoke_cli()
      │  └─ 公式cli.main() → _evaluate() → EvaluationConfig → _backend()
      │     └─ evaluate_episode() → Recording backend → super().complete()
      │        → 実送受信の記録 → 公式parser/scorer → 標準出力保存
      ├─ collect_scores() → query-level records
      └─ summarize_scored() → 公式_summarize() → 条件全体のsummary
```

CLI・評価・送信の出典はそれぞれ [`llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py)、[`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py)、[`llm_x/backends.py`](../../upstream/LLM-Xavier/llm_x/backends.py)。ログの細かい項目は [Exp.2の出力説明](../02_gpt35_single/README.md)、N別の再解析は [Exp.4](../04_sample_size_analysis/README.md) を参照してください。
