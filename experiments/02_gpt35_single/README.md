# Exp.2 — GPT-3.5 single-query pilot

Exp.1で確認したものと同じ8 queryを `gpt-3.5-turbo` に送り、応答・使用token・時間・公式スコアを確認します。**フラグなしではAPIを呼ばず、計画だけ保存します。**

実行済みpilotの時間・input/output tokens・精度・分散などは [結果まとめ：RESULTS.md](results/RESULTS.md) に整理しています（ローカルの生成結果。現在のresultsはGit ignore対象）。

## 1. 実行方法

既存のMental Modeling / LLM-Xavier用環境を有効化します。Notebookの `Python (mental-modeling)` と同じ環境です。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/02_gpt35_single/run.py
```

上は計画だけ確認するコマンドです。API keyは不要です。環境を新規作成する必要はありません。[01の環境説明](../01_prompt_preview/README.md)と同じ `.venv` を使います。

`results/dry_run/manifest.json` の `prerequisites` が `ready` であること、条件・件数・promptを確認します。Exp.1が不足・不一致の場合、通常はその理由をmanifestに記録します。計画が保存されたことだけでは、送信可能とは限りません。

実際に送る場合だけ、`OPENAI_API_KEY` を安全な方法で環境変数に設定したうえで、両方のフラグを指定します。キーをコードやCLI引数へ書かないでください。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/02_gpt35_single/run.py --execute --confirm-paid-api
```

全件成功・retryなしなら8 API試行です。`--execute` だけ、または `--confirm-paid-api` だけでは拒否されます。`--dry-run` という引数はなく、フラグなしがdry-runです。

既定の `--retries 0` は追加retryなし。明示的に増やすと、1 queryに複数のAPI試行が発生し得ます。既存の実行結果がある場合は上書き・自動resumeせず停止します。再実行前に結果を手動で退避し、再課金の可能性を確認してください。

> このREADMEの作成に伴う実験実行・dry-run・API送信は行っていません。以下は現行コードの説明です。

## 2. 入力元

| 入力 | 場所・用途 |
| --- | --- |
| 公式raw episodes | `data/llmx_data/offline_data/<dataset>/raw_transitions/<task>/episodes/*.npz` |
| Exp.1のpreview manifest | `experiments/01_prompt_preview/results/manifest.json`。送信予定queryとの一致確認 |
| 再現性の情報 | `configs/sources.json`、関連ソースのSHA256、Git revision |

rawからpromptを再生成します。**01の `.txt` を読み込んで送るのではありません。** 既存の [`dataset_adapters/llmx.py`](../../dataset_adapters/llmx.py) の `discover_episodes()` と [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) の `build_prompt_queries()` を利用します。

episodeをpath順に並べ、MountainCar/Pendulumの各metric・H条件で先頭の有効queryを1件選びます。rawの書換え・ダウンロードは行いません。

[`experiments/common.py`](../common.py) の `check_inputs()` が、01の8件全体とidentity・model ID・prompt SHA・履歴範囲・関連ソースhashなどを照合します。不一致なら有料送信を開始しません。送信時にも `verify_prompt()` で実際のpromptを照合します。

## 3. 出力先とファイルの見方

出力先は `experiments/02_gpt35_single/results/` です。正常実行後の概略は次のとおりです。

```text
results/
├── .gitkeep
├── dry_run/                 # dry-runした場合のmanifest.json/csv
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
    └── 3.5__MountainCar-v0__next-action__H5/  # 8条件のうち1つ
        └── episode_000/
            ├── run.json
            ├── config.effective.json
            ├── metrics.json
            └── predictions.jsonl
```

生成結果はGitのignore対象です。Exp.1のpreview結果を共有する例外とは異なります。

| ファイル | 中身・粒度 |
| --- | --- |
| `manifest.json/csv` | 計画。query一覧、source/prompt SHA、H、条件別予定件数。JSONにはprompt全文も入る |
| `.started.json` | 実行開始の記録。再実行による上書きを防ぐ |
| `requests.jsonl` | 1 API試行ごとの実kwargs（model・messages・temperatureなど） |
| `responses.jsonl` | 各試行の全SDK response、assistant text、usage、時刻、経過時間、例外 |
| `records.jsonl` | 1 queryごとの公式scoreと、request/raw response/usage/timingをまとめた記録。失敗queryの記録も含み得る |
| `summary.json` | 実験全体の完了状態・実績件数・時間と条件別の集計 |
| `summary.csv` | 条件別の件数・時間・Accuracyなど。正常終了なら8行 |
| `run.log` | 公式CLIの標準出力・エラー出力。秘密情報はredact |
| `runs/.../` の4ファイル | 公式CLIが生成する実効設定、スコア、予測などの標準出力 |

JSONLは1行ごとに独立したJSONです。`query_id` でqueryを、`attempt_id` でretryを含む個別試行を対応付けます。manifestの `api_requests_made=0` は計画時点の値で、**実行後の実績は `summary.json`** を確認します。

| 主要項目 | 意味 |
| --- | --- |
| `logical_queries` | 実際に処理したquery数。retryは別queryに数えない |
| `api_attempts` / `retry_attempts` | SDK create試行の総数 / 2回目以降の試行数 |
| `successful_queries` | 採点済みquery数。正解数ではなく、`ignored`も含む |
| `failed_queries` | backend処理などで失敗として記録されたquery数 |
| `status`（query） | `match`＝一致、`mismatch`＝不一致、`ignored`＝解析できなかった応答、`failed`＝実行失敗 |
| `prediction` / `ground_truth` | 公式parserの予測値 / 公式評価関数の正解値 |
| `input_tokens` / `output_tokens` / `total_tokens` | 最終的に利用したAPI responseのusage。tiktokenの概算ではない。未取得はnull |
| `query_elapsed_seconds` | query処理の経過時間。retry待ちなどを含む |
| `request_elapsed_seconds` | そのqueryのSDK create所要時間の合計 |

API試行が通信失敗したとき、サーバー到達・課金の有無までログだけで確定できるとは限りません。token統計は最終responseのusageであり、retry分の課金総量を保証しません。各試行のusageはrawログに残します。

エラー時は後続送信を止めます。取得済みprefixの応答があれば、APIを使わない同じCLIのreplayでスコアを回収し、`episode_000_partial/` のような場所に保存します。強制終了・安全性違反・ディスク障害では回収を保証せず、未完了結果を完了扱いしません。

### 3.1 実ファイルを調べた結果：出力元のファイル・関数

2026-09-10に、既存の `results/` をread-onlyで読み、保存処理と照合しました。実験を再実行したのではありません。今回の結果は8 query・8 API試行・retry 0・失敗0です。以下の件数・キー数は**今回保存されたファイルの実測値**で、将来の失敗・retry・SDK変更時まで固定のschemaを保証するものではありません。

出力元は大きく2つです。

- [`experiments/common.py`](../common.py)：計画、実送受信ログ、統合record、実験全体のsummaryを保存。
- [`upstream/LLM-Xavier/llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py)：`runs/<condition>/episode_000/` の4ファイルを保存。

次の表の `common.py` と `cli.py` は、それぞれ上記のファイルを指します。「作る関数」と「ファイルへ書く関数」を区別しています。

| 出力ファイル | 内容を作る／保存を呼び出す関数 | 実際の書込み経路 | 今回の形状・粒度 |
| --- | --- | --- | --- |
| `.started.json` | `common.py:run()` | `write_json()` → `write_text()` | JSON object 1個、1キー（開始時刻） |
| `manifest.json` | `common.py:make_plan()` が計画を作り、`run()` が実行mode等を追加 | `save_manifest()` → `write_json()` → `write_text()` | object 1個、16キー。`conditions` は8要素、`queries` は8要素 |
| `manifest.csv` | `common.py:save_manifest()` がplanのqueryからprompt本文2項目を除外 | `write_csv()` → `write_text()` | データ8行×23列、別にheader 1行 |
| `dry_run/manifest.json` | `common.py:run()` のdry-run分岐。計画は `make_plan()` | `save_manifest()` → `write_json()` → `write_text()` | object 1個、16キー。条件8・query 8。modeは `dry_run` |
| `dry_run/manifest.csv` | `common.py:save_manifest()` | `write_csv()` → `write_text()` | データ8行×23列、別にheader 1行 |
| `requests.jsonl` | `common.py:RecordingSession.backend_class()` が定義する `RecordingOpenAIBackend.complete()` 内の `recording_create()` | `append_jsonl()` | 8行＝8試行。各行object、7キー |
| `responses.jsonl` | 同じ `recording_create()`。SDK responseを `model_dump(mode="json")` で取得 | `finally` 節から `append_jsonl()` | 8行＝8試行。各行object、13キー |
| `records.jsonl` | 上記 `complete()` が応答・時間・usageを用意し、`common.py:collect_scores()` が公式scoreを結合 | `collect_scores()` → `append_jsonl()` | 8行＝8 query。今回の各行は48キー。入れ子object/listを含む |
| `summary.json` | `common.py:execute_plan()`。指標は `summarize_scored()` → 公式 `_summarize()` | `write_json()` → `write_text()` | object 1個、16キー。`conditions` は8要素、各条件は24キー |
| `summary.csv` | `common.py:execute_plan()` が条件別summaryを平坦化 | `write_csv()` → `write_text()` | データ8行×31列、別にheader 1行 |
| `run.log` | `common.py:invoke_cli()` が公式CLIのstdout/stderrを捕捉 | `finally` 節の `Path.open("a")` → `handle.write()` | 通常のテキスト。表ではなく、公式CLIの出力を順に連結 |
| `runs/.../run.json` | `cli.py:_write_result()` | `_atomic_json()` → `_atomic_text()` | 各directoryにobject 1個、10キー。全8ファイル |
| `runs/.../config.effective.json` | `cli.py:_write_result()` が `asdict(result.config)` にbackend/model/data hashを追加 | `_atomic_json()` → `_atomic_text()` | 各objectは18キー。全8ファイル |
| `runs/.../metrics.json` | `llm_x/evaluation.py:evaluate_episode()` → `_summarize()` が指標を作り、`cli.py:_write_result()` が保存 | `_atomic_json()` → `_atomic_text()` | 各objectは9キー。全8ファイル |
| `runs/.../predictions.jsonl` | `llm_x/evaluation.py:_score_response()` と `evaluate_episode()` がrecordを作り、`cli.py:_write_result()` が保存 | `_atomic_text()` | 各ファイル1行＝1 query、全8ファイル。行のキー数は7または8 |
| `.gitkeep` | 実験実行による出力ではない | repositoryで追跡する空ファイル | 0 bytes |

`requests.jsonl`、`responses.jsonl`、`records.jsonl`、`run.log` は、最初に `common.py:execute_plan()` が `write_text(..., exclusive=True)` で空ファイルを作り、その後に追記します。失敗queryの場合は、`complete()` の例外処理から直接 `records.jsonl` に失敗recordを書きます。今回の8行は全て通常の採点済みrecordです。

公式の `_atomic_text()` は一時ファイルへ書いてから置き換えます。共通のJSONL writerは追記ごとにflush/fsyncします。これらは保存方法の違いで、promptや採点定義の違いではありません。

### 3.2 「形状」の意味：JSONは行列とは限らない

CSVの「8×23」は行数×列数です。一方、JSONは `dict`、JSONLは行ごとに独立した `dict` であり、NPZのような数値配列の `shape` はありません。読み込んだ後のPython型で、`dict` / `list` / `str` / `int` / `float` / `bool` / `None` を区別すると構造が分かります。

JSONLを全行読み込めば `list[dict]` になりますが、ファイル自体は `[...]` で囲まれたJSON配列ではありません。入れ子のlist/dictを1列に保持したDataFrameと、入れ子を展開したDataFrameでは列数が異なります。

#### manifest：条件一覧とquery一覧

生成元：[`experiments/common.py`](../common.py) の `make_plan()` / `save_manifest()`。以下は型と要素数を示す説明図です。

```text
manifest.json: dict（16キー）
├── provenance: dict（5キー）
│   └── source/codeのhashやrevision（数値配列ではない）
├── conditions: list[dict]（長さ8）
│   └── 各条件: dict（11キー）
│       └── evaluation_config: dict（公式EvaluationConfigの15項目）
├── queries: list[dict]（長さ8）
│   └── 各query: dict（25キー）
│       ├── system_prompt: str
│       ├── user_prompt: str
│       ├── H / query_index / history_start等: int
│       └── model / task / episode_path / SHA等: str
└── mode / planned_api_requests等: scalar
```

`conditions` は1条件1要素、`queries` は1 query 1要素です。Exp.2ではN1なので両方8要素ですが、一般には同じ長さとは限りません。`manifest.csv` は `queries` の25項目から `system_prompt` と `user_prompt` を除いた23列です。

CSVの23列は次の順です（出典：`save_manifest()` と実ファイルheader）。

```text
condition_id, ordinal, model_alias, model, task, metric, question_name,
H, requested_H, upstream_history_size, actual_history_steps,
history_start, history_end, history_end_is_exclusive,
dataset, episode, episode_path, episode_sha256, query_index,
system_prompt_sha256, user_prompt_sha256, query_id, reuse_required
```

#### requests：1試行に2つのmessage

生成元：[`experiments/common.py`](../common.py) の内部関数 `recording_create()`。実kwargsを作る元は [`llm_x/backends.py`](../../upstream/LLM-Xavier/llm_x/backends.py) の `OpenAIChatBackend.complete()` です。

```text
requests.jsonl: 8行
└── 1行: dict（7キー）
    ├── query_id: str
    ├── attempt_id: str
    ├── attempt: int（今回1）
    ├── request_started_at_utc: str（UTC ISO8601）
    ├── api_request_made: bool
    ├── reused: bool
    └── kwargs: dict（3キー）
        ├── model: str
        ├── temperature: int（今回0）
        └── messages: list[dict]（長さ2）
            ├── [0]: {role: "system", content: str}
            └── [1]: {role: "user", content: str}
```

2つのmessageは1つのAPI requestにまとめて送ります。systemとuserを別々に2回送っているわけではありません。

#### responses：全文・usage・時間

生成元：[`experiments/common.py`](../common.py) の内部関数 `recording_create()`。今回の1行は13キーです。

| 項目 | Pythonで読んだ型・構造 |
| --- | --- |
| `query_id`, `attempt_id` | `str` |
| `attempt` | `int`。今回全て1 |
| `api_request_made`, `reused` | `bool` |
| `request_started_at_utc`, `response_received_at_utc`, `attempt_finished_at_utc` | UTC ISO8601の `str` |
| `request_elapsed_seconds` | `float`、秒 |
| `assistant_text` | 応答全文の `str` |
| `exception` | 今回全て `None`。例外時にはtype/messageを持つdictなどが入る |
| `raw_response` | 今回9キーの `dict`。`choices` は長さ1の `list[dict]`、その `message.content` が応答全文 |
| `usage` | 今回5キーの `dict`。`raw_response.usage` と同じ使用量を取り出したもの |

`usage` は `prompt_tokens`、`completion_tokens`、`total_tokens` の整数3項目と、`prompt_tokens_details`、`completion_tokens_details` の入れ子dictからなります。今回のdetailsはそれぞれ3キー、4キーです。SDKの追加項目を固定列へ削らず、rawのまま保持します。

#### records：request・response・scoreを1 queryへ統合

生成元：[`experiments/common.py`](../common.py) の `RecordingOpenAIBackend.complete()` と `collect_scores()`。

```text
records.jsonl: 8行
└── 1行: dict（今回48キー）
    ├── query identity / H / source / prompt情報
    ├── request: dict（messages/model/temperature）
    ├── raw_response: dict（全SDK response）
    ├── assistant_text: str
    ├── usage: dict
    ├── input_tokens / output_tokens / total_tokens: int
    ├── attempts: list[dict]（今回各1要素、試行のresponseログ）
    ├── score: dict（公式predictionsの1行と同じ内容）
    ├── prediction / ground_truth / status / element_accuracy
    ├── query/requestの時間・時刻・試行数
    └── upstream_output: str（対応するruns内directory）
```

`prediction` 等は参照しやすいよう `score` から上位にも取り出しています。したがって、`score` 内と上位を別queryとして数えません。`element_accuracy` がないscalar actionでも、上位の項目自体は存在し `None` になります。

### 3.3 `runs/` 内の4ファイルの詳細

保存元は全て [`upstream/LLM-Xavier/llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py) の `_write_result()` です。8条件に各1 episode directoryがあり、**8×4＝32ファイル**を確認しました。今回 `max_queries=1` なので、どの `predictions.jsonl` も1行です。

| ファイル | object内の構造・内容 |
| --- | --- |
| `run.json` | 10キー。`schema_version` はint、`raw_responses_stored` はbool、その他は実行時刻・version・model/backend・data hash・system promptのstr |
| `config.effective.json` | 18キー。公式 `EvaluationConfig` の15項目＋`data_sha256`・`backend`・`model`。履歴設定は `history_size=4`、query上限は `max_queries=1` |
| `metrics.json` | 9キー。query/parsed/ignored/match件数はint、各精度・parse率はfloatまたはnull。値の作成元は公式 `_summarize()` |
| `predictions.jsonl` | 各1行。`index`・履歴開始/終了はint、`prompt`・`status` はstr、予測/正解は下表のscalarまたはlist。vectorでは `element_accuracy` も入る |

`run.json` の `raw_responses_stored=false` は「公式の4ファイルは応答全文を保存しない」という意味です。追加の `responses.jsonl` / `records.jsonl` には応答全文があるため、Exp.2全体でraw応答が失われたという意味ではありません。

#### 予測・正解の形状はタスクとmetricで異なる

値の作成元：[`upstream/LLM-Xavier/llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) の `_score_response()`。以下は今回の全8出力を確認した結果です。shape表記はJSON listの長さを説明したもので、保存された値はNumPy配列ではありません。

| task | metric | `prediction` / `ground_truth` の型・形状 | `score` のキー数 |
| --- | --- | --- | ---: |
| MountainCar-v0 | next-action / last-action | `int` scalar。離散actionの値 | 各7 |
| MountainCar-v0 | next-state / last-state | `list[int]`、長さ2（形状相当 `(2,)`）。state各要素の変化方向 | 各8 |
| Pendulum-v1 | next-action / last-action | `list[int]`、長さ1（形状相当 `(1,)`）。連続actionのbin番号 | 各8 |
| Pendulum-v1 | next-state / last-state | `list[int]`、長さ3（形状相当 `(3,)`）。state各要素の変化方向 | 各8 |

state予測は生の次状態ベクトルの回帰値ではなく、公式の変化方向ラベルです。Pendulum actionも生の制御値ではなくbin番号です。応答全文はどの場合も `str` であり、この表はparserを通した後の値です。

今回の `score` の共通7キーは `index`、`prediction`、`ground_truth`、`status`、`history_start`、`history_end_exclusive`、`prompt`。vectorでは `element_accuracy` が加わります。将来 `ignored` が発生した場合は `parse_error` が入り、`prediction` がないなど構造が変わり得ます。

### 3.4 summaryの形状と出力までの対応

生成・保存元：[`experiments/common.py`](../common.py) の `execute_plan()`。

`summary.json` は実験全体のdictで、`conditions` に8個の条件別dictを持ちます。各条件dictは24キーで、その中の `evaluation_config` は15キー、`metrics` は公式集計の9キーです。

`summary.csv` は各条件から `evaluation_config` と入れ子の `metrics` を取り除き、`metrics` の9項目を列へ展開します。したがって **24−2＋9＝31列**。実験全体の `experiment_elapsed_seconds` などはJSON側の上位情報で、CSVに全て入るわけではありません。

以下は保存経路の説明図です。`common.py` 内の関数を起点に、公式出力と追加出力がどこで分かれるかを示します。

```text
common.run()
├─ make_plan() → save_manifest() → manifest.json / manifest.csv
├─ write_json() → .started.json
└─ execute_plan()
   ├─ invoke_cli() → 公式cli.main() → _evaluate()
   │  ├─ evaluate_episode()
   │  │  ├─ Recording backend.complete() → super().complete()
   │  │  │  └─ recording_create() → requests.jsonl / responses.jsonl
   │  │  ├─ _score_response() → queryごとのscore（メモリ上）
   │  │  └─ _summarize() → episodeのmetrics（メモリ上）
   │  └─ cli._write_result() → runs内の4ファイル
   │  → 捕捉したstdout/stderrをrun.logへ追記
   ├─ collect_scores() → 公式predictionsと記録済み応答を結合 → records.jsonl
   └─ summarize_scored() → 公式_summarize() → summary.json / summary.csv
```

`summary` はepisode単位のAccuracyの単純平均ではなく、条件内のscoreを集めて公式 `_summarize()` に渡した結果です。今回のN1では条件とepisodeの粒度が一致しますが、Nを増やして複数episodeを使う場合は別の粒度になります。

## 4. `run.py` の中身

出典：[`experiments/02_gpt35_single/run.py`](run.py)。

```python
EXPERIMENT = common.Experiment(
    name="02_gpt35_single", models=("3.5",),
    tasks=common.TASKS, metrics=common.METRICS, histories=(common.PILOT_H,), n=1,
    preview_from="01_prompt_preview",
)
```

型の定義は [`experiments/common.py`](../common.py) の `Experiment` dataclassです。上は型定義ではなく、条件を入れたインスタンスの生成です。生成だけではAPIを呼びません。`frozen=True` により属性の再代入を防ぎます。

同じ `run.py` の次の部分が、起動時に共通関数を呼びます。

```python
if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
```

`common.run()` は `experiments/common.py` の `run(spec, argv=None)`。`EXPERIMENT` を `spec` として受け取り、フラグによって計画保存か有料実行かを分けます。別実験を自動実行するものではありません。先頭のpath設定はrepositoryのimportを可能にし、`sys.dont_write_bytecode=True` は公式ソースへのpycache書込を防ぎます。

## 5. 各設定がどこで使われるか

指定元は [`run.py`](run.py)。処理側の出典は、以下の各引用に示したファイル・関数です。

| 設定 | 値 | 主な使用場所・意味 |
| --- | --- | --- |
| `name` | `02_gpt35_single` | `common.py:run()` の出力先、manifestの実験名 |
| `models` | `("3.5",)` | `common.py:make_plan()` で `MODELS` により `gpt-3.5-turbo` へ解決 |
| `tasks` | MountainCar-v0 / Pendulum-v1 | `common.py:TASKS`、`make_plan()` のepisode選択 |
| `metrics` | next/last-action、next/last-state | `common.py:METRICS`、`QUESTIONS` による質問選択 |
| `histories` | `(5,)` | `common.py:PILOT_H`。01と共有する実際の履歴長 |
| `n` | `1` | `common.py:make_plan()` で各条件の先頭1 queryを選択 |
| `preview` | `False`（既定） | 01のpreview分岐ではなく、フラグでdry-run/有料実行を分岐 |
| `preview_from` | `01_prompt_preview` | `common.py:check_inputs()` で01のmanifestとの一致検査 |
| `reuse_from` | `None`（既定） | 過去応答の再利用はしない |

条件数は `1 model × 2 tasks × 4 metrics × 1 H = 8`、各 `n=1` で8 queryです。

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
question = QUESTIONS[task][metric]
upstream_h = history_size(H)
```

`history_size()` も同ファイルにあり、H=5を公式パラメータ4へ変換します。履歴範囲と `Step ...:` 行数が実際に5であることを検査します。`QUESTIONS` はMountainCarのactionにdiscrete版、Pendulumのactionにcontinuous_bins版を指定します。4 familyの正式な質問名は [01の対応表](../01_prompt_preview/README.md) を参照してください。

出典：[`experiments/common.py`](../common.py) の `check_inputs()` 内。

```python
source_name = spec.preview_from or spec.reuse_from
```

今回は01が確認元になります。`preview_from` は「過去の応答を再利用」ではなく「過去に確認したpromptと同じかの検査」です。

出典：[`experiments/common.py`](../common.py) の `run()` 内（抜粋）。

```python
if not spec.preview and not args.execute:
    plan["mode"] = "dry_run"
    # manifestを保存してreturnする
```

実行が許可された場合は同ファイルの `execute_plan()` → `invoke_cli()` が公式CLIを呼びます。Loggingは `RecordingSession.backend_class()` が返す `OpenAIChatBackend` のsubclassで追加し、実処理は `super().complete()` を通します。独自prompt/parser/scorerへの置換ではありません。

集計は [`experiments/common.py`](../common.py) の `summarize_scored()` → [`upstream/LLM-Xavier/llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) の `_summarize()` です。primary `Accuracy` は `legacy_compatible_match_rate`。actionは全query数、stateはparsed数が分母で、parsed数0のstate Accuracyはnullです。

## 6. 呼び出し順

```text
02_gpt35_single/run.py
└─ common.run(EXPERIMENT)
   ├─ make_plan() → discovery → build_prompt_queries() → 公式prompt関数
   ├─ check_inputs() → 01のmanifestとの照合
   ├─ フラグなし → dry_run/にmanifest保存 → 終了（API 0）
   └─ 二重フラグ＋前提条件OK
      ├─ 既存結果の検査・manifest保存
      └─ execute_plan() → invoke_cli()
         └─ llm_x.cli.main() → _evaluate() → EvaluationConfig → _backend()
            └─ evaluate_episode() → Recording backend.complete()
               └─ super().complete() → SDK createを記録
            → 公式parser/scorer → 標準出力保存
         → collect_scores() → records/summary保存
```

公式CLIの出典は [`llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py)、評価処理は [`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py)、送信処理は [`llm_x/backends.py`](../../upstream/LLM-Xavier/llm_x/backends.py) です。結果を確認した後、必要に応じて [Exp.3](../03_gpt35_history_n30/README.md) へ進みます。
