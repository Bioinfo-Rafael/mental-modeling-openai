# Exp.6 — Sol / Terra / Luna、全条件N=10

新3モデルで、MountainCar/Pendulum・4 metric・H=5/10/20/30を各10 query評価します。Exp.5の12 queryを再利用するため、**最終960 recordsに対して新規API queryは948件（316/model）**です。

## 1. 実行方法

既存のMental Modeling / LLM-Xavier用環境を有効化します。[01の環境説明](../01_prompt_preview/README.md)と同じ `.venv`（Notebookでは `Python (mental-modeling)`）で、新規作成は不要です。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/06_new_models_n10/run.py
```

フラグなしはdry-runでAPI 0。`results/dry_run/manifest.json/csv` に条件とquery一覧を保存します。`--dry-run` という引数はありません。API keyは不要です。

**Exp.5の結果との照合が全12件で通ることが必要**です。dry-runの `prerequisites` を確認してください。計画上は948件でも、前提が揃っていなければ実行可能という意味ではありません。

実際に送信する場合は、`OPENAI_API_KEY` を環境変数に設定し、条件・費用を確認したうえで両フラグを指定します。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/06_new_models_n10/run.py --execute --confirm-paid-api
```

全件成功・既定 `--retries 0` なら948 API試行です。両フラグが揃わない場合や、05の不足・不一致がある場合は送信を始めません。欠けた05の分を新規送信するfallbackはありません。

retryを明示的に増やした場合、その試行数はN10に含めず別記録します。通信失敗時のサーバー到達・課金の有無は、試行ログだけで断定できないことがあります。

既存の06の結果があれば上書き・自動resumeせず停止します。再実行する場合は手動退避と再課金の確認が必要です。このREADMEの作成では実験・dry-run・API送信を行っていません。

## 2. 入力元

| 入力 | 場所・用途 |
| --- | --- |
| 公式raw episodes | `data/llmx_data/offline_data/<dataset>/raw_transitions/<task>/episodes/*.npz`。新しい計画とpromptを生成 |
| Exp.5 manifest | `experiments/05_new_models_single/results/manifest.json`。再利用すべきidentity全体を照合 |
| Exp.5 summary | 同directoryの `summary.json`。全12件が採点済みで完了しているか確認 |
| Exp.5 records | 同directoryの `records.jsonl`。実request・全raw response・scoreなどを再利用 |
| 再現性情報 | 関連ソースhash、`configs/sources.json`、Git revision |

rawは既存 [`dataset_adapters/llmx.py`](../../dataset_adapters/llmx.py) の `discover_episodes()` で取得済みファイルを探します。path順に並べ、各条件の先頭10有効queryを選びます。promptは [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) の `build_prompt_queries()` 経由で公式関数が生成します。

1 episodeで不足したら次へ進みます。履歴はepisode境界を跨ぎません。ランダム抽出でも、rawを上書き加工する処理でもありません。

### Exp.5と何を照合するか

出典：[`experiments/common.py`](../common.py) の `check_inputs()`。

- 05の実験名と関連ソースの `semantics_sha256`。
- 再利用対象全12件の集合。重複・不足がないこと。
- model ID、task、metric、質問名、H、episode path/hash、query index、両prompt SHAからなるquery identity。
- 履歴開始/終了、実履歴長、公式history parameter、条件内の順番。
- 05が完了し、採点済みrecordが12件あること。
- raw responseのassistant text、保存score、prompt、index、取得元が整合すること。
- 保存された実requestのmodel・messages・temperatureなどが現行の公式backendの引数と一致すること。
- SDK responseをローカルで復元できること。

これらは**06の最初のAPI送信より前**に確認します。再利用時にも実kwargsを比較します。05の `match` だけでなく `mismatch` / `ignored` も整合していれば再利用します。正解だけを選ぶ処理ではありません。

01〜04の結果は入力にしません。03と同じなのはtask/metric/Hのgridであり、03のGPT-3.5応答は再利用しません。

## 3. 出力先とファイルの見方

出力先は `experiments/06_new_models_n10/results/` です。

```text
results/
├── dry_run/                 # 計画確認時のmanifest.json/csv
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
    └── sol__Pendulum-v1__next-action__H20/  # 条件例
        └── episode_000/
            ├── run.json
            ├── config.effective.json
            ├── metrics.json
            └── predictions.jsonl
```

結果はGitのignore対象です。Exp.5のファイルは変更しません。再利用したrecordも06のresultsへ含め、06だけで最終N10を確認できるようにします。

| ファイル | 主な内容 |
| --- | --- |
| `manifest.json/csv` | 全960 queryの計画。JSONでは96条件の件数、新規948・再利用12を記録 |
| `requests.jsonl` / `responses.jsonl` | 新規API試行の実kwargs/responseに加え、再利用であることを示す記録 |
| `records.jsonl` | 正常完了時、再利用12件＋新規948件＝960 query records |
| `summary.json` | 全体の完了状態、実logical/API/retry/成功/失敗/再利用件数と時間 |
| `summary.csv` | 3 models×32条件＝96行の条件別集計（正常完了時） |
| `runs/.../` | 公式CLIによるepisode別設定・指標・予測。再利用応答も同じ公式scorerに通す |
| `.started.json` / `run.log` | 開始記録 / 公式CLIの出力・エラー |

manifestは計画時点のスナップショットです。実API試行数は `summary.json` の `api_attempts` / `api_requests_made` を確認します。JSONLの行数をそのまま新規API件数とみなさないでください。

### 再利用recordの見分け方

出典：[`experiments/common.py`](../common.py) の `RecordingSession.backend_class()` 内。

| 項目 | 再利用queryでの値・意味 |
| --- | --- |
| `reused` | `true` |
| `api_request_made` | `false`。06ではこのqueryを新規送信していない |
| `source_experiment` | `05_new_models_single` |
| `source_record` | 05の `records.jsonl` とquery IDを示す参照文字列 |
| `api_attempts` / `retry_attempts` | 今回は両方0 |
| `raw_response` / `usage` | 元の応答と元のusageを保持。今回の新規課金tokenという意味ではない |
| `source_query_elapsed_seconds` | 05で記録した元query時間 |
| `query_elapsed_seconds` | 06でのreplay処理時間。05のAPI推論時間とは異なる |
| `request_elapsed_seconds` | 今回のSDK create実通信はないため0 |

requests/responsesの再利用行には `attempt=0`、`attempt_id="<query_id>:reuse"` を付けます。新規API試行のattemptは1から始まります。`records.jsonl` のusageを全件合計すると、05のusageも含む最終datasetの量になり、06だけの追加課金量とは異なります。

Accuracyは公式 `legacy_compatible_match_rate`。actionは全件、stateはparsed件数を分母にし、all/parsed/element精度・parse率も分けて保存します。API失敗をmismatchと置き換えたり、欠損usageを0で捏造したりしません。

エラー時は以降の送信を止め、取得済みprefixは可能な場合だけAPIなしで採点回収します。部分出力は `episode_000_partial/` などに保存します。強制終了・安全性違反・ディスク障害では回収を保証しません。再利用があることと、06の途中結果を自動resumeできることは別です。

## 4. `run.py` の中身

出典：[`experiments/06_new_models_n10/run.py`](run.py)。

```python
EXPERIMENT = common.Experiment(
    name="06_new_models_n10", models=common.NEW_MODELS,
    tasks=common.TASKS, metrics=common.METRICS, histories=common.H_VALUES, n=10,
    reuse_from="05_new_models_single",
)
```

`Experiment` の型定義は [`experiments/common.py`](../common.py) にあります。上は条件のインスタンスを作って `EXPERIMENT` に代入する処理です。

同じ `run.py` の `common.run(EXPERIMENT)` で、`common.py` の `run(spec, argv=None)` に渡します。`spec` にこの設定が入ります。インスタンスを作るだけではAPIを呼ばず、`run()` が二重フラグ・入力・出力を検査します。05を自動実行して不足を埋める機能はありません。

先頭のpath設定はrepositoryをimport可能にし、`sys.dont_write_bytecode=True` は公式ソースへのpycache書込を防ぎます。

## 5. 各設定がどこで使われるか

| 設定 | 値 | 使用場所・意味 |
| --- | --- | --- |
| `name` | `06_new_models_n10` | `common.py:run()` の出力先とmanifestの実験名 |
| `models` | sol / terra / luna | `common.py:NEW_MODELS`、`MODELS`。各 `gpt-5.6-*` IDへ解決 |
| `tasks` | MountainCar-v0 / Pendulum-v1 | `common.py:TASKS` と `make_plan()` の対象選択 |
| `metrics` | 4 family | `common.py:METRICS` と `QUESTIONS` による質問選択 |
| `histories` | `(5, 10, 20, 30)` | `common.py:H_VALUES`。実際の履歴長。公式パラメータはH−1 |
| `n` | `10` | 最終的に各条件へ揃えるquery数。新規送信だけの件数ではない |
| `preview` | `False`（既定） | フラグなしはdry-run、二重フラグで有料実行 |
| `preview_from` | `None`（既定） | 01とのpreview照合はしない |
| `reuse_from` | `05_new_models_single` | `common.py:make_plan()` で必須再利用queryを指定し、`check_inputs()` で05を照合 |

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内（整形した抜粋）。

```python
row["reuse_required"] = bool(
    spec.reuse_from
    and task == "Pendulum-v1"
    and H == 20
    and row["ordinal"] == 0
)
```

再利用対象は「Pendulum、H20、各条件の先頭query」です。これは現在の05の条件に合わせた指定で、任意の過去実験を自動検索する汎用cacheではありません。05の条件を変更すると、06の前提検査が止まる可能性があります。

### 件数はどう計算するか

1 modelあたりの計算は次のとおりです（コード引用ではなく条件の内訳）。

| 条件群 | 条件数 | 最終query/条件 | 再利用/条件 | 新規API query数 |
| --- | ---: | ---: | ---: | ---: |
| Pendulum・H20・4 metrics | 4 | 10 | 1 | 4×9＝36 |
| 残りのtask/metric/H | 28 | 10 | 0 | 28×10＝280 |
| 1 model合計 | 32 | — | 計4 | 316 |
| 3 models合計 | 96 | — | 計12 | 948 |

最終件数は320/model、960合計です。05の12件をさらに足して「972個の異なるquery」と数えるものではありません。06の960件に05の12件が含まれます。

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
reused = sum(row["reuse_required"] for row in selected)
new = 0 if spec.preview else len(selected) - reused
```

この値を条件ごとに足してmanifestを作るため、948を送信件数としてhard-codeしていません。

### 同じ公式評価に通し、送信だけ省く

出典：[`experiments/common.py`](../common.py) の `RecordingSession.backend_class()` 内、`recording_create()` の再利用分岐（抜粋）。

```python
if cached:
    if kwargs != cached["request"]:
        raise SafetyStop("Replay API parameters differ from the saved request")
    from openai.types.chat import ChatCompletion
    latest.update(request=kwargs, raw_response=cached["raw_response"])
    return ChatCompletion.model_validate(cached["raw_response"])
```

再利用時は保存済みSDK responseを復元して返すため、実ネットワーク送信に進みません。呼出元は同じファイルの `super().complete()` であり、その後も公式 [`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) のparser/scorerを通ります。05のscoreを無条件にコピーして採点を省く実装ではありません。

## 6. 呼び出し順

```text
06_new_models_n10/run.py
└─ common.run(EXPERIMENT)
   ├─ make_plan() → 最終960 query、必須再利用12、新規948
   ├─ check_inputs() → 05の全12件を照合・cachedへ格納
   ├─ フラグなし → dry_run/manifest保存 → 終了（API 0）
   └─ 二重フラグ＋前提条件OK → execute_plan() → invoke_cli()
      └─ 公式cli.main() → _evaluate() → EvaluationConfig → _backend()
         └─ evaluate_episode() → Recording backend → super().complete()
            └─ recording_create()
               ├─ cachedあり → kwargs照合 → 保存response復元（新規API 0）
               └─ cachedなし → 実SDK create → request/response記録
            → 同じ公式parser/scorer → 標準出力保存
      → collect_scores() / summarize_scored() → 最終N10のrecords/summary
```

公式CLIは [`llm_x/cli.py`](../../upstream/LLM-Xavier/llm_x/cli.py)、公式backendは [`llm_x/backends.py`](../../upstream/LLM-Xavier/llm_x/backends.py) にあります。入力pilotは [Exp.5](../05_new_models_single/README.md)、共通ログの詳細は [Exp.2](../02_gpt35_single/README.md) を参照してください。
