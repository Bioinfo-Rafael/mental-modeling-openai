# Exp.4 — N30 / N20 / N10の再解析

Exp.3で取得した各条件30件の結果について、全30件・先頭20件・先頭10件の統計を比較します。**APIを呼ばないanalysis-only実験**です。新しい応答は取得せず、promptの再生成もしません。

Exp.3を `--resume` で再開した場合、`common.latest_results()` が `results/resumes/0001/`、`0002/` などの最新世代を入力元にします。再開履歴がなければ従来どおり `results/` を読みます。最新世代が未完了なら、古い結果へ戻らず分析を停止します。

## 1. 実行方法

既存のMental Modeling / LLM-Xavier用環境を使います。[01の環境説明](../01_prompt_preview/README.md)と同じ `.venv`（Notebookでは `Python (mental-modeling)`）で、新規作成は不要です。図作成には、projectの `notebook` extraに含まれるmatplotlibが必要です。

入力が揃っているかだけ確認する場合：

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/04_sample_size_analysis/run.py --dry-run
```

入力を検査して `results/dry_run/manifest.json` を保存します。統計表・図はまだ作りません。**dry-runでもExp.3の完了結果が必要**で、不足していれば停止します。

統計表と図を生成する場合：

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/04_sample_size_analysis/run.py
```

有料実験と異なり、04はフラグなしで分析を実行します。どちらのコマンドもAPI 0で、API keyは不要です。`--execute`・`--confirm-paid-api`・`--retries` は受け付けません。

既存の分析結果があれば上書きせず停止します。再解析する際は、結果を手動で退避してください。このREADME作成では実験・dry-run・描画を実行していません。

## 2. 入力元

```text
experiments/03_gpt35_history_n30/results/
├── manifest.json
├── summary.json
└── records.jsonl
```

| ファイル | 読む理由 |
| --- | --- |
| `manifest.json` | 元の条件grid、query identity・順序、実装hashを確認 |
| `summary.json` | Exp.3が `status="complete"` か確認 |
| `records.jsonl` | 保存済みの公式score、query時間、API usageを取得 |

出典：[`experiments/analysis.py`](../analysis.py) の `load_source()`。

32条件が揃い、各条件に順序どおりの30件があり、query IDの重複やmanifestとの不一致がないことを確認します。また、現在の [`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) のhashを読み、Exp.3実行時から集計実装が変わっていないことを検査します。

`data/` のNPZ、01のprompt、02のpilot、05/06の応答は分析入力にしません。公式parserで応答を再解析するのでもなく、03が保存した公式scoreを使って集計し直します。

## 3. 出力先とファイルの見方

出力先は `experiments/04_sample_size_analysis/results/` です。

```text
results/
├── .gitkeep
├── dry_run/manifest.json         # dry-runした場合のみ
├── .started.json
├── manifest.json
├── statistics.csv
├── statistics.json
├── timing.json
├── figures/
│   ├── accuracy.png
│   ├── elapsed_time.png
│   ├── input_tokens.png
│   ├── output_tokens.png
│   └── total_tokens.png
└── cache/matplotlib/             # 描画用のcache
```

生成結果はGitのignore対象です。APIのrequest/responseファイルや、公式CLIの `runs/` は新たに生成しません。

| ファイル | 中身・粒度 |
| --- | --- |
| `manifest.json` | 入力ファイルhash、N候補、条件数、予定統計行数、API 0の記録 |
| `statistics.csv` | 1 condition × 1 N＝1行。32条件×3種類のN＝96行 |
| `statistics.json` | CSVと同じ統計行に加え、分母・分散・時間・token・図の定義 |
| `timing.json` | 今回の分析開始/終了UTC時刻と分析所要時間。03のquery時間とは別 |
| `figures/*.png` | Hを横軸、N30/20/10を系列とした比較図。各図は2 task×4 metricの8パネル |
| `.started.json` | 分析開始記録。既存結果を上書きしないためのmarker |

### 統計表のカラム

出典：[`experiments/analysis.py`](../analysis.py) の `calculate()` / `describe()`。

| カラム・接頭辞 | 意味 |
| --- | --- |
| `condition_id`, `model`, `task`, `metric`, `question_name`, `H` | 元の実験条件 |
| `N` | 今回のsubset件数。30、20、10 |
| `query_ids` | 集計に使ったquery IDを順番付きで保持 |
| `Accuracy` / `legacy_compatible_match_rate` | 公式のprimary Accuracy。actionは全N件、stateはparsed件数が分母 |
| `query_count`, `parsed_count`, `ignored_count`, `exact_matches` | 公式のquery数・解析成功数・解析不能数・完全一致数 |
| `exact_match_rate_all_queries` | 完全一致数÷N |
| `exact_match_rate_parsed_queries` | 完全一致数÷parsed件数。parsedが0ならnull |
| `mean_element_accuracy_parsed_queries` | 要素精度が存在するparsed queryの平均 |
| `parse_rate` | parsed件数÷N |
| `correct_all_queries_*` | 全queryについてmatch=1、mismatch/ignored=0とした統計 |
| `element_accuracy_parsed_*` | parsed vector応答の `element_accuracy` の統計。scalar actionなど、値がないものは除外 |
| `query_elapsed_seconds_*` | 03で記録したquery時間の統計。retry待ちなどを含む |
| `request_elapsed_seconds_*` | 03で記録したquery内のSDK create所要時間合計の統計 |
| `input_tokens_*`, `output_tokens_*`, `total_tokens_*` | 03の最終response usageの統計。今回tokenizerで測り直すのではない |

`*` 部分には、次の統計名が付きます。例えば `input_tokens_mean` は入力token数の平均です。

| 接尾辞 | 計算・欠損時の扱い |
| --- | --- |
| `count` | nullを除いた観測値数 |
| `mean` | 算術平均。観測なしならnull |
| `variance_population_ddof0` | 母分散。偏差平方和÷観測値数 |
| `variance_sample_ddof1` | 不偏標本分散。偏差平方和÷(観測値数−1)。観測が2未満ならnull |
| `std_population_ddof0` | 母分散の平方根 |
| `std_sample_ddof1` | 標本分散の平方根。観測が2未満ならnull |

欠損usageや要素精度は0埋めしません。JSONではnull、CSVでは空欄になります。retryに伴う過去のusageは03のrawログに残りますが、この統計は最終responseのusageです。

**stateのprimary Accuracyと `correct_all_queries_mean` は別の値になり得ます。** 例えば10件のうちmatch=6、mismatch=2、ignored=2なら、state Accuracyは6/8、全件correct平均は6/10です。

Accuracy図はprimary値のみでerror barなし。時間・token図は平均±母標準偏差（ddof=0）です。標準偏差はquery間のばらつきであり、信頼区間ではありません。同じepisodeの近接queryには相関もあり得ます。

## 4. `run.py` の中身

出典：[`experiments/04_sample_size_analysis/run.py`](run.py)。

```python
if __name__ == "__main__":
    raise SystemExit(common.analysis_main())
```

04には `EXPERIMENT = common.Experiment(...)` がありません。`common.run()` も呼びません。代わりに、[`experiments/common.py`](../common.py) の次の関数を呼びます。

```python
def analysis_main(argv=None):
    from experiments.analysis import main
    return main(argv)
```

分析の本体は [`experiments/analysis.py`](../analysis.py) の `main()` です。`run.py` 先頭のpath設定・pycache抑止は他の実験と同じです。

## 5. 各設定と処理がどこにあるか

出典：[`experiments/analysis.py`](../analysis.py) のmodule定数。

```python
SOURCE = "03_gpt35_history_n30"
DESTINATION = "04_sample_size_analysis"
SAMPLE_SIZES = (30, 20, 10)
```

| 設定 | 使用場所・意味 |
| --- | --- |
| `SOURCE` | `analysis.py:load_source()` で入力experimentのresultsを決める |
| `DESTINATION` | `analysis.py:main()` で出力先を決める |
| `SAMPLE_SIZES` | `analysis.py:calculate()` でsubsetのNを展開。`load_source()` は最大N=30の完備を確認 |
| `common.TASKS/METRICS/H_VALUES` | `common.py` の定数。入力gridの検証と図のパネル・横軸に使う |
| `--dry-run` | `analysis.py:main()` で入力確認だけ行い、統計・描画前に終了 |

出典：[`experiments/analysis.py`](../analysis.py) の `calculate()` 内。

```python
for N in SAMPLE_SIZES:
    selected = ordered[:N]
    metrics = common.summarize_scored(selected, condition["metric"])
```

`ordered` はその条件の保存済み30 records。`[:N]` は先頭N件です。N10はN20に、N20はN30に含まれます。別sampleの抽選やAPI再送はありません。

`common.summarize_scored()` の出典は [`experiments/common.py`](../common.py)。保存済みの `score` を取り出し、[`llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) の公式 `_summarize()` を呼びます。したがって、stateとactionの分母の違いも保持します。

出典：[`experiments/analysis.py`](../analysis.py) の `calculate()` 内（正誤列の値を作る式）。

```python
[int(r["status"] == "match") for r in selected]
```

この値を同ファイルの `describe()` へ渡し、mean/variance/stdを計算します。これは追加の記述統計であり、公式primary Accuracyの置き換えではありません。

図の作成は同ファイルの `figures()`。matplotlibのcache先も `results/cache/matplotlib/` に指定し、raw directoryには派生物を保存しません。

## 6. 呼び出し順

```text
04_sample_size_analysis/run.py
└─ common.analysis_main()
   └─ analysis.main()
      ├─ load_source() → 03のmanifest/summary/records・実装hashを確認
      ├─ --dry-runあり → dry_run/manifest保存 → 終了
      └─ 通常実行
         ├─ 既存出力の検査・開始記録・manifest保存
         ├─ calculate()
         │  ├─ 条件ごとに ordered[:30], [:20], [:10]
         │  ├─ common.summarize_scored() → 公式_summarize()
         │  └─ describe() → 正誤・要素精度・時間・tokenの統計
         ├─ statistics.csv/json保存
         ├─ figures() → 5枚の図を保存
         └─ timing.json保存 → 終了（API 0）
```

入力実験は [Exp.3](../03_gpt35_history_n30/README.md)、全体の流れは [experiments/README.md](../README.md) を参照してください。
