# Exp.1 — prompt preview

公式のepisodeから、LLMへ送る前のsystem/user promptを8件作り、内容を確認する実験です。**API送信・応答の取得・採点・token計測は行いません。**

## 1. 実行方法

作成済みの **Mental Modeling / LLM-Xavier用の仮想環境**を使います。新しい環境を作る必要はありません。既存環境を有効化し、repository直下から実行してください。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/01_prompt_preview/run.py
```

確認した既存環境は次のとおりです（2026-09-10）。

| 項目 | 値 |
| --- | --- |
| 環境directory | `/Users/cls-lab/Git/Matsuo/mental-modeling-openai/.venv` |
| Python | `3.14.5` |
| LLM-Xavierのインストール済みpackage | `llm-x-evaluator==0.1.0`（editable install） |
| `llm_x` の実際の参照先 | `/Users/cls-lab/Git/Matsuo/mental-modeling-openai/upstream/LLM-Xavier/llm_x/` |
| この環境内に登録されたNotebook kernel表示名 | `Python (mental-modeling)` |

`Python (mental-modeling)` は別のconda環境ではなく、この `.venv/bin/python` を使うNotebook kernelの表示名です。元の `.venv/bin/python experiments/01_prompt_preview/run.py` も、この同じ環境を直接指定するコマンドでした。

有効化後、使用中のPythonは次のコマンドで確認できます。これは実験を実行しません。

```bash
python -c 'import sys; print(sys.executable)'
# /Users/cls-lab/Git/Matsuo/mental-modeling-openai/.venv/bin/python
```

API keyは不要です。`--execute` / `--confirm-paid-api` は付けません。Exp.1はpaid実行を拒否します。

このコマンドは「何も保存しないdry-run」ではありません。APIなしで、実際にpromptとmanifestを保存します。Exp.2以降の有料実験にある通常のdry-run分岐とは異なります。Exp.1に `--dry-run` 引数はありません。

既存結果がある場合は、上書きせず停止します。再実行が必要なら、まず以前の結果を別の場所へ退避してください。隠しファイル `.started.json` も実行開始の記録として含まれます。`results/` に `.gitkeep` と `dry_run/` 以外があると再実行は拒否されます。

> このREADMEの作成時には、上記コマンドも実験も実行していません。以下はコードに基づく説明です。

## 2. 入力元：取得済みの公式raw data

入力のルートは次の場所です。Exp.1自身はデータをダウンロードしません。

```text
/Users/cls-lab/Git/Matsuo/mental-modeling-openai/data/llmx_data/
└── offline_data/
    └── <dataset>/
        └── raw_transitions/
            └── <task>/
                └── episodes/
                    └── *.npz
```

1つのNPZを1 episodeとして扱います。既存の [`discover_episodes()`](../../dataset_adapters/llmx.py) でepisodeを探し、path順に並べ、次の2タスクに該当するものを選びます。

- `MountainCar-v0`
- `Pendulum-v1`

各条件について、先頭のepisodeから公式関数が生成する有効queryを順に取得します。必要数に足りなければ次のepisodeへ進みます。ランダム抽出ではなく、ファイル名の手打ちでもありません。履歴がepisode境界を跨ぐことはありません。

Exp.1では各条件 `n=1` なので、最初の有効queryを1件ずつ使用します。`n=1` は「episodeを必ず1個使う」という指定ではありません。

raw dataはread-onlyで読みます。加工データを `data/` に保存したり、NPZを書き換えたりしません。prompt生成は [`preprocessing/llmx_original.py`](../../preprocessing/llmx_original.py) 経由で公式コードを再利用します。再現性の記録には、Git revision、`configs/sources.json`、関連ソースのSHA256も読みます。

## 3. 出力先とファイルの見方

出力はすべて次の場所に保存します。

```text
/Users/cls-lab/Git/Matsuo/mental-modeling-openai/experiments/01_prompt_preview/results/
├── .gitkeep
├── .started.json
├── manifest.json
├── manifest.csv
├── 3.5__MountainCar-v0__next-action__H5.txt
├── 3.5__MountainCar-v0__last-action__H5.txt
├── 3.5__MountainCar-v0__next-state__H5.txt
├── 3.5__MountainCar-v0__last-state__H5.txt
├── 3.5__Pendulum-v1__next-action__H5.txt
├── 3.5__Pendulum-v1__last-action__H5.txt
├── 3.5__Pendulum-v1__next-state__H5.txt
└── 3.5__Pendulum-v1__last-state__H5.txt
```

これは初期設定で正常終了した場合の構成です。ユーザーの指定により、Exp.1の `.started.json`、`manifest.json/csv`、promptの `.txt` は `.gitkeep` とともにGitで共有します。他の実験の生成結果は引き続きignore対象です。

| ファイル | 中身・用途 |
| --- | --- |
| `.started.json` | 実行開始時刻。既存結果の上書き・重複実行を防ぐための記録 |
| `manifest.json` | 実験名、条件別の件数、query一覧、prompt全文、source/codeのSHA256、revision。`mode` は `preview`、API予定数は0 |
| `manifest.csv` | 1 query＝1行、計8行。JSONのquery一覧からsystem/user prompt本文を除いたもの。条件・使用episode・query index・履歴範囲・SHA256などを確認する表 |
| 8つの `.txt` | 人間が確認するためのsystem/user prompt全文。1条件＝1ファイル |

`.txt` は次の形式です。区切り見出しは表示用で、モデルへ送るprompt本文の一部ではありません。

```text
SYSTEM MESSAGE
================
（タスク説明などのsystem prompt）

USER MESSAGE
==============
（episodeの履歴と質問を含むuser prompt）
```

まず `.txt` で実際の文面を確認し、その入力がどのepisode・indexに由来するかを `manifest.csv` で確認すると読みやすいです。

| manifestの主要項目 | 意味 |
| --- | --- |
| `condition_id` | model alias・task・metric・Hを組み合わせた条件名。`.txt` のファイル名にも使う |
| `model_alias` / `model` | `3.5` / `gpt-3.5-turbo`。Exp.1では記録だけで送信しない |
| `task` / `metric` / `question_name` | 対象タスク・評価family・公式の質問名 |
| `H` / `requested_H` | 指定した実際の履歴step数。初期値5 |
| `upstream_history_size` | 公式関数に渡す履歴パラメータ。初期値4 |
| `actual_history_steps` | 生成した履歴範囲の長さ。指定Hとの一致を検査する |
| `history_start` / `history_end` | 履歴の半開区間 `[start, end)`。end自身は履歴に含まれない |
| `episode` / `episode_path` | NPZのファイル名 / repositoryルートからの相対パス |
| `episode_sha256` | 読み込んだNPZの内容を識別するhash |
| `query_index` | 公式評価関数が使用するepisode内の0-based index。意味・有効範囲はmetricによる |
| `ordinal` | 条件内で選んだ順序。Exp.1は各1件なので全て0 |
| `system_prompt_sha256` / `user_prompt_sha256` | 各prompt本文のhash。Exp.2との一致確認に使う |
| `query_id` | model・task・metric・質問名・H・episode path/hash・index・両prompt hashをまとめた識別子 |
| `reuse_required` | 保存済み応答の再利用が必須か。Exp.1では `false` |

Exp.1ではAPIを呼ばないため、`requests.jsonl`、`responses.jsonl`、採点結果などは生成しません。

## 4. `run.py` の中身

[`run.py`](run.py) は「条件をまとめたオブジェクトを作り、共通関数へ渡す」ための短いファイルです。

### importとパス設定

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.dont_write_bytecode = True
from experiments import common
```

`parents[2]` はrepositoryルートです。ここをPythonのmodule検索先に追加して、`experiments.common` をimportできるようにします。`dont_write_bytecode=True` は、公式ソースなどに `__pycache__` を作らないための設定です。

### `Experiment` は型の定義、`Experiment(...)` はインスタンス生成

型を定義しているのは [`common.py`](../common.py) 内のクラスです。

```python
@dataclass(frozen=True)
class Experiment:
    name: str
    models: tuple[str, ...]
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    histories: tuple[int, ...]
    n: int
    preview: bool = False
    preview_from: str | None = None
    reuse_from: str | None = None
```

`@dataclass` によって、これらの値を受け取る初期化処理などが生成されます。`frozen=True` は、作成後に属性を再代入しない設定です。型注釈そのものが、全ての引数の型を実行時に自動検査するわけではありません。

一方、Exp.1の `run.py` では、既に定義された型のインスタンスを作っています。

```python
EXPERIMENT = common.Experiment(
    name="01_prompt_preview",
    models=("3.5",),
    tasks=common.TASKS,
    metrics=common.METRICS,
    histories=(common.PILOT_H,),
    n=1,
    preview=True,
)
```

`EXPERIMENT` は、そのインスタンスを入れる変数名です。`common.Experiment(...)` を呼んだだけでは、データ読込やprompt作成は始まりません。
`("3.5",)` や `(5,)` の末尾のカンマは、要素が1つのtupleを作るためのものです。

### 実行開始部分

```python
if __name__ == "__main__":
    raise SystemExit(common.run(EXPERIMENT))
```

このファイルをスクリプトとして起動したときに、[`common.py` の `run(spec, argv=None)`](../common.py) を呼びます。`run.py` というファイル名と、`common.run` という関数は別物です。

渡した `EXPERIMENT` は、関数内では `spec` として受け取られます。例えば `spec.preview` は `True`、`spec.n` は `1` です。戻り値は終了コードになり、正常終了は0、共通処理で捕捉したエラーは2です。別実験の `run.py` を続けて起動する処理はありません。

## 5. 各設定がどこで使われるか

条件の指定元は [`experiments/01_prompt_preview/run.py`](run.py) の `EXPERIMENT` です。この節で引用する処理コードは、特記のない限り [`experiments/common.py`](../common.py) のものです。各引用の直前にも、ファイル名と関数名を記載します。`...` を含むコードは説明用に省略した抜粋で、そのまま実行するコードではありません。

| 設定 | Exp.1の値 | 意味・主な使用場所 |
| --- | --- | --- |
| `name` | `"01_prompt_preview"` | `run()` のCLI説明と出力directory、manifestの実験名 |
| `models` | `("3.5",)` | `make_plan()` でモデルaliasを展開。API model IDとquery identityに使う |
| `tasks` | `("MountainCar-v0", "Pendulum-v1")` | `make_plan()` でタスク別のepisodeを選択 |
| `metrics` | `next-action`, `last-action`, `next-state`, `last-state` | `make_plan()` で評価familyを展開し、taskと組み合わせて質問名を決める |
| `histories` | `(5,)` | 実際にpromptへ入れる履歴長Hの候補。`history_size()` で公式パラメータへ変換 |
| `n` | `1` | 1条件あたりに取得するquery数。`selected` がこの数に達したら選択を止める |
| `preview` | `True` | `run()` でpaid実行を拒否し、prompt保存後に終了する |
| `preview_from` | `None`（省略時の値） | 過去previewとの一致確認元。Exp.1では使わない |
| `reuse_from` | `None`（省略時の値） | 過去API応答の再利用元。Exp.1では使わない |

### `name`：出力先を決める

出典：[`experiments/common.py`](../common.py) の `run()` 内。

```python
directory = EXPERIMENTS / spec.name / "results"
```

`EXPERIMENTS` はrepositoryの `experiments/`。したがって、Exp.1の出力先は `experiments/01_prompt_preview/results/` になります。

### `models`：aliasをmodel IDへ変換する

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
for alias in spec.models:
    model = MODELS[alias]
```

共通の対応表は次のとおりです。出典：[`experiments/common.py`](../common.py) のmodule定数 `MODELS`。

| alias | model ID |
| --- | --- |
| `3.5` | `gpt-3.5-turbo` |
| `luna` | `gpt-5.6-luna` |
| `terra` | `gpt-5.6-terra` |
| `sol` | `gpt-5.6-sol` |

Exp.1は送信しませんが、解決後のmodel IDをquery identityに含めます。[`experiments/02_gpt35_single/run.py`](../02_gpt35_single/run.py) も `3.5` を指定するため、同じデータ・条件・promptなら同じquery IDになります。`3.5` は文字列のaliasで、数値の `3.5` ではありません。alias変更で `condition_id` と出力ファイル名は変わりますが、model IDは `gpt-3.5-turbo` のままです。

### `tasks` と `metrics`：対象データと質問を選ぶ

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。`TASKS`・`METRICS`・`QUESTIONS` も同ファイルのmodule定数です。

```python
question = QUESTIONS[task][metric]
```

| metric | MountainCar-v0のquestion | Pendulum-v1のquestion |
| --- | --- | --- |
| `next-action` | `next_action_prediction` | `next_action_prediction_continuous_bins` |
| `last-action` | `last_action_prediction` | `last_action_prediction_continuous_bins` |
| `next-state` | `next_state_prediction` | `next_state_prediction` |
| `last-state` | `last_state_prediction` | `last_state_prediction` |

この質問名の対応表は `experiments/common.py` の `QUESTIONS` に基づきます。episodeは同ファイルの `make_plan()` 内で `s.task == task` のものだけ選びます。MountainCarとPendulumではactionの質問が異なりますが、質問文自体は [`upstream/LLM-Xavier/llm_x/questions.py`](../../upstream/LLM-Xavier/llm_x/questions.py) の公式 `render_question()` が生成します。

### `histories`：Hは実際の履歴step数

[`experiments/common.py`](../common.py) のmodule定数 `PILOT_H=5` なので、[`experiments/01_prompt_preview/run.py`](run.py) の `histories=(common.PILOT_H,)` は `(5,)` と同じです。Exp.1とExp.2がこの定数を共有します。

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。

```python
upstream_h = history_size(H)
```

[`upstream/LLM-Xavier/llm_x/evaluation.py`](../../upstream/LLM-Xavier/llm_x/evaluation.py) の `_history_range()` が作る履歴長はパラメータに対して1つ多くなるため、[`experiments/common.py`](../common.py) の `history_size()` が `H - 1` に変換します。次はこの関係を説明した図で、コードの引用ではありません。

```text
指定H=5 → upstream history_size=4 → 実際の履歴5 steps
```

`make_plan()` は `q.history_end - q.history_start` と履歴中の `Step ...:` の行数を検査し、Hと異なれば停止します。これは実行時の検査であり、このREADME作成時にpromptを生成して確認したという意味ではありません。

### `n`：各条件で選ぶquery数

出典：[`experiments/common.py`](../common.py) の `make_plan()` 内。選択処理の要点を示す説明用の抜粋です。

```python
selected = []
for source in ...:
    for q in build_prompt_queries(...):
        selected.append(row)
        if len(selected) == spec.n:
            break
    if len(selected) == spec.n:
        break
```

1条件は「model × task × metric × H」の1組です。Exp.1では、

```text
1 model × 2 tasks × 4 metrics × 1 H = 8 conditions
8 conditions × n=1 = 8 queries
```

となります。各metricの有効indexは公式関数に従うため、8件がすべて同じ `query_index` になるとは限りません。

### `preview`：API実行へ進まない分岐

出典：[`experiments/common.py`](../common.py) の `run()` 内。

```python
if spec.preview and args.execute:
    parser.error("Exp.1 is local preview only; paid flags are not accepted")
```

paidフラグは拒否されます。なお `--execute` だけを付けた場合は、その前にある二重フラグの検査で拒否されます。

正常なpreview実行は通常の有料実験向けdry-run分岐を通らず、最後の次の分岐に入ります。出典：[`experiments/common.py`](../common.py) の `run()` 内（省略した抜粋）。

```python
if spec.preview:
    for row in plan["queries"]:
        write_text(...)
    return 0
```

ここでpromptを保存して終了するため、有料評価を行う `execute_plan()` には進みません。

### `preview_from` と `reuse_from`：後続実験との関係

出典：[`experiments/common.py`](../common.py) の `check_inputs()` 内。次のように確認元を探します。

```python
source_name = spec.preview_from or spec.reuse_from
if not source_name:
    return cached
```

Exp.1は両方 `None` なので、前の実験のmanifestや応答を読むことなく、空の辞書を返します。

- [`experiments/02_gpt35_single/run.py`](../02_gpt35_single/run.py) は `preview_from="01_prompt_preview"` を指定します。再生成したqueryとExp.1のmanifestについて、identity・両prompt SHA・履歴範囲・関連ソースhashなどの一致を送信前に確認します。01の `.txt` をAPIへ直接送る設計ではありません。
- [`experiments/06_new_models_n10/run.py`](../06_new_models_n10/run.py) は `reuse_from="05_new_models_single"` を指定します。保存された応答を照合して再利用するための設定です。現在の再利用対象は `experiments/common.py` の `make_plan()` 内でPendulum・H20・各条件の先頭queryに限定されており、任意の過去実験を自動再利用する汎用機能ではありません。

## 6. Exp.1の呼び出し順

```text
01_prompt_preview/run.py
  Experiment(...) のインスタンスを作る
  └─ common.run(EXPERIMENT)              # 関数内では spec として受け取る
      ├─ 引数・n・paidフラグの検査
      ├─ make_plan(spec, retries=0)
      │   ├─ discover_episodes()        # rawのepisode一覧をpath順に取得
      │   ├─ model × task × metric × H を展開
      │   ├─ history_size(H)            # H → H-1
      │   └─ build_prompt_queries()    # 各条件の先頭n件を選択
      │       ├─ Episode.load()
      │       ├─ resolve_task() / resolve_question()
      │       ├─ system_prompt()
      │       ├─ _query_indices() / _history_range()
      │       ├─ render_question()
      │       └─ user_prompt() → Episode.history_text()
      ├─ check_inputs()                # Exp.1では確認元なし
      ├─ resultsの既存ファイル検査
      ├─ .started.jsonを保存
      ├─ save_manifest()               # JSON/CSV保存・予定件数表示
      └─ preview=True
          ├─ 8つのprompt .txtを保存
          └─ return 0                  # execute_plan()には進まない
```

要するに、**`Experiment` は実験条件を保持する型、`make_plan()` はその条件を具体的なquery一覧へ変換する関数、`run()` は検査・保存・実行方法の分岐を担当する関数**です。

全実験の説明は [experiments/README.md](../README.md)、未検証事項を含む静的レビューは [REVIEW.md](../REVIEW.md) を参照してください。
