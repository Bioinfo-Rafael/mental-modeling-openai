# 入出力の生データビューア

clone後、[view_rawdata.html](view_rawdata.html) をブラウザで開いてください。サーバー、Python、APIキー、元のJSONLは不要です。

HTMLにはSol / Terra / Lunaの各320件と、GPT-3.5 Jointの各条件先頭10件（計320件）、合計1,280件を埋め込んでいます。元JSONLへのアクセスや外部通信は行いません。SolのAPI失敗6件も含みます。

モデル・タスク・予測対象・Hを選択し、上部に固定された「前へ／次へ」で10件を切り替えられます。入力・回答は改行を保って表示し、条件や数値は別表に表示します。

データは2026-09-16時点のスナップショットです。元データの更新はHTMLへ自動反映されません。

[view_rawdata.ipynb](view_rawdata.ipynb) はローカルJSONLを読み込む別版です。元JSONLは通常Gitの除外対象なので、Notebookの再実行にはセル内で指定した元データを別途配置する必要があります。HTML版の閲覧には不要です。

## 1. 回答取得・採点・records.jsonl保存の呼び出し経路

```text
各実験のrun.py（06はsplit_execution.py経由）
  → experiments/common.py の execute_plan()
  → invoke_cli()
  → upstream/LLM-Xavier/llm_x/cli.py の _evaluate()
  → evaluation.py の evaluate_episode()
      → backend.complete()   ← モデルの回答全文を取得
      → _score_response()    ← 旧parserでその回答を採点
  → cli.py の _write_result()
      → predictions.jsonl に採点結果を保存
  → common.py の collect_scores()
      → 入出力と採点結果を結合してrecords.jsonlに保存
```

直接の呼び出し元は [evaluation.py の evaluate_episode()](../../upstream/LLM-Xavier/llm_x/evaluation.py#L68) です。

```python
response = backend.complete(system_prompt=scene, user_prompt=prompt)
record = _score_response(
    episode,
    config,
    index=index,
    response=response,
    drop_last_feature=drop_last,
    presented_action=presented_action,
    presented_is_correct=presented_is_correct,
)
```

**この経路では保存済みのrecords.jsonlを読んで採点するのではなく、回答が返ってきた時点で採点してからrecordsに保存します。** 後述するanalysisでの再解析は別の経路です。

06の `merged/<ID>/records.jsonl` は [split_execution.py の merge_results()](../split_execution.py#L197) が05と06のbatchのrecordをコピー・統合したものです。統合時に再採点せず、元のstatusを引き継ぎます。

## 2. _score_response()への入力とassistant_textの関係

[_score_response()](../../upstream/LLM-Xavier/llm_x/evaluation.py#L155) の入力は以下です。

| 引数 | 内容 |
| --- | --- |
| `episode` | 元の軌跡データ。正解となる状態や行動を取り出す |
| `config` | タスク、予測対象、質問名、状態変化の閾値、bin数など |
| `index` | 元episode上の採点対象の時刻 |
| `response` | モデルの回答全文の文字列 |
| `drop_last_feature` | Fetch系タスクの状態次元調整。今回の2タスクではfalse |
| `presented_action` / `presented_is_correct` | argue-action用。今回の4種類の予測では使わない |

同じ回答本文を [common.py の RecordingOpenAIBackend.complete()](../common.py#L525) が保存対象に含めます。

```python
finished = {
    # 条件・入力・応答・時間などの他のフィールド
    "assistant_text": text,
}
```

HTMLの「出力 — assistant_text」は、採点関数に渡された回答全文です。モデルの応答の `choices[0].message.content` に由来し、Reasoningや最終回答ラベルも含みます。HTML表示のために採点用predictionへ置き換えているわけではありません。

[collect_scores()](../common.py#L631) はupstreamのpredictions.jsonlを読み、`score["status"]` をrecord直下のstatusにコピーしてrecords.jsonlへ追記します。API等の実行失敗では [complete()の例外処理](../common.py#L511) がstatus=failedを設定します。

## 3. analysisは回答全文をJoint用parserで読み直す

06の4モデル比較は [comparison.py](../common_analysis/comparison.py#L106) → [joint_data.py の load_and_parse()](../common_analysis/joint_data.py#L95) → [parse_response()](../common_analysis/joint_data.py#L80) の経路を使います。

`load_and_parse()` はassistant_textとraw_response内の回答本文の一致を検証し、次の各成分を独立に抽出します。next/lastによるparserの違いはありません。

| 対象 | 読み取る出力ラベル | 解析上の名前 | 期待する内容 |
| --- | --- | --- | --- |
| MountainCar action | `Final action choice` | `action` | 0〜2の行動番号1個 |
| Pendulum action | `predictions` | `action_value` | 連続action値1個 |
| Pendulum action | `Final action bins` | `action_bin` | 0〜9のbin番号1個 |
| 両タスク state | `predictions` | `direction` | INC / DEC / UNCHのリスト |
| 両タスク state | `Final state values` | `state_value` | 状態値の数値リスト |
| 両タスク state | `Final state deltas` | `state_delta` | 状態差分の数値リスト |

stateの要素数はMountainCarが2、Pendulumが3です。例えば、

```text
predictions = [-1.45]       → action_valueとして取得
>>Final action bins: [1]    → action_binとして取得
```

のように連続値とbinを分けて読みます。**analysisでは連続値をbin番号として誤って読もうとする問題はありません。元のstatus=ignoredだけを理由に除外せず、assistant_textから再解析します。**

## 4. analysis側の正規表現と抽出後の検査

実装は [joint_data.py の parse_component()](../common_analysis/joint_data.py#L41) です。全成分で同じ正規表現の骨格を使い、上表のラベルだけを差し替えます。

```python
prefix = r"^[ \t]*(?:(?:\d+[.)]|[-*])[ \t]*)?(?:>>[ \t]*)?(?:\*\*)?(?:\[)?"
suffix = r"(?:\])?(?:\*\*)?[ \t]*[:=][ \t]*(?:\*\*)?\s*(\[[^\]]*\]|[^\n]+)"
matches = re.findall(prefix + re.escape(label) + suffix, text, re.M | re.I)
```

| 部分 | 意味 |
| --- | --- |
| `^` と `re.M` | 回答内の各行の先頭からラベルを探す。説明文の途中の数値を自由に拾わない |
| `[ \t]*` | 空白・タブを許可 |
| `(?:(?:\d+[.)]\|[-*])[ \t]*)?` | `1.`、`1)`、`-`、`*` のような先頭の番号・箇条書きを許可（表の `\|` はMarkdown上のエスケープ。実コードは上のブロック参照） |
| `(?:>>[ \t]*)?` | ラベル前の `>>` を許可 |
| `(?:\*\*)?` と角括弧の任意部分 | `**predictions**` や `[predictions]` などのラベル装飾を許可 |
| `re.escape(label)` | 指定ラベルを文字どおり探す |
| `[:=]` | ラベルの後の `:` または `=` を許可 |
| `(\[[^\]]*\]\|[^\n]+)` | 角括弧付きリスト、または行末までの値を取得。リスト内の改行も許可（表の `\|` はMarkdown用） |
| `re.I` | ラベルの大文字・小文字を区別しない |
| `re.findall()` | 最初の1個だけでなく、同じラベルの全出現を取得 |

例えば以下のラベル付き出力が抽出対象です。

```text
3. >>Final action choice: [2]
predictions = [-1.45]
>>Final action bins: [1]
**predictions** = ["INC", "DEC", "UNCH"]
>>Final state values: [0.9, 0.1, -0.2]
>>Final state deltas: [0.01, -0.01, -0.3]
```

取得した文字列は `ast.literal_eval(token.strip())` で値に変換します。Pythonコードを実行するevalではなく、リストや数値などのリテラルとして読みます。その後、以下を検査します。

1. **次元数**：期待する要素数のlistか。MountainCarのactionだけは単一の数値も1要素listに変換して受け付けます。
2. **増減方向**：各要素がINC/DEC/UNCHかを検査し、DEC=0、INC=1、UNCH=2へ変換します。
3. **連続値**：全要素が有限のint/floatか。bool、NaN、無限大等は受け付けません。ここで物理的な上下限への適合までは判定しません。
4. **行動・bin番号**：整数値かつ、actionは0〜2、binは0〜9かを検査します。
5. **重複ラベル**：同じ成分を複数回書いた場合、解釈後の値が全て同じなら採用し、食い違えば `conflicting_markers` として失敗にします。

結果は成分ごとの `{"value": ..., "ok": True/False, "error": ...}` です。マーカーなしは `missing_marker`、数値化できない説明・プレースホルダー等は `non_literal_or_placeholder`、次元違いは `wrong_dimension` として記録します。一つの成分が失敗しても、他の成分は独立して判定されます。

**ok=trueは「読み取れた」という意味で、予測が正解という意味ではありません。** 抽出後の評価は [joint_aggregate.py](../common_analysis/joint_aggregate.py) / [joint_metrics.py](../common_analysis/joint_metrics.py) が担当します。成分別の読み取り成功・失敗件数は06の `analysis/comparison_4models/tables/parse_summary.csv` にあります。API失敗は、明示的に許可された4モデル比較で欠損として扱い、回答を捏造しません。

## 5. records.jsonlのstatusは旧parserの判定：Jointの評価結果としては使わない

**records.jsonlのstatusは旧parserで記入されており、Joint出力の評価としては誤った判定を含みます。analysis側で再解析・評価した結果ではありません。**

- Pendulum action：旧parserは連続値用の `predictions` を整数binとして読み、別出力の `Final action bins` を読みません。指定どおり `predictions = [-1.45]` と回答していてもignoredになります。実データにも、正しいbinを別に出しているのにignoredとなった例があります。
- state：旧parserのmatch/mismatchは増減方向だけの一致判定です。state値・delta値の正しさを示しません。
- MountainCar actionの判定など、statusが正しく付いている場合もあります。「全件のstatusが誤り」という意味ではありません。
- failedはAPI等の実行失敗を表す別の状態です。analysisもAPI失敗の確認にはこの値を使います。

元データを保存記録として保持するため、statusは書き換えていません。06の評価を確認するときはanalysisの成分別解析結果を参照してください。

HTML上部は、回答本文が保存されている場合の「応答あり」だけを表示します。回答がない場合は空欄とし、採点結果やAPI失敗のステータス文言は表示しません。詳細表・折りたたみJSONには保存済みのstatusを保持していますが、これは旧parserの結果であり、Joint全出力の評価として解釈しないでください。Notebookは元のままです。
