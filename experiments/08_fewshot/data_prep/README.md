# Few-shot候補の準備

07の [ビューア](../../07_view_rawdata/view_rawdata.html) で例を選び、JSONに保存し、件数・条件を変えてfew-shotブロックを生成する。API呼び出しは行わない。

## ビューアの操作

1. モデル・Task・予測対象・Hを選び、上部のScore・一致／誤差と、回答直後の正解比較表を確認する。
2. **正解例**または**不正解例**を押す。色が変わり、同じボタンの再クリックで解除、もう片方で区分を変更する。これは手動の例の区分であり、自動評価を書き換えない。
3. 右上の**選択済み**から一覧を開く。回答への移動、メモ、区分変更、削除、並べ替えができる。選択はブラウザのlocalStorageに保存する。ブラウザ・ファイル移動等で使えなくなることがあるためJSONも保存する。
4. **出力**を開く。**保存先を選択（data_prep）**からこのディレクトリを選ぶ。Chrome等のフォルダ選択に対応したブラウザでは、以後このページを開いている間、そのフォルダに保存する。非対応の場合はブラウザのダウンロード先に保存されるため、このフォルダへ移動する。
5. **候補集JSON（全選択）**はフィルタに関係なく選択全件を保存する。**指定件数のJSON/TXT**はTask・予測対象・モデル・H・Score、正解例／不正解例の件数、抽出方法・seed・並び順を反映する。出力画面のTaskと予測対象は、閲覧中の条件を初期値にする。
6. 一覧の**JSONを読み込む**で保存した候補集／subsetを復元できる。現在の選択を置き換える。回答・入力の一致、ID重複、回答欠損を検査し、表示データの正解・Scoreを用いて復元する。

ファイル名には時刻と識別子を付け、既存の選択ファイルを上書きしない。保存先フォルダはブラウザから任意のパスを自動指定できないため、最初にユーザーが選ぶ必要がある。

## ファイル

| ファイル | 内容 |
|---|---|
| `examples.json` | 全1,920件の共通データ。配列形式。まだ例として未選択なので `example_type=null` |
| `validation.json` | 元ファイルSHA256、対応付け、件数、欠損件数、誤差の定義 |
| `build_examples.py` | 元回答と正解・スコアを結合し、共通データと07の自己完結HTMLを再生成 |
| `prepare_prompt.py` | 選択済みJSONから条件・件数を指定してTXTとsubset JSONを生成 |
| `test_data_prep.py` | 入出力の同一性・正解・誤差・抽出のオフライン検証 |
| `selected_examples_*.json` | ブラウザで選んだ全候補。ユーザーが出力した時点で作成 |
| `fewshot_examples_*.json` / `fewshot_prompt_*.txt` | 指定した条件・件数の出力。出力した時点で作成 |

HTML本体は `07_view_rawdata/view_rawdata.html`。編集元は同ディレクトリの `viewer_template.html` / `viewer.css` / `viewer.js`。HTMLはCSS・JS・全データを内包し、ネットワークも元JSONLも不要。

## データと正解

- 03_1のGPT-3.5全960件と06統合版の960件。06に含まれる05の120件は追加結合しない。
- 同梱の `common_analysis.joint_data.load_and_parse()` を利用。manifest・raw episode SHA・元応答・保存された基本正解の整合性を検証する。完了済み06に保存されたAPI失敗6件も保持するが、few-shotとして選択できない。
- MountainCar action：行動ID。Pendulum action：トルクと10-bin ID。state：DEC/INC/UNCH、状態値、forward delta。
- Next Stateの正解は後の状態、Last Stateの正解は前の状態。deltaは両方とも後 − 前。増減方向の閾値は既存解析の `1e-4`。
- 離散値は次元別と全要素の一致、連続値は次元別絶対誤差と平均絶対誤差。連続値の「正解」閾値は置かない。bin番号の絶対誤差も保存する。抽出失敗は値・誤差・正誤をnullとして保持する。
- 既存の `records.status` はJointの全出力の評価として使わない。HTMLの旧parser詳細だけにそのまま残す。
- Reasoningスコアは `07_view_rawdata/reasoning_for_scoring_scored_astra_high.csv` の値を保持。再採点しない。Score 1=定量・数学、2=具体的観測、3=policy傾向の一般化、4=一般的直感・推測。正誤とは独立。
- スコアCSVにはIDがないため、既存のモデル順（sol/terra/luna/3.5）→Task→metric→H→ordinalと、未採点CSVの列の同一性、空欄以外のReasoningと元回答の一致を検証して `query_id` を付ける。`score_csv_record` はヘッダを除いた1始まりのCSVレコード番号（複数行フィールドがあるので物理行番号ではない）。Reasoning欠損126件もフラグと既存スコアを保持。

## 行動IDで自動選択する

`select_by_id.py` はユーザー指定の選択規則で、Task×モデルごとに以下を各3件まで選ぶ。

- 正解例：ID一致かつScore 1を3件、さらにScore 2を3件。
- Score 2の正解例が不足する場合だけ、Score 3→4で補完する。元のScoreは変更しない。
- 不正解例：ID不一致かつScore 3を3件、さらにScore 4を3件。他Scoreで補完しない。
- 各Score内ではnext-actionを優先し、足りなければlast-action。Hは制限しない。同順位はH・episode・index・ordinal・query_id順。
- MountainCarは行動ID、Pendulumはbin IDで判定。連続トルクの完全一致は要求しない。抽出できないIDやAPI失敗は不正解例に含めない。
- 各Task・各モデル内で選び、足りない分を別Task・モデルから埋めない。異なるH／モデルの回答はquery_idが異なれば別例。

```bash
.venv/bin/python -B experiments/08_fewshot/data_prep/select_by_id.py
```

既定出力は `selected_examples_by_id.json`（既存exportと同じcandidate_pool形式）。選択根拠は `selection_policy.selection_audit`、各例の `note` に保存する。07での読み込みと08のrun.pyの自動検出に対応する。

2026-09-28実行：96件中74件を選択、22件不足。正解Score 2の補完は0件。詳細は [選択結果](selected_examples_by_id.md)、[集計CSV](selected_examples_by_id.summary.csv)、[監査JSON](selected_examples_by_id.summary.json)。元Reasoningが空欄で既存Score 4の例を3件含み、フラグを保持している。

再実行時は既存ファイルを上書きしない。必要なら `--output` に別名を指定する。複数の候補集JSONをdata_prepに置いた場合は、08の `INPUT_FILES` で使うファイルを指定する。

## JSONの構造

共通データ `examples.json` は配列。選択／subsetの出力は `{schema_version, kind, options, examples: [...]}`。1例の主なキー：

- 条件：`query_id`, `task`, `metric`, `model`, `model_id`, `H`, `ordinal`
- 手動選択：`example_type`（`correct` / `incorrect`）, `note`, `selection_order`
- 入出力：`question` はuser全文だけ、`answer` はモデル回答全文。system入力は含まない。
- スコア：`reasoning_score`, `reasoning`, `reasoning_available`
- 正解・評価：`ground_truth`, `comparison`（予測値、正解、抽出成否、一致、誤差）
- 出典：元recordsとその行番号、episodeパス／SHA／query_index、回答SHA、score元CSVとレコード番号

選択JSONの配列順が現在の順序。`selection_order` は出力時の候補集内の順序も記録する。任意の連続値許容誤差やモデル混合は自動で決めない。

## 08からプロンプトを生成

リポジトリのルートで実行する。`selected_examples_....json` は実際に出力したファイル名に置き換える。

```bash
.venv/bin/python -B experiments/08_fewshot/data_prep/prepare_prompt.py \
  experiments/08_fewshot/data_prep/selected_examples_....json \
  --task Pendulum-v1 --metric next-action \
  --scores 1 2 --correct 3 --incorrect 2 \
  --sampling random --seed 42 --order alternate \
  --output experiments/08_fewshot/data_prep/pendulum_next_action_3_2.txt
```

同名の `.txt` と `.json` を生成する。既存ファイルがあれば停止する。省略した条件は全条件、件数の初期値は0。候補不足・重複・未選択データ・回答なしはエラー。`--model`、`--history` も指定できる。

抽出方法は `first` / `random`。ブラウザとPythonは同じseed付きFNV-1a順位を使い、同じ候補順・条件で同じIDを選ぶ。並び順は `selection` / `correct-first` / `alternate` / `shuffle`。モデル間の同一問題は別回答として保持する。

TXTは簡単な使い方の説明に続けて、以下を例ごとに並べる。

```text
Question:
元のuser入力全文

Answer:
元のモデル回答全文

Label:
選択区分・Reasoning score・正解・項目別の比較結果（JSON）
```

元の回答は不正解例でも修正しない。systemは例に含めず、本番リクエスト側で一度設定する。本番の質問は生成した例の後へ別途追加する。Labelは過去の例の評価で、新しい質問への回答フォーマットではない旨を冒頭に記載する。

pandasでの読み込み例：

```python
import json
import pandas as pd
with open('selected_examples_....json') as f:
    payload = json.load(f)
df = pd.json_normalize(payload['examples'])
subset = df[(df['task'] == 'Pendulum-v1')
            & (df['metric'] == 'next-action')
            & (df['example_type'] == 'correct')
            & df['reasoning_score'].isin([1, 2])]
```

評価側でfew-shotと同一の問題を除く場合は `query_id` だけでなくepisodeとquery_indexも比較する（query_idはモデルやHでも変わる）。履歴窓の重複をどこまで除くかは08の実験設計で決める。

## 再生成・検証

```bash
.venv/bin/python -B experiments/08_fewshot/data_prep/build_examples.py
.venv/bin/python -B -m pytest experiments/08_fewshot/data_prep/test_data_prep.py -q -p no:cacheprovider
```

再生成にはローカルの元results、runs、raw episodes、スコアCSVが必要。06の元snapshotを変更する場合は `--source06` を明示する。共通データ・検証JSON・生成HTMLを更新し、選択済みファイル、元結果、採点CSV、Notebookは変更しない。
