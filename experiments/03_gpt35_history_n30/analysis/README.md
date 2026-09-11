# Exp.3のAction Matching Rate / State-change Accuracy解析

保存済みのExp.3結果を読み、2タスク×4種類の予測を2行×4列に配置します。各条件の30 queryから、N=30・20・10の図を1枚ずつ作ります。**API送信・実験再実行はしません。raw data、既存の実験ログ、公式コードも変更しません。**

今回の結果の解釈は [RESULT.md](RESULT.md) を参照してください。

## 1. 実行方法

既存のPython環境を使います。追加インストールは不要です。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/03_gpt35_history_n30/analysis/plot_metrics.py
```

既に生成物がある場合は上書きを拒否します。別の解析結果として保存するには、例えば以下です。

```bash
python experiments/03_gpt35_history_n30/analysis/plot_metrics.py \
  --output-dir experiments/03_gpt35_history_n30/analysis/recheck
```

生成済みのCSV・JSON・図だけを再生成する場合は `--overwrite` を付けます。`README.md` / `RESULT.md` や `results/` は、この指定でも書き換えません。`--seed 42` が既定値です。元ログが変わった場合、手書きの `RESULT.md` は自動更新されないので見直してください。

テスト：

```bash
python -m pytest experiments/03_gpt35_history_n30/analysis/test_plot_metrics.py -q
```

## 2. 入力元と今回の実データ

`common.latest_results()` で最新の再開世代を選びます。今回の入力は `../results/resumes/0001/` の `manifest.json` / `summary.json` / `records.jsonl` / `responses.jsonl` です。最新が未完了なら古い世代へ戻らず停止します。

| 項目 | 確認した内容 |
| --- | --- |
| モデル | GPT-3.5 Turbo |
| タスク／行順 | MountainCar-v0 → Pendulum-v1 |
| 予測／列順 | Next Action (NA)、Last Action (LA)、Next State (NS)、Last State (LS) |
| H | manifestから取得。今回5、10、20、30（実際の履歴timestep数） |
| 条件数とquery数 | 32条件×30件＝960件。重複query IDなし |
| raw LLM response | `raw_response`。本文は `assistant_text` と照合 |
| prediction / ground truth | 保存済みの `prediction` / `ground_truth` / `score` を使用・検証 |
| 状態の回答 | INC/DECの方向クラスのみ。連続state予測値ではない |
| MountainCarの行動回答 | discrete action ID（0 / 1 / 2） |
| Pendulumの行動回答 | bin IDの直接回答。absolute torque予測ではない |
| 元episode | 各タスク1ファイル。manifestの `episode_path` とSHA256を照合 |
| API応答とparse | 応答960件取得済みだが、44件は公式parserの `ignored` |

stateの3クラスground truthを再評価するため、該当するraw NPZを `Episode.load()` で読みます。読み取りだけで、raw directoryに加工データを保存しません。

## 3. 2つの主要指標

[Mental Modeling論文のAppendix B.4 / C](https://arxiv.org/html/2406.18505v1) は、離散行動の一致、連続行動のbin比較、連続stateの変化方向を評価します。直接bin回答とabsolute action値のbin化は別の設定です。今回の解析では、以下の実データに対応するものだけを用います。

### Action Matching Rate（NA / LA）

MountainCarは `predicted action == ground truth action`。Pendulumは **直接回答したbin IDとground-truth bin IDの一致**です。Pendulumのabsolute value予測からのpost-hoc matchingは今回のデータに存在しません。

各queryの一致を100%、不一致を0%とし、その平均を取ります。既存の公式action集計に合わせ、parse失敗も「一致できなかったquery」として分母に含め、accuracyへの寄与を0にします。**欠損predictionをaction=0と補完するわけではありません。** 保存predictionは欠損のままです。

公式 `_score_response()` → `bin_actions()` の再利用で、Pendulumのground truthは範囲[-2, 2]の10 binsになります。実際のedgesを [bin_edges.json](bin_edges.json) に保存します。`np.digitize(..., right=True)-1` を0〜9へclipする仕様で、内部境界上は下側binに入ります。promptに書かれた左閉右開の区間表記とは境界規則が異なりますが、解析だけで公式採点を変更しません。直接回答のbin IDにはclipをしません。

### State-change Accuracy（NS / LS）

添付依頼の定義に従い、各dimensionをINC=1、DEC=0、UNCH=2の3クラスとして比較します。ground truthは公式 `state_directions()` を **`allow_unchanged=True`** で呼び出して再評価します。

1. stateを小数点以下5桁に丸める（公式関数の既定値）。
2. `delta = state[index+1] - state[index]` を計算する。
3. `abs(delta) < state_threshold` はUNCH。今回のthresholdはmanifest記載の `1e-4`。
4. それ以外はdelta>0ならINC、それ以外ならDEC。
5. 1 query内の各dimensionの一致率を平均し、query間でmean/stdを計算する。

MountainCarは2 dimensions、Pendulumは3 dimensionsです。同じタスク内ではdimension数が固定なので、この平均は有効queryの全dimensionをまとめた一致率と一致します。LSも公式と同様に `state[index] → state[index+1]` の変化方向を評価し、逆向きに符号を反転しません。

**重要：今回の実験promptはINC/DECの2択で、UNCHを回答させていません。** 保存された元のground truthも `allow_unchanged=False` の2クラスです。したがって今回の3クラス値は「論文の分類形式に合わせた追加再評価」であり、**3択promptで実施した論文実験の厳密な再現ではありません**。UNCH正解の要素では今回のモデル回答は一致できません。

比較できるよう、次の値を別々にCSVへ残しています。

| `measure` | 意味 |
| --- | --- |
| `accuracy_pct` | 図の主要値。action一致率／stateの3クラス・dimension平均 |
| `original_element_accuracy_pct` | 保存済みの2クラスstate・dimension平均。actionではN/A |
| `official_exact_match_pct` | 元の公式scoreがquery全体で完全一致したか。stateは全dimension一致が必要で、dimension平均とは別 |

stateのparse失敗はaccuracyを欠損にし、平均の分母から除外します。選んだ件数 `subset_n` と実際の分母 `N`、`missing_count`、`parse_failures` を保存し、図の各パネル上にもHごとのNを表示します。全件parse失敗なら0%にせずN/Aです。

## 4. subsetとstdの定義

各条件のordinal=0〜29を1回だけshuffleします。条件ごとの乱数seedは `SHA256("42:condition_id")` の先頭8 bytesを整数化して作るので、他条件の追加や読み取り順序に依存しません。

- N=30：全query。
- N=20：shuffle順の先頭20件。
- N=10：同じshuffle順の先頭10件。必ずN10 ⊂ N20 ⊂ N30。

選択したordinalとquery IDを [query_subsets.json](query_subsets.json) に保存します。主要accuracy、元のstate accuracy、dimension別accuracyのすべてで同じsubsetを使います。parse失敗を見てから取り替えることはしません。

線はquery値の平均、帯は **平均±母標準偏差（ddof=0）** です。帯は信頼区間や標準誤差ではありません。表示だけ0〜100%へclipし、CSVのmean/stdはclipしません。actionはquery単位の0/100なので帯が広くなります。

元の30 queryはExp.3がepisodeの有効prefixから選んだものです。今回のshuffleはその30件内だけであり、dataset全体の無作為抽出ではありません。Hを変えると対象query indexも変わり、同一episode内の履歴も重なります。H効果の因果推定や、独立した30反復の実験とは区別してください。

## 5. ファイルとコードの読み方

| ファイル | 内容／今回の形状（ヘッダ除く） |
| --- | --- |
| [plot_metrics.py](plot_metrics.py) | 読込→検証→採点→subset→集計→作図の実行コード |
| [test_plot_metrics.py](test_plot_metrics.py) | 手計算できる小さな6つのテスト |
| [computed_metrics.csv](computed_metrics.csv) | 1 query＝1行、960行。元prediction/GTと追加accuracyを区別 |
| [state_dimension_metrics.csv](state_dimension_metrics.csv) | 1 state query×dimension＝1行、1,200行 |
| [aggregated_metrics.csv](aggregated_metrics.csv) | 32条件×3 subsetについて、Accuracy、実行時間、token使用量などを集計 |
| [state_dimension_aggregated.csv](state_dimension_aggregated.csv) | 条件×subset×state dimensionのmean/std/N、120行 |
| [query_subsets.json](query_subsets.json) | seed・条件seed・選択ordinal/query ID |
| [bin_edges.json](bin_edges.json) | Pendulumの公式action bin edges・境界規則 |
| [analysis_metadata.json](analysis_metadata.json) | 入力先・raw/ログ/コードhash・H・seed・図の一覧 |
| [validation.json](validation.json) | 件数、再採点照合、parse欠損、raw/ログ不変の確認結果 |
| [RESULT.md](RESULT.md) | 図と数値を確認した結果・解釈・制約。手書きなので自動再生成しない |

CSVはUTF-8、数値欠損は空欄、配列はJSON文字列です。`mean` / `std` は百分率／percentage points、`N` はそのmeasureの有効query数です。`original_element_accuracy_pct` がaction行で空欄なのは適用外で、API失敗ではありません。

コードの関数は次の役割に分けています。

| 関数（すべて `plot_metrics.py`） | 入力 → 出力／役割 |
| --- | --- |
| `load_results()` | 最新の保存ファイル → source path、manifest dict、records list。完全性・hash・対応関係を検証 |
| `validate_official_score()` | 保存record・Episode・EvaluationConfig → 公式 `_score_response()` との一致検査 |
| `score_queries()` | records・manifest → query行、dimension行、raw hash。3クラス評価だけ追加 |
| `select_subsets()` | query行・seed → 条件別の固定nested subset |
| `summarize_values()` / `aggregate()` | 同じsubsetの値 → mean/std/N。既存 `experiments.analysis.describe()` を再利用 |
| `action_bin_settings()` | manifest → 公式のbin設定を記録するdict |
| `plot_figures()` | 集計行 → Accuracy・実行時間・token使用量の2×4パネルを各3図、PNG/SVGに保存 |
| `main()` | 上記を順に呼び、CSV/JSON出力と前後のsource hash照合を行う |

既存の `experiments/common.py`、`experiments/analysis.py`、`upstream/LLM-Xavier/llm_x/` の関数はimportして再利用するだけで、この解析のために変更していません。Exp.4の「先頭N」方式とは別で、ここは固定seedによるnested samplingです。

## 6. Figure一覧

各図は2行×4列、横軸H、縦軸accuracy (%)、線mean、帯±stdです。PNGは2520×1260 pixels、SVGはvector形式です。

| subset | PNG | SVG |
| --- | --- | --- |
| N=30 | [図](figures/paper_matching_accuracy/paper_matching_accuracy_n30.png) | [vector](figures/paper_matching_accuracy/paper_matching_accuracy_n30.svg) |
| N=20 | [図](figures/paper_matching_accuracy/paper_matching_accuracy_n20.png) | [vector](figures/paper_matching_accuracy/paper_matching_accuracy_n20.svg) |
| N=10 | [図](figures/paper_matching_accuracy/paper_matching_accuracy_n10.png) | [vector](figures/paper_matching_accuracy/paper_matching_accuracy_n10.svg) |

同じsubset・パネル構成で、query単位の実行時間と総token使用量も出力する。

| 指標 | N=30 | N=20 | N=10 |
| --- | --- | --- | --- |
| 実行時間（秒/query） | [図](figures/paper_matching_execution_time/paper_matching_execution_time_n30.png) | [図](figures/paper_matching_execution_time/paper_matching_execution_time_n20.png) | [図](figures/paper_matching_execution_time/paper_matching_execution_time_n10.png) |
| 総token（input＋output/query） | [図](figures/paper_matching_token_usage/paper_matching_token_usage_n30.png) | [図](figures/paper_matching_token_usage/paper_matching_token_usage_n20.png) | [図](figures/paper_matching_token_usage/paper_matching_token_usage_n10.png) |

MSE・Pearson相関・10-bin単独指標の図は作成していません。特にstateは方向ラベル、Pendulum actionはbin IDなので、それを連続値とみなしたMSEや相関は計算していません。
