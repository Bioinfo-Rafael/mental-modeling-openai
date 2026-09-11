# Joint実験のオフライン解析

## 共通化後のコード配置

評価実装は [experiments/common_analysis](../../common_analysis/README.md) へ移設した。
このdirectoryの`analyze.py`は共通`runner.main()`を呼び、`joint_*.py`は互換importだけを残している。
以下の説明に登場するparse・集計・作図関数の実体は共通directoryにある。計算式と既定N=10/20/30は変更していない。
05/06も同じ実装を使い、06ではN=10のみをモデル別に生成する。
既存のCSV・図・RESULTS・監査JSONは共通化前のスナップショットとして保持し、上書きしていない。
共通化後に再計算したい場合は、新しい`--output-dir`（このanalysisの下）か明示的な`--overwrite-derived`を使う。

取得済み960回答を、保存されたLLM回答から再parseして評価する。追加API送信はしない。
既存の実験コード・`results/`・raw datasetは変更せず、この`analysis/`以下だけに派生成果物を保存する。
数値結果と注意点は [RESULTS.md](RESULTS.md)、全ファイルの一覧は [figure_index.json](figure_index.json)。

## 1. 実行方法と入出力

Repository rootで、既存のPython環境を使う。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python -B -m pytest experiments/03_1_gpt35_history_n30_joint/analysis/test_joint_analysis.py -q -p no:cacheprovider
python -B experiments/03_1_gpt35_history_n30_joint/analysis/analyze.py
```

すでに成果物が存在する場合、最後のコマンドは上書きを拒否する。解析の派生ファイルだけを再生成する場合：

```bash
python -B experiments/03_1_gpt35_history_n30_joint/analysis/analyze.py --seed 42 --bootstrap 1000 --overwrite-derived
```

`--overwrite-derived`は、この解析が生成するCSV・図・JSONのみを同じ名前で更新する。
README、RESULTS、元の実験結果、他の解析を削除・上書きしない。実験の`run.py`は呼ばない。
`RESULTS.md`は今回のseed=42・bootstrap=1000で確認したスナップショットであり、再生成時に自動更新されない。

入力はすべてread-only：

| 入力 | 使用目的 |
|---|---|
| `../results/records.jsonl` | 960件の`assistant_text`、`raw_response`、query識別情報、実測token・時間 |
| `../results/runs/<condition>/episode_000/predictions.jsonl` | 指定されたrunsとの対応確認。index、prompt、GTをrecordsと全件照合 |
| `../results/manifest.json` | 条件・query ID・episode・index・threshold・action bin設定 |
| `../results/summary.json` | 完了状態の確認 |
| `data/llmx_data/.../episodes/*.npz`（repository root基準） | 元trajectoryから正解のstate/actionを取得。記録済みSHA-256を確認 |
| `upstream/LLM-Xavier/llm_x/task.py` | 正規化範囲を確認したtask定義 |

`runs/.../predictions.jsonl`には生の回答全体がないため、対応する`records.jsonl`の回答を正本にする。
`assistant_text`と`raw_response.choices[0].message.content`の一致も全件確認する。
旧`prediction`・`status`・`element_accuracy`・`metrics.json`を今回の採点には使わない。

出力は、3種類×N=10/20/30の9 directory、`tables/`、subset・監査JSON。
`protected_sha256.json`に、実験結果全ファイル・既存experiment/upstream Pythonファイルのhashを保存し、処理前後の一致を確認する。
raw episodeも別途前後のhashを確認する。

## 2. コードの読み順・関数の対応

```text
analyze.py: main()
  joint_data.py: load_and_parse()
    parse_response() → parse_component()  # 回答componentを独立に抽出
    Episode.load()                       # rawデータを読むだけ
    state_directions() / bin_actions()    # 公式のGT定義を再利用
  joint_data.py: select_subsets()
  joint_aggregate.py: aggregate_all()
    accuracy_rows() / diagnostic_rows()
    continuous_rows()
      joint_metrics.py: nrmse_statistics() / pearson() / cosine()
      joint_metrics.py: angular_pair() → theta() / wrap_to_pi()
  analyze.py: validate_tables() / flatten_records() / write_csv()
  joint_plots.py: make_figures()
    draw_line() / confusion_plot() / save()
  analyze.py: validate_saved_outputs() / write_json()
```

`load_and_parse()`は`(source: Path, manifest: dict, records: list[dict], episode_hashes: dict)`を返す。
`aggregate_all()`は`dict[table_name, list[row_dict]]`、作図関数は生成ファイルの相対path一覧を返す。
解析データと作図を分け、作図側ではsubsetの再抽出・採点をしない。
数値計算のNumPy入力は、有効query数V×次元D。ActionはD=1、MountainCar StateはD=2、Pendulum StateはD=3。

## 3. N=10 / 20 / 30とseed

各Task×Metric×Hに30件、32条件で960件あることをassertする。
taskごとに全条件の`(episode_path, query_index)`の和集合をsortし、`random.Random(42).shuffle()`する。
この共通順位で各条件の30件を並べ、先頭10/20/30件を使う。
したがって各条件で`subset_10 ⊂ subset_20 ⊂ subset_30`が成立する。

同じtaskの同じepisode/indexにはMetricやHをまたいで同じ順位を与える。
ただし条件によって対象indexが違うため、異なるH・Metricで完全に同じquery集合になるとは限らない。
物理的に異なる2 task間で同じindexを同一観測とはみなさない。
全図・全指標は共通の [query_subsets.json](query_subsets.json) を参照し、parse失敗時の補充抽出はしない。
このNは独立実験の反復数ではなく、各条件の保存済み30回答から使うquery数。

## 4. Joint回答のparse

| 対象 | component | 読むfinal marker | 値・shape |
|---|---|---|---|
| MountainCar Action | `action` | `Final action choice: [2]` | action ID 0/1/2、(1,) |
| Pendulum Action | `action_value` | `predictions = [0.6]` | raw torque、(1,) |
| Pendulum Action | `action_bin` | `>>Final action bins: [6]` | bin ID 0〜9、(1,) |
| State | `direction` | `predictions = ["INC", "DEC", ...]` | DEC=0 / INC=1 / UNCH=2、(D,) |
| State | `state_value` | `>>Final state values: [...]` | absolute state、(D,) |
| State | `state_delta` | `>>Final state deltas: [...]` | forward delta、(D,) |

数値は`ast.literal_eval()`で読み、有限のint/floatだけを受理する。`eval()`やreasoning中の数字の推測は行わない。
行頭の明示markerだけを読む。`>>`の省略、番号・箇条書き、太字、`[Final state values]`のような見出し括弧、直後の改行は書式差として許容する。
別名の`Next state`や`Predicted numerical values`からの推測はしない。
`[[...]]`を勝手にflattenせず、指定した1次元listを要求する。MountainCar actionのみ単一scalarも許容する。

markerなし、次元数不一致、placeholder（v0/d0）、非数値、非有限値、範囲外bin、未定義directionは失敗。
同じmarkerが複数ある場合、同一の有効値なら受理し、矛盾する値や未完成templateが混在する場合は失敗にする。
これは保守的なfinal-marker parserであり、「人間には読めるが規定書式でない回答」も失敗になり得る。
各componentのvalue / ok / errorを保存し、他componentの失敗によって有効値を捨てない。
連続値は範囲外でも有限値ならそのまま採点し、clip・0埋め・補完しない。
`all_required`はそのTask/Metricに必要なすべてのcomponentがparse成功した場合のみTrue。

## 5. 正解値とAccuracy

query indexをiとし、元episodeの`early = state[i]`、`late = state[i+1]`を取得する。
Next Stateのabsolute GTはlate、Last Stateはearly。
両方ともdeltaは必ず`late - early`。連続値の演算はfloat64で行うがrawファイルには書き込まない。
Action GTは元episodeの`action[i]`。

Direction GTは公式`llm_x.metrics.state_directions()`を再利用する。
元dtypeのstateを小数5桁でroundし、manifestの`state_threshold`より差の絶対値が小さければUNCH、正ならINC、それ以外DEC。
`allow_unchanged=True`。記録されたGTと一致を確認する。
Pendulum bin GTも公式`llm_x.metrics.bin_actions()`を呼ぶ。
`np.linspace(-2,2,11)`のedgesに`np.digitize(..., right=True)-1`、最後に0〜9へclipする既存仕様。
境界は下側binに入り、例としてtorque=0はbin4、-1.6はbin0、-2はbin0、+2はbin9。
独自の四捨五入や、promptにある区間表記への変更はしていない。

| Accuracy | 計算 |
|---|---|
| MountainCar Action | action IDのexact match割合 |
| Pendulum Action | 明示されたaction binのexact match割合（torqueをbinとして使わない） |
| State Direction | 全query×全state次元をpoolしたDEC/INC/UNCHのexact match割合 |

図は**有効予測のみを分母にした割合×100**、線のみでstd帯なし。
Stateでは分母`valid_units = valid_n × D`。全次元同時正解率ではない。
parse失敗を不正解とする別値`accuracy_all_selected`もCSVに併記する。
主図を読む際には必ずparse successも確認する。有効分母の異なる旧解析の値とは単純比較しない。
Directionの次元別AccuracyはCSVのみ。Confusion MatrixはGT行・予測列、Stateは次元pool。
各cellにcountとGT行内の割合を表示し、GTの出現0件の行は割合N/A（0%を捏造しない）。

## 6. 連続指標、正規化、bootstrap

`NRMSE_d = sqrt(mean((prediction_d - GT_d)^2)) / range_width_d`。
deltaも対応する元stateの幅で割る。サンプルの観測min/maxや、deltaの倍の幅で割らない。
範囲は既存 [task.py](../../../upstream/LLM-Xavier/llm_x/task.py) のMountainCar/Pendulum定義で確認済み。

| task | 次元 | 理論範囲 | 正規化幅 |
|---|---|---|---:|
| MountainCar | position | [-1.2, 0.6] | 1.8 |
| MountainCar | velocity | [-0.07, 0.07] | 0.14 |
| Pendulum | cos(theta) | [-1, 1] | 2 |
| Pendulum | sin(theta) | [-1, 1] | 2 |
| Pendulum | angular_velocity | [-8, 8] | 16 |
| Pendulum | action torque | [-2, 2] | 4 |

Macro NRMSEは次元別NRMSEの算術平均。raw RMSEを次元横断で平均しない。

NRMSEの線は選択subsetのpoint estimate。帯はpaired-query bootstrap 1,000回の標本標準偏差（ddof=1）。
subset内で当該componentが有効なV queryについて、予測とGTのペアをV件復元抽出する。
全次元を同じ抽出indexで再計算し、Macroも各bootstrap内で次元平均を取ってからstdを求める。
次元別stdの単純平均でMacroのstdを代用しない。`bootstrap_mean`もCSVに保存する。
基準seedは42で、condition ID・N・componentからSHA-256で決定した子seedを使う（CSVにも保存）。
これは**parse成功に条件付けた**不確実性であり、parse失敗率の揺らぎや時系列依存を補正するものではない。
V=1ならbootstrap stdは0になるが、十分な精度・安定性を意味しない。
帯の下側が負になる場合、描画のみ0で打ち切る。元のstdは変更しない。帯は信頼区間ではない。

Pearson rとCosineは、各Task×Metric×H×N×dimensionについて、V queryを並べた予測vectorとGT vector間で計算する。
scalarごとのCosineを平均しない。PearsonはV<2または定数列、Cosineはzero normなら欠損とし、0に置換しない。
CSVの数値欠損は空欄（pandasで読むとNaN）、JSONはnull、`na_reason`に理由を保存する。図は欠損点をつながない。
Pearson/Cosineは線のみでstdなし。NRMSEの縦軸は0–1固定にしない。subplotごとの自動scaleなので目盛りも比較する。

Pendulum Bin MAEは`mean(abs(predicted_bin - GT_bin))`。有効な明示binに対して計算し、線のみで表示する。
torque評価とbin評価は別の有効query集合になり得る。

## 7. Pendulumのforward Δtheta

`theta = atan2(sin(theta), cos(theta))`、`wrap(x) = (x + pi) % (2*pi) - pi`。
GTは`wrap(theta_late - theta_early)`。

| 復元元 | Next State | Last State |
|---|---|---|
| absolute state | 予測lateと真のearlyから角度差 | 真のlateと予測earlyから角度差 |
| state delta | `pred_late = true_early + pred_delta` | `pred_early = true_late - pred_delta` |

復元したcos/sinからatan2で角度を作り、必ずforward-timeの差をwrapする。
cos/sin=(0,0)の復元は角度未定義として除外し、理由をparsed_recordsに残す。勝手に正規化・角度0としない。
angular NRMSEは`error = wrap(pred_delta_theta - GT_delta_theta)`のRMSEをpiで割る。bootstrap帯あり。
Pearson/Cosineはwrapped Δthetaのquery vector同士で計算し、線のみ。
これは通常のPearsonであり、別定義の円周相関ではない。
absolute由来とdelta由来の2本を同じ図に描くが、componentごとの有効件数はCSVで確認する。

## 8. DiagnosticsとCSV

Parse Success Rateは`100 × component有効query数 / selected_n`。all_requiredも併記、線のみ。
input/output/total tokensは保存済みAPI usageに由来する実測値で、tokenizer見積もりではない。
processing_time図は`query_elapsed_seconds`（query開始から完了まで、API以外の処理も含む）。
`request_elapsed_seconds`（API要求の経過時間）もCSVに残すが、図には別線を足さない。
tokenと時間は選択した全queryの測定値を使用し、parse失敗した回答も除外しない。線=平均、帯=通常のquery間sample std（ddof=1）、bootstrapはしない。

| `tables/`のCSV | 1行の単位・内容 |
|---|---|
| `parsed_records.csv` | 1回答。960行。query ID、元ファイル/行、各componentの値・成否・理由、GT、token/time、角度復元 |
| `accuracy_metrics.csv` | 条件×NのAccuracy、Pendulum bin MAE。120行 |
| `continuous_metrics.csv` | 条件×N×component×dimension×連続指標。1,032行 |
| `state_dimension_metrics.csv` | State absolute/deltaの次元別NRMSE・Pearson・Cosine。720行 |
| `state_macro_metrics.csv` | State absolute/deltaのMacro NRMSE。96行 |
| `pendulum_action_metrics.csv` | torque連続3指標とbin MAE。96行 |
| `pendulum_delta_theta_metrics.csv` | 2復元方法×angular連続3指標。144行 |
| `direction_dimension_accuracy.csv` | 次元別Direction Accuracy。120行 |
| `diagnostics.csv` | component parse率・token/time平均std。792行 |
| `confusion_matrices.csv` | 条件×N×GT class×predicted classのcell。648行 |
| `all_metrics_long.csv` | 指標の統合long table。2,064行（confusion cellと個別回答は別表） |
| `parse_summary.csv` | Task×Metric×componentの全120回答parse件数・失敗理由。26行 |

多くの表は`continuous_metrics`等の目的別viewなので、複数CSVを足すと同じ結果を重複集計する。
共通columnは`task, metric, H, subset_n, component, dimension, metric_name, value, std, selected_n, valid_n`。
`valid_n`はquery数であり、Stateの次元pool後の要素数ではない。
continuousの`parse_success_rate`は元componentのparse率で、角度が未定義ならparse済みでも`valid_n`から除かれる。
resource行の同名columnは測定値の存在率を表す。`std_kind`でbootstrapと通常stdを区別する。
vectorはJSON文字列としてセルに保存する（例`[0.2, 0.01]`）。欠損やnot_applicableを0と解釈しない。
CSV作成ではSpreadsheetsスキルの欠損・有効件数・出典追跡の方針を反映した。科学図・集計は指定どおりPythonで実装した。

## 9. Figure一覧

全図のH順は5,10,20,30。PNGと編集可能なSVGを同時出力する。
各Nに22図、N=10/20/30で66図（PNG66 + SVG66）。ファイル名は以下の`.png`に対応する`.svg`もある。

| directory | 用途・レイアウト | Figure |
|---|---|---|
| `cross_task_N/` | 2 Taskの比較。Accuracyは2×4 | `accuracy.png` |
| 同上 | MountainCar action confusion、各Hを2×2 | `cm_mountaincar_next_action.png`, `cm_mountaincar_last_action.png` |
| 同上 | State direction confusion、各Hを2×2 | `cm_mountaincar_next_state_direction.png`, `cm_mountaincar_last_state_direction.png`, `cm_pendulum_next_state_direction.png`, `cm_pendulum_last_state_direction.png` |
| 同上 | absolute/delta Macro NRMSE、task行×NS/LS列の2×2 | `state_absolute_macro_nrmse.png`, `state_delta_macro_nrmse.png` |
| 同上 | absolute次元別、2×2 | `state_absolute_dimension_nrmse.png`, `state_absolute_dimension_pearson.png`, `state_absolute_dimension_cosine.png` |
| 同上 | delta次元別、2×2 | `state_delta_dimension_nrmse.png`, `state_delta_dimension_pearson.png`, `state_delta_dimension_cosine.png` |
| `pendulum_N/` | Pendulum固有のcontinuous action、4指標行×NA/LA列 | `pendulum_action_metrics.png` |
| 同上 | Pendulum angular analysis、3指標行×NS/LS列 | `pendulum_delta_theta_metrics.png` |
| `diagnostics_N/` | parse/token/time、task行×4 Metric列の2×4 | `parse_success_rate.png`, `input_tokens.png`, `output_tokens.png`, `total_tokens.png`, `processing_time.png` |

## 10. Validationと解釈上の注意

`test_joint_analysis.py`の18ケースを実行済み。action exact match、binの独立parse・公式境界・MAE、direction pool、state value/delta parse、異常値拒否、NRMSE/Macro/bootstrap、Pearson/Cosineの未定義、atan2/wrap、NS/LS両方のforward符号、nested subsetを確認した。
実データでも全480 State GT delta、960件のrun対応、subset、accuracy分母、Macro定義、有効件数・欠損理由をassertする。
全12CSVを読み戻し、全PNGを画像として検査し、全SVGをXMLとしてparseする。結果は [validation.json](validation.json)。

1条件の30件は同一episode内のqueryであり、history windowも重なる。queryを独立同分布とみなす通常bootstrapは時系列依存を補正しない。
NやHの増加による有意な改善、別episodeへの一般化をこの図だけで主張しない。
小さなNRMSEと高いdirection/相関は別物。特に広い理論幅で正規化した微小な状態変化は、NRMSEだけでは良く見える場合がある。
API成功960件は、全componentが正しい形式・正確な値で返ったことを意味しない。
API再送信、既存scorerの変更、commit/pushは行っていない。
