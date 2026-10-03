# GPT-3.5 / Terra / Luna / Sol 比較（N=10、PNGのみ）

このREADMEは解析出力directoryにもコピーする。結果の数値は同じdirectoryの`RESULTS.md`、全図の一覧は`figure_index.json`。

## 実行方法・入力元・出力先

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
source .venv/bin/activate
python experiments/06_new_models_n10/analysis/compare_models.py --allow-api-failures
```

保存先：`experiments/06_new_models_n10/analysis/comparison_4models/`。
既存directoryは上書きしない。再実行は`--output-dir experiments/06_new_models_n10/analysis/comparison_4models_v2`など新しい場所を指定する。
`--allow-api-failures`は「全件試行済みのAPIエラーを欠損として残し、有効応答だけで指標を計算する」明示指定。未指定ならAPI失敗がある結果の評価を拒否する。API呼出し・再送・補充抽出は一切しない。

入力は03_1の取得済みGPT-3.5結果と、05および06の完了/全件試行済みbatch。退避batch・probeは使わない。
05+06を検証して新しい`results/merged/<識別子>/`へオフライン統合する。05は各モデル40件だけを1回含める。
元のresults、raw NPZ、既存の図やCSVは変更しない。保護対象のhashと画像/CSV読戻し検査を保存する。

## 選択と欠損の扱い

- GPT-3.5：03_1の各条件30件から、既存`select_subsets(seed=42)`で10件抽出。既存03_1/analysis/query_subsets.jsonのN10集合と一致確認する。
- Terra/Luna/Sol：05+06の各条件10件全部。API失敗も選択10件の中に残し、別queryで補充しない。
- 各モデル32条件×10件＝320件、計1,280件。GPT-3.5と新モデルで入力query集合が同じとは限らないため、paired-model比較ではない。
- APIエラーとparse失敗は区別する。APIエラーには応答・score・token数を作らず、required componentの値を欠損とする。
- Accuracy/NRMSE/Pearson/Cosine等はcomponentごとの有効応答で計算。`selected_n`、`valid_n`、`na_reason`を残す。図の`n=...`は10件未満の有効件数、`N/A`は定義不能。
- `accuracy_all_selected`は従来からの「無効を不正解とした代替値」でCSVにのみ残す。主図はこれを使わない。
- token/time図は成功応答の実測値のみ。API失敗の通信時間は`api_failures.json`に別保存し、0秒/無料にはしない。05再利用分は当初の推論時間を保持する。
- parse_success_rate図は、全required componentが成功したquery数÷選択10件。API失敗も含めるので、純粋な「応答取得後のparser成功率」ではない。component別内訳もCSVに保存する。

NRMSE等の式・正規化範囲・bootstrapは03_1共通コードをそのまま利用する。NRMSE帯は既定1,000回paired-query bootstrapのsample SD、token/time帯はquery間sample SD。Accuracy/Pearson/Cosine/bin MAEには帯を付けない。
有効1件のbootstrap SD=0を安定性の証拠と解釈しない。モデル・H・componentごとに有効集合が異なる点にも注意する。

## 色・図の構成

| モデル | 色 | HEX | marker |
|---|---|---|---|
| GPT-3.5 Turbo | 青 | #0072B2 | 丸 |
| GPT-5.6 Terra | 緑 | #009E73 | 四角 |
| GPT-5.6 Luna | 橙 | #E69F00 | 三角 |
| GPT-5.6 Sol | 紫 | #CC79A7 | 菱形 |

線系列は必ず4モデル。状態成分や角度の復元方法をモデルと同じ図の系列に混ぜない。混同行列は各モデル色の濃淡（0〜100%共通尺度）を使う。

| directory / 図 | 枚数 | レイアウト |
|---|---:|---|
| cross_task_10/accuracy.png | 1 | 2 Task × 4 Metric、4モデル比較 |
| cross_task_10/state_absolute_macro_nrmse.png、state_delta_macro_nrmse.png | 2 | 2 Task × Next/Last State、4モデル比較 |
| cross_task_10/state_dimensions/{Task}/state_{absolute,delta}_{成分}_{指標}.png | 30 | position/velocity/cos/sin/angular_velocity別。1行×Next/Last State。NRMSE/Pearson/Cosine別 |
| cm/ | 24 | 6 Task/Metric組 × 4モデル。各図H=5/10/20/30の2×2 |
| pendulum_10/pendulum_action_metrics.png | 1 | NRMSE/Pearson/Cosine/bin MAE × Next/Last Action |
| pendulum_10/pendulum_delta_theta_metrics_from_{absolute,delta}.png | 2 | 復元方法別。NRMSE/Pearson/Cosine × Next/Last State |
| diagnostics_10/ | 5 | parse成功率、input/output/total tokens、processing_time。2 Task × 4 Metric |
| 合計 | 65 | PNGのみ。SVGは生成しない |

ファイル名は元実装と同じ`nrmse`表記（Normalized Root Mean Square Error）。

生成後、画像の内容を変えず、`state_dimensions/`の下をTask別に整理した。
`MountainCar-v0/`直下に12枚、`Pendulum-v1/`直下に18枚を移動した。
一時的に設けた`state/`・`action/`は、画像を移動し空であることを確認してから空ディレクトリのみ削除した。
画像は削除・再生成せず、`figure_index.json`のパスのみ追従した。生成コード自体は変更していない。

### 混同行列の番号

| 番号 | Task | Metric | 例 |
|---|---|---|---|
| 01 | MountainCar | Last Action | 01_35_mountaincar_last_action.png |
| 02 | MountainCar | Next Action | 02_luna_mountaincar_next_action.png |
| 03 | MountainCar | Last State | 03_terra_mountaincar_last_state_direction.png |
| 04 | MountainCar | Next State | 04_sol_mountaincar_next_state_direction.png |
| 05 | Pendulum | Last State | 05_35_pendulum_last_state_direction.png |
| 06 | Pendulum | Next State | 06_sol_pendulum_next_state_direction.png |

元の03_1と同じ6種。MountainCar Actionは3 action ID、Stateは3 direction classの次元pool。
Pendulum Actionの10-bin混同行列は元図にないため追加しない。

## ファイルと関数

| ファイル | 関数 | 処理 |
|---|---|---|
| experiments/06_new_models_n10/analysis/compare_models.py | mainの呼出し | 実行入口 |
| experiments/common_analysis/comparison.py | main | 読込み・統合・モデルごとの抽出/集計、CSV/JSON/RESULTSと図の出力、hash検証 |
| experiments/common_analysis/joint_data.py | load_and_parse | 応答のmarkerを解析、rawからGT算出。明示許可時のみAPI失敗を欠損として保持 |
| 同上 | select_subsets | 03_1と同一のseed=42抽出 |
| experiments/common_analysis/joint_aggregate.py | aggregate_all | モデルごとに同じ評価式を実行。モデルをpoolしない |
| experiments/common_analysis/joint_metrics.py | nrmse_statistics等 | 既存の数値評価・bootstrap |
| experiments/common_analysis/comparison_plots.py | make_comparison_figures | long tableだけを使い65図の構成を作る |
| 同上 | draw_models / confusion_figure / save_png | 固定色4系列 / モデル別混同行列 / PNGのみ新規保存 |

CSVは`tables/`に13種類。`all_metrics_long.csv`等にモデル名、値、std、有効件数があり、`condition_coverage.csv`はAPI失敗と有効回答の件数。
`query_subsets.json`はモデル別の抽出query ID、`analysis_metadata.json`は色・seed・入力/コードhash・評価方針。
`api_failures.json`は失敗queryの元record（送信prompt・error・request ID等を含む）。成功例や0点の予測に置換しない。
