# Pendulum：Reasoning score × Action metric

行はScore 1〜4（1が最も高度）、列はLast Action / Next Action。H=5,10,20,30をpoolし、各評価指標につきGPT-3.5・Luna・Terra・Sol・全モデルの5枚、計25枚の300 dpi PNGを生成する。PDFは生成しない。

## 入力と対応付け

リポジトリルートからのパス：

- 採点CSV：`experiments/07_view_rawdata/reasoning_for_scoring_scored_astra_high.csv`
- 新3モデル：`experiments/06_new_models_n10/analysis/comparison_4models/tables/parsed_records.csv` の `model_alias != "3.5"`
- GPT-3.5全件：`experiments/03_1_gpt35_history_n30_joint/analysis/tables/parsed_records.csv`

4モデル比較表のGPT-3.5はn=10への抽出済みなので使用しない。採点CSVにIDがないため、保存時の行順を前提にModel・Task・Metrics・H内の出現順を0始まりordinalとして復元する。5列の一意性、全1,920件の一対一結合、元JSONLのquery ID、およびReasoning本文が元応答にそのまま含まれることを確認する。不一致時は停止する。全体には空Reasoningもあるが、今回のPendulum Action対象には存在しない。

結合後に `Task == "Pendulum-v1"`、`Metrics in ["last-action", "next-action"]` を抽出する。採点自体の変更やAPI呼び出しは行わない。

## 件数・除外

| Model | 対象 | bin有効 | トルク有効 |
|---|---:|---:|---:|
| GPT-3.5 | 240 | 237 | 230 |
| Luna | 80 | 80 | 80 |
| Terra | 80 | 80 | 80 |
| Sol | 80 | 80 | 80 |
| 全体 | 480 | 477 | 470 |

bin解析失敗3件、トルク解析失敗10件を各指標から個別に除外する。既存のcomponent成功フラグを使用し、失敗を不正解や0誤差に置換しない。セルの `n=有効件数 / 対象件数` は成分の有効件数であり、Pearson等が数学的に定義不能でもその件数は表示する。

## 指標と色

| ファイル名のprefix | 式 | 良い方向・共通色範囲 |
|---|---|---|
| bin_accuracy | `100 × mean(pred_bin == true_bin)` | 高い、0〜100% |
| bin_mae | `mean(abs(pred_bin − true_bin))` | 低い、0〜9 |
| nrmse | `sqrt(mean((pred_torque − true_torque)^2)) / 4` | 低い、0〜全5図の最大値 |
| pearson | 予測トルクと正解トルクのPearson相関 | 高い、−1〜1 |
| cosine | `dot(pred, true) / (norm(pred) × norm(true))` | 高い、−1〜1 |

binは回答に明示された0〜9の予測binと正解binを使用する。予測トルクからbinを再生成しない。トルクの正規化幅4は範囲−2〜2に対応する。MAE・NRMSEは正解率ではなく誤差指標である。

モデル別の色範囲は同一指標で統一し、低誤差が良好に見えるよう誤差指標では配色を反転する。範囲は `validation.json` に保存する。NRMSE最大値が全て0の場合のみ描画範囲を0〜1とする。

全モデルの図は、各Score × Metricセルについて算出可能なモデル別指標を単純平均する（モデル等重み）。GPT-3.5の件数が多くても重みは増えない。NRMSEもモデル別NRMSEの平均であり、全サンプルからのpooled NRMSEとは異なる。PearsonはFisher変換せず相関係数を、Cosineもモデル別類似度を算術平均する。値を算出できないモデルは除外し、図にはmodels=寄与モデル数 / 4を表示する。nは元サンプルの参考件数であり、平均の重みではない。モデル別では引き続きHをpoolする。Reasoning scoreと予測性能の関連を記述する図であり、因果効果やモデル間の同一query比較を示すものではない。

サンプル0件は灰色N/A。Pearsonは有効n<2または予測・正解のいずれかが定数の場合、Cosineはどちらかのノルムが0の場合にN/Aとする。少数セルの相関・類似度は不安定で、特にn=1の非ゼロCosineは±1になるため、件数と併せて読む。除外・N/A理由は `cell_summary.csv` に保存する。

## 成果物と再実行

- `plot_heatmaps.py`：照合・集計・作図コード。
- `joined_samples.csv`：480行。score、予測・正解、component有効フラグ、query ID、元JSONLパス・行番号、Reasoning、採点CSV上のレコード番号を含む。`score_csv_row` はヘッダーを含む論理レコード番号で、複数行CSVの物理行番号ではない。
- `cell_summary.csv`：5モデル区分 × 5指標 × 8セル＝200行。対象n・有効n・除外n・N/A理由を記録。
- `validation.json`：入力SHA256、件数、色範囲、N/Aセル一覧。
- `figures/{指標}_{3.5,luna,terra,sol,all}.png`：25枚。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python experiments/07_view_rawdata/reasoning_score_analysis/pendulum_score_performance/plot_heatmaps.py \
  --output-dir /tmp/pendulum_score_performance_new
```

Python・pandas・NumPy・Matplotlibを使用する。引数なしではスクリプトと同じフォルダへ出力する。既存成果物がある場合は停止するため、再実行には未使用の出力先を指定する。別のリポジトリ配置には `--repo-root` も指定可能。

全25枚の生成・目視確認、1,920件の結合と元応答照合、480件のセル件数合計、成分別有効件数、入力SHA256の実行前後一致を確認済み。既知の小例で完全一致率・MAE・NRMSEの式と、空配列・定数・ゼロノルム・n=1の定義不能条件も確認した。

旧sample-pooled版の図と集計は `sample_pooled_archive/` に保存。現在の `figures/*_all.png` と `cell_summary.csv` はモデル等重み版。集計CSVの `n_models` と `aggregation` で寄与数・集約方式を確認できる。
