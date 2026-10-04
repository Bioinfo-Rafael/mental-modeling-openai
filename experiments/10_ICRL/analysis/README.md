# 10_ICRL 可視化・HTMLの計画と実行手順

09の `analysis/plot_comparisons.py`、`analysis/build_results_viewer.py`、`templates/results_viewer_template.html` を基にした。推論評価済み72回答を反映した [HTML](../view_rawdata.html) と比較図7枚を生成済み。

テスト10件が成功し、全図を目視確認した。HTMLは72件のscore、24条件の絞り込み、詳細／一覧切り替え、前後移動、検索、JSON保存、0件時の表示をJavaScriptの操作ロジックテストで確認。OSの操作権限が付与されていないため、実ブラウザでの操作・レイアウト確認は未実施。

## 作図するもの

09と同様、各PNGは横に3パネル（正解率・MAE・推論レベル）を並べる。**7枚、合計21パネル**を生成する。

| PNG | 横軸またはmatrix | poolする条件 | 1本／1セルのn |
| --- | --- | --- | --- |
| [01_prompt_pooled.png](../results/analysis/comparisons/figures/01_prompt_pooled.png) | Prompt 4群の棒 | E・モデル・repeat | 18 |
| [02_prompt_by_model.png](../results/analysis/comparisons/figures/02_prompt_by_model.png) | Prompt 4群、各群にTerra/Lunaの2本 | E・repeat | 9 |
| [03_episodes_pooled.png](../results/analysis/comparisons/figures/03_episodes_pooled.png) | E=1/3/9の3群の棒 | Prompt・モデル・repeat | 24 |
| [04_episodes_by_model.png](../results/analysis/comparisons/figures/04_episodes_by_model.png) | E=1/3/9の3群、各群にTerra/Lunaの2本 | Prompt・repeat | 12 |
| [05_matrix_pooled.png](../results/analysis/comparisons/figures/05_matrix_pooled.png) | 行Prompt × 列Eの4×3 matrix | モデル・repeat | 6 |
| [06_matrix_terra.png](../results/analysis/comparisons/figures/06_matrix_terra.png) | Terraの4×3 matrix | repeat | 3 |
| [07_matrix_luna.png](../results/analysis/comparisons/figures/07_matrix_luna.png) | Lunaの4×3 matrix | repeat | 3 |

Promptの表示順は **Direct → PPO → ICRL (K=2) → ICRL (K=5)**。対応する内部値は `direct, ppo_prior, icrfqi_k2, icrfqi_k5`。Eは **1 → 3 → 9**。matrixは上からPrompt、左からEとし、モデル別にさらに条件分解しない。

集約は各回答を等重みとするsample pooling。各モデル・条件で3 repeatsが揃うため、モデルをpoolする場合も両モデルの寄与は等しい。たとえばモデル別Promptの棒は、3つのE × 3 repeats = **n=9**。

| 指標 | 定義 | 軸・色の範囲 |
| --- | --- | --- |
| 正解率 | final action binが正解binと一致した割合 × 100 | 0–100%。棒の上に注記用余白を付ける。 |
| MAE | `mean(abs(predicted torque - ground-truth torque))` | 0から、モデル別12条件セルの最大MAEを0.1刻みで切り上げた共通上限まで。全て0なら0.1。 |
| 推論レベル | 手動採点した `score` の平均 | score 1–4、高いほど具体的・定量的。matrixの色は1–4。棒は0を基点にする。 |

正解率とMAEは既存Joint scorerの保存値を使い、bin IDの差をMAEにしない。推論レベルは08・09と同じ番号体系をそのまま使い、反転しない。平均scoreは順序尺度を等間隔とみなした記述的要約。内部のFQI実行有無を示す値ではない。

棒と各セルには値とnを記載。Terra/Lunaは09と同じ青／橙。正解率・推論レベルは `YlGnBu`、MAEは `YlOrRd`。各指標の範囲を全図で共通にする。nは同じtarget queryへの回答数であり、独立した評価episode数ではないため、信頼区間・有意差検定は付けない。

## 入力と検証

入力は `results/api/manifest.json`、APIの各journal、`reasoning_for_scoring_scored_astra_high.csv`。`query_id` で対応づけ、実験条件、Reasoning本文、72件の一意性、2 models × 4 methods × 3 E × 3 repeatsを検証する。

作図には全72件の有効なbin・continuous actionと、1–4の推論scoreが必要。欠落、空欄、重複、不一致があれば**出力ファイルを作る前に停止**する。未評価を0扱いしたり、欠損行を除いてnを変えたりしない。CSVは読み取りだけで、採点自体は行わない。

`results_data.py` は09のjournal readerとReasoning抽出処理を再利用。`plot_comparisons.py` は09の `aggregate`、指標定義、軸設定、値/nの注記、CSV writerを直接再利用する。09のファイルは今回変更していない。

## HTML

09と同じ、データを埋め込んだ単一のオフラインHTMLを `10_ICRL/view_rawdata.html` に生成する。

- 詳細表示／回答一覧の切り替え、前後移動、回答選択、JSON保存。
- モデル・Prompt・K・E・repeat・推論レベル・正誤で絞り込み、ID・入力・回答を検索。
- system、user全文（補助episodeと追加指示を含む）、target history/question、回答全文、正解・予測・誤差、tokens・時間、raw responseを表示。
- 補助episodeのpathと各episodeのSHA256も条件欄に表示する。
- 採点CSVがない／一部scoreが空欄の場合も閲覧でき、推論レベルは「未評価」。評価済み件数と作成日時を表示する。不正な値や対応の不一致は拒否する。

HTMLは生成時点のスナップショット。採点完了後の内容を反映するには、新しい出力先へ再生成する。09のテンプレートをコピーし、few-shot/reward-ablation用の表示をPrompt/K/Eへ変更した。外部CDN・サーバー・API接続は使わない。

## 推論採点が完成した後の実行

repo rootで以下を実行する。実験APIは送信しない。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python -B experiments/10_ICRL/analysis/plot_comparisons.py
.venv/bin/python -B experiments/10_ICRL/analysis/build_results_viewer.py
```

作図出力は `results/analysis/comparisons/`：`figures/` に7 PNG、`joined_samples.csv`、`prompt_summary.csv`、`episode_summary.csv`、`matrix_summary.csv`、入力SHA256・n・検証結果を記した `validation.json` を保存する。集計CSVの `model=pooled` はモデルを統合した行。

既存の出力は上書きしない。再生成時は新しい保存先を指定する（相対pathは実行時のcwd基準）：

```bash
.venv/bin/python -B experiments/10_ICRL/analysis/plot_comparisons.py \
  --output-dir experiments/10_ICRL/results/analysis/comparisons_v2
.venv/bin/python -B experiments/10_ICRL/analysis/build_results_viewer.py \
  --output experiments/10_ICRL/view_rawdata_v2.html
```

コードのテストは `.venv/bin/python -B -m pytest experiments/10_ICRL/tests/test_analysis.py -q`。テスト用scoreは一時ディレクトリ内だけに作成し、実験の採点CSVは変更しない。

Codexに生成・確認まで依頼する場合：

```text
10_ICRLの推論採点が完了したので、analysis/README.mdに従ってテストを実行し、
比較図7枚とview_rawdata.htmlを生成してください。
図の目視確認とHTMLのフィルター・詳細／一覧切り替えも確認してください。
既存の採点CSV・API結果は変更せず、実験APIも再送信しないでください。
```
