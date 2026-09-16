# Reasoning score analysis

## 入力・定義

入力: `../reasoning_for_scoring_scored_astra_high.csv`。既存scoreを可視化するもので、Reasoningの再採点は行わない。採点依頼の原文は [Prompt.md](Prompt.md) に保存する。

- 実際の列名: `model`, `Task`, `Metrics`, `H`, `Reasoning`, `score`。
- Model表示名: `3.5` → GPT-3.5、`luna` → Luna、`terra` → Terra、`sol` → Sol。この順で統一。
- Task: `MountainCar-v0`, `Pendulum-v1`。
- Metrics: `last-action`, `last-state`, `next-action`, `next-state`。
- Pendulum比較の抽出条件: `Task == "Pendulum-v1"` かつ `Metrics` が **`last-action` / `next-action`**。図ではLast Action / Next Actionと表示。
- H: 5, 10, 20, 30。この順で表示し、折れ線のx位置は実際のHに対応する。
- Score 1: 明示的な定量・数学的推論。Score 2: 具体的な履歴・観測に基づく推論。Score 3: Agent/policy傾向の一般化。Score 4: 一般的な物理直感・曖昧な推測など。最終回答の正誤を表す指標ではない。
- `reasoning_level = 5 - score`。1〜4で、高いほど高度。
- `high_level_reasoning = score <= 2`、その平均が `P(score <= 2)`。
- `guessing_rate = P(score == 4)`。一般的・弱い推論の割合であり、ランダム回答と同義ではない。低いほど良い。

全stacked barで、下からScore 1→2→3→4。色は濃青・水色・淡橙・濃赤で統一し、各barの上にnを表示する。12%以上の区画に割合を丸めて表示するため、表示値の合計は100%からずれる場合がある。区画の高さは丸め前の割合を用いる。

## データ検証・集約

全1,920行、欠損score 0件、除外0件。128個の `Model × Task × Metrics × H` 条件が全て存在する。

| Model | 条件数 | 各条件のn | 各Model × Hのn | 全行数 |
|---|---:|---:|---:|---:|
| GPT-3.5 | 32 | 30 | 240 | 960 |
| Luna | 32 | 10 | 80 | 320 |
| Terra | 32 | 10 | 80 | 320 |
| Sol | 32 | 10 | 80 | 320 |

想定n=10に一致するのはLuna・Terra・Solの96条件。GPT-3.5の32条件はn=30。各条件の件数・平均・率は `condition_summary.csv`、Model × Hの集約値は `model_h_summary.csv`、全unique値・入力SHA256・検証結果は `validation.txt` に保存する。

集約は各行を等重みとするsample pooling。Model × H内では8個のTask × Metric条件のnが等しいため、条件平均の単純平均と一致する。一方、Modelをpoolする図09・10ではGPT-3.5が全体の50%、他3モデルが各16.7%を占める。モデル均等重みの結果ではない。

scoreは順序尺度であり、平均levelは隣接レベルを等間隔と置く記述的な要約。分布と率を併せて確認する。図01の薄い点は8個のTask × Metric平均であり、個別サンプルや信頼区間ではない。サンプル対応・独立性を示すIDがないため、信頼区間や有意差検定は付けていない。図からモデル能力の序列やHの因果効果を確定しない。

## 図の読み方

各図を `figures/` に300 dpi PNGで保存する。PDFは生成しない。

| 図 | 比較・poolする列 | 見る点 |
|---|---|---|
| 01_model_h_interaction | Model × Hの平均level。Task・Metricsをpool。薄い点はTask × Metric平均 | Hに対する傾き、同じHでのモデル差、集約に隠れる条件差。点の横位置には視認用の小さなずらしを加える |
| 02_model_h_heatmap | Model × Hの平均level。Task・Metricsをpool | 各cellの絶対水準とH方向の変化。色範囲を1〜4に固定 |
| 03_model_h_stacked_distribution | ModelごとにHの4本を配置。Task・Metricsをpool | 平均の変化がScore 1の増加か、Score 4の減少か。モデル間には広めの間隔 |
| 04_pendulum_last_vs_next_stacked | Pendulum限定。行Model、列Action Metric、barはH。追加poolなし | 同じModel・Hの左右でScore分布を比較。n=10では1件が10ポイント |
| 05_pendulum_last_vs_next_interaction | Pendulum限定。Model facet、Metric別の平均level。追加poolなし | Last/Nextの水準差と、Hへの依存性の違い |
| 06_high_level_reasoning_rate | Model × HのP(score ≤ 2)。Task・Metricsをpool | 具体的・定量的根拠を用いる割合がHで変わるか |
| 07_guessing_rate | Model × HのP(score = 4)。Task・Metricsをpool | 一般的・弱い推論の残存率。低いほど良い |
| 08_model_score_distribution | Model別。H・Task・Metricsをpool | モデル全体のScore構成。Hによる違いは隠れる |
| 09_h_score_distribution | H別。Model・Task・Metricsをpool | 全体のH依存。ただしモデル別の傾向が相殺されうる |
| 10_metric_score_distribution | Pendulumのlast-action / next-actionのみ。Model・Hをpool | Action Metric間の全体差。Model別の差は図04・05で確認 |

## 今回の記述的な観察

- H=5→30の平均levelはGPT-3.5が1.488→1.567、Lunaが3.338→3.238、Terraが3.138→3.113、Solが3.300→3.450。全モデルがH増加に伴い向上するパターンではない。
- 同じHでの最高・最低モデルの差はH=5で1.850、H=10で1.913、H=20で1.913、H=30で1.883。短いHで差が大幅に縮小する傾向は、この集約値では見えない。
- 全てのModel × Hで、PendulumのLast Actionの平均levelがNext Actionを上回る。ただしHに対する曲線の形はモデル・Metricによって異なる。
- GPT-3.5の集約平均の上昇は小さいが、Pendulum Last Actionは1.700→2.200と変化しており、poolした平均だけでは条件固有の変化を見落とす。

## 再実行

使用ライブラリ: Python、pandas、NumPy、Matplotlib。実行確認環境はpandas 3.0.5、NumPy 2.5.3、Matplotlib 3.11.1。

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python experiments/07_view_rawdata/reasoning_score_analysis/plot_reasoning_scores.py \
  --output-dir experiments/07_view_rawdata/reasoning_score_analysis_rerun
```

入力は `--input /path/to/file.csv` で変更できる。出力先に同名の成果物がある場合は実行を停止し、上書きを防止する。再実行には新しい `--output-dir` を指定する。初回は引数なしでスクリプトと同じディレクトリへ生成できる。

全10図（PNG）の生成、図の目視確認、入力CSVのSHA256が実行前後で一致することを確認した。元CSVと作業開始前から存在する成果物は変更していない。
