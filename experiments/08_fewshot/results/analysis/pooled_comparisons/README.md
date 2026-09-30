# 08 few-shot：他の2条件をpoolした比較

各Taskについて、行はScoreパターン・few-shot総数・H、列は正解率・平均絶対誤差（MAE）・平均推論スコア。青がTerra、橙がLuna。各比較軸の残り2条件をまとめている。

## 図

| Task | 3軸の一覧 | Scoreパターン | few-shot総数 | H |
|---|---|---|---|---|
| MountainCar-v0 | [PNG](MountainCar-v0_overview.png) | [PNG](MountainCar-v0_by_score_pattern.png) | [PNG](MountainCar-v0_by_shots.png) | [PNG](MountainCar-v0_by_H.png) |
| Pendulum-v1 | [PNG](Pendulum-v1_overview.png) | [PNG](Pendulum-v1_by_score_pattern.png) | [PNG](Pendulum-v1_by_shots.png) | [PNG](Pendulum-v1_by_H.png) |

## 集計方法

- Scoreパターン別：few-shot総数3水準 × H 2水準 × 3問 = 各モデル・群18回答。
- few-shot総数別：Scoreパターン2水準 × H 2水準 × 3問 = 各モデル・群12回答。
- H別：Scoreパターン2水準 × few-shot総数3水準 × 3問 = 各モデル・群18回答。
- 全条件で回答が揃っているため、回答単位の平均は各条件平均の等重み平均と一致する。
- 同じTaskでは同じ3問を全条件で共有する。独立した12問・18問ではないため、独立標本を仮定した誤差棒や有意差検定は付けていない。

## 指標

**正解率**：MountainCarは`Final action choice`、Pendulumは`Final action bins`のIDと正解IDの完全一致。`runner/scoring.py`（共通Joint parserを利用）でAPI実行・再採点と同じ採点を行う。解析不可は正解数に含めず分母に含める。今回144件すべて解析成功。

**誤差（MAE）**：各回答の予測値とraw episodeの実際の行動の絶対差を平均する。MountainCarは行動ID（0・1・2）の差で、隣接行動の誤りは1、逆方向の誤りは2。Pendulumは`predictions`に記載された連続トルク値の差。単位が異なるためTask間でMAEの大きさを直接比較しない。数値解析不可はMAEから除外し、`error_n`に有効件数を記録する。今回除外0件。

**推論スコア**：`reasoning_for_scoring_scored_astra_high.csv`の新score（1〜4、大きいほど具体的・定量的）の算術平均。query_idと条件列を照合して結合する。順序尺度の記述的要約として表示する。今回144件すべて採点済み。

Scoreパターンはfew-shot例選択時の旧Score体系の条件名で、今回の出力推論スコアとは別。

- Pattern 1：正解例は旧S1、不正解例は旧S3・S4。
- Pattern 2：正解例は旧S1・S2、不正解例は旧S1・S2・S3。

## 既存の正誤記録との差

API実行時の既存parserはPendulumのトルク予測をbin IDとして比較していた。例えば`predictions = [2.00]`、`Final action bins: [9]`、正解bin `[9]`に対し、保存済みpredictionは`[2]`で不正解になっていた。

修正後はPendulumが59/72正解（修正前0/72）、MountainCarが65/72正解（変化なし）。保存済みrecords・summaryも再採点し、`accuracy_tables.*`、通常の正解率図、pool図、ビューアを同じ採点に統一済み。旧ファイルは`results/scoring_backups/`に保存し、`per_query_metrics.csv`の`legacy_correct`にはrecord内の`legacy_evaluation`から旧判定を残している。API応答全文・プロンプト・Reasoning採点は変更していない。

## データと再生成

- [群ごとの集計](pooled_metrics.csv)：全28群、分母・正解数・MAE・平均推論スコア。
- [回答ごとの指標](per_query_metrics.csv)：全144件のID・正解値・予測値・絶対誤差・score。
- [検証情報](validation.json)：入力SHA256、解析件数、新旧正誤の差。

08_fewshotから実行：

```bash
python analysis/plot_pooled_comparisons.py
```

入力は`results/api/records.jsonl`、採点済みCSV、出典raw episode。API呼び出しなし。入力のハッシュ・完全な条件グリッド・query_idの一意性・採点CSVの条件列・episode由来の正解を照合する。
