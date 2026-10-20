# 07（既存n=10）と08（既存n=3）の比較

追加APIは使用していない。07の既存回答160件と08の既存回答144件を使用する。H=5・20は等重みでpoolする。

## 一覧図

- [MountainCar-v0：4行×3指標](MountainCar-v0_overview.png)
- [Pendulum-v1：4行×3指標](Pendulum-v1_overview.png)

列は正解率・MAE・平均推論レベル。行は①全条件、②few-shot総数をpool、③Scoreパターンをpool、④Terra/Lunaのみで全条件をpool。PNGのみ。

①〜③の左群は07のGPT-3.5/Sol/Terra/Luna、中央群は08のTerra、右群は08のLuna。①は例数ごとにP1/P2の棒を隣接させる。②はP1/P2の2本、③は4/8/12例の3本。④はモデルごとに07/08を隣接させる。

Terraは07/08共通で青系、Lunaは橙系。08のScoreパターンはP1＝無地、P2＝斜線で区別し、few-shot総数4→8→12は同系色の薄→中→濃で示す。④の07/08比較もモデル別の同系色で、07を薄く、08を濃く表示する。

## 指標ごとの拡大図

| Task・比較 | 正解率 | MAE | 推論レベル |
|---|---|---|---|
| MountainCar-v0 ①全条件 | [PNG](MountainCar-v0_all_accuracy.png) | [PNG](MountainCar-v0_all_mae.png) | [PNG](MountainCar-v0_all_reasoning_score.png) |
| MountainCar-v0 ②例数pool | [PNG](MountainCar-v0_pool_shots_accuracy.png) | [PNG](MountainCar-v0_pool_shots_mae.png) | [PNG](MountainCar-v0_pool_shots_reasoning_score.png) |
| MountainCar-v0 ③パターンpool | [PNG](MountainCar-v0_pool_pattern_accuracy.png) | [PNG](MountainCar-v0_pool_pattern_mae.png) | [PNG](MountainCar-v0_pool_pattern_reasoning_score.png) |
| MountainCar-v0 ④全pool | [PNG](MountainCar-v0_pool_all_accuracy.png) | [PNG](MountainCar-v0_pool_all_mae.png) | [PNG](MountainCar-v0_pool_all_reasoning_score.png) |
| Pendulum-v1 ①全条件 | [PNG](Pendulum-v1_all_accuracy.png) | [PNG](Pendulum-v1_all_mae.png) | [PNG](Pendulum-v1_all_reasoning_score.png) |
| Pendulum-v1 ②例数pool | [PNG](Pendulum-v1_pool_shots_accuracy.png) | [PNG](Pendulum-v1_pool_shots_mae.png) | [PNG](Pendulum-v1_pool_shots_reasoning_score.png) |
| Pendulum-v1 ③パターンpool | [PNG](Pendulum-v1_pool_pattern_accuracy.png) | [PNG](Pendulum-v1_pool_pattern_mae.png) | [PNG](Pendulum-v1_pool_pattern_reasoning_score.png) |
| Pendulum-v1 ④全pool | [PNG](Pendulum-v1_pool_all_accuracy.png) | [PNG](Pendulum-v1_pool_all_mae.png) | [PNG](Pendulum-v1_pool_all_reasoning_score.png) |

## 対象と採点

- 07：next-action、H=5/20、4モデル。各条件ordinal 0〜9の10件。GPT-3.5の30件も先頭10件に限定する。07ビューアが参照する03_1/06統合recordsを出典として照合する。
- 08：next-action、H=5/20、Terra/Luna、2パターン×4/8/12例、各条件ordinal 0〜2の3件。
- 正解率：共通の08用scoringを使用して回答全文から最終action/bin IDを解析し、raw episodeの正解と比較する。旧statusをそのまま使わない。解析不可は非正解として分母に含める。
- MAE：MountainCarは行動IDの絶対差、Pendulumは連続トルクの絶対差。解析不可は欠測とし、補完しない。単位が異なるためTask間でMAEをまとめない。
- 推論レベル：07は採点CSVの旧scoreを1回だけ`5 − score`で変換。08は新scoreをそのまま使用。モデル名や正誤から再採点しない。原文Reasoningが空欄なら平均から除外する。今回選択した304件にはReasoning欠測はない。

## poolと分母

| 図 | 07の各棒 | 08の各棒 |
|---|---:|---:|
| ①全条件 | 20 | 6 |
| ②例数pool | 20 | 18 |
| ③パターンpool | 20 | 12 |
| ④全pool | 20 | 36 |

各元条件で指標を平均し、条件平均を等重みでpoolする。H=5と20も同じ重み。欠測がない群では回答単位の平均と一致する。07のMountainCar/LunaはH=5に1件、H=20に5件の行動ID解析失敗があり、MAEはそれぞれ有効9件・5件の平均を求めてから等重みでpoolする（合計有効14件）。図の該当棒にn=14を表示する。正解率の分母は20件のまま。

07の基準は複数図に再掲しているが、再掲回数をサンプル数に加算しない。同じ指標・Taskの縦軸は4種類の図で統一する。独立性を仮定した誤差棒は付けない。

## 比較上の意味

07はepisode_0、08はepisode_1で、評価時点も異なる。これは既存の0-shot/few-shot結果の記述的比較であり、同一問題を使った対応付き比較やfew-shot単独の因果効果の推定ではない。08は同じ3問を条件間で共有しており、pool後の36回答は独立した36問ではない。新スコアは平均を表示するが、1〜4の順序尺度である。

## データと再生成

- [使用した全回答の指標](per_query_metrics.csv)：重複なしの304回答。元query_id・episode・時点・旧score・変換後score・正誤・誤差を保存。
- [各棒の集計](group_metrics.csv)：条件平均、分母、有効件数、欠測数。基準の再掲を含む228行。
- [検証情報](validation.json)：入力SHA256・件数・欠測数。

08_fewshotから：

```bash
python analysis/plot_previous_comparison.py
```

追加取得用に誤って作ったrun_comparison_n10.py、plot_comparison_n10.py、test_comparison_n10.py、results/comparison_n10/の送信計画は削除済み。既存結果は保持している。
