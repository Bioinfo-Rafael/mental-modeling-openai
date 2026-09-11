# Joint共通評価コード

03_1の解析実装をここへ移し、03_1・05・06から呼ぶ。評価式を実験ごとに複製しない。
API clientを作る処理や、有料実験を開始する処理はない。入力は取得済みrecordsとraw trajectoryのみ。

## 実行入口

repository rootで既存の`.venv`を有効化して使う。

| 入口 | 対象・出力 |
|---|---|
| `python experiments/03_1_gpt35_history_n30_joint/analysis/analyze.py` | GPT-3.5。従来どおりN10/20/30、03_1/analysis以下 |
| `python experiments/05_new_models_single/analysis/analyze.py` | 新3モデルのPendulum H20。N10のpilot CSVをモデル別に出力 |
| `python experiments/06_new_models_n10/analysis/analyze.py` | 新3モデルの全grid。N10だけのCSV・図をモデル別に出力 |

05は1 Task・1 Hのみなので、未取得のTask/Hを埋めた比較図は作らない。06で全gridの図を作る。

06に`results/batches/`がある場合、`runner.main()`は`experiments/split_execution.py:merge_results()`を先に呼ぶ。05の120件と完了batchの840件を検証し、新しい`results/merged/<識別子>/`に全960件を統合してから同じ評価系へ渡す。不足・重複・未完了なら停止する。元の05/batchは変更せず、統合にAPIは使わない。旧方式のroot直下の結果は従来の読込経路を維持する。
各実験のresultsがcompleteで、期待するModel×Task×Metric×H×Nが揃っていなければ停止する。
既存の03_1のCSV・図・RESULTSは移動・上書きせず、取得当時のスナップショットとして残した。

共通引数は`--seed 42 --bootstrap 1000`、`--output-dir <その実験のanalysis内のpath>`、`--overwrite-derived`。
既存成果物への再出力は明示的な上書き指定が必要。results・raw・別実験のanalysisを出力先にはできない。
`--output-dir`は新しい解析保存先を作るためにも使える。元の結果は保持できる。

## 所有する処理

| ファイル・関数 | 責務 |
|---|---|
| `runner.py:main()` | 実験specと完了結果を読む。モデル別に分ける。06は05の再利用内容も検証 |
| `runner.py:export_model()` | 1モデルを計算・保存・監査。複数モデルのpoolを拒否 |
| `joint_data.py:load_and_parse()` | final markerをparseし、元episodeからGTを取得。旧questionやgrid不一致を拒否 |
| `joint_data.py:parse_response()/parse_component()` | torque/bin、direction/absolute/deltaを独立に抽出 |
| `joint_data.py:select_subsets()` | seed固定・stableなepisode/index順位。sizesとexpected_nを引数で受ける |
| `joint_metrics.py` | NRMSE、paired bootstrap、Pearson、Cosine、角度復元。03_1から式を変更せず移設 |
| `joint_aggregate.py:aggregate_all()` | 同一subsetのlong table集計。model/model_aliasも保持 |
| `joint_plots.py:make_figures()` | long tableから作図。モデルの混在を拒否、Nは引数で限定 |

03_1/analysisの旧`joint_*.py`は互換importのみ。新しいコードは`experiments.common_analysis`から直接importする。
questionの対応は別の`experiments/joint_questions.py`。prompt本文は既存`upstream/LLM-Xavier/llm_x/feedback.py`が所有する。

## 変えない評価定義

詳細な数式・range・図一覧は [03_1の解析定義](../03_1_gpt35_history_n30_joint/analysis/README.md)。

- 旧`prediction/status`を正解判定に流用せず、`assistant_text/raw_response`の明示markerを解析する。
- MountainCar ActionはID exact match、Pendulum Actionは明示bin exact match。bin GTは公式`bin_actions()`。
- DirectionはDEC/INC/UNCHの次元pool Accuracy。主値は有効回答分母、失敗を不正解と数える別値も併記。
- State deltaはNext/Lastとも`later - earlier`。GTはraw trajectoryからfloat64で計算。
- NRMSEは理論range幅で正規化。Macroは次元別NRMSEの平均。
- NRMSE帯は1,000回のpaired-query bootstrap std。Pearson/Cosine・Accuracy・bin MAEには帯を付けない。
- Token/time帯はquery間sample std。欠損は0補完しない。
- Pendulum Δthetaはatan2とwrapでforward-timeの差を復元。Last Stateでは`early = late - delta`。
- Parse成否はcomponentごと。`selected_n / valid_n / parse_success_rate / na_reason`を残す。

03_1のN10は30件からseed固定で10件を抽出する。05/06のN10は保存された先頭10有効queryをすべて使用する。
したがって03_1のN10と05/06のquery集合が同一とは限らない。比較時はquery_subsets.jsonを確認する。

## 06のモデル別出力

`06_new_models_n10/analysis/{sol,terra,luna}/`のそれぞれに出力する。
モデルを同じ行に混ぜて平均・bootstrapしない。各CSVにもmodel/model_aliasを保存する。

| 出力 | 数・内容（1モデル） |
|---|---|
| `cross_task_10/` | Accuracy、6 confusion、absolute/delta Macro NRMSE、次元別NRMSE/Pearson/Cosine＝15図 |
| `pendulum_10/` | torque/bin MAE、Δtheta＝2図 |
| `diagnostics_10/` | parse、input/output/total tokens、処理時間＝5図 |
| `tables/` | 下記12 CSV |
| JSON | query_subsets、figure_index、analysis_metadata、validation、protected_sha256 |

22図×PNG/SVG＝44画像ファイル/model。3モデル合計66図、132画像ファイル。
05の応答を含む06の320 query/modelだけを解析し、05のrecordsをもう一度結合しない。
06に記録された再利用集合が05と一致すること、再利用recordが`api_request_made=false`でraw_responseが同じことも検証する。

CSV：`parsed_records.csv`、`accuracy_metrics.csv`、`continuous_metrics.csv`、`state_dimension_metrics.csv`、
`state_macro_metrics.csv`、`pendulum_action_metrics.csv`、`pendulum_delta_theta_metrics.csv`、
`direction_dimension_accuracy.csv`、`diagnostics.csv`、`confusion_matrices.csv`、`all_metrics_long.csv`、`parse_summary.csv`。
一部は同じ結果の目的別viewなので、複数CSVを足して件数や値を重複集計しない。

## 再利用の時間・token

05からの再利用では元の`query_elapsed_seconds / request_elapsed_seconds / usage`を保持する。
今回の短い再生処理は`replay_elapsed_seconds`に分離し、API推論時間の平均を押し下げない。
`api_request_made=false / api_attempts=0 / source_experiment / source_record`で新規通信と区別する。
診断表は05を含む最終datasetの性能・消費量であり、06単独の追加課金表ではない。

## オフライン検証

```bash
python -B -m pytest tests/test_joint_modern_experiments.py tests/test_experiment_resume.py experiments/03_1_gpt35_history_n30_joint/analysis/test_joint_analysis.py -q -p no:cacheprovider
```

取得済み03_1を再計算し、共通化前の全集計CSVの既存column・数値と一致することを検証する。
さらにquery非重複、Joint question一致、05前提不足時の送信前停止、全cached条件のclient未生成、元時間の保持、N10だけの作図を検証する。
テスト用の図やモック応答はpytestの一時directoryに出し、新モデルの実測結果として保存しない。
実APIでのモデル利用可否は別の確認事項であり、このテストの成功は有料APIの成功を保証しない。
