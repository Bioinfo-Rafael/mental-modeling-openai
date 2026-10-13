# Experiment 15: LAGの10 step trajectory予測

14のデータ読込・環境説明・入力範囲・窓選択・API記録方式を引き継ぎ、未来10 stepの **stateとactionを1回の回答で同時に予測** する。データは13のprocessed CSVを参照する。

## 条件と件数

| 項目 | 設定 |
|---|---|
| 試合 | `mhi_v1.3_ic100601_win` のみ |
| 対象 | A0100 / A0200 |
| モデル | sol / terra / luna（14と同じモデルID） |
| 履歴H | 20 step |
| 予測長 | 10 step（1 step = 0.2秒） |
| 入力範囲 | `self` / `all_aircraft_missiles` |
| n | 有効な開始時刻を5等分した各区間の先頭から5窓 |
| 回答 | 各窓につき1回、10 step分のstateとactionを一括回答 |

**3モデル × 1試合 × 2機体 × 1種類のH × 1課題 × 2入力範囲 × 5窓 = 60リクエスト。**
12条件、モデル間で共有する入力は20件。同一問題を5回反復する意味ではない。
selfと全機体条件は同じ開始step・予測step・正解を使い、モデル間でも問題を共有する。

設定は [run.py](run.py) と [config.py](trajectory_experiment/config.py) に置く。
モデルIDは `gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna`、`reasoning_effort=medium`。
14の設定をそのまま使用し、temperature・出力トークン上限は指定しない。

## 時刻と入力

履歴開始をbとし、履歴には `b ... b+19` の20組 `(s_t, a_t)` を渡す。
回答は `b+20 ... b+29` の10組 `(s_t, a_t)`。
例えばb=0なら、step 0〜19を入力し、step 20〜29を予測する。
同じ回答行のstateとactionは **同一step** を指す。`a_t`は`s_t`で選ばれ、t→t+1に作用する。
最初の未来状態`s_(b+20)`は履歴最後の状態・行動から予測する。

未来10 stepの状態・行動は、対象機体以外も含めて一切提示しない。逐次API呼び出しや、途中の正解を与えるteacher forcingは行わない。14のnext/lastの4課題を、1つの`trajectory`課題に置き換える。

- `self`：履歴の対象機体のみ。
- `all_aircraft_missiles`：同じ履歴時刻の全4機と記録済みの全ミサイル、および利用可能な機体行動。欠損はnull、無効状態もnull。effectは除外。
- 状態は正規化・補完・追加の桁丸めをしない。
- 履歴のactionはshootを含む4成分。**予測するactionは14と同じ移動3成分**（altitude / heading / speed）。
- 履歴と未来10 stepすべてについて、対象機体の有効な状態10成分と記録済み行動4成分がそろう窓だけを採用する。
- n未満の窓しかない場合や、有効な開始時刻に途中の穴がある場合はエラーにする。

状態の順序：

```text
lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg,
tas_mps, vn_mps, ve_mps, vu_mps
```

予測するactionの順序とID：

```text
altitude_action: 0=up, 1=hold, 2=down
heading_action: 0=left30, 1=left15, 2=hold, 3=right15, 4=right30
speed_action: 0=up, 1=hold, 2=down
```

## 回答と採点

14と同じReasoning / Prediction / Formattingの枠を使い、Formattingには次のマーカーとJSON配列を1行で出力する。

```text
>>Final trajectory: [{"step":20,"state":[120.0,60.0,6096.0,0.0,0.0,10.0,243.8,240.0,40.0,0.0],"action":[1,2,0]}, ...]
```

上は省略例。実際は指定されたstep番号の10要素すべてが必要。各要素のキーは`step / state / action`のみとする。
stepの欠落・重複・順序違い、非有限値、範囲外の行動ID、4成分のaction、片方だけの回答などは形式失敗にする。部分的に10 step未満を採点しない。

各回答について、以下を保存する。

- 各stepのaction3成分の完全一致・成分別一致、10 step全体のaction完全一致、step平均正解率。
- 各stepのstate成分別絶対誤差と、10 stepの成分別MAE。単位が異なる成分を1つの誤差に混ぜない。
- yaw / rollの循環角度誤差も補助指標として保存。主指標は14と同じ生の座標差。

条件ごとに5窓を等重みで集計し、予測距離1〜10ごとの成績も保存する。形式失敗・APIエラー・未採点数を分離し、全5件が有効な場合のみ`complete=true`。actionには全計画数を分母にしたtrajectory一致率も記録する。state MAEの分母は有効回答のみで、欠損を0点扱いしない。

## 実行（Batch API）

リポジトリルートから実行する。通常起動はローカルの計画・プロンプト・Batch用JSONL生成のみでAPIを呼ばない。
`--execute`は最初の1件からBatch APIを使う。デフォルトではモデルごとに20件、合計3 Batchを投入する。

```sh
# デフォルトの60件を計画。20件のプロンプトを確認可能。
.venv/bin/python experiments/15_trajectory_LAG/run.py

# 60件を3 Batchへ投入して終了。
.venv/bin/python experiments/15_trajectory_LAG/run.py --execute

# 投入から完了待ち・結果回収・採点まで行う場合。
.venv/bin/python experiments/15_trajectory_LAG/run.py --execute --wait

# 送信済みBatchの完了を待ち、結果を回収・採点（新規投入なし）。
.venv/bin/python experiments/15_trajectory_LAG/run.py --collect --wait

# 一度だけ状態を更新し、完了分を回収する場合。
.venv/bin/python experiments/15_trajectory_LAG/run.py --collect

# 保存済みの状態を確認（API不要、リモートの最新状態は--collectで取得）。
.venv/bin/python experiments/15_trajectory_LAG/run.py --status

# 中断後は同じ実行コマンドで再開。--resumeも互換フラグとして使用可能。
.venv/bin/python experiments/15_trajectory_LAG/run.py --execute --resume --wait

# 保存済み回答をオフラインで再採点。
.venv/bin/python experiments/15_trajectory_LAG/run.py --rescore

# 小さい計画の例。これもAPI不要。
.venv/bin/python experiments/15_trajectory_LAG/run.py \
  --models sol --targets A0100 --scopes self --n 1 --output smoke

# オフラインテスト。
.venv/bin/python -m pytest experiments/15_trajectory_LAG/tests -q
```

`OPENAI_API_KEY`が未設定なら実行時に非表示入力を求める。
SDKの自動リトライは0、各API操作のタイムアウトは180秒。Batchには`completion_window="24h"`を指定する。
Batch用のJSONLを`purpose="batch"`でアップロードし、`/v1/chat/completions`へ投入する。返却順序は入力順とは限らないため、`custom_id`（query_id）で対応付ける。仕様は[OpenAI公式Batch APIガイド](https://developers.openai.com/api/docs/guides/batch)を参照。

ローカルの待機を中断しても送信済みBatchはリモートで継続する。再実行時は保存済みBatch IDから再開し、同じリクエストを重複投入しない。`--wait`はデフォルト30秒間隔で確認する（`--poll-seconds`で1〜60秒）。
送信前にjournalと送信意図を保存する。Batch作成の応答が失われた場合はmetadataから既存Batchを探す。特定できなければ自動再送を止め、ダッシュボードで確認したIDを次のように関連付けられる。

```sh
.venv/bin/python experiments/15_trajectory_LAG/run.py \
  --adopt --job sol_000 --batch-id batch_...
```

失敗・期限切れ・取消の部分結果も保存し、欠落は明示的なエラーとして記録する。失敗リクエストは自動再送しない。成功した生回答は採点前に保存する。
計画・プロンプト・ソースの整合性を検証し、別の条件やコードで既存計画を上書きしない。設定やプロンプトを変更したら新しい`--output`名を使う。

出力先はこの実験の`results/trajectory_batch_v1/`。先に作成した逐次実行用の`results/trajectory_v1/`は保存したまま、新しい計画を使う。

```text
manifest.json                  条件・正解・全リクエスト・データ/コードhash
prompts/*.txt                  モデル間で共有する20件の入力
api/requests.jsonl             送信前journal
api/responses.jsonl            APIの生回答・エラー
api/summary.json               実行状況（送信・回収時に作成）
api/batch/state.json            Batchの準備・送信・回収状態とBatch ID
api/batch/*.input.jsonl         モデル別の送信用JSONL（デフォルト3ファイル）
api/batch/*.remote.json         リモートBatchの状態
api/batch/*.output.jsonl        Batchの生出力
api/batch/*.error.jsonl         Batchの生エラー
analysis/<scoring_version>/
  per_query.json               各回答の予測・正解・step別誤差
  conditions.json             12条件の集計（予測距離別成績を含む）
  conditions.csv              条件ごとの集計
  horizons.csv                条件×予測距離1〜10の集計
  metadata.json               採点コード・生回答のhash
```

実装は独立した`trajectory_experiment`パッケージとし、14のコードや既存の結果には依存しない。13のCSV定義・読み込みと、未変更のLLM-Xavierのsystem/user外枠を再利用する。

## 実測・予測の時系列HTML

[時系列ビュー](results/trajectory_batch_v1/analysis/trajectory_timelines_v1/index.html)で、対象機体の実測値とsol / terra / lunaの10 step予測を重ねて比較できる。機体・問題番号・入力範囲を選び、履歴と周辺の実測値を含む図／予測区間の拡大図を切り替える。
状態10成分と行動4成分を表示し、shootは予測対象外なので実測のみ。形式不正の回答は線を描かず、理由と原文を表示する。実測値の欠損は補完せず、角度の折り返しでは線を切る。予測のstep 0〜9は正解の同一stepへ対応付ける。

```sh
.venv/bin/python experiments/15_trajectory_LAG/plot_trajectory_timelines.py

# 更新時は別の出力先へ作成（既存の図は上書きしない）。
.venv/bin/python experiments/15_trajectory_LAG/plot_trajectory_timelines.py \
  --output-dir experiments/15_trajectory_LAG/results/trajectory_batch_v1/analysis/trajectory_timelines_v2
```

デフォルトは20条件×2表示区間＝40枚のPNGとHTML。`--before`は最初の予測stepより前の表示step数（既定50）、`--after`は10 step予測の末尾以降に追加する実測step数（既定10）。CSVのhash・予測対象の正解・query IDと送信記録の整合性を検証してから描画する。
図とHTMLを移動・共有するときは、同じディレクトリ内の`figures/`・JSONを含めて保持する。

## 入出力のRaw data HTML

[view_rawdata.html](view_rawdata.html)は14と同じ詳細／回答一覧／3モデル×2入力範囲の比較ビュー。10 stepの行動比較と状態100値の誤差、送信したsystem/userプロンプト、回答全文、API応答JSON、token使用量とBatch IDを確認できる。
各回答のJSON保存、モデル・機体・問題・状態による絞り込み、入力／回答の全文検索に対応する。同じ問題の時系列図へのリンクも表示する。

```sh
.venv/bin/python experiments/15_trajectory_LAG/build_results_viewer.py

# Batchの新しい回答を回収した後、HTMLスナップショットだけを更新。
.venv/bin/python experiments/15_trajectory_LAG/build_results_viewer.py --overwrite
```

両ビューアはローカルの保存済みデータだけを読む。API送信・実験計画・生回答・既存の採点結果の変更は行わない。Raw data HTMLはデータを内包する単一ファイルで、オフラインで開ける。

## 10地点の誤差合計による比較図

直下の[view_results.html](view_results.html)から、14と同じself／全機体／全機体−selfの比較図を開ける。H=20のみなので横軸は3モデル。機体別（計画n=5）と2機体pool（計画n=10）を切り替える。

```sh
.venv/bin/python experiments/15_trajectory_LAG/plot_results.py

# 再生成時は新しい出力先と入口HTMLを指定する。
.venv/bin/python experiments/15_trajectory_LAG/plot_results.py \
  --output-dir experiments/15_trajectory_LAG/results/trajectory_batch_v1/analysis/comparison_plots_sum10_v2 \
  --entry experiments/15_trajectory_LAG/view_results_v2.html
```

各問題で先に未来10地点の誤差を**合計**し、その値を問題間で等重み平均する。時間方向では10で割らず、符号付き誤差も相殺しない。

- 状態・成分別：`mean_q(sum_t(abs(pred[q,t,d] - truth[q,t,d])))`。生の単位、固定範囲で正規化、固定標準偏差で正規化の3種類。
- 状態・10成分全体：正規化した各地点の10成分MAEまたはRMSEを計算し、その10地点分を合計してから問題間で平均。範囲基準／標準偏差基準の計4種類。RMSEの平方根は**各地点の成分平均の後、時間合計の前**に取る。
- 行動：3成分のいずれかが違うstep数（0〜10）、または各stepの3成分平均絶対ID差の10地点合計。成分別の不一致step数・絶対ID差合計も保存。
- 混同行列：有効な各回答の10地点を加算。図のnは回答数、セルは地点数。10地点を独立な試行とは扱わない。
- token：1リクエストあたりの実測使用量を平均。10倍しない。形式不正でも取得できたusageを使う。

状態のroll・yawは14の比較図と同じ最短角度差を使う。範囲・標準偏差は全計画の予測対象となる100個の固有正解状態から固定する（モデル／入力範囲の重複を除く）。角度の正規化基準も14と同じ最大空白区間で切った円弧に基づく。既存の採点ファイルの生値誤差とは別の描画用集計であり、元の採点結果は変更しない。

形式不正は回答全体を除外し、全ての棒に有効n／計画nを記載。self／全機体の差は、同じ問題で両方の指標が有効なペアのみから再計算する。モデル間や単独パネルでは有効な問題集合が異なる場合がある。

`results/trajectory_batch_v1/analysis/comparison_plots_sum10_v1/`にPNG・HTML・`definitions.json`（計算式／正規化／出典）・`metrics.json`（集計値と採用問題ID）・`per_query.json`（問題別の10地点誤差合計）を保存する。入口HTMLを共有する際は参照先の図も保持する。
