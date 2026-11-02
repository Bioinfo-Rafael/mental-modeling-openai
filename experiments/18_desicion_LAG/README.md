# 18_desicion_LAG — 説明を始める地点の検出

`gpt-6-luna` に履歴だけを渡し、人が選んだ地点を再現できるか調べる。
指定されたディレクトリ名 `18_desicion_LAG` を使用している。
元の段階1＋2を `develop`、段階3＋4を `evaluate` に統合した。

| モード | 目的 | 入力 | API・出力 | 既定のリクエスト数 |
|---|---|---|---|---:|
| `develop` | 誤答・理由を見てプロンプトを調整 | 固定の正例5窓＋負例5窓、H=40 | Responses：YES/NO、複数のstep、理由 | 10窓 × 1回 = 10 |
| `evaluate` | 固定したプロンプトを時系列に適用して比較 | 全試合をH=40、5step刻みで走査 | Responses診断、Decisions機体4択 | 62窓 × 2条件 × 3回 = 372 |

**実装とオフライン検証まで完了。実APIの正答率・速度は未測定。**
`develop` / `evaluate` は実行計画を保存するだけで、API送信は `execute` だけが行う。

## タスク・入力窓

問いは「**渡した窓全体に、A0100またはA0200の動きの傾向が明確に変わった地点、または新しいミサイル発射があるか**」。
正例はYES、負例はNOを期待する。1つの窓を1クエリとして独立に渡す。
正解ラベル、注釈内容、正例・負例という区別、注釈の中心stepは入力に含めない。
全4機と記録されたミサイルを文脈として渡し、検出対象はAチームの行動に限定する。
敵機の動き、相対位置、推測した役割が変わっただけではYESにしない。

操作指令のID・符号・振れ幅の変化だけを検出するタスクではない。
例えば方位指令が `-15,+15,-15` や `-15,+30,-15` と振動しても、
実際の進行方向がほぼ直進を保つなら、それだけでは行動傾向の変化としない。
前後の複数時点で操作指令と機体の動きを合わせて比較し、直進から持続的な旋回、
旋回の終了・反転、上昇・下降・加減速の傾向の開始・終了などを検出する。
操作から動きへの遅れは許すが、動きの裏付けがない単発の指令だけで判断しない。
大きい旋回でも窓全体で継続しているだけなら新しい変化ではない。

発射は別の検出条件であり、旋回などの変化を伴わなくてもYESになり得る。
入力に発射指令はないため、弾数減少や発射元が分かるミサイル記録を根拠とする。
厳密な発射stepが不明なら、最初に根拠を確認できるstepを返して理由に時刻の限界を記す。
同じ機体が同じ開始stepで旋回と発射を始めた場合は、1イベントに両方を記載する。

Hは**時点数**。現在step=tで `[t-H+1, t]` の連続したH点を渡す。
1step=0.2秒なのでH=40の両端の時間差は7.8秒、H=30なら5.8秒。
窓の末尾より未来のデータは渡さない。ただし、窓内の地点を、その後に観測された窓内データから説明することは許す。
これは履歴からの検出であり、未来の行動予測ではない。

| 組 | 参考アノテーション | 正例 H=40 | 負例 H=40 | 正例 H=30 | 負例 H=30 |
|---|---|---|---|---|---|
| 1 | A0100@46、A0200@49 | 16–55 | 1–40 | 26–55 | 11–40 |
| 2 | A0100@83、A0200@95 | 61–100 | 100–139 | 71–100 | 110–139 |
| 3 | A0200@143 | 111–150 | 150–189 | 121–150 | 160–189 |
| 4 | A0100@243 | 211–250 | 250–289 | 221–250 | 260–289 |
| 5 | A0100@311 | 281–320 | 270–309 | 291–320 | 280–309 |

各正例は指定した組の地点だけを含む。各負例は7地点を1つも含まない。
負例の幅を固定したため、組2・3は注釈後の窓とした。組4・5の負例には重複区間がある。
**「注釈がない」は「対象となる行動変化・発射が本当にない」と同義ではない。** FPはまず未注釈の対象事象か、モデルの過剰検出かを確認する。

83と95を同じ正例で扱うため、末尾10stepだけに判定対象を制限しない。
この定義をsliding windowでも維持する。過去の検出結果は次のモデル入力へ渡さない。

入力CSVは `../13_LAG/data/processed/mhi_v1.3_ic100601_win.csv`。
13の型付きローダーでCSVとmanifestのハッシュを照合する。
不正・非活動の状態はnull、未記録の行動もnullとし、発射指令、標的ID、命中結果を補完しない。
完全なCSV行動時点0–340を使用し、ACMIだけの終端341は使用しない。
参考ラベルは [data/annotations.json](data/annotations.json) に分離した。

## 構造

```text
18_desicion_LAG/
  run.py                       # CLIの入口
  configs/
    develop.json               # 元段階1+2
    evaluate.json              # 元段階3+4
    pricing.json               # 日付・出典付きの料金設定
  data/annotations.json        # 7地点、5組、10窓の指定
  prompts/
    reference/tactical_short_v1/# 17の参照原文と出典ハッシュ
    v001/                      # 共有判定指示、問い、二択、出力指示
    v002/                      # clone-promptで作る次の仮説
  decision_experiment/
    dataset.py                 # 入力と参考ラベルを分離
    prompts.py                 # プロンプト・APIリクエスト組み立て
    planning.py                # 送信しない計画作成・固定・検証
    execution.py               # 両APIの実行、再開、送信記録
    outputs.py                 # 出力検証・料金推定
    metrics.py                 # 二択・step対応・重複集計
    evaluation.py              # HTML/CSV/JSON/Markdown出力
    cli.py
  tests/
  results/
    01_develop/v001/<run_name>/ # 元段階1+2
    02_evaluate/v001/<run_name>/# 元段階3+4
```

各runは `manifest.json`、`prompt/` の原文・差分、`inputs/` の入力全文、
`samples.json` の評価用ラベル、料金設定、実装のスナップショットを保存する。
プロンプト・入力・リクエストのSHA256を記録するので、同名版を編集しても過去runに使用した本文を確認できる。
API結果は `responses/`、送信記録は `journal.jsonl`、集計は `report/` に置く。
同じstage・prompt・run名への上書きは禁止。試行ごとにrun名を変える。

## 実行手順

以下はリポジトリ直下で実行する。既存の `.venv` を使用できる。
依存は `openai`、`pandas`、テスト用の `pytest`（HTTPモックはSDK依存の`httpx`）。
検証時のSDKはopenai 2.54.0。Decisions専用メソッドがなくてもSDKの汎用POSTを使用する。

`execute` は `OPENAI_API_KEY` が未設定なら、ターミナルでAPIキーの入力を求める。
貼り付けてEnterを押す。入力文字は表示せず、入力したキーをファイルや環境変数へ保存しない。
環境変数が設定済みならそれを使う。非対話実行では環境変数を設定する。

LAGの仕様説明を復元し、systemの6・7節だけを短くした段階1＋2のオフライン計画を保存済み：
[固定10例の入力確認](results/01_develop/v001/concise_h40/report/review.html)。
`execute --run 01_develop/v001/concise_h40` で送信できる。実APIには未送信。
以前の `baseline_h40`、`behavior_change_h40`、`behavior_change_short_h40`、`sliding_h40` は今回のsystem修正前のプロンプトを固定した計画なので、
今回の基準の実験には使わない。段階3＋4は調整完了後に `evaluate` で新しいrunを作る。
実装検証はPythonの41テスト、グラフ集計のNode.jsの4テストが成功。機体4択・機体別取り逃し・旧形式互換、キー入力・中断・非保存、HTTPモックで両APIの送信・再開・タイムアウト継続・集計、HTMLの入力保持と文字列の安全な埋め込みまで確認した。

### 1＋2：10例を見ながら調整

```bash
# オフラインで計画作成。API送信なし
.venv/bin/python experiments/18_desicion_LAG/run.py develop --prompt v001 --run trial01

# 正確な送信内容を1件確認。API送信なし
.venv/bin/python experiments/18_desicion_LAG/run.py request --run 01_develop/v001/trial01

# 保存した10件を送信。キー未設定なら、表示される入力欄へ貼り付けてEnter
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v001/trial01

# 結果確認
open experiments/18_desicion_LAG/results/01_develop/v001/trial01/report/review.html

# 次の仮説を別の版として作り、該当ファイルを編集
.venv/bin/python experiments/18_desicion_LAG/run.py clone-prompt --from v001 --version v002 --note "継続旋回と新しい回避開始の区別を明確化"
# prompts/v002/system.txt 等を編集
.venv/bin/python experiments/18_desicion_LAG/run.py develop --prompt v002 --run trial01
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v002/trial01

# 版ごとの比較。どの段階・本文かをハッシュ付きで表示
.venv/bin/python experiments/18_desicion_LAG/run.py compare --runs 01_develop/v001/trial01 01_develop/v002/trial01
```

`review.html` はFN/FP/TP/TNやエラーで絞り込める。各窓の正解地点、指摘地点、根拠step、理由、入力・生応答へのリンクを表示する。
`predictions.csv` には全結果、`summary.json` には詳細指標を保存する。
`review_template.csv` を別名コピーして人手のレビューを記入する。

### 17と同形式の入出力ビューア

[view_rawdata.html](view_rawdata.html) は入力・応答を埋め込んだ単一HTML。ブラウザで直接開ける。
詳細表示、回答一覧、FP/FNなどの絞り込み、同じ窓のrun/API比較、注釈と予測stepの対照、
実測方位・方位指令のグラフ、各stepのデータ、送信したプロンプト全文と生応答を確認できる。
二択の一致率と注釈地点の検出率は分けて表示する。元データ・実験結果の変更やAPI送信は行わない。

実行結果を追加したら、次のコマンドでHTMLを更新する。

```bash
.venv/bin/python experiments/18_desicion_LAG/build_results_viewer.py
```

既定では実行記録のあるrunだけを収録する。`--run 01_develop/v001/concise_h40` でrunを指定でき、
`--include-planned` を付けると未送信の計画も含む。表示するプロンプトは各runに保存した原文を使用する。
HTMLは生成時点のスナップショットで、自動更新はしない。

### 段階3＋4の時系列・確率・混同行列

評価runでは「時系列・確率・混同行列」タブを最初に表示する。r01/r02/r03を切り替えられ、
全反復を選ぶ場合も時系列は反復ごとに描画する。このタブは独自の反復選択を使い、詳細表示の絞り込みは適用しない。

- **回答時系列**：DecisionsとResponsesを別の図にし、下からNo・A2・A1・Bothの4段に点を描く。
  横軸は入力窓の右端step。各時系列図の薄い赤の縦線は注釈の発生stepを示す。
  灰色の破線は各40step窓に含まれる注釈の機体集合で、発生stepの推定値ではない。
- **差分時系列**：A1・A2それぞれのYES/NOの不一致数を0/1/2で描く。A1↔A2、No↔Bothは2、A1↔Bothは1。
  各APIと注釈の差分、および同一窓・同一反復におけるAPI間の差分を別々に表示する。
  API間の差分0は正解を保証しない。比較は両側が有効な4択を持つ場合のみ。
- **確率時系列**：DecisionsのNo/A2/A1/Bothの確率を4系列で表示する。エラーや未実行は欠測とし、線を切る。
- **所要時間**：横軸を窓の右端step、縦軸を秒とし、Responses（青丸）とDecisions（橙の四角）を共通軸で比較する。
  既定の縦軸はlog(1＋秒)による対数表示で、外れ値を残したまま短い応答の差を読みやすくする。目盛り・ツールチップは実測の秒数。線形表示に切り替え可能。
  送信開始から応答またはエラーまでの実測時間を使用し、失敗には赤い×を重ねる。
  タイムアウトは打ち切りまでの待ち時間で、正常な応答時間とは区別する。時間未記録・未実行は欠測。
- **混同行列**：API別に窓内の注釈なし/ありとNO/YESの2×2、および機体集合の4×4を表示する。
  行が注釈、列が回答。件数と行内割合を併記し、エラー・未実行は別に数える。
  全反復では応答件数を合算する。同じ注釈を含む重複窓や同一試合の反復なので、独立したイベント数ではない。
  この集計は初回検出の早さを測るものではない。
- **API間の混同行列**：行をResponses、列をDecisionsとしたYES/NOの2×2と機体集合の4×4。
  同じ入力範囲・sample・反復で両APIの回答が有効なペアだけを使い、件数・行内割合・一致率・除外ペア数を表示する。
  反復選択に従って集計する。対角線はAPI間の一致であり、注釈に対する正解を意味しない。

グラフの点はクリックまたはキーボードで選択できる。その入力窓の生データと回答詳細に移動する。
詳細の「前の時間へ／次の時間へ」は、現在のAPI・反復を固定し、絞り込みに含まれる窓を右端step順に移動する。
最後の窓で止まり、別APIや次の反復には移らない。APIや反復を変える場合は回答選択欄またはフィルターを使う。
詳細の「表示項目」では方位・高度・TAS速度・ロール・ピッチ・北向き/東向き/上向き速度を切り替えられる。
高度・方位・TASは実測値と対応actionの2段表示。actionは元IDと概算オフセット（高度±100m、方位±15/30度、TAS±17m/s）の対応を併記する。
ロール・ピッチ・速度成分には直接対応するactionがないため実測値のみ表示する。欠損actionを維持指令とは扱わない。
方位とロールは360度境界を連続化し、ピッチは記録値のまま表示する。表示項目の選択は窓を移動しても保持する。
Responsesの詳細には、抽出した機体・step・理由に加えて、形式エラー時も含む**回答本文**を表示する。
旧プロンプトの結果も同じviewerから選択できる。

グラフの差分定義、欠測、混同行列、本文保持の回帰テストは外部依存なしで実行できる。

```bash
node --test experiments/18_desicion_LAG/tests/test_analysis.cjs
```

### v003：全体の行動傾向を先に判定する改善案（未評価）

v002のcase06・10では、繰り返す方位指令と運動の振動の一部分を新しいイベントとして報告した。
case07ではA0200のheading ID=4をleft30と説明し、実際に243で`[1,2,0] → [1,0,1]`へ変化した
A0100を除外した。記録値、解釈、未確認事項は
[v002の確認記録](analysis/20261026_v002_global_pattern_review.json) に分離して保存している。

[v003/system.txt](prompts/v003/system.txt) は、窓全体の行動傾向を把握してから、
その傾向を変える事象か判定し、最後に開始stepを特定する。反復振動の一相、小さな段差、
同じ動きの収束だけをイベントとして分割しない。各機体のobject_idを固定して指令・状態を読み、
理由には同じ機体の変更前後のaction IDとstepを記載する。
末尾付近の本当の開始を落とさないため、新しい動きが窓の大半を占めることや完成することは要求しない。
発射は独立した対象とする。LAGの仕様説明・冒頭・1〜5節・8節は変更していない。

case08の253は実際に高度hold→downがあり、その後も下降するため、単なる振動とは扱わない。
既存旋回内の高度変更を独立イベントと数えるかは、注釈粒度の論点として残している。
ラベルの書き換えや、全ての高度変化を無視するルールは追加していない。

同じ10例・H=40・モデル・推論設定・出力schemaで未送信の計画を作成済み。

```bash
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v003/global_trend_h40
.venv/bin/python experiments/18_desicion_LAG/run.py compare --runs 01_develop/v002/onset_localization_h40 01_develop/v003/global_trend_h40
.venv/bin/python experiments/18_desicion_LAG/build_results_viewer.py
```

### v002：開始地点の特定を優先した改善案（初回実行済み）

`01_develop/v001/concise_h40` の10件を、回答と実際の入力値を照合して確認した。
二択はTP=5、TN=1、FP=4、FN=0だが、同じ機体・±2stepでの地点一致は7注釈中3地点。
予測12イベントの右端一致は0、左端一致は4。case05とcase08には、理由がNOを述べる一方で
decision=YES・events非空という内容上の矛盾があった。数値・各ケース・入力根拠・解釈の区別は
[監査JSON](analysis/20261026_concise_h40_error_audit.json) と
[ケース別CSV](analysis/20261026_concise_h40_cases.csv) に保存している。

[v002/system.txt](prompts/v002/system.txt) は、Aの2機を個別に確認し、前後の指令傾向から開始stepを決め、
後の運動を確認根拠に使う。動きの完成を待つことで末尾付近の開始を落とさないようにし、
既存の補正振動や同じ指令下の収束を新規開始と混同しないようにした。
diagnosticでは採用イベントだけを返し、decision・events・reasonを一致させる。
冒頭・1〜5節・8節はv001から変更せず、判定部分の6・7節と問い・選択肢・出力指示を修正した。
[親版からの差分](prompts/v002/parent.diff) を参照。

同じ10窓・設定で初回の実行結果を保存済み。v003で扱う残存課題は上記の確認記録を参照。

```bash
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v002/onset_localization_h40
.venv/bin/python experiments/18_desicion_LAG/run.py compare --runs 01_develop/v001/concise_h40 01_develop/v002/onset_localization_h40
.venv/bin/python experiments/18_desicion_LAG/build_results_viewer.py
```

採用判断では、既存の発射関連3地点を維持した上で83・95・143・243の局所化を改善し、
正例5窓の取り逃しを増やさないことを優先する。その後にFPを減らす。
最初の1反復を確認してから同一プロンプトを複数回再実行し、たまたま一致した可能性を確認する。
理由の矛盾を後処理でNOへ書き換えるだけでは、case05がFNになるため改善とみなさない。

調整時に見る点は以下。

1. **取り逃し**：正例でNO、またはYESでも83だけを拾い95を落としていないか。
2. **過剰検出**：小さな制御変更や同じ旋回の継続を、新しい変化と呼んでいないか。
3. **位置**：機体と開始stepが合うか。発射とミサイル初出の1step差も区別する。
4. **理由**：記録にある変化と、その意味の推測が分かれているか。標的・命中・連携を捏造していないか。

レビューの0/1/2は「根拠なし／部分的に支持／十分に支持」の作業用評点。
誤り分類は `missed_change`、`continuation_as_change`、`wrong_step`、`wrong_aircraft`、
`unsupported_intent`、`unannotated_event`、`insufficient_observation` 等を使える。
理由の正しさは自動採点していない。未注釈事象を正解へ追加する場合は、新しいラベル版・runを作り、同じ評価問題の比較と混ぜない。

まず10例のFNと根拠の不整合を減らし、同じプロンプトを `--repeats 3` で再実行して安定性も確認する。
この小標本に適合した事実と、未知試合でも再現するという仮説を区別する。

### 3＋4：固定プロンプトをsliding windowで評価

```bash
.venv/bin/python experiments/18_desicion_LAG/run.py evaluate --prompt v003 --run sliding01
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 02_evaluate/v003/sliding01
open experiments/18_desicion_LAG/results/02_evaluate/v003/sliding01/report/review.html

# Hの効果はプロンプト変更と分けて別runで比較
.venv/bin/python experiments/18_desicion_LAG/run.py evaluate --prompt v003 --run sliding_h30 --history 30

# Decisionsの4択だけを実行する場合
.venv/bin/python experiments/18_desicion_LAG/run.py evaluate --prompt v003 --run decisions_only --conditions decisions
```

H=40では窓末尾を39,44,…,339,340とする。最後の340を追加するため末尾の間隔だけ1step。
**62窓中、注釈を含む窓は41、含まない窓は21**。H=30では末尾29,34,…,339,340の64窓（34正例・30負例）。
5step刻みはシミュレーション時間の1秒間隔であり、API送信を実時間で1秒おきに行うものではない。
1リピートずつ時系列にリプレイし、各窓内のAPI条件の順番だけ固定seedでシャッフルする。
同時実行は1、隠れた自動リトライは0。3回は生成・計測の反復で、独立した試合3本ではない。

| 条件 | 出力 | 用途 |
|---|---|---|
| `responses_diagnostic` | YES/NO、複数event、step、根拠step、短い理由 | 検出地点と理由、重複の検証 |
| `decisions` | NONE / A1 / A2 / BOTH、各選択肢の確率、confidence | 検出の有無・変化した機体・速度・費用 |

Responses二択の条件は使用しない。古い保存済みrunを読むための互換処理のみ残す。
Decisionsの4択は、NONE=どちらもなし（NO）、A1=A0100のみ（YES）、A2=A0200のみ（YES）、
BOTH=両方（YES）。BOTHは同じstepでの変化に限らず、窓内の別々のstepでもよい。
参考機体集合は窓内の注釈から、Responsesの予測集合はeventsのaircraft_idから導く。
同じ機体の複数イベントは集合では1機として数える。

Decisionsの自由文・step出力は要求しない。機体集合の一致と開始地点の一致は別に評価する。
共有systemと入力履歴は同じ。Responsesには診断出力、Decisionsには機体集合を尋ねる。
このため速度・費用は、理由生成込みと4択の運用条件の比較であり、APIだけの純粋な効果とは解釈しない。
4択の問い・選択肢は `prompts/v003/aircraft_question.txt` と `aircraft_choices.json` に保存する。
既存のsystem・user・診断プロンプトは変更せず、runの `decisions_task=aircraft_set` で4択を選ぶ。
旧runでは保存された二択リクエストを再現する。

YES/NOは選ばれた4択から導く。`P(YES)=P(A1)+P(A2)+P(BOTH)` も保存するが、
この合計に閾値を適用して選択結果を上書きしない。各機体の周辺確率も保存する。
Responsesのreasoningは既定`low`、出力上限4096token。必要なら `--reasoning-effort none` 等の別runを作る。
DecisionsへResponses固有のreasoning設定は送らない。

## 指標と解釈

- **二択**：TP/TN/FP/FN、recall、precision、F2、FNR/FPR。取り逃しを重視してrecallとF2を見る。
- **機体4択**：NONE/A1/A2/BOTHの完全一致率と4×4混同行列、A0100・A0200別のTP/TN/FP/FN・recall。例えば正解BOTHに対してA1なら、二択ではTPでもA0200はFNと数える。
- **障害**：未実行、APIエラー、拒否・不正出力、送信済みか不確定を分離。エラーをNOへ変換しない。有効出力のみの指標と、試行したエラーも取り逃し扱いにする指標の両方を残す。
- **step**：同じ機体、完全一致および±2step（0.4秒）。予測と注釈を一対一で最大対応させ、同数ならstep誤差を最小化する。1つの指摘に2地点分の点を与えない。
- **検出遅延**：各注釈について、最初に指摘できた窓の末尾−注釈stepと、最初の検出機会からの超過stepを保存。API処理時間は別に測定する。
- **重複**：診断条件では同じ注釈を指摘した2窓目以降を数える。Decisionsでは窓のYESと機体選択のcoverage proxyを保存する。機体選択から個別事象の開始stepを識別したとは扱わない。
- **速度**：有効応答のp50/p95と全試行のp95。SDK呼び出し開始から応答受信・SDKデコードまで。待ち行列、後段の説明生成、実時間の継続配信は含めない。両APIとも成功した同じ窓・リピートのペアで判定一致率と時間差も集計する。
- **費用**：usageからUSDを推定。cache read/writeと通常入力を重複加算しない。usage不明はnullとし、既知費用の部分合計・不明件数を併記する。料金設定はrun内に保存する。

`summary.json` の `streams` はリピートごと、`paired_comparisons` は条件ペアごとの結果。
窓には重複があり、7注釈も同じ1試合から来るため、窓を独立標本とみなした信頼区間は出さない。
**今回の全試合評価にも調整に用いた地点が入る。これは運用に近いリプレイ評価であり、独立テストの正答率ではない。**
汎化性能を主張するには、プロンプト固定後に別試合の注釈を追加し、試合単位で評価セットを分離する。

### 重複と後段説明への接続

生のYES/NO評価には抑制を入れない。0/10/40stepの単純cooldownでトリガーを減らした場合も別途集計するが、
これは感度分析であり採用済み運用方針ではない。特に46と49のような近い別事象を抑制する可能性がある。
運用ではYES時に賢い説明モデルへ窓と直近の説明済み事象を渡し、同じ事象か新しい事象かを確認する構成を検討する。
この実験は検出器の評価までを実装しており、後段の大モデル呼び出しは実行しない。
17の前後30stepをそのまま再現するなら、検出時点で未観測の後半がある場合に待ち時間が必要になる。
現在窓だけで即時説明する条件とは、精度・遅延を分けて検証する必要がある。

## プロンプトの編集点

- `system.txt`：Aチームの行動傾向の変化・発射の基準、操作の振動との区別、環境・行動・欠損値の解釈。
- `user.txt`：履歴の共通テンプレート。評価ラベルは入れない。
- `question.txt` / `choices.json`：両APIで共通の問い・YES/NOの意味。
- `diagnostic.txt` / `choice.txt`：Responsesだけの出力契約。
- `metadata.json`：親版と今回の仮説。変更ごとに更新する。

17の `tactical_short_v1` から、環境、状態・行動の定義、ミサイルの注意点、推測の留保を引き継いだ。
中心±30step・4機の解説というタスク部分を、現在までの窓・Aチームの行動傾向の変化と発射の検出へ変更した。
理由は前後の動きと発射の観測根拠を中心にし、戦術的な目的の推測は補助情報に限定する。
現在のsystemは、`behavior_change_h40` の冒頭・1〜5節（LAG仕様）・8節（根拠と解釈）を原文どおりに復元した。
短縮対象は6節（検出対象）・7節（操作の振動と行動変化の区別）だけで、両節を688語から292語に整理した。
system全体は1,246語（いずれも空白区切り）。diagnosticは出力形式、question／choicesは問いと二択の定義に絞り、
判定基準の詳細はsystemを参照する。短縮による精度への影響は未測定。
[初期版のsystem差分](prompts/v001/system.diff) と [user差分](prompts/v001/user.diff) を参照。
各runにも実際の編集結果から計算し直した差分を保存する。過去版の原文は [reference](prompts/reference/tactical_short_v1/) に固定した。

## 再開・検証

```bash
# まず1件だけ送る場合
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v001/trial01 --limit 1
# 続きは、まだ一度も開始していないリクエストだけを送る
.venv/bin/python experiments/18_desicion_LAG/run.py execute --run 01_develop/v001/trial01 --resume
.venv/bin/python experiments/18_desicion_LAG/run.py status --run 01_develop/v001/trial01
.venv/bin/python experiments/18_desicion_LAG/run.py list
.venv/bin/python -m pytest experiments/18_desicion_LAG/tests -q
```

開始前にjournalへ書き込み、結果を保存してから終了を書き込む。タイムアウト・中断後、開始済みリクエストは自動再送しない。
APIエラー時は残りの送信を止める。`--resume` は成功・失敗・不確定を問わず開始済みを飛ばす。
`--continue-on-timeout` を付けると、`APITimeoutError` は失敗として保存した上で次の未実行queryへ進む。
同じqueryの再送はしない。認証・レート制限・サーバーエラーなど、それ以外のAPIエラーでは引き続き停止する。
この実行オプションはsessionのjournalに記録する。保存済みのプロンプト・リクエスト・180秒の設定は変更しない。

```bash
.venv/bin/python experiments/18_desicion_LAG/run.py execute \
  --run 02_evaluate/v003/sliding_h40 --resume --continue-on-timeout
```
失敗分を再測定したい場合は新runにする。拒否・形式エラーは集計に残し、続くクエリは実行する。
run内の固定入力を編集すると検証に失敗する。変更は元のプロンプトを編集し、新runで行う。

料金・API仕様の確認日：2026-10-26。利用権限と実環境での応答は初回の実API実行で確認する。
[Responses Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、
[Decisions API](https://developers.openai.com/api/docs/guides/decisions)、
[Decisionsリクエスト・応答仕様](https://developers.openai.com/api/reference/resources/decisions/methods/create)、
[gpt-6-lunaと料金](https://developers.openai.com/api/docs/models/gpt-6-luna) を参照した。
