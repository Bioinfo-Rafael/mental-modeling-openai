# Experiment 14: LAGでのmental modeling

13の統合CSVを使い、「対象機体のみ」と「全機体＋ミサイル」の2条件でnext/last action・next/last stateを予測する。
`run.py → runner.run() → make_plan() → execute()`の構成を引き継ぐ。プロンプトはレビュー済みの英文（`lag-context-v3`）を使用する。

通常の起動は**ローカルでの計画・プロンプト生成のみ**。API送信は`--execute`を指定したときだけ行う。

## 1. 今回の条件

条件の入口は[run.py](run.py)の`EXPERIMENT = Experiment(...)`。設定の型・検証・モデルIDは[config.py](lag_experiment/config.py)に置く。

| 項目 | 初期設定 |
|---|---|
| データ | 13のprocessed CSV、3試合 |
| 予測対象 | 各試合のA0100、A0200を別々に評価：計6軌跡 |
| 入力範囲 | `self`＝対象機体のみ、`all_aircraft_missiles`＝全4機＋両陣営の全ミサイル |
| モデル | sol / terra / luna |
| 履歴行数H | 2, 5, 10, 20, 30 |
| 課題 | next-action / last-action / next-state / last-state |
| n | 各条件の有効時刻範囲から等間隔に選んだ異なる5窓 |
| 各窓の推論回数 | 1回。同一問題の5回反復ではない |
| 時間間隔 | 履歴内は連続step、1step = 0.2秒 |
| few-shot / reward | 追加しない。rewardはデータにない |

**3モデル × 5種類のH × 3試合 × 2機体 × 4課題 × 2入力範囲 = 720条件。各5問で3,600リクエスト。**
モデル間では同一の問題・プロンプトを使うため、異なる入力は1,200件（各入力範囲600件）。2入力範囲の間では同じ開始step・予測対象step・正解を使う。

モデルIDは既存実験の設定を踏襲して`gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna`。
`reasoning_effort=medium`を指定し、temperatureは指定しない。利用環境でのモデル提供状況は別途確認する必要がある。
SDKの自動リトライは0、タイムアウトは180秒、送信は逐次実行。出力トークン上限は明示設定していない。
変更する場合は`config.MODELS` / `REQUEST_OPTIONS`を編集し、新しい出力名で計画を生成する。

### 2つの入力範囲

- `self`：従来どおり対象機体の状態・行動のみ。
- `all_aircraft_missiles`：同じ時刻に記録された全4機の状態、利用可能な機体行動、両陣営の全ミサイルの状態。対象機体と予測する成分はselfと同じ。

機体・ミサイルの状態は共通の10列順で表示する。ミサイルの未収録の速度4列はnull、Bチームの未収録行動もnullのままにする。補間や速度差分による推定は行わない。
全機体条件では`entity_type / team / alive / launcher_id / event`を併記し、撃墜後やremove行の無効な状態値は全成分nullにする。ミサイルが未記録のstepに行を追加しない。撃墜・爆発の表示オブジェクト（effect）は含めない。

履歴の過去時刻には記録済みの機体行動を提示する。追加情報では状態が与えられる時刻に全オブジェクトの状態を加え、行動が既知の時刻だけ他機体の行動も加える。next/last actionの対象時刻には他機体の行動も提示しない。
last-stateの対象時刻では対象機体の状態行全体を隠し、他機体・ミサイルの状態は提示する。next-stateでは正解時刻の全オブジェクトを入力に含めない。last課題で与える後続状態以外の未来情報は追加しない。

`observation_scopes`が入力範囲の設定。CLIでは`--scopes self`または`--scopes all_aircraft_missiles`で片方だけにもできる。ファイル名・condition_id・manifest・採点結果に入力範囲を記録する。

### 状態・行動の順序

状態は次の連続10次元。入力も正解も出力も同じ順序で、正規化・離散化・追加の桁丸めをしない。

```text
lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg,
tas_mps, vn_mps, ve_mps, vu_mps
```

経緯度・姿勢は度、高度はm、速度はm/s。vn / ve / vuは北 / 東 / 上方向。
CSVの元の数値文字列は13に保持し、プロンプトには型付き読み込み後のPython floatをJSONで記載する。

履歴に渡す行動は`[altitude_action, heading_action, speed_action, shoot_action]`の4整数ID。
行動予測の出力・正解は最初の3成分のみとする。

| 成分 | IDの意味 |
|---|---|
| altitude_action | 0=up, 1=hold, 2=down |
| heading_action | 0=left30, 1=left15, 2=hold, 3=right15, 4=right30 |
| speed_action | 0=up, 1=hold, 2=down |
| shoot_action | 0=no, 1=yes。履歴・既知行動には提示、行動予測では出力しない |

next/last actionでは予測対象時刻の行動を隠すため、その時刻のshootも提示しない。
next/last stateでは既知の行動として対象遷移のshootも提示する。
行動の確率、CSVの`missile_launched`列、health、残弾数、勝敗を含むファイル名は入力に入れない。全機体条件では、すでに記録されたミサイルの存在と発射元は提示する。

## 2. 時刻と4課題の定義

`s_t`はstep tの状態、`a_t`はt→t+1に作用する行動。`b`は履歴の開始step。
**Hは実際に渡す履歴の時点数**。selfでは各時点に対象機体の`(s_t, a_t)`、全機体条件では同じ時点に他機体・ミサイルの記録も加える。全課題の履歴は`b ... b+H-1`である。

| 課題 | 基準時刻q | 履歴に加えて見せる情報 | 予測する正解 |
|---|---|---|---|
| next-action | b+H | s_q | a_qの移動3成分 |
| last-action | b+H | s_q, s_(q+1) | a_qの移動3成分 |
| next-state | b+H−1 | s_q, a_q（履歴最終行と重複） | s_(q+1)の10成分 |
| last-state | b+H | a_q, s_(q+1) | s_qの10成分 |

LLM-Xavierの`_history_range()`と各質問の情報提示に合わせた時刻関係である。
上流の`history_size`に換算すると`H−1`。たとえばH=2、b=0なら履歴はstep0,1。
next-actionはa2、last-actionはs2,s3からa2、next-stateはs2、last-stateはa2,s3からs2を予測する。
last-stateでは正解のs2を入力に含めない。

上流next-stateのreward提示は、今回のデータにrewardがないため省く。
last課題の後続状態は、推定のために意図的に与える条件。通常の将来予測と混ぜて集計しない。

### n=5の選び方

試合・機体・課題・Hごとに、必要な履歴・既知情報・正解がすべて得られる開始stepを列挙する。
有効な開始stepが`b_min ... b_max`の連続M個なら、選ぶ開始stepは次の通り。

```text
b_k = b_min + floor(k * M / n),  k = 0, ..., n-1
n = 5
```

M=400、b_min=0なら`0,80,160,240,320`。終端を含むlinspaceではなく、5等分した各区間の先頭を取る。
各窓の内部は連続stepで、窓同士の開始時刻をずらす。重なる窓も許す。
履歴長と末尾の正解可用性があるため、CSVの全step数をそのままMにしない。
たとえばMHI試合のA0100、H=2、next-actionでは有効開始は0〜338の339個なので、`0,67,135,203,271`となる。

H・課題によって有効範囲が変わるので、**H間・課題間で正解時刻が一致するとは限らない**。
同じ条件のモデル間および2入力範囲の間は同じ5問で比較できる。Hだけを同一正解で比較したい場合は、今後`select_uniform_starts()`と計画作成側で共通の正解時刻を選ぶ方式を追加する。
実際の開始step、正解step、有効範囲、除外理由別件数はmanifestの`coverage`に残す。

### 死亡・欠損・末尾

以下の完全性条件は予測対象機体に適用する。他機体・ミサイルの欠損で窓を除外しない。

- 対象機体の履歴の状態10成分と行動4成分が完全であることを要求する。
- 既知情報と正解にも必要な成分の完全性を要求する。状態は`state_valid=1`、行動は`action_status=recorded`が必要。
- 死亡後の古い座標、欠損、NaN/Infinityは使用せず、補完しない。
- ACMIだけの末尾状態はvn / ve / vuが欠けるため、10次元状態として利用しない。
- loss試合のA0200は途中で死亡するため、生存中の有効な範囲から5窓を選ぶ。
- 有効窓がn未満なら計画をエラーにする。nを黙って減らさない。
- 有効開始stepに途中の穴がある新規データもエラーにする。時刻上の等間隔を、有効レコードの順位上の等間隔へ黙って変更しない。

## 3. 既存コードを使う部分、新しく実装する部分

| 処理 | 実装 |
|---|---|
| CSVスキーマ・状態と行動の定義 | 13の`lag_data/definitions.py`を再利用 |
| CSV読込・検証・索引化 | 13の`load_episode()` / `index_records()`を再利用 |
| systemの外枠 | **LLM-Xavierの`llm_x.prompting.system_prompt()`を変更せず直接呼ぶ**。生成後、空のReward節だけをLAG側で削除 |
| userの外枠 | **LLM-Xavierの`llm_x.prompting.user_prompt()`を変更せず直接呼ぶ** |
| LAGの説明、履歴の整形、質問、回答形式 | 14の`prompts.py`で新規定義 |
| 窓抽出、4課題の可視情報と正解の分離、条件の展開 | 14の`runner.py` |
| API実行・記録・再開 | 09の記録方式を参考に14の`execution.py`で実装 |
| 3成分の離散行動・10成分の連続状態の採点 | 14の`evaluation.py` |

上流の`Episode`はNPZ前提なので作らず、`user_prompt()`が必要とする`history_text()`だけを持つ小さな`HistoryView`を渡す。
履歴の外枠は上流が生成し、中身はLAGのstep・時刻・状態配列・行動配列として描画する。
上流の`evaluate_episode()`、`feedback.py`、既存の単一離散行動やPendulum向けの分類・採点は呼ばない。
既存01〜12とupstreamのソースは変更しない。

### プロンプトを変更する

[lag_experiment/prompts.py](lag_experiment/prompts.py)の次の関数を編集する。

1. `system_prompt(query)`：タスク、状態・行動、遷移などの説明。説明辞書を上流のsystem関数へ渡す。
2. `user_prompt(query)`：質問文と既知情報の配置。履歴と質問を上流のuser関数へ渡す。
3. `_render_block()`：状態・行動の表示形式を変える場合。

`build_messages()`は可視データだけを受け取る。正解値を渡さない構造にしている。
LAG固有の文面・履歴をここで変更する。systemは環境説明を中心とし、Reward節を含めない。状態の参考範囲は`STATE_DESCRIPTIONS`で管理し、経度は`[119, 121]`、緯度は`[59, 61]`とする。データ自体の正規化やクリップは行わない。
変更後は`PROMPT_VERSION`を更新し、新しい`--output`名で生成する。

全4課題の回答は`1. [Reasoning]:`、`2. [Prediction]:`、`3. [Formatting]:`の順に求める。各見出しに追加の説明指示は付けず、Formattingに次の最終回答行を置く。

```text
>>Final action IDs: [1, 2, 0]
>>Final state values: [120.0, 60.0, 6096.0, 0.0, 0.0, 10.0, 243.8, 240.0, 40.0, 0.0]
```

各回答には対応するマーカーを1回だけ出す。配列はJSONで、行動は範囲内の整数3個、状態は有限数10個。
4行動を返した場合にshootを黙って切り捨てることはせず、形式失敗とする。
回答形式まで変えるときは`evaluation.parse_prediction()`と`SCORING_VERSION`も更新する。
生成済み`prompts/*.txt`は確認用で、直接編集して送信する方式ではない。

## 4. 実行方法

すべてリポジトリルートから実行する。既存`.venv`を使用する。
新しい環境なら`python -m pip install -e '.[data,test]'`でpandas等とテスト依存を導入する。

```sh
# 全720条件・3,600件の計画と1,200件のプロンプトを保存。API不要。
.venv/bin/python experiments/14_mental-modeling_LAG/run.py

# プロンプトを更新した後は、出力名も変更する。
.venv/bin/python experiments/14_mental-modeling_LAG/run.py --output prompt_v2

# 小さい条件でプロンプトを確認する例。これもAPI不要。
.venv/bin/python experiments/14_mental-modeling_LAG/run.py \
  --models sol --histories 2 --episodes mhi_v1.3_ic100601_win \
  --targets A0100 --tasks next-action --n 2 --output smoke
```

送信するときは同じ設定・同じ出力名に`--execute`を付ける。全件送信のコマンドは以下。
`OPENAI_API_KEY`がなければ非表示入力を求める。キーを設定ファイルやmanifestへ保存しない。

```sh
.venv/bin/python experiments/14_mental-modeling_LAG/run.py --execute

# 中断後：一度も送信を試みていないリクエストだけを続行する。
.venv/bin/python experiments/14_mental-modeling_LAG/run.py --execute --resume

# 採点の更新だけ。保存済み回答を利用し、APIを呼ばない。
.venv/bin/python experiments/14_mental-modeling_LAG/run.py --rescore
```

上の`smoke`など条件を絞った実験を実行・再開する場合、`--execute` / `--resume`に加え同じ条件引数を指定する。
`--rescore --output smoke`は保存済み計画を読むため、条件引数を付けない。
`--output`はこの実験の`results/`内のディレクトリ名。既存と異なる計画の上書きは拒否する。

### 再開の挙動

送信直前にリクエストを記録し、回答は解析・採点より先に原文を保存する。
API例外が出たらそこで停止し、エラーを記録する。タイムアウトなど、課金・完了したか不明な呼出しも自動再送しない。
`--resume`は未試行の残りだけを送るため、失敗した条件は5件の完全な平均にならない場合がある。
その場合は`complete=false`、成功数・失敗数・未評価数を確認する。再送が必要な場合は対象と重複を確認して別実験として計画する。
送信直前に、データ・コードのSHA-256と保存済み計画の一致を確認する。
プロンプトや条件、コードを変えた後は旧計画の続行ではなく新しい出力名を使う。
同じ出力ディレクトリを使う同時実行はロックで拒否する。

## 5. 採点と5問の平均

集計単位は**モデル × 試合 × 対象機体 × H × 課題 × 入力範囲**。各条件の5窓を等しい重みで平均する。
同一試合内の5窓は別問題であって、5個の独立した試合ではない。

| 予測 | 主な集計 |
|---|---|
| 行動 | 3成分すべて一致した割合、各成分の正解率 |
| 状態 | 10成分それぞれのMAE（絶対誤差の平均、各成分の元の単位） |
| 姿勢の補助評価 | rollとyawについて循環角誤差`abs((pred−true+180) % 360−180)`の平均も保存 |

全5件が正常なら、その5件の平均となる。形式失敗を状態値0として扱わない。
失敗がある場合、`parsed`件だけの平均と分母を明示し、`complete=false`とする。
行動は、正常回答内の正解率に加えて計画5件を分母にした`exact_match_rate_all_planned`も保存する。
状態の経緯度はdegree誤差であり、mへの投影はしていない。単位の異なる10成分を一つのMAEに合算しない。
循環角誤差は補助列とし、生値MAEと区別する。pitchは循環角集計に含めない。
既存実験のINC/DEC/UNCH分類や離散ビンへの変換は今回の連続値評価では行わない。

各条件の`planned / responses / parsed / parse_failures / api_errors / unscored / complete`を保存する。
`responses`はAPIエラー記録も含む応答ジャーナルの件数、`parsed`は正常な予測の件数。
送信記録だけが残り応答がない中断は`unscored`に含み、API summaryの`unresolved`でも追跡する。

## 6. 保存内容とレビューの入口

```text
14_mental-modeling_LAG/
  run.py                          # 条件と入口（直下のPythonはこれだけ）
  lag_experiment/
    config.py                     # 設定の型・検証、モデルと生成設定
    runner.py                     # 時刻関係、抽出、計画、CLIの流れ
    prompts.py                    # レビュー済みsystem/user、上流への接続
    execution.py                  # 保存・送信・再開
    evaluation.py                 # 回答解析とオフライン採点
  tests/test_experiment.py
  results/<output>/
    manifest.json                 # 全条件・時刻・正解・messages・ハッシュ
    prompts/<実験条件>.txt          # 確認用。モデル間で同じ入力を共有
    api/                          # --execute後に作られる
      identity.json
      requests.jsonl              # 送信前の正確なリクエスト
      responses.jsonl             # 生の回答・エラー・経過時間
      summary.json
    analysis/<scoring_version>/   # 実行後または--rescore後に作られる
      per_query.json
      conditions.json
      conditions.csv              # 1条件1行、成分別の指標を列に展開
      metadata.json               # 評価コードと回答ジャーナルのハッシュ
```

manifestの`samples`はモデル共通の問題と正解、`queries`はモデル別の送信予定。
プロンプトのファイル名は`<episode_id>__<target_id>__<task>__<observation_scope>__H<履歴長>__sample<1始まりの問題番号>__start<履歴開始step>__step<予測対象step>.txt`。
例：`mhi_v1.3_ic100601_win__A0100__next-action__self__H02__sample01__start0000__step0002.txt`。
モデル間で本文が共通のためモデル名は付けず、1,200個のファイルを保存する。勝敗を含む試合名は確認用ファイル名のみで、LLMへ渡す本文には含めない。
`problem_id`は元CSV・機体・課題・正解時刻・正解列から作り、Hが違っても同一正解の問題を照合できる。
`repeat_id=1`は記録用で固定し、反復生成しない。
`coverage`は採用・除外の根拠、`source_sha256`はデータと使用コード、`plan_id`は計画全体のハッシュ。
採点結果だけを再生成しても、manifestとAPIジャーナルは変更しない。

レビューは`run.py`の条件 → `runner.build_query()`の情報境界 → `select_uniform_starts()` → `prompts.py` → 採点・実行の順で読める。
継承階層やプラグイン機構を作らず、データ依存は13に、実験の判断は14にまとめた。
間引き履歴、few-shotは未実装。今後は14の入力構築・プロンプトに追加する。
共通に使えるCSV操作を追加するときだけ13へ置く。

```sh
.venv/bin/python experiments/13_LAG/run.py --check
.venv/bin/python -m pytest experiments/13_LAG/tests experiments/14_mental-modeling_LAG/tests -q
```

テストは実データでの全条件数・時間選択・正解非混入・死亡と終端欠損、回答形式・集計、仮のAPIクライアントでの記録・再開を検証する。外部APIは呼ばない。
統合CSVの64列スキーマと元データの保持方針は[13の詳細README](../13_LAG/data/processed/README.md)を参照。

## 7. 通常API 220件からBatch APIへの切り替え

`batch_run.py`は保存済みの`manifest.json`をそのまま使用する専用入口。
元の計画・プロンプト・モデル・`reasoning_effort=medium`を変更しない。
既存のソースhashも検証するため、Batchアダプターは`lag_experiment/`の外に置いている。

前提は、通常APIの先頭220件の送信が終了し、220件すべてに応答記録があること。
通常runnerが動いている間は出力ディレクトリのロックによりBatch側は実行できない。
221件目以降の3,380件がBatchの対象になる。通常APIのエラー記録も送信済みとして除外し、自動再送しない。

repository rootから実行する。`OPENAI_API_KEY`を設定済みなら利用し、未設定なら非表示入力で尋ねる。

```bash
# オフライン準備・分割内容の確認（API送信なし）
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py prepare --output sliding_window_v1

# 残りを送信 → 完了を待つ → 結果回収・採点。中断後も同じコマンドで再開。
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py run --output sliding_window_v1
```

`run`は終了まで起動しておく。Ctrl+Cでローカル処理を止めても、送信済みBatchはサーバーで継続する。
再実行すると保存済みBatch IDを照合して結果を回収し、未送信の分割ファイルを順次送信する。
端末を閉じる場合も、後で同じコマンドから再開できる。通常runnerの`--execute --resume`には戻さない。

### 分割・上限

- 入力JSONLはモデルごとに分ける。`custom_id`は既存の`query_id`。
- 1ファイルは既定で最大500件、入力推定100万tokens、190MB以下。
- token数は`o200k_base`によるrequest JSONの計数に10%と128 tokens/requestを足した準備用推定。実測usage・請求額ではない。
- 同時送信は各モデル1 Batchまで。終了・回収後にそのモデルの次の分割を送る。
- 利用アカウントのBatch queue上限と、別ジョブが使用中のtoken枠は別途適用される。
- 初回prepare時に`--max-input-tokens`と`--max-requests`を変更可能。準備済みの分割は再実行で変更しない。
- `--after-sync`の既定値は220。先頭からこの件数の通常API記録がそろっていることを厳密に検証する。

### 個別操作と保存先

```bash
# ローカルに保存済みの状況を表示（APIなし）
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py status

# サーバーの状態確認・結果回収後、各モデルで空いた枠に1 Batchずつ送信し終了
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py submit

# 状態確認・結果回収・採点のみ。新しいBatchは送信しない
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py collect
```

`api/batch/`に分割入力、状態ファイル、Batch情報、出力・エラー原文を保存する。
回収した応答は既存の`api/responses.jsonl`へ追記し、通常API分と合わせて既存採点器で集計する。
`transport=batch`と`batch_id`を記録する。Batchの待機時間を通常APIの推論時間に混ぜないため、
Batchの`elapsed_seconds`はnullとする。usageは応答本文の実測値を保持する。
`api/requests.jsonl`のBatch行は送信前の予約記録でもあるため、件数だけで課金済みと判断しない。

SDKの自動再送は無効。Batch作成前に予約・送信意図を永続化する。
作成応答を失った場合、再開時にmetadataの`lag_job_key`を使って最新1,000 Batchを照合する。
一意に特定できなければ自動再送せず停止する。Dashboardで該当Batch IDを確認できた場合は、次で関連付ける。

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py adopt --job terra_000 --batch-id batch_...
```

関連付け時には入力file ID・endpoint・metadataも検証する。
未作成だったことの確認、期限切れ、キャンセル、APIエラーの再送は自動化していない。
部分成功は保存し、欠損・失敗は明示的なエラーとして記録する。`run`は失敗を検出したら新しい分割の送信を停止する。
既に送信済みの別Batchは継続するため、`collect`で回収できる。

仕様参照（2026-10-12確認）：[OpenAI Batch API](https://developers.openai.com/api/docs/guides/batch)。
Batchは通常API比50%割引、処理枠24h。実際の利用料金はLAGの入力・出力usageに依存する。

### 最初の比較セット補完だけを送る（2026-10-12追加）

今回まず使用するのは次のコマンド。先頭220件で着手済みの条件だけを調べ、
不足分を送信して終了する。後続の未着手条件へは進まない。

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py complete-prefix --output sliding_window_v1
```

現在の対象は`mhi_v1.3_ic100601_win/A0100/H5/last-state/self`の不足5件
（sol 1件、terra 2件、luna 2件）。モデル別の3 Batchを作成し、サーバー側で並行して処理する。
既存220件は再送しない。同じコマンドを再実行しても送信済みBatchを二重作成しない。

送信せずに準備だけする場合：

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py prepare --completion-only --output sliding_window_v1
```

結果回収・採点は`batch_run.py collect --output sliding_window_v1`を使う。
全5件が成功すれば15条件×3モデル×5問＝225件がそろう。
状態ファイルには`scope=prefix-completion`を保存し、通常の全件用`run`による範囲変更は拒否する。
後続条件の優先順位・セット単位の送信計画は、補完分とは別に決定する。

### 30件セットを最大4組、完了ごとに補充する（推奨）

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/batch_run.py rolling \
  --output sliding_window_v1 --parallel-sets 4 --poll-seconds 30
```

このコマンドは送信・状態確認・回収・採点を最後まで行う。停止後も同じコマンドで再開する。
初回は補完専用計画をローカルで移行する。既存Batch ID・送信記録・応答原文を引き継ぎ、
旧状態を`api/batch/state.before_rolling.json`に保存する。準備だけなら`prepare-rolling`を使う。
移行後は`rolling`で続行する（旧`run` / `submit` / `complete-prefix`で全件送信しない）。
`status`・`collect`・`adopt`は移行後も利用できる。

- 1セット＝同じ試合・対象機体・H・課題で、2入力範囲×3モデル×同じ5問＝30件。
- 通常はモデル別の10件Batchを3本投入する。最大4セット・12本が同時処理対象。
- 3モデルが全て返ったセットだけ枠を解放し、次のセットを追加する。4セット全部の完了は待たない。
- 優先順位は保存済み実験設定の試合→対象機体→H→課題。self／全機体は必ず同じセット。
- 優先順位は**投入順**に適用する。あるHが処理中でも、そのHの4課題を投入済みなら次のHへ進める。
- 既定30秒ごとに状態確認。ローカルの回収・通信時間が加わるため、厳密なリアルタイム処理ではない。
- 入力token推定を全activeセットで合計し、各モデル100万tokens以内に制限する。
  `--max-queued-input-tokens`で変更可能。別のプログラムが使用中の枠は含まない。
- 送信済みの補完5件は先に回収してから新規投入する。その後、同じ条件の全機体15件を優先し、
  次の30件セットを空き枠に入れる。既存成功分を再送しない。
- 送信途中でプロセスが落ちてもadmission・予約・Batch IDを保存しているので、同じセットの残りから再開する。
  Batch IDが不明なら既存のmetadata照合を使い、重複送信しない。
- API失敗・期限切れがあれば新規投入を停止し、エラー内容を残す。別の送信済みBatchは`collect`で回収可能。

`api/batch/comparison_sets.json`に各30件セットの応答取得数・成功数・パース成功数を保存する。
`responses_complete`と`predictions_complete`を分け、応答がそろっただけで全予測が有効とは扱わない。

Batchはサーバー側で非同期処理されるが、待機時間もある。逐次送信より必ず速いという保証はない。

### 実行中に経過時間・予想終了時刻を表示する

送信側を止めず、別ターミナルで次を実行する。APIキーは不要。

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/batch_watch.py \
  --output sliding_window_v1 --interval 10
```

10秒ごとにローカルの保存済み状態を読み、rolling開始からの経過時間、モデル別成功件数、
比較セットの回収状況、予想残り時間・終了時刻（JST）を表示する。1回だけ表示するなら`--once`。
送信・API呼出し・ロック取得・ファイル書込みは行わない。Ctrl+Cは表示だけを止める。
サーバー情報の鮮度は送信側の状態確認間隔に依存し、最終保存からの時間も表示する。

ETAは通常API220件・補完5件を除いたrolling分を対象に、最初のBatch作成からの経過時間と
各モデルの成功件数で処理速度を算出し、最も遅いモデルの残り時間を使う。
途中の再開・ローカル停止時間も経過に含む。入力長、並列枠、API待ち時間の将来変化は予測しない。
特に序盤は大きく動く暫定値。各モデルの成功実績が未取得、またはAPI失敗がある場合は算出を停止する。
API処理の完了予想であり、最後のローカル回収・採点に要する時間は含まない。

### Question前後の状態・行動の時系列

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/plot_question_timelines.py
```

`analysis/question_timelines_<日時>/index.html`から試合・機体・問題番号・H・課題を選択する。
既定の表示区間は、回答対象の正解stepをtとして`t−50 ... t+10`（61点、1step＝0.2秒）。
`--before 50 --after 10`で区間、`--output-dir <新規ディレクトリ>`で保存先を指定できる。

- 5問×3試合×2機体の30グループ。H・課題によってtが異なるため、5種類のH×4課題を切り替える全600条件のPNGを保存する。現行計画の異なる正解時刻は177通り。
- 図は対象機体の状態10成分・行動4成分。対象系列と正解は3モデル・self／全機体で共通なので重複生成しない。他機体・ミサイルの系列は含めない。
- 青背景は入力履歴、橙の菱形は追加で与えた値、赤い星は予測させた成分の正解。赤い破線は全パネル共通でtを示す。
- 描画する全時系列は正解時刻の前後を含む実測値であり、全区間をモデルに見せた意味ではない。shootは既知行動に含まれるが行動予測の出力対象外。
- 試合開始前・不在・無効・死亡後の値は空白。補間しない。行動はstep間の階段表示。roll・yawは±180度を超える折り返しで接続線を切り、点の値は元の角度を保持する。
- manifest・CSVのハッシュを検証し、600条件すべてのselfプロンプトと正解を元データから再構成して保存済み内容と照合する。self／全機体の正解・履歴・対象時刻の一致も検証する。
- `timeline_index.json`に各条件の入力履歴・追加情報のstep・正解を、`metadata.json`に出典ハッシュと検証情報を保存する。API送信・元データの変更は行わない。

PNGは`<試合>/<機体>/question_01〜05/H02_next-action.png`のように分類する。

### 隣接する実測値との比較（コピー傾向の確認）

```bash
.venv/bin/python experiments/14_mental-modeling_LAG/plot_neighbor_differences.py
```

既存の`analysis/comparison_plots_v1/`に次の12図（3指標×pool・試合別3組）を追加する。
同じデータ・正規化基準であることを確認してから追加し、既存のPNGは変更しない。
`--output-dir <新規ディレクトリ>`なら独立したレポートを生成する。

ここではtを予測対象step、予測をŷ、実測状態をS、実測行動をaとする。

- 行動：nextは`mean(abs(ŷ_t − a_(t−1)))`、lastは`mean(abs(ŷ_t − a_(t+1)))`。
  3移動成分の絶対ID差を平均してから、問題を等重みで平均する。0は隣接する実測行動と同一。
- 状態：nextは`||D(ŷ_t − S_(t−1))||₂ / ||D(S_t − S_(t−1))||₂`、lastは
  `||D(S_(t+1) − ŷ_t)||₂ / ||D(S_(t+1) − S_t)||₂`。
  Dは成分ごとに既存図と同じ固定範囲、または固定標準偏差で割る処理。roll・yawは最短角度差を使う。
  10次元のL2ノルム比を各問題で計算してから平均する（変化量の平均同士の比ではない）。
- 状態の比が0なら隣接状態のコピー、1なら変化量の大きさが同じ。ただし方向の一致や予測の正しさは意味しない。
- 実際の状態変化が0の場合は0/0も含め比が未定義。有効nから除外して別集計し、微小値への置換や上限クリップはしない。
- 配置はnext／last×self／全機体／全機体−self、3モデルの折れ線。各点pool n=30・試合別n=10を基本とし、解析失敗・隣接値欠損・未定義の比で減ったnを注記する。
- これらは正解との誤差ではなく隣接値との比較であり、小さいほど良い指標ではない。差分の符号にも改善・悪化を割り当てない。
- last-actionの`t+1`の行動はプロンプトに提示していない。その値との一致だけでは入力コピーとは判定できない。また実際の行動が変化しない場合、隣接行動への一致と正解は両立する。

図の保存先は`action/adjacent_mae/`と`state/change_ratio/{range,sd}/`。
`neighbor_definitions.json`に式・出典・除外件数、`neighbor_per_query.json`に隣接実測値・予測・正解・変化量・比を保存する。
