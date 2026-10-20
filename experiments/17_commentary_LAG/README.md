# Experiment 17: LAGの場面・行動意図の解説

作成日: 2026-10-17 JST。抽出・プロンプト生成・Batch実行コードを実装。2026-10-20にユーザー共有の実験概要・結果を追記。以下の実装・計画と、共有結果の条件の差分は概要末尾に記載する。

## 実験概要・共有結果（2026-10-20）

先方のモデル対ルールベースの対戦について、特に動きの変化が大きかった **45, 145, 245, 315 step** を選択した。

- [アノテーション](https://docs.google.com/spreadsheets/d/1EIgeVnxUKSuLsN2aV1Vqb0Tmjwi7bY85CtlCrYRPyIE/edit?usp=sharing)
- 対戦ゲームの解説を作成する[論文（ACM）](https://dl.acm.org/doi/10.1145/3628454.3629551)を参考にプロンプトを作成した。

プロンプトの主な内容は次のとおり。

- LAGの説明。
- 行動の意図を説明させるための例示など。
- 上記timestepの前後30 step、合計61時点における全機体の状態・行動の数値。

上記アノテーションの情報や、特定のtimestepに着目すべきといったヒントは与えていない。

[結果スプレッドシート](https://docs.google.com/spreadsheets/d/1IIi_8UT3nvX4hkYIybChtHcyS2cpC7OVmGL_dkFgMvA/edit?usp=sharing)について、ユーザーによる所見は以下のとおり。

- 特に動きの大きかったA100については、自身でアノテーションした結果をほとんど再現した。
- 動きの変化が少ないA200については、解釈が分かれる場面が存在した（145, 245 step）。
- 自身のアノテーションでは、「囮・援護の役割」や「ミサイルが誰を狙っているか」などの点は説明できていない。

上記はユーザー共有内容の記録。A100・A200はログ／ビューアーではそれぞれ `A0100`・`A0200` と表記される。共有内容では「前後30 step（5秒）」とされているが、本リポジトリの時間定義は1 step = 0.2秒なので、前後30 stepは前後6秒（窓全体で12秒）に相当する。また、以下の既存実装・計画は **45, 85, 145, 245, 310 stepの5窓** を使用しており、共有された4場面とは異なる。315 stepの前後30 stepは既存のMHI試合の採用範囲を超えるため、共有結果との対応は未確認。ここでは実行設定を変更していない。

## 試合ビューアーの起動

録画と状態・行動のグラフは、13_LAGの試合ビューアーで確認できる。リポジトリルートから実行する（pandas、ffmpeg、ffprobeが必要）。

```sh
.venv/bin/python experiments/13_LAG/viewer/serve.py
```

ブラウザで <http://127.0.0.1:8130/viewer/> が開く。自動で開かない場合は、このURLにアクセスする。終了はターミナルで `Ctrl+C`。初回起動時は録画から表示用フレームを生成する。

試合に `MHI v1.3 / IC 100601 / WIN` を選び、機体とstepを切り替えて確認する。[スクリーンショット付きの案内](../13_LAG/README.md#試合ビューアー)と[詳しい操作方法](../13_LAG/viewer/README.md)も参照。

## 目的

13の状態・行動の数値時系列から、LLMが場面の推移、機体間の関係、各機体の行動意図をどの程度説明できるかを調べる。各試合の指定stepを中心とする61時点を1問とし、全機体をまとめて解説させる。

中心時点より後の情報も提示するため、評価対象は後続情報を踏まえた回顧的な状況解釈である。時点tまでの情報だけによるオンライン意図推定とは区別する。また、ログには意図の正解ラベルや方策の内部状態がないため、意図の真偽を直接測ることはできない。観測事実の正確さと、意図仮説の根拠・不確実性の扱いを分けて評価する。

## 13〜16から引き継ぐもの

| 実験 | 確認した役割 | 17での利用 |
|---|---|---|
| 13_LAG | 3試合の統合CSV、列定義、読み込み、動画同期ビューアー | データと定義を直接参照する |
| 14_mental-modeling_LAG | next/last action・state、self/全機体条件 | 全機体＋ミサイルの表示、欠損・無効値の扱いを参考にする |
| 15_trajectory_LAG | 未来10 stepの状態・行動を一括予測 | 計画・プロンプト・生回答の保存とBatch実行方式を参考にする |
| 16_dynamics_LAG | 15に姿勢角の経験的回帰式を追加 | 初期条件では回帰式を追加しない。追加する場合は別条件とする |

14〜16の数値予測用の回答形式・採点は、解説用に新設する。13の読み込みAPIは再利用できるが、14〜16の「対象機体の全状態・行動が完全な窓のみ」という選別は採用しない。消滅や死亡を含む場面も本実験の対象とする。

## 入力窓D_t

試合e、中心step tについて、次を定義する。

```text
D_(e,t) = {その試合のstep kに記録された全機体・全ミサイルの状態と利用可能な行動
           | k = t-30, ..., t, ..., t+30}
```

試合が明らかな場合はD_tと略す。61時点、60遷移であり、1 step = 0.2秒なので、最初と最後の時刻差は12秒（中心の前後6秒）。61行ではなく、各時点に複数オブジェクトがある。

- `episode_id, center_step`の組をユーザー指定の設定に保存する。同一stepでも試合が違えば別問題。
- 指定された範囲をそのまま使う。間引き、中心stepの自動移動、短い窓への切り詰めは行わない。
- 初期案では全61時点が元actions CSVの範囲内にある窓を採用する。ACMIのみの最終時点を使う条件は別扱いとする。
- この規則で中心stepの範囲は、MHI試合が30〜310、ours lossが30〜263、ours winが30〜465。これは時刻範囲の条件であり、全機体が生存していることは要求しない。
- ミサイルはその区間に記録のある全個体を含め、区間開始前に発射された個体も含む。命中・勝敗による選別はしない。
- `effect`は入力から除外。区間外の状態・行動・イベントは提示しない。
- 最終時点の行動も含めるが、その行動が作用した次状態は窓外であることを明記する。

### 指定された中心step

中心stepは **45, 85, 145, 245, 310** に固定する。

| 中心step t | 入力step（両端を含む） | 中心時刻 | 入力時刻 |
|---|---|---|---|
| 45 | 15〜75 | 9秒 | 3〜15秒 |
| 85 | 55〜115 | 17秒 | 11〜23秒 |
| 145 | 115〜175 | 29秒 | 23〜35秒 |
| 245 | 215〜275 | 49秒 | 43〜55秒 |
| 310 | 280〜340 | 62秒 | 56〜68秒 |

対象試合は`mhi_v1.3_ic100601_win`の1試合に固定する。指定の5窓すべてを使用する。

最初の2窓は21時点、2番目と3番目は1時点が重複する。manifestに重複を記録する。

## 入力項目と欠損

14の全機体条件に準じ、以下を時刻順・種別順・ID順に表示する。

| 区分 | 項目 |
|---|---|
| 時刻 | step、time_s、中心stepからの相対step |
| 識別・可用性 | object_id、entity_type、team、launcher_id、alive、state_valid、acmi_event、action_status |
| 状態10成分 | lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg, tas_mps, vn_mps, ve_mps, vu_mps |
| 残弾数 | missiles_remaining（元ACMIに記録された時点の値、未記録はnull） |
| 機体行動3成分 | altitude_action, heading_action, speed_actionの整数ID |

元CSVには全機体・ミサイルの全行動がそろっているわけではない。Bチームの選択行動とミサイルの制御指令は未収録であり、復元しない。ミサイルの速度4成分も未収録。未収録はnullと理由を表示する。ミサイルに機体用の3行動IDを割り当てない。

`state_valid=0`の数値は全成分nullにして、生存フラグとイベントは残す。撃墜後にCSVに残る古い座標を飛行として解釈させない。`state_valid=1`でも個別に欠損する成分はnull。記録のないミサイル行は作らず、補間・forward-fill・差分速度の追加をしない。元の数値を追加で丸めない。

初期案ではshoot_action、行動確率、health、`missile_launched`、controllerの種別説明、勝敗を含むファイル名はプロンプトに含めない。モデルには中立な試合別名を使い、実ファイル名との対応はmanifestに保存する。特に「Bは追跡ルール」という説明は意図の回答を誘導するため、14〜16のsystemプロンプトからそのまま継承しない。

射撃指令は入力から除外し、残弾数を提示する。残弾数の欠損を補完しない。ミサイルの初出は区間内で初めて見えた記録であり、窓先頭なら発射時刻とは判断できない。`remove`も命中の証拠ではない。標的ID・命中先は元データにない。

## プロンプトと回答

system/userプロンプトは英語で作成し、回答も英語で統一する。本番用の本文を以下に定義する。`{...}`はデータ生成時に置換する値であり、例示の数値や省略記号を送信しない。prompt versionは`lag-commentary-v2`。

systemは14〜16の環境、状態、行動、遷移の説明を参考にする。予測タスクの指示、controller種別による意図の誘導、16の回帰式は加えず、場面解説用の役割と根拠の扱いを定義する。userは対象窓と実データ、回答の構成を指定する。実行時の正本は`commentary_experiment/system.txt`と`user.txt`。以下に同じ本文を掲載し、変更時は両者を同期する。

### System prompt

```text
You analyze recorded engagements in Light Aircraft Game (LAG), an
aircraft combat simulation game using JSBSim. Produce evidence-based
commentary on the situation and the plausible intentions behind each
aircraft's behavior. Distinguish observed facts from interpretations.

Environment:
The recording contains a two-versus-two engagement between F-16 aircraft
equipped with missiles. Team A consists of A0100 and A0200. Team B
consists of B0100 and B0200. The task objective is to defeat the opposing
aircraft while surviving.

Control is hierarchical. High-level altitude, heading, and speed
commands are translated into aircraft control inputs by a low-level
controller. Recorded telemetry is not the complete observation available
to each aircraft's policy. Do not assume that an aircraft knew every
state shown in this recording.

State representation:
Each state vector has the following fixed field order:
[lon_deg, lat_deg, alt_m, roll_deg, pitch_deg, yaw_deg,
 tas_mps, vn_mps, ve_mps, vu_mps]

Longitude and latitude are in degrees. Altitude is in meters above mean
sea level. Roll, pitch, and yaw are in degrees. True airspeed (tas_mps)
and northward, eastward, and upward velocity (vn_mps, ve_mps, vu_mps)
are in meters per second.

Heading is measured clockwise from north: 0 degrees is north, 90 is east,
180 is south, and 270 is west. Account for angular wraparound. Aircraft
heading and direction of travel need not coincide. Longitude and latitude
are angular coordinates; their raw differences are not distances in meters.

Missile states use the same field order and units. Their speed and
velocity components are unrecorded and appear as null.

Action representation:
Supplied aircraft actions contain three integer IDs in this order:
[altitude_action, heading_action, speed_action]

altitude_action: 0 = up, 1 = hold, 2 = down.
heading_action: 0 = left30, 1 = left15, 2 = hold,
                3 = right15, 4 = right30.
speed_action: 0 = up, 1 = hold, 2 = down.

Movement commands correspond to altitude offsets of +100 / 0 / -100
meters, heading offsets of -30 / -15 / 0 / +15 / +30 degrees, and
forward-speed offsets of approximately +17 / 0 / -17 meters per second.
These are controller commands, not state changes completed within one
recorded step. The speed command concerns forward speed, whereas tas_mps
records true airspeed. Firing commands are not supplied.

missiles_remaining is the ammunition count recorded for that aircraft at
that step. A null count is unavailable, not zero. Counts are not filled
forward. A decrease may support an inference of firing between recorded
counts; it does not by itself establish an exact launch step, target, or hit.

Time and data availability:
Steps are separated by 0.2 seconds. State s_k precedes action a_k, which
acts from step k to step k+1. The final action in the supplied window is
included, but its resulting next state is outside the window.

The input contains 61 consecutive time points centered on step t,
from t-30 through t+30 inclusive. You may use the entire window to explain
the scene retrospectively. Distinguish the situation at t from earlier
and later developments, and identify later evidence when using it to
interpret behavior at t. Do not describe events outside the window.

A null value means missing or unavailable information, not zero.
state_valid = 0 means that the state must not be interpreted as active
flight; its numerical components are masked as null. state_valid = 1
does not guarantee that every component is available.

alive = 0 marks an inactive aircraft; alive = 1 marks an aircraft recorded
as alive. A null alive flag is unavailable and does not establish death.
Aircraft action_status may be recorded, not_recorded, inactive, or
state_only. Team B's selected actions are not recorded. Missile control
actions are not recorded and the aircraft action space does not apply
to missiles. Never reconstruct missing action IDs as observed facts.

Metadata identifies each object's ID, entity type, team, missile launcher,
and recorded event. An update is a recorded update; remove records
disappearance; none means no corresponding ACMI record for that row.
Missing object rows are not interpolated. A missile present at the start
of the window may have been launched earlier. Its first appearance in
this window alone does not establish its launch time. A remove event
alone does not establish a hit. Missile target IDs and hit attribution
are not provided. launcher_id identifies the launching aircraft, not
the target.

Evidence and interpretation:
Support substantive claims with object IDs, exact steps or step ranges,
and relevant recorded values or changes. Clearly distinguish quantities
you calculate from recorded fields. Explain spatial and temporal relations
when supported by the data, without assuming a target or intention from
proximity alone.

Treat intentions as hypotheses about behavior, not directly observed
internal states. Give the best-supported interpretation for each aircraft,
with a brief explanation of the evidence. Where materially different
interpretations remain possible, state them and explain the uncertainty.
Do not invent cooperation, awareness, missile targets, hits, or outcomes.
When evidence is insufficient, identify the specific missing information.

Write the commentary in English. Be specific and concise, summarize the
meaningful developments rather than narrating every row, and provide
brief evidence-based explanations.
```

### User prompt

```text
Analyze the following engagement window.

Episode: {episode_alias}
Center step t: {center_step}
Center time: {center_time_s} seconds
Window: steps {start_step} through {end_step}, inclusive
Window times: {start_time_s} through {end_time_s} seconds
Time points: 61
Elapsed time from the first to the last point: 12 seconds

The records contain all aircraft and missiles recorded within this
window, with available movement actions and recorded ammunition counts. relative_step is step minus t.
State and action field orders are defined in the system message.

BEGIN WINDOW DATA
{window_data}
END WINDOW DATA

Explain what is happening around the center step and how the engagement
evolves across the supplied window. Explain the plausible intentions
behind each aircraft's behavior using the recorded states and available
actions. Address all four aircraft, including any aircraft that is
inactive or has insufficient data for an interpretation.

Use the following sections:

1. Scene overview
Summarize the situation at the center step and the main developments
before and after it.

2. Key observed developments
Describe the important changes in chronological order. Cite object IDs,
steps or step ranges, and relevant numerical evidence. Distinguish
commanded actions from observed motion.

3. Aircraft-by-aircraft interpretation
Provide one subsection for each of A0100, A0200, B0100, and B0200.
For each, state the observed behavior, the most plausible intention,
the supporting evidence, and any material alternative interpretation
or uncertainty. Explicitly identify evidence after the center step
when using it to interpret the aircraft's behavior at the center step.

4. Missile activity and relationships
Explain the recorded missile motion, appearances, and removals, and
supported relationships to aircraft. Separate observed facts from
hypotheses about targets or effects. If no missiles are recorded in
the window, say so.

5. Limits of the interpretation
Identify the specific conclusions that cannot be established from this
window and the missing evidence that prevents a stronger conclusion.
```

`window_data`は14の`_render_block()`を参考に、各stepの時刻・相対step、全対象の状態・可用性・メタデータ、利用可能な機体行動を表示する。データは省略しない。回答は上記の固定見出しによる英文とし、JSONは要求しない。本文変更時はprompt versionを更新する。

## 評価計画

文章の流暢さだけでは能力を評価しない。回答を確認する前に、各窓について入力に含まれる根拠だけで確認可能な事実と、判断不能な項目を評価者が整理する。

| 評価軸 | 確認すること |
|---|---|
| 事実の正確さ | ID・チーム・時刻・高度変化・旋回・指令などに誤りがないか |
| 主要事実の網羅性 | 評価者が事前に列挙した、その窓の重要な変化に言及できたか |
| 時間関係 | 中心tの状況と、その前後の変化を区別しているか |
| 根拠との対応 | 引用したstep・数値が実在し、主張を支えているか |
| 意図仮説の妥当性 | 行動・状況と整合し、代替解釈を考慮しているか |
| 不確実性 | 未収録の行動、標的、命中、内部意図を断定していないか |

事実主張は「支持／矛盾／入力から確認不能」に分けて数える。重要事実の網羅率と併記し、何も具体的に述べない回答が高評価にならないようにする。意図仮説は別枠の人手評価（例: 0=根拠なし・矛盾、1=一部支持、2=根拠と限界が明確）とし、「真の意図の正解率」とは呼ばない。

可能ならモデル名を伏せて複数評価者が独立評価し、不一致を記録する。動画・13のビューアーは照合を助けるが、入力外の情報を正解の必須根拠にしない。LLMによる自動評価は補助に留める。

集計単位は窓。61時点や4機を独立な61件・4件として数えない。重なる窓は重複幅を記録し、少数の指定場面の結果を全状況へ一般化しない。

## 実施順序と成果物

1. MHI試合と指定済みの中心step 45, 85, 145, 245, 310を使用する。指定意図がある場合は人間用メモに保存し、LLM入力には入れない。
2. 13からD_tを抽出し、窓・欠損・消滅・実際の提示項目を検査する。
3. 全61時点を含むプロンプトをローカル生成し、入力長と表示を確認する。長すぎる場合は黙って間引かず、表示形式またはモデル選択を見直す。
4. 少数の窓で1モデル・各1回答のパイロットを行い、回答形式と評価票を固定する。調整に使った窓は本評価と区別する。
5. 同じ窓・同じプロンプトでモデル比較を行う。既定は3モデル、各1回。窓N個、モデルM個、反復R回ならN×M×Rリクエスト。各機体の解説は同じ回答に含む。
6. 生回答と使用量を保存し、人手評価・比較表を作成する。

実装構成:

```text
17_commentary_LAG/
  README.md
  run.py
  commentary_experiment/
    config.py       # 試合・中心step・モデル・反復
    dataset.py      # 61時点の抽出と入力マスク
    prompts.py      # 解説用system/user
    runner.py       # ローカル計画生成
    execution.py    # 生回答保存・再開
    batch.py        # Batch投入・回収・重複防止
    evaluation.py   # 人手評価用の回答エクスポート
    system.txt      # 英語system本文
    user.txt        # 英語userテンプレート
  tests/            # 境界、61時点、欠損、全個体、入力範囲の検査
  results/<run_name>/
    manifest.json   # 条件、CSV hash、窓、提示項目、prompt/model設定
    windows/        # 実際の可視データ（元記録は13を参照）
    prompts/
    api/
    evaluation/    # 事前の事実リスト、評価票、評価結果
```

通常起動はローカル生成、送信は明示的な実行オプションで行う設計を引き継ぐ。モデルIDは既存実験の設定を採用し、提供状況は送信時に確認する。

追加比較として、将来は状態のみ／状態＋行動、tまで／前後30、controller説明なし／ありを分離できる。初回はユーザー指定の全機体・全ミサイルのD_tに固定する。

## 未確定事項

- 本番送信時のモデル提供状況と実行予算（既定は15・16と同じ3モデル、各1回）。
- 評価者、主要事実リスト、評価基準の最終版。

## ローカル参照先

- [13の概要](../13_LAG/README.md)、[統合CSVの定義](../13_LAG/data/processed/README.md)、[読込API](../13_LAG/lag_data/dataset.py)
- [14の設計](../14_mental-modeling_LAG/README.md)、[プロンプト](../14_mental-modeling_LAG/lag_experiment/prompts.py)、[窓と可視データ](../14_mental-modeling_LAG/lag_experiment/runner.py)
- [15の設計](../15_trajectory_LAG/README.md)
- [16の設計](../16_dynamics_LAG/README.md)、[プロンプト](../16_dynamics_LAG/dynamics_experiment/prompts.py)

## ユーザー提供の正解資料

[正解スプレッドシート](https://docs.google.com/spreadsheets/d/1EIgeVnxUKSuLsN2aV1Vqb0Tmjwi7bY85CtlCrYRPyIE/edit?usp=sharing)を評価用の参照先とする。2026-10-17にユーザーが指定。シート本文はこの環境から取得できておらず、内容・対象stepとの対応は未検証。URLをmanifestにも保存する。正解の本文・URLはLLMのsystem/user入力には含めない。自動採点は実装せず、回答と人手評価用テンプレートを出力する。

## 実行方法

```sh
# 5窓のプロンプト、可視データ、評価票、15リクエストのBatch入力をローカル生成
.venv/bin/python experiments/17_commentary_LAG/run.py
# 本番送信（モデルの利用可否を確認して実行）
.venv/bin/python experiments/17_commentary_LAG/run.py --execute
# 結果を回収、原文とレビュー用JSONを保存
.venv/bin/python experiments/17_commentary_LAG/run.py --collect --wait
# ローカルの実行状況
.venv/bin/python experiments/17_commentary_LAG/run.py --status
# 保存済み回答からレビュー用JSONを再生成（API不要）
.venv/bin/python experiments/17_commentary_LAG/run.py --export
# オフラインテスト
.venv/bin/python -m pytest experiments/17_commentary_LAG/tests -q
```

既定モデルIDは15・16と同じ`gpt-5.6-sol / gpt-5.6-terra / gpt-5.6-luna`。利用可否は今回API照会していない。`--models`で任意の実モデルIDも指定可能。`--repeats`の既定は1。`reasoning_effort=medium`。変更した条件は`--output`で別名に保存する。Batch送信・再開・重複防止は15の実装を複製して利用する。`--adopt --job ... --batch-id ...`で応答不明の投入を回復できる。

`windows/`には入力として実際に提示するデータのみ保存し、元の全64列は13を正本として参照する。`evaluation/reference.json`に正解資料URL、`evaluation/rubric_template.json`に未記入の評価票、回収後の`evaluation/responses.json`に原文・usage・失敗状態を保存する。評価票は再実行で上書きしない。

## 入出力ビューアー

[view_rawdata.html](view_rawdata.html)は16のRaw dataビューアーを参考にした、外部通信なしで開ける単一HTML。回答全文、system/user入力、モデル・中心step・反復・状態による絞り込み、全文検索、同じ窓のモデル比較、token使用量、API応答JSON、回答のJSON保存に対応する。正解シートは参照リンクとして表示し、自動採点値は表示しない。

```sh
.venv/bin/python experiments/17_commentary_LAG/build_results_viewer.py
# 回答回収後にHTMLだけ更新
.venv/bin/python experiments/17_commentary_LAG/build_results_viewer.py --overwrite
```

生成時に計画と送信記録の一致を検証する。実験データとAPI記録は変更しない。HTMLは生成時点のスナップショットなので、追加回答の回収後には再生成する。

## 戦術的意図の再構成条件（追加実験）

ユーザー提供のsystem/user promptを用いる`lag-tactical-v1`を追加した。既存の`commentary_batch_v1`、ビューアー、抽出CSVは保持する。MHIの5窓、入力データ、3モデル、reasoning_effort=medium、各1回は同じで、プロンプトだけを変更する。

- [tactical_system.txt](commentary_experiment/tactical_system.txt): 添付「貼り付けたテキスト.txt」の原文を変更せず保存。
- [tactical_user.txt](commentary_experiment/tactical_user.txt): ユーザー提示文を採用。例示のt=85だけに限定せず、時刻・範囲・61時点のデータを各窓に合わせて置換。
- 中心時点よりも窓全体の意図、行動目的、周囲の状況、行動が変化した理由とその後の展開を説明させる。
- 回答は`A0100 / A0200 / B0100 / B0200`の4文字列フィールドのみのJSON。生回答を保存し、回収時にキー・型・重複・終了理由を検証する。形式不正を自動修正せず`evaluation/responses.json`の`intention_format`に記録。意図内容の正しさは自動採点しない。

```sh
# 新プロンプトの生成・確認（API不要）
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical
# 新条件の15件を送信
.venv/bin/python experiments/17_commentary_LAG/run.py --prompt-variant tactical --execute
# 回収／状態確認は出力名を明示
.venv/bin/python experiments/17_commentary_LAG/run.py --output tactical_batch_v1 --collect --wait
.venv/bin/python experiments/17_commentary_LAG/run.py --output tactical_batch_v1 --status
# 既存ビューアーを保持して新しいHTMLへ出力
.venv/bin/python experiments/17_commentary_LAG/build_results_viewer.py --run tactical_batch_v1 --output experiments/17_commentary_LAG/view_rawdata_tactical.html
```

新条件の保存先は`results/tactical_batch_v1/`。`--prompt-variant`未指定では従来プロンプトを選ぶ。コード更新前の既存計画は上書きせず、従来条件を再生成する場合も新しい`--output`名を指定する。保存済み旧結果の閲覧・回収は元manifestを使用できる。

### 同じHTMLで実験を切り替える

`view_rawdata.html`上部の「実験 / Batch」で`commentary batch v1`と`tactical batch v1`を切り替えられる。回答、入力全文、件数、モデル比較、JSON保存は選択した実験に対応する。今回のtacticalを初期表示とする。モデル比較は同じ実験内の同じ窓を対象とし、別実験の同名sample IDと混ぜない。

生成コマンドは既定で`results/`直下のmanifestがある全実験を読み込む。

```sh
.venv/bin/python experiments/17_commentary_LAG/build_results_viewer.py --overwrite
# 収録対象を指定する場合（最後に指定した実験を初期表示）
.venv/bin/python experiments/17_commentary_LAG/build_results_viewer.py --run commentary_batch_v1 tactical_batch_v1 --overwrite
```
