# LAG 2v2 統合CSV：設計と実験での利用

作成日：2026-10-12 JST。schema version：`1.0`。

1試合につき1つのCSVに、A・B両チームの4機体、両チームの全ミサイル、撃墜・爆発の表示オブジェクト、元データにある状態・行動・行動確率・発射結果・消滅記録を保存する。
履歴長、提示時点、対象機体、特徴量を固定したデータセットではない。全記録を保持したこのCSVから、実験条件に従って入力と正解を切り出す。

**Bチームの選択行動とミサイルの制御指令は元資料にない。保存対象の「全情報」は、受領したACMI・actions CSVに記録されている情報を指す。** 機体の方策入力観測ベクトル、RNN内部状態、rewardも未収録であり、生成していない。

## 1. ファイルと実測件数

| ファイル | 全行数 | 機体行 | ミサイル行 | 表示行 | 元CSVの最後 | ACMIの最後 |
|---|---:|---:|---:|---:|---:|---:|
| `mhi_v1.3_ic100601_win.csv` | 2,124 | 1,366 | 750 | 8 | step 340 / 68.0秒 | step 341 / 68.2秒 |
| `ours_seed100100_ic100601_loss.csv` | 1,893 | 1,178 | 707 | 8 | step 293 / 58.6秒 | step 294 / 58.8秒 |
| `ours_seed100100_ic100606_win.csv` | 3,016 | 1,986 | 1,022 | 8 | step 495 / 99.0秒 | step 496 / 99.2秒 |

行数はヘッダーを除く。ミサイル行には消滅行を含む。ミサイルの個体数は順に6、5、6発で、どの試合にも機体4機がある。
各試合の機体撃墜表示とミサイル消滅時の表示は、合わせて8オブジェクトある。
試合ごとの削除レコード数も8件だが、これは表示行数とは別の集計である。

- [manifest.json](manifest.json)：64列の順序と読み込み型、行動ID対応表、元ファイルと出力のSHA-256、件数、発射対応、ACMIヘッダー、変換コードのハッシュ、検証結果。
- [lag_data/prepare.py](../../lag_data/prepare.py)：元データを結合・検証して再生成する。Python標準ライブラリのみ。
- [lag_data/dataset.py](../../lag_data/dataset.py)：型を指定した読み込み、履歴時点の選択、機体・特徴量の抽出。pandasを使用。
- [tests/test_dataset.py](../../tests/test_dataset.py)：元データ保持、ミサイル・終端状態、欠損、抽出、再現性のテスト。

原本は[../raw/20261005_LAG_2v2_ACMI/](../raw/20261005_LAG_2v2_ACMI/)に残す。元ACMI・CSV・動画・受領READMEは変更しない。
この文書の時間と件数は実ファイルから計算した値であり、受領メモに書かれた概略時刻を採用したものではない。

## 2. 行の単位・キー・時間

### 2.1 1行＝1時刻×1オブジェクト

主キーは`(episode_id, step, object_id)`。
同じ機体IDは別試合でも使われるので、複数試合を結合した後に`step, object_id`だけをキーにしない。

| entity_type | 対象 | 例 |
|---|---|---|
| `aircraft` | A・Bチームの機体 | `A0100`, `A0200`, `B0100`, `B0200` |
| `missile` | 発射されたミサイル | `A01002`, `B02002` |
| `effect` | 撃墜・爆発などの表示 | `B0200X`, `A01002F` |

CSVとACMIのキーの**和集合**を保存する。CSVにしかない撃墜後の機体行も、ACMIにしかないミサイル・表示・消滅・最終状態も残る。
元CSVの全stepには4機分の行がある。ACMIだけの最終stepは、実際に更新・消滅が記録されたオブジェクトのみで、4機分を人工的に作らない。
ミサイルや表示は、記録されていない時刻の行を作らない。したがって各stepの行数は一定ではない。

並び順は`step`昇順、`aircraft → missile → effect`、`object_id`昇順。
この並び順は表示・抽出用であり、元ACMIの同時刻内の記載順は`acmi_line_number`から復元する。

### 2.2 時刻の意味

`time_s = step × 0.2`。`step`は0始まりで、整数のstepとIDで結合する。
ACMI時刻のstep化にはDecimalを使い、0.2秒刻みから外れた記録はエラーにする。

元actions CSVの同じ行は、時刻tの状態`s_t`と、その状態を見て選んだ行動`a_t`を表す。

```text
s_t -- a_tを適用 --> s_(t+1)
     [t, t+0.2秒)
```

`missile_launched`はその行動による実際の発射結果である。現在の行動を予測する前に既知の情報ではない。
今回の全17発では、CSVの発射行がstep t、ACMIのミサイル初出がstep t+1だった。両者を時刻補正せず、IDで結び付ける。
各対応はmanifestの`episodes[].launches`に保存する。

ACMIの`ReferenceTime=2020-04-01T00:00:00Z`は元ヘッダー値としてmanifestに保存する。実際の試合実施日と解釈しない。

## 3. 列定義（全64列）

CSVはUTF-8、カンマ区切り、1行のヘッダー、LF改行。空欄は欠損／非該当で、数値0や行動`hold`・`no`とは異なる。
`Int64`、`Float64`はpandasで使う欠損可能な読み込み型であり、CSVファイル内には型情報がない。型定義の正本はmanifestの`columns`。

### 3.1 識別・所属（11列）

| 列 | 型 | 意味・出典 |
|---|---|---|
| `episode_id` | string | 元ACMIの拡張子なしファイル名。勝敗文字列を含むので通常のLLM入力には使わない |
| `step` | Int64 | 0始まりの時刻番号 |
| `time_s` | Float64 | 試合開始からの秒数。CSV行があればCSV、なければACMIの時刻 |
| `object_id` | string | 元CSVの`uid`またはACMIのID |
| `entity_type` | string | `aircraft / missile / effect` |
| `team` | string | `A / B`。元ColorのRed→A、Blue→B |
| `color` | string | 元の`Red / Blue` |
| `name` | string | ACMIのName。`F16`, `AIM-9L`, `SHOTDOWN`など。Nameのない表示は空欄 |
| `acmi_type` | string | ACMIのType。表示の`Misc+Explosion`など。元にない場合は空欄 |
| `launcher_id` | string | ミサイル行のみ、CSVの`missile_launched`と照合した発射元機体ID |
| `controller` | string | 機体の`learned / rule_pursue`。CSVから機体ごとの固定属性として取得 |

`name / color / acmi_type / controller / entity_type`はオブジェクトの固定属性として消滅行などにも付ける。
これは時変の状態補完ではない。受領データでName・Color・Typeやcontrollerが途中で変わった場合は生成を停止する。
`launcher_id`はID文字列の接頭辞だけから推定せず、発射CSVで照合する。ミサイルの標的IDや命中先は保存元にないため、列を推定生成しない。

### 3.2 状態（13列）

| 列 | 型・単位 | 意味 |
|---|---|---|
| `alive` | Int64、0/1 | **元CSVの生存フラグのみ**。ACMIだけの行・ミサイル・表示は空欄 |
| `lon_deg`, `lat_deg` | Float64、度 | 経度・緯度 |
| `alt_m` | Float64、m | 高度 |
| `roll_deg`, `pitch_deg`, `yaw_deg` | Float64、度 | 姿勢角。保存時には角度正規化しない |
| `tas_mps` | Float64、m/s | 真対気速度 |
| `vn_mps`, `ve_mps`, `vu_mps` | Float64、m/s | 速度の北・東・上方向成分。CSVにだけ存在 |
| `health` | Float64、元値 | ACMIのHealth。CSVのaliveとは別情報 |
| `missiles_remaining` | Int64、発 | ACMIのMissiles（残弾数） |

採用規則：

1. CSV行がある場合、位置・姿勢・速度・aliveをCSVからそのままコピーする。CSVの欠損をACMIで勝手に埋めない。
2. CSV行がなく、ACMIにTがある場合、Tの6成分を位置・姿勢として使う。TASがあれば速度に使う。
3. Health・Missilesは、その時刻のACMIレコードに存在する場合だけ保存する。
4. ミサイルACMIには今回TASや速度成分がない。位置差分などから作らず空欄とする。
5. 機体の終端状態では位置・姿勢・TAS・Health・Missilesがある場合も、CSVのaliveと速度3成分はない。`alive == 1`だけで絞るとこの有効な終端状態を落とすため、`state_valid`も参照する。

### 3.3 行動ラベルとID（8列）

| ラベル列 | ID列 | ID→ラベル |
|---|---|---|
| `altitude_action` | `altitude_action_id` | 0→up、1→hold、2→down |
| `heading_action` | `heading_action_id` | 0→left30、1→left15、2→hold、3→right15、4→right30 |
| `speed_action` | `speed_action_id` | 0→up、1→hold、2→down |
| `shoot_action` | `shoot_action_id` | 0→no、1→yes |

ラベルは元CSVの文字列、IDはこの対応から作るnullable整数。**行動は4種類それぞれの選択であり、13択の単一分類ではない。**
行動の意味・上位指令と下位制御の関係は[元actions README](../raw/20261005_LAG_2v2_ACMI/actions/README.md)を参照。
高度upや速度upは指令であり、次の0.2秒で必ず高度・速度が増加するというラベルではない。

### 3.4 行動確率（13列、すべてFloat64）

| 出力ヘッド | 列（ID順） |
|---|---|
| 高度 | `p_altitude_up`, `p_altitude_hold`, `p_altitude_down` |
| 方位 | `p_heading_left30`, `p_heading_left15`, `p_heading_hold`, `p_heading_right15`, `p_heading_right30` |
| 速度 | `p_speed_up`, `p_speed_hold`, `p_speed_down` |
| 射撃 | `p_shoot_no`, `p_shoot_yes` |

各グループの和が約1、選択行動がそのグループのargmaxであることを確認する。
空欄のグループを0や一様分布で埋めない。

### 3.5 発射・ACMIイベント（3列）

| 列 | 型 | 意味 |
|---|---|---|
| `missile_launched` | string | CSVに記録された、その行の行動で発射されたミサイルID。Bチームの記録もある |
| `acmi_event` | string | `update`＝ACMI更新、`remove`＝`-ID`、`none`＝その行に対応するACMI記録なし |
| `radius_m` | Float64、m | 表示オブジェクトのRadius |

ミサイルが消滅したことだけから命中を推定しない。`effect`の位置を、元機体・ミサイルの未収録の次状態として代入しない。
今回の`effect`は発生時の記録であり、以後全stepに持続するオブジェクトとして展開しない。

### 3.6 有効性・出典（7列）

| 列 | 型 | 意味 |
|---|---|---|
| `has_csv_record` | Int64、0/1 | 元actions CSVの行が対応しているか |
| `has_acmi_record` | Int64、0/1 | 元ACMIの更新または削除レコードが対応しているか |
| `state_valid` | Int64、0/1 | 位置・姿勢6成分がすべてあり、CSVのaliveが0ではなく、消滅行でもない場合に1 |
| `state_source` | string | 位置・姿勢などの採用元。`csv / acmi / none` |
| `action_status` | string | 下表の行動可用性 |
| `csv_row_number` | Int64 | 元CSVの物理行番号。ヘッダーを1として最初のデータは2 |
| `acmi_line_number` | Int64 | 元ACMIの1始まりの物理行番号 |

`state_valid=1`は全特徴量がそろっていることを保証しない。ミサイルの速度や終端の速度成分は、個別に欠損を確認する。
表示オブジェクトも位置・姿勢があれば1になるので、機体の抽出では`entity_type`も指定する。

| action_status | 意味 |
|---|---|
| `recorded` | 生存中の学習モデル機体で4種類の行動が記録されている |
| `not_recorded` | CSV行はあるが行動は未収録。今回の生存中のBチーム |
| `inactive` | CSVのaliveが0、または機体のACMI消滅行 |
| `state_only` | CSV行がなく、消滅行ではない機体状態。今回のACMI最終状態 |
| `not_applicable` | ミサイル・表示など、機体の4行動の対象外 |

撃墜後も元CSVには最後の位置・速度が残る。**保存CSVではその値を保持し、`state_valid=0`とする。**
読み込み・抽出の共通処理は値を変更しない。実験側で無効状態を除外・マスクする。
Experiment 14では、対象機体の無効状態を含む履歴窓・正解を除外する。

### 3.7 ACMIの元値（9列）

| 列 | 型 | 意味 |
|---|---|---|
| `acmi_time_s` | Float64、秒 | 元ACMIレコードの時刻 |
| `acmi_lon_deg`, `acmi_lat_deg`, `acmi_alt_m` | Float64 | ACMIのTの位置3成分 |
| `acmi_roll_deg`, `acmi_pitch_deg`, `acmi_yaw_deg` | Float64 | ACMIのTの姿勢3成分 |
| `acmi_tas_mps` | Float64、m/s | ACMIのTAS。CSVより丸められている |
| `acmi_properties_json` | string | そのACMI行の全プロパティを、元の文字列値のJSONとして保存 |

`acmi_properties_json`にはTそのもの、Name、Color、Health、Missiles、Type、Radiusなど、そのレコードの全項目を保存する。
削除行では`{}`、ACMI記録のない行では空欄。JSON中のカンマ・引用符は標準CSVの引用規則でエスケープするので、単純な`split(',')`ではなくCSVパーサーを使う。
ヘッダー・グローバル属性はmanifestの`acmi_metadata`、元レコードの記載順は行番号で保持する。

## 4. 精度と補完の方針

- 変換時は、元CSVのセルとACMIの数値文字列をそのまま書く。小数桁を丸めない。
- 元CSVの全列を保存する。`uid`だけ`object_id`へ改名し、その他は同名で保持する。
- 位置・姿勢のCSV/ACMI照合は絶対誤差`1e-8`以下、TASはACMIの0.1 m/s刻みの丸めを考慮し`0.050000001`以下を要求する。
- 補間、時変状態のforward-fill、角度正規化、速度差分推定、行動の穴埋め、rewardの生成をしない。
- Pythonで読み込んだ後はFloat64の浮動小数値になる。元の数値表記を確認したい場合は`csv.DictReader`または全列stringで読み込む。
- プロンプトへの数値整形は実験側で行う。Experiment 14はPythonのfloatを標準JSONで表現し、追加の桁丸めをしない。保存CSVの数値文字列は変更しない。

## 5. 再生成と検証

以下のコマンドはリポジトリルート`mental-modeling-openai`から実行する。

```sh
# 生成。同じ内容の既存ファイルは書き換えない。
.venv/bin/python experiments/13_LAG/run.py

# 書き込みなし。元データから再計算し、3つのCSVとmanifestをバイト単位で照合。
.venv/bin/python experiments/13_LAG/run.py --check

# 入力または変換コードを変更した場合だけ、派生CSVとmanifestを更新。
.venv/bin/python experiments/13_LAG/run.py --overwrite

# ソース保持、時刻境界、欠損、抽出のテスト。
.venv/bin/python -m pytest experiments/13_LAG/tests -q
```

生成器は全試合の変換と検証をメモリ内で完了してから保存する。変更済みの出力があり`--overwrite`がなければ、書き込み前に停止する。
原本とREADMEは上書き対象にしない。実行日時をmanifestに埋め込まないので、同じ入力・コードでは同じバイト列を再生成できる。
manifestの`generator`は`lag_data/prepare.py`と`lag_data/definitions.py`の相対パス→SHA-256の辞書。変換コードや定義の編集後はCSVが同一でもmanifestの再生成が必要になる。

生成時には次を確認する。

1. CSVのstepが連続し、各stepに既知の4機が重複なく存在する。
2. 時刻が0.2秒格子に一致し、ACMIとCSVの生存機体状態が整合する。
3. 元CSVの全セルが文字列として一致し、ACMIの全更新・削除レコードと全プロパティが保存される。
4. 4行動の記録が完全で、確率の範囲・和・argmaxが正しい。
5. CSVの発射ID集合とACMIのミサイルID集合が一致し、発射元・陣営が整合する。
6. 最後のACMI時刻が最後のCSV時刻の1step先である。
7. 数値列に不正な数値やNaN/Infinityの文字列がない。

同一step・同一IDに複数ACMIレコードが出現したり、未知の物体種別・CSV列構成になった場合は、既存の行で上書きせずエラーにする。今回の3試合向けのスキーマであり、新しい資料ではスキーマの見直しが必要な場合がある。

## 6. Pythonでの読み込みと抽出

### 6.1 基本の読み込み

`13_LAG`は数字から始まるため、import文の代わりにimportlibを使う。

```python
from importlib import import_module
from pathlib import Path

lag = import_module("experiments.13_LAG.lag_data.dataset")
path = Path("experiments/13_LAG/data/processed/ours_seed100100_ic100606_win.csv")
df = lag.load_episode(path)

print(df.shape)  # (3016, 64)
print(df[["step", "object_id", "entity_type", "action_status"]].head())
```

隣接するmanifestを読み、CSVハッシュと列構成を検証する。IDは文字列、空欄はpd.NAとして扱う。`verify_hash=False`は明示的に編集したコピーの確認などに限定する。

### 6.2 機体・ミサイル・履歴の選択

```python
# 全4機。撃墜後の行も含む保存データ。
aircraft = df[df["entity_type"].eq("aircraft")]

# 有効な状態が記録された自機、味方2機、全4機。
active_aircraft = aircraft[aircraft["state_valid"].eq(1)]
self_states = active_aircraft[active_aircraft["object_id"].eq("A0100")]
team_a_states = active_aircraft[active_aircraft["team"].eq("A")]

# 両チームの全ミサイル。消滅行も残す。
missiles = df[df["entity_type"].eq("missile")]
self_missiles = missiles[missiles["launcher_id"].eq("A0100")]
removals = df[df["acmi_event"].eq("remove")]
effects = df[df["entity_type"].eq("effect")]

# 20点、0.2秒刻み。基準step100に対し80〜99。
steps = lag.history_steps(100, length=20)
history = df[df["step"].isin(steps)]

# 20点、1秒刻み。基準step100に対し0,5,...,95。
sparse_steps = lag.history_steps(100, length=20, stride=5)

# 任意の相対時点。
custom_steps = lag.history_steps(100, offsets=[-20, -10, -5, -1])
```

「全機体」と「全機体＋ミサイル」は別条件にする。前者は`entity_type == aircraft`、後者は`entity_type in [aircraft, missile]`。
`effect`は元記録として保存するが、通常の予測入力からは除外する。

ミサイルの軌跡を含める場合、その時点までの情報でオブジェクトを選ぶ。将来命中したミサイルだけを選んで過去の入力へ入れると、選択自体が未来情報になる。

### 6.3 再利用する抽出API

```python
# 時刻・機体・列を指定。存在するレコードだけが、時刻順・指定機体順に返る。
selected = lag.select_records(
    df,
    steps=lag.history_steps(100, length=5),
    object_ids=("A0100", "A0200"),
    columns=(*lag.STATE_FEATURES, *lag.ACTIONS, "state_valid", "action_status"),
)
assert len(selected) == 10  # この試合・時刻では5時刻×2機

# 多数の窓を作るときは、1試合を一度だけキーで索引化する。
records = lag.index_records(df)
row = records[(99, "A0100")]
print(row["lon_deg"], row["shoot_action_id"])
```

`STATE_FEATURES`は位置・姿勢・速度の10成分。`ACTIONS`は射撃を含む4行動のラベル列名で、整数ID列は各列名に`_id`を付ける。
`select_records()`は欠損行を生成せず、死亡状態をマスクせず、正解時刻を選ばない。要求したキーの有無と特徴量の完全性は実験側で確認する。
`index_records()`も値を改変せず、`(step, object_id)`→レコード辞書を返す。試合は混ぜず、試合ごとに読み込む。

### 6.4 予測問題の構築はExperiment 14へ

[Experiment 14のREADME](../../../14_mental-modeling_LAG/README.md)に4課題の時刻関係、H、n、プロンプトの編集、実行・採点を記載した。
旧`dataset.make_query()`は削除した。再利用するデータ操作を13に、どの情報を見せて何を答えさせるかを14に分離するためである。

`history_steps(q, length=H)`はqより厳密に前のH点を返す汎用関数である。
Experiment 14のnext-stateはLLM-Xavier同様、現在の`(s_q, a_q)`を含むH点を使うため、汎用関数のqと予測課題のqを混同しない。
14の`runner.build_query()`では、開始stepと実際の履歴行数Hから各課題の時刻を明示的に決める。

## 7. 今後、条件を増やすときの注意

- 履歴の行数H、履歴内のstride、評価窓の開始時刻の選び方は別の条件。strideが5でも元の行動の適用期間は0.2秒であり、間引き後の隣接行を1step遷移と見なさない。
- Hだけの効果を比較したければ、各Hで共通の正解時刻を選ぶ。14の初期実装は指定に従って各条件の有効範囲から10点を等間隔に選ぶため、H間で正解時刻が変わる。
- 全4機、Aチーム2機、自機のみ、ミサイルを含むか等は実験側の入力選択で変える。保存CSVは条件ごとに複製しない。
- 生存しているBチームの行動、ミサイルの制御指令、rewardは未収録。欠損を0やholdで埋めない。
- 今回の3試合はMHIモデル1試合とoursモデル2試合。同一方策の別episodeと、別方策のepisodeを区別する。few-shotを加える場合は評価試合との重複を記録する。
- 次状態10成分を予測する場合、ACMIだけの最終状態は`vn_mps / ve_mps / vu_mps`が欠損するため使用できない。
- 位置・高度・速度は単位を区別して誤差を計算する。経緯度の度差をmと解釈せず、循環角の誤差規則も明記する。

## 8. 実験結果側へ残す情報

| 項目 | 内容 |
|---|---|
| 入力の識別 | CSVのSHA-256、schema version、対象episode |
| 問題の識別 | task、対象機体、基準step、予測対象step・列 |
| 履歴 | 実際に使ったstep配列、H、strideまたはoffsets |
| 可視範囲 | 提示する機体・ミサイル・状態列・行動列、欠損の扱い |
| context | few-shot ID・出典・順序、参考episode、方策の同異 |
| 実行 | モデル・生成設定、送信したプロンプト全文とハッシュ |
| 評価 | 正解、回答、評価方法、除外理由、計画件数と成功件数 |

14ではmanifestに正解を保存するが、APIへ渡すのは許可した項目から構築した`messages`だけ。
CSVの全列やmanifest全体をプロンプトにしない。行動確率、発射結果、勝敗情報などは予測入力から除く。
last-action/last-stateで意図的に見せる後続状態は課題定義に明記し、next課題とは別に集計する。
