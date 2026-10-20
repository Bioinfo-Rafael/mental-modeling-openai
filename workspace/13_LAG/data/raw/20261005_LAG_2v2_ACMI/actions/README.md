# モデルの出力と環境が受け取るaction

2026-10-09追記。以下は上流リポジトリ `liuqh16/LAG` のcommit `ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08` を確認した説明。対象は **2v2・階層制御・ShootMissile**。今回のCSVの列構成・説明と整合するが、CSV生成に使われた実行コードと上流の差分までは未確認。

## 1機・1時刻のactionは4個の離散整数

環境の `HierarchicalMultipleCombatShootTask.load_action_space()` は次のように定義されている。

```python
self.action_space = spaces.MultiDiscrete([3, 5, 3, 2])

# 環境に渡す選択済みの行動
# action = [高度の選択, 方位の選択, 速度の選択, 発射の選択]
```

CSVの `*_action` は、この整数を読める名前に置き換えたものに対応する。

| 要素 | CSV列 | 整数とラベルの対応 | 下位方策へ与える指令の意味 |
|---|---|---|---|
| `action[0]` | `altitude_action` | 0: up、1: hold、2: down | 高度差 +100 / 0 / −100 m相当 |
| `action[1]` | `heading_action` | 0: left30、1: left15、2: hold、3: right15、4: right30 | 方位差 −30 / −15 / 0 / +15 / +30度 |
| `action[2]` | `speed_action` | 0: up、1: hold、2: down | 機体前後方向速度の差 +17 / 0 / −17 m/s相当 |
| `action[3]` | `shoot_action` | 0: no、1: yes | 発射要求 |

高度と速度の物理量換算は、上位指令の正規化値と下位方策の入力スケールを照合したもの。高度は `[0.1, 0, -0.1] × 1000 m`、速度は `[0.05, 0, -0.05] × 340 m/s` に相当する。

これらは下位方策に与える目標との差であり、「0.2秒で100m上昇する」「1ステップで17m/s加速する」という状態変化の保証ではない。環境は各ステップでこの差分指令を直接下位方策へ渡すため、固定の絶対目標を保持したり、前回の目標へ差分を累積したりする処理とも異なる。

## logits・確率・選択済みactionの関係

上流の2v2実装では、4つのCategorical出力ヘッドがそれぞれ3・5・3・2個のlogitsを出す。それぞれのグループ内で確率に変換し、決定論のrolloutでは各グループの最大確率の選択肢を選ぶ。非決定論の場合は各分布からサンプリングする。

```text
各ヘッドのlogits：3個・5個・3個・2個
        ↓ 各グループ内で確率化
CSVの p_*：合計13列（グループごとに確率の合計が1）
        ↓ 決定論の場合、各グループでargmax
選択済みactionの例：[2, 3, 0, 1]
        ↓ 名前に対応付け
CSV：down, right15, up, yes
```

**13択から1個選ぶのではなく、4種類それぞれから1個ずつ選ぶ。** モデルがCSVの文字列を直接生成しているわけではない。

バッチサイズを `B` とすると、各ヘッドのlogitsは `(B, 3)`、`(B, 5)`、`(B, 3)`、`(B, 2)`。確率を連結すると `(B, 13)`、選択後のaction tensorは `(B, 4)` となる。MAPPOの `PPOActor.forward()` は、選択した `actions` に加えて `action_log_probs` と更新後の `rnn_states` も返す。

これは2v2の `MultiDiscrete([3, 5, 3, 2])` の分岐についての説明。1v1 ShootMissileで使われるTuple型・Beta/Bernoulliの分岐と混同しない。

## 選択したactionから実際の操縦・発射まで

高度・方位・速度の指令3個は、自機情報9個と合わせた12次元の入力として、学習済みの固定下位方策 `baseline_model.pt` に渡される。下位方策がエルロン・エレベータ・ラダー・スロットルの離散値を選び、環境が舵3系統を `[-1, 1]`、スロットルを `[0.4, 0.9]` の操作値に変換する。

射撃は別処理で、`action[3] > 0` が発射要求となる。実際の発射には、生存・残弾・攻撃角・距離・発射間隔の条件を満たす必要があり、対象は最も近い敵機。したがって `shoot_action=yes` の行数は発射回数ではなく、実際の発射はCSVの `missile_launched` で確認する。

## 確認したコード

- [2v2の行動空間・指令値・下位方策への変換・発射処理（multiplecombat_task.py）](https://github.com/liuqh16/LAG/blob/ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08/envs/JSBSim/tasks/multiplecombat_task.py#L129-L274)
- [下位方策の入力スケール（heading_task.py）](https://github.com/liuqh16/LAG/blob/ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08/envs/JSBSim/tasks/heading_task.py#L85-L99)
- [出力ヘッドの構成と行動選択（act.py）](https://github.com/liuqh16/LAG/blob/ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08/algorithms/utils/act.py#L33-L82)
- [Categoricalのlogitsと最大確率の選択（distributions.py）](https://github.com/liuqh16/LAG/blob/ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08/algorithms/utils/distributions.py#L13-L72)
- [MAPPO Actorの戻り値（ppo_actor.py）](https://github.com/liuqh16/LAG/blob/ebd3a38e64d49c38c3e4ee04cdc1a1541be0da08/algorithms/mappo/ppo_actor.py#L35-L47)

---

以下はコピー時のREADME本文。`data/raw/manifest.json` の本READMEのハッシュ・サイズは、上記追記前のコピー時点のもの。

# LAG 2v2 行動の時系列（ACMI 3 試合に対応）

`20261005_LAG_2v2_ACMI` の ACMI 3 本と同じ試合の、学習したモデルの行動（方策の出力）と状態の時系列。試合を決定論で流し直して書き出した。流し直した試合の ACMI は、渡した ACMI とファイルの中身が完全に一致することを確かめてある。

| ファイル | 試合 |
|---|---|
| `ours_seed100100_ic100606_win_actions.csv` | こちらで学習したモデル（seed 100100）、初期配置 100606、勝ち |
| `ours_seed100100_ic100601_loss_actions.csv` | こちらで学習したモデル（seed 100100）、初期配置 100601、負け |
| `mhi_v1.3_ic100601_win_actions.csv` | MHI 先方モデル v1.3、初期配置 100601、勝ち |

## 形式

- 0.2 秒ごと・機体ごとに 1 行（4 機）。`step` が ACMI の時刻 `#(step×0.2)` の状態に対応する。
- 各行は「時刻 t の状態」と「その状態を見て選んだ時刻 t の行動」。行動は t から t+0.2 秒の間に効く。
- 赤（A0100・A0200）が学習したモデル、青（B0100・B0200）が固定ルール（追跡）。行動の列は赤の生きている機体だけに入る（青と撃墜後の機体は空欄）。

## 列

| 列 | 意味 |
|---|---|
| `step`・`time_s` | ステップ番号と時刻（秒） |
| `uid`・`color`・`controller` | 機体 ID、陣営、`learned`（学習したモデル）か `rule_pursue`（固定ルール） |
| `alive` | 生きていれば 1 |
| `lon_deg`・`lat_deg`・`alt_m`・`roll_deg`・`pitch_deg`・`yaw_deg` | ACMI の 6 次元と同じ（経度・緯度・高度 m・ロール・ピッチ・ヨー 度） |
| `tas_mps`・`vn_mps`・`ve_mps`・`vu_mps` | 真対気速度と速度の北・東・上成分（m/s） |
| `altitude_action` | 高度の目標の変化：`up` / `hold` / `down` |
| `heading_action` | 機首方位の目標の変化：`left30` / `left15` / `hold` / `right15` / `right30`（度） |
| `speed_action` | 速度の目標の変化：`up` / `hold` / `down`（約 ±17 m/s） |
| `shoot_action` | 発射の意思：`yes` / `no` |
| `p_<行動>_<選択肢>` | その時刻の方策の確率（各行動で合計 1）。選んだ行動は確率が最大のもの（決定論） |
| `missile_launched` | この時刻の行動で実際に発射されたミサイルの ID（青も含む） |

## 注意

- 学習したモデルが出すのは、速度・加速度そのものではなく「目標の変化」の離散選択。目標に沿って舵（エルロン・エレベータ・ラダー・スロットル）を出すのは、学習しない固定の低層コントローラ。
- `shoot_action = yes` は方策が撃とうとしたことを表す。実際の発射は、環境の条件（角度 45° 以内・距離 14 km 以内・残弾あり）を満たしたときだけ起こり、`missile_launched` に出る。
- 確率は方策の出力層の値から計算したもので、選んだ行動は各時刻で確率最大の選択肢と一致することを確かめてある。
