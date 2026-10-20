# Experiment 16: 回帰式を与えたLAGの10 step trajectory予測

`15_trajectory_LAG` と同じ入力・正解・モデル設定・採点方法を使い、systemプロンプトの `transition_dynamics` に姿勢角の経験的な回帰式を追加する。userプロンプトは変更しない。

## 実験条件

| 項目 | 設定 |
|---|---|
| 試合 | `mhi_v1.3_ic100601_win` |
| 対象 | A0100 / A0200 |
| 履歴 | 20 step（4秒） |
| 予測 | 10 step（2秒）の状態10成分と移動行動3成分を一括回答 |
| 履歴開始step | 0, 62, 124, 187, 249 |
| 入力範囲 | `self` / `all_aircraft_missiles` |
| モデル | `gpt-5.6-sol` / `gpt-5.6-terra` / `gpt-5.6-luna` |
| API設定 | `reasoning_effort=medium`、temperature・出力トークン上限は未指定 |
| 件数 | 20入力 × 3モデル = 60リクエスト、12条件 |

条件名・出力先は `dynamics_batch_v1`、prompt versionは `lag-dynamics-v1`。既存15の結果・コードは変更しない。採点versionは共通の `lag-trajectory10-action3-state10-v1` を維持する。

## 追加した回帰式

[解析レポート](../13_LAG/analysis/attitude_prediction_20261013/report.md) の「姿勢角履歴＋速度」モデルを採用する。

$$
\widehat{\mathbf q}_{n+1}=\mathbf q_n+\mathbf b+B^\top\mathbf x_n.
$$

`q = [roll, pitch, yaw]`。入力24成分は、直前・その前の角度増分（各3）、現在角度のsin/cos（各3）、現在の北・東・上速度（3）、直前・その前の速度差分（各3）、現在TAS（1）、直前・その前のTAS差分（各1）。単位は度・m/sで、差分を0.2で割らない。係数は元の単位に戻した値であり、再標準化しない。

- [追加プロンプトのテンプレート](dynamics/attitude_prior.txt)：定義、式、再帰予測の使い方、機首と速度の方向の違いを記載。
- [固定した全係数](dynamics/attitude_model.json)：24×3行列と切片、原資料hash、学習データの由来。
- [レンダリング処理](dynamics_experiment/dynamics.py)：固定JSONから数値を省略せず埋め込む。実行時に再学習しない。

係数は全機体で共通。姿勢角3成分それぞれに異なる係数列を使う。ミサイルには適用しない。最初の予測は実測履歴の最後3時刻から、その後はモデル自身が予測した姿勢角・速度から継続する。CLIの履歴長は最低3 step。

位置・速度・TAS・行動を直接決定する回帰式は今回追加しない。LLMは元の課題どおり状態10成分と移動行動3成分を回答する。コードでLLMの回答を回帰値に置き換えることはしない。レポート全体、実測未来値、診断用の両端速度平均、画像、回帰の評価スコアはプロンプトに含めない。

## 比較の意味と限界

**固定係数は全3試合から推定され、評価対象のMHI試合とその未来区間を含む。** 本実験は「同じ記録から得た経験的な式を追加で提供した効果」を測るものであり、未知の試合への汎化評価ではない。係数を通じた評価データとの重複は、未来の実測行をプロンプトへ渡していないこととは区別する。

回帰は1 step先の姿勢角向けで、行動・舵角を明示的な入力としていない。10 stepの再帰適用精度や厳密な力学を保証するものではない。

## 実行

リポジトリルートから実行する。通常起動と `--dry-run` はローカル保存のみで、API送信しない。

```bash
# 20入力と60リクエストを準備。モデルごとに20件のBatch用JSONLを作る。
.venv/bin/python experiments/16_dynamics_LAG/run.py --dry-run

# オフラインテスト（APIクライアントはモック）。
.venv/bin/python -m pytest experiments/16_dynamics_LAG/tests -q

# 保存済みのローカル状態のみ確認。
.venv/bin/python experiments/16_dynamics_LAG/run.py --status
```

API実行が必要になったときだけ `--execute` を使う。`--collect --wait` は投入済みBatchの回収、`--rescore` は保存済み回答の再採点。15と同じ再開・重複投入防止を引き継ぐ。設定やプロンプトを変更した場合は新しい `--output` を使い、既存計画を上書きしない。

## 保存先と可視化

`results/dynamics_batch_v1/` に以下を保存する。

```text
manifest.json                 全問題・正解・送信内容・データ/コード/係数/template hash
prompts/*.txt                 system/userの全文（20ファイル）
api/batch/*.input.jsonl        送信前のBatch入力（3ファイル、各20件）
api/batch/state.json          Batch準備・送信・回収状態
baseline_comparison.json      15との問題・正解・設定・プロンプト差分の確認結果
```

送信後はAPI生回答・採点結果を15と同じ形式で保存する。可視化スクリプトは16の結果を既定の入力とする。

```bash
.venv/bin/python experiments/16_dynamics_LAG/build_results_viewer.py
.venv/bin/python experiments/16_dynamics_LAG/plot_results.py
.venv/bin/python experiments/16_dynamics_LAG/plot_trajectory_timelines.py
```

15との対応付けには `problem_id`・`observation_scope`・`model_alias` を使う。プロンプトが違うため `sample_id`・`query_id` は異なる。状態・行動・予測距離別の誤差、形式成功率、トークン量を同じ問題で比較する。まだAPI回答がないdry-run段階では成績比較は作らない。

## 検証と由来

継承した窓選択・未来入力の遮断・厳格な回答解析・Batch再開・可視化集計のテストに加えて、次を確認する。

- 保存済み15のmanifestと20問題の入力履歴・正解・時刻・設定が同一。
- userプロンプトは完全一致し、systemプロンプトから追加ブロックを除くと15と完全一致。
- 固定係数と元の `results.json` の値が一致し、プロンプトの行列から同じ数値計算を再現できる。
- 係数とテンプレートがmanifestのhash対象に含まれる。

[baseline_provenance.json](baseline_provenance.json) にコピー元コードと比較元manifestのSHA-256を保存する。追加式の出典は固定係数JSON内に記録する。禁止指定された原文メモは参照しない。
