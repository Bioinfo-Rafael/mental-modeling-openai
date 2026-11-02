# Experiment 20: 固定30遷移の目的・行動予測

17の保存済み `tactical_short_batch_v2` をベースに、短い履歴から「何を目的にどう動くか」だけを出力する。
説明対象は常に **[T,T+30]**。31時点・30遷移・6秒であり、入力末尾から30step先へずらさない。

| 条件 | 入力 | 用途 |
|---|---|---|
| n=-1 | [T-30,T-1]、30時点 | 対象区間を観測する前の予測 |
| n=1 | [T-30,T+1]、32時点 | 最初の1遷移を観測した後の予測 |
| n=30 | [T-30,T+30]、61時点 | 後続情報込みの参照回答 |

3条件は同じsystem/userテンプレートを使う。4機のJSON、1機1文、各55英単語以内。
履歴・現状の要約、根拠説明、代替シナリオは出力しない。推測の短い留保は残す。
61時点条件は同じ対象区間の説明であり、未来予測の成功や正解そのものとは扱わない。
旧17の回答は窓全体を説明しているため、今回の参照回答として流用しない。

## 注釈と条件数

`data/annotations_v1.json` は18の保存済み注釈のコピー。元出典はそのファイルに記録する。
Tは機体別注釈の46/49/83/95/143/243/311。**厳密な開始時点か代表時点かは未確認**。
Tはモデルに対象区間の境界として提示するが、注釈・変化点とは説明しない。
注釈ラベル、n、注釈対象機体は評価側のみに保持する。

完全な61時点が取れる注釈だけを全条件で共通採用する。切り詰め・中心移動はしない。
現在はT=311が341stepを必要とし、完全な行動データ範囲0〜340を超えるため3条件とも除外する。
除外はrunの `excluded_events.json` とHTMLに記録する。

**6注釈 × 3入力条件 × 3モデル = 54リクエスト。反復なし。**
モデルと `reasoning_effort=medium` はv2の保存manifestから引き継いだ。
同じ試合の重なる窓を独立試合として扱わない。n=-1→1はTとT+1の2時点を追加する比較。

## 操作

リポジトリルートから実行する。

```sh
# オフライン。入力全文、Batch送信用JSONL、比較HTMLを作る。
.venv/bin/python experiments/20_foresight_LAG/run.py plan --run foresight_short_v1

# ここから実API送信。計画を編集せず、保存された内容を実行する。
.venv/bin/python experiments/20_foresight_LAG/run.py execute --run foresight_short_v1

# 送信済み分の回収。APIを使用するが新規Batchは送信しない。
.venv/bin/python experiments/20_foresight_LAG/run.py collect --run foresight_short_v1

# 未送信の後続シャードを送信。既存Batchは再送しない。
.venv/bin/python experiments/20_foresight_LAG/run.py execute --run foresight_short_v1

# ローカルのみで状態・回答・HTMLを再出力。
.venv/bin/python experiments/20_foresight_LAG/run.py status --run foresight_short_v1
.venv/bin/python experiments/20_foresight_LAG/build_results_viewer.py --run foresight_short_v1
```

`results/foresight_short_v1/report/review.html` をブラウザで開く。
注釈・モデル・表示機体を選び、3条件を横並びにする。未送信の段階でも入力全文とプロンプト差分を確認できる。
サーバーや外部通信は不要。HTMLは生成時点のスナップショットなので、回収後は再読込する。

`--wait` はexecute/collectでのみ使用できる。ポーリング間隔は1〜60秒。
Batchが分割された場合、collectだけでは未送信のシャードを実行しない。
送信結果が不確定な場合は17の復旧処理を使用し、必要なら以下で既知Batchを採用する。

```sh
.venv/bin/python experiments/20_foresight_LAG/run.py adopt --run foresight_short_v1 --job JOB --batch-id BATCH_ID
```

## 正本・再現性

- 編集するプロンプト：`prompts/foresight_short_v1/system.txt` と `user.txt`。
- v2との全差分・原文・出典：同ディレクトリの `changes/`。
- 計画時に本文・diff・設定・注釈・入力・コードをrunに保存。
- 実行時は保存されたリクエストを使用し、依存する17/18/13のコード・元データもハッシュ検証。
- 設定やプロンプトを変える場合は版・provenance・diffを更新し、別runへ保存。
- 同一runの変更計画・差分の改変・既存コードスナップショットの上書きは拒否。

17のBatch/実行/回答解析、18の記録整形、13の読み込みを利用する。既存実験は編集しない。
17の共通Batchが出す `evaluation/` は汎用エクスポートであり、20固有の比較正本は `report/`。

## 評価

形式・語数を自動検査する。文数は句読点による参考値で、内容や言語的な一文を保証しない。
拒否・打切り・不正JSONは形式エラー、生の応答は保持する。APIエラー・未送信・回答待ちを分ける。
意味の正解点は自動生成しない。`report/review_template.csv` をコピーして人手評価する。

1. 実際の後続記録との行動一致。
2. 目的の説明に根拠があるか。注釈の目的ラベルも解釈であり、単語一致を正解としない。
3. 根拠のない標的・連携・結果等の断定。
4. 61時点の参照回答との一致（実際の正しさとは別）。
5. 予測だけという形式に反し、履歴要約や根拠を出していないか。

対象区間内の別注釈をHTMLにも表示する。別の局面に移った回答変化を単純な不安定性と扱わない。
3反復や飽和曲線の評価は行わない。入力長と情報量は同時に変わる。
Batchの処理時間から2〜3秒の即時応答要件を満たしたとは判断しない。

## 検証

```sh
.venv/bin/python -m pytest experiments/20_foresight_LAG/tests -q
```
