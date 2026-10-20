# 行動ID一致／不一致による例の選択

目標 96件 / 選択 74件 / 不足 22件。
Score 2不足の補完: 0件。元Scoreは変更していません。

各Task×モデルで、正解Score 1・正解Score 2・不正解Score 3・不正解Score 4を各3件まで選択。
Pendulumはbin ID、MountainCarは行動IDを比較。ID抽出失敗・API失敗・state予測は除外。
各Score内でnext-actionを優先し、不足ならlast-action。正解Score 2だけは両予測対象でも不足した後にScore 3→4で補完。
Hは制限せず、同順位はH・episode・index順。異なるH/modelの回答はquery_idが別なら別例として数えます。
候補不足を他モデル・Task・未許可Scoreで埋めません。希望数はTaskごとに各区分12件（4モデル×3件）、計48件です。

| Task | モデル | 正解 S1 | 正解 S2枠 | 不正解 S3 | 不正解 S4 | 合計 |
|---|---|---:|---:|---:|---:|---:|
| Pendulum-v1 | sol | 3 | 3 | 3 | 1 | 10 |
| Pendulum-v1 | terra | 3 | 3 | 3 | 0 | 9 |
| Pendulum-v1 | luna | 3 | 3 | 3 | 3 | 12 |
| Pendulum-v1 | 3.5 | 0 | 3 | 3 | 3 | 9 |
| MountainCar-v0 | sol | 3 | 3 | 1 | 0 | 7 |
| MountainCar-v0 | terra | 3 | 3 | 1 | 0 | 7 |
| MountainCar-v0 | luna | 3 | 3 | 3 | 2 | 11 |
| MountainCar-v0 | 3.5 | 0 | 3 | 3 | 3 | 9 |

## 不足

- Pendulum-v1 / sol / incorrect / 希望Score 4: 1/3件（不足2）
- Pendulum-v1 / terra / incorrect / 希望Score 4: 0/3件（不足3）
- Pendulum-v1 / 3.5 / correct / 希望Score 1: 0/3件（不足3）
- MountainCar-v0 / sol / incorrect / 希望Score 3: 1/3件（不足2）
- MountainCar-v0 / sol / incorrect / 希望Score 4: 0/3件（不足3）
- MountainCar-v0 / terra / incorrect / 希望Score 3: 1/3件（不足2）
- MountainCar-v0 / terra / incorrect / 希望Score 4: 0/3件（不足3）
- MountainCar-v0 / luna / incorrect / 希望Score 4: 2/3件（不足1）
- MountainCar-v0 / 3.5 / correct / 希望Score 1: 0/3件（不足3）

元Reasoningが空欄の選択例: 3件。既存Scoreを保持しています。

07への読み込み／08のdry-run入力: `selected_examples_by_id.json`。
選択JSONは既存exportと同じcandidate_pool形式です。summary JSON/CSVは監査用で、例の入力として自動検出されません。
