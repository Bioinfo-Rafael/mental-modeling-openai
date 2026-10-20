# 08 Few-shot dry-run

Status: `ready`

予定: 48条件 × 3問 = 144件。生成済み: 144件。API呼び出し: 0。

- 全条件のプロンプトを生成しました。

[preview.html](preview.html)で送信予定system/user全文を確認できます。改行はそのまま表示します。

- `prompts/`: 1件ごとのTXT（ready時のみ）
- `manifest.json`: 全条件・送信予定request・使用例ID・入力SHA
- `candidate_inventory.csv`: Task・Score・区分別の必要数と候補数
- `selected_candidates.json`: 読み込んだ候補のスナップショット

これは計画のみで、送信・回答取得・精度評価は行っていません。
