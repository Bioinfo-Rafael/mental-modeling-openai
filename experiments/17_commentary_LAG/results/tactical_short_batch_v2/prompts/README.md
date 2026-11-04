# このディレクトリについて

`tactical_short_batch_v2` の計画生成時点のプロンプトと差分を保存した場所。編集用原本は `commentary_experiment/tactical_short_v2_system.txt` / `tactical_short_v2_user.txt`。このコピーを直接編集せず、原本と比較資料を更新して別の計画を作る。

| 見たいもの | ファイル |
|---|---|
| 今回のsystem本文 | [templates/system.txt](templates/system.txt) |
| 今回のuserテンプレート | [templates/user.txt](templates/user.txt) |
| short v1 → short v2のsystem差分 | [changes/from_tactical_short_batch_v1/system.diff](changes/from_tactical_short_batch_v1/system.diff) |
| short v1 → short v2のuser差分 | [changes/from_tactical_short_batch_v1/user.diff](changes/from_tactical_short_batch_v1/user.diff)（変更なし・0バイト） |
| tactical v1 → short v2のsystem差分 | [changes/from_tactical_batch_v1/system.diff](changes/from_tactical_batch_v1/system.diff) |
| tactical v1 → short v2のuser差分 | [changes/from_tactical_batch_v1/user.diff](changes/from_tactical_batch_v1/user.diff) |
| 設計理由と資料との関係 | [保存した設計メモ](changes/from_tactical_short_batch_v1/README.md) |

`templates/` はv2の本文。`changes/from_.../baseline_*.txt` はフォルダ名に示した比較元の本文であり、今回モデルに渡す本文ではない。`provenance.json` は保存済み比較元manifestとの照合記録。`---` が旧本文、`+++` がv2。

直下の長いファイル名の `.txt` は、各窓の履歴を入れた `[SYSTEM]` / `[USER]` 全文。5窓それぞれの入力を3モデルで共有する。比較資料やこのメモはモデルには送らない。
