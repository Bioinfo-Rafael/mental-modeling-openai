# このディレクトリについて

ここは、**`tactical_short_batch_v1` がモデルに渡すプロンプトと、その比較資料を保存する場所**です。
プロンプトを編集するための原本ではなく、このrunの計画生成時点の記録です。
dry runでも作成されるため、このディレクトリがあるだけではAPI送信済み・回答取得済みとは限りません。

## まずどれを見ればよいか

| 確認したいこと | 見る場所 |
|---|---|
| 今回使うsystem本文 | [templates/system.txt](templates/system.txt) |
| 今回使うuserテンプレート | [templates/user.txt](templates/user.txt) |
| 履歴データまで入った入力全文 | このディレクトリ直下の長い名前の `.txt`（5窓分） |
| 元のtactical条件からsystemをどう変えたか | [changes/system.diff](changes/system.diff) |
| 元のtactical条件からuserをどう変えたか | [changes/user.diff](changes/user.diff) |
| 元のsystem本文 | [changes/baseline_system.txt](changes/baseline_system.txt) |
| 元のuserテンプレート | [changes/baseline_user.txt](changes/baseline_user.txt) |
| 変更理由 | [changes/README.md](changes/README.md) |

## サブディレクトリの意味

### `templates/`：今回使うテンプレートの保存コピー

`system.txt` は今回のsystem本文です。`user.txt` は時刻や履歴データを入れる前のテンプレートです。
`{center_step}` や `{window_data}` などを各窓の値に置き換えて、実際のuser入力を作ります。

### `changes/`：元のtactical条件との比較資料の保存コピー

比較元は `results/tactical_batch_v1/` です。
`baseline_system.txt` と `baseline_user.txt` は変更前、`system.diff` と `user.diff` は変更前から今回の短縮版までの差分です。
`provenance.json` には、比較元の出典と照合に使ったハッシュなどを保存しています。
これらはレビュー用であり、モデルへ送る入力には含めません。

### 直下の長い名前の `.txt`：窓ごとの入力全文

各ファイルに `[SYSTEM]` と `[USER]` の両方が入り、user部分には実際の履歴データを挿入済みです。
例えば `sample01__start0015__step0045` は「1番目の窓、開始step 15、中心step 45」を示します。
同じ窓の入力を3モデルで共有するため、入力全文は5ファイル、リクエストは5窓×3モデルの15件です。
APIへ送るJSON形式のリクエストは、別の `../api/batch/*.input.jsonl` に保存されます。

## 原本はどこか

- 編集用system：[commentary_experiment/tactical_short_v1_system.txt](../../../commentary_experiment/tactical_short_v1_system.txt)
- 編集用user：[commentary_experiment/tactical_short_v1_user.txt](../../../commentary_experiment/tactical_short_v1_user.txt)
- 比較資料の原本：[commentary_experiment/prompt_changes/](../../../commentary_experiment/prompt_changes/README.md)

ここにある保存コピーを直接編集しても、原本は更新されません。保存済み計画との不一致になるため、変更時は原本を編集して計画を更新します。
このREADMEは場所を説明するメモで、モデルへの入力には含めません。
