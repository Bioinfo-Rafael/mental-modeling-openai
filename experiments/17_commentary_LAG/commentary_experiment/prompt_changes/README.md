# このディレクトリについて

ここは、**プロンプトを何からどう変更したかを確認するための比較資料**の置き場です。
モデルに送るプロンプトの原本は、このディレクトリの外、`commentary_experiment/` 直下にあります。

## 目的を中心にした短縮版 v2

[tactical_short_v2/README.md](tactical_short_v2/README.md)に本文の所在、資料との関係、実行方法を記載しています。`tactical_short_v1/` とその保存済みrunは従来版です。

| 比較 | system | user |
|---|---|---|
| short v1 → short v2 | [diff](tactical_short_v2/from_tactical_short_batch_v1/system.diff) | [diff（変更なし）](tactical_short_v2/from_tactical_short_batch_v1/user.diff) |
| tactical v1 → short v2 | [diff](tactical_short_v2/from_tactical_batch_v1/system.diff) | [diff](tactical_short_v2/from_tactical_batch_v1/user.diff) |

v2の原本は [system](../tactical_short_v2_system.txt) と [user](../tactical_short_v2_user.txt)。新しいrunは `results/tactical_short_batch_v2/` に保存します。

## 従来の短縮版 v1

| 確認したいこと | 見るファイル |
|---|---|
| 短縮版v1のsystem本文 | [../tactical_short_v1_system.txt](../tactical_short_v1_system.txt) |
| 短縮版v1のuser本文 | [../tactical_short_v1_user.txt](../tactical_short_v1_user.txt) |
| 元のtactical条件からsystemをどう変えたか | [tactical_short_v1/system.diff](tactical_short_v1/system.diff) |
| 元のtactical条件からuserをどう変えたか | [tactical_short_v1/user.diff](tactical_short_v1/user.diff) |
| 変更の理由 | [tactical_short_v1/README.md](tactical_short_v1/README.md) |

## `tactical_short_v1/` の中身

| ファイル | 意味 |
|---|---|
| `baseline_system.txt` | `results/tactical_batch_v1/` に保存された変更前のsystem本文 |
| `baseline_user.txt` | 元のtactical条件のuserテンプレート。実データを挿入すると、保存済み5窓のuser入力と一致することを確認済み |
| `system.diff` | 元のtactical system → 現在の短縮版systemの差分 |
| `user.diff` | 元のtactical user → 現在の短縮版userの差分 |
| `provenance.json` | 比較元の保存先、照合した窓、本文・manifestのハッシュなどの出典情報 |
| `README.md` | 変更内容と理由の説明 |

diffの `-` は元の文章から削った行、`+` は短縮版に追加した行です。
`baseline_*` は比較用です。今回の短縮版の入力として送信するものではありません。

## `results/.../prompts/` との違い

- **ここ（`commentary_experiment/prompt_changes/`）**：比較資料の原本。プロンプトを編集・レビューするときに使います。
- **[results/tactical_short_batch_v1/prompts/](../../results/tactical_short_batch_v1/prompts/README.md)**：そのrunの計画生成時に保存したコピーと、実データを入れた入力全文。そのrunが何を使うかを確認します。

比較資料は計画生成時にmanifestへ取り込まれ、`results/.../prompts/changes/` にコピーされます。
同じ資料が2か所にあるのは、編集用とrunごとの記録を分けるためです。あとで原本を変更しても、保存済みrunのコピーは自動では変更されません。
