# tactical_short_v1 のプロンプト変更

## 出発点

`results/tactical_batch_v1/manifest.json` の5窓の保存済みメッセージに照合した原文を基準とする。
確認・保存した原文と出典をそのまま使用し、採用された短縮指示を残しながら、アノテーションをヒントとして与える表現を除いた。

- [元条件からのsystem差分](system.diff)
- [元条件からのuser差分](user.diff)
- [元のsystem本文](baseline_system.txt)、[元のuserテンプレート](baseline_user.txt)
- [出典・照合した5窓・ハッシュ](provenance.json)

diffの `---` が元のtactical条件、`+++` が今回の短縮版である。数値データの変更は含まない。

## 今回修正した内容

1. system冒頭を元の文章へ完全に戻した。

   > Your task is to reconstruct the most plausible tactical intentions behind each aircraft's behavior over an entire recorded time window.

2. systemの中心時点の説明を元の文章へ完全に戻した。

   > The center step t is a reference point for constructing the window, NOT the exclusive subject of the interpretation.

3. user冒頭を元の `Center step t` / `Center time` / `relative_step = step - t` に戻した。「Human-annotated change point」「Change-point time」を削除した。
4. systemに残っていた `around T` を `within the window` に変更し、中心時点に変化があると暗示しない形にした。
5. systemとuserの重複指示をsystemへ集約した。userのデータ後は「systemの分析・出力要件に従って、この窓を説明する」という依頼だけにした。

## 重複指示の整理

今回の整理ではsystem本文は変更していない。分析対象の範囲、主要行動・変化と理由、他機・ミサイルの扱い、数値・操作履歴の省略、結果の採否、後続情報の扱い、推測の留保、文数・語数、JSON形式はsystemだけに記載する。
userには対象窓のメタデータ、実データ、systemを参照する短い依頼を残す。system内部の節間の繰り返しは今回の変更対象に含めない。

`system.diff` と `user.diff` は、どちらも `tactical_batch_v1` の原文から現在の短縮版までの差分である。編集途中の版との比較ファイルは残さない。

## 採用済みの短縮方針

- 全61時点を分析し、主要な行動と重要な変化、その目的・根拠となる状況を残す。
- 行動を変えたきっかけは、根拠がある場合にWHYの中に含める。
- 目的には、行動で実現したい近い将来の状況を含める。
- 敵・味方・ミサイルは行動の理由に直結する情報だけを残す。
- 数値・細かな操作履歴は原則省略し、その後の結果は解釈を左右する場合だけ残す。
- 推測を示す表現を短く残し、短縮のために断定を強めない。
- 原則1機2〜3文、目安60〜90英単語、上限90語。短くて足りる場合は下限なし。全窓非活動の機体は短い1文。
- 出力は4機の文字列フィールドだけのJSONとする。

環境、状態・行動の仕様、欠損値、ミサイル情報、根拠のない標的・命中・認知・連携を作らない規則は維持している。
アノテーションの存在・変化点ラベル・正解説明文はモデルへのsystem/userメッセージに含めない。窓そのものの選定方法を変更するものではない。

## 識別と実行

| 項目 | 値 |
|---|---|
| CLI variant | `tactical_short_v1` |
| Prompt version | `lag-tactical-short-v1` |
| 既定の結果ディレクトリ | `results/tactical_short_batch_v1/` |
| system | `commentary_experiment/tactical_short_v1_system.txt` |
| user | `commentary_experiment/tactical_short_v1_user.txt` |

ユーザーの指定に従い、API未送信のv1を修正版で上書きした。実行時は `--prompt-variant tactical_short_v1` を指定する。
manifestの `prompt_spec` と `prompts/templates/` に新本文、`prompts/changes/` にこの説明・原文・diff・出典を保存する。本文・ハッシュ・出力条件と実際の入力全文を保存し、どの結果にどのプロンプトを使ったかを確認できるようにする。

入力は前後30ステップの振り返り条件であり、時刻t以前だけを使うリアルタイム条件ではない。語数短縮と内容の選別を同時に行うため、結果の差を語数制限だけの因果効果とは結論できない。
