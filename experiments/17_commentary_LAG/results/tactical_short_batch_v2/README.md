# tactical_short_batch_v2

目的を先に示し、その実現手段として行動を説明する条件。原則1文、必要な場合のみ2文、1機あたり目安35〜50英単語・合計55語以内。

2026-10-26にdry runで5窓×3モデル＝15リクエストを準備した。**API送信・回答取得はまだ行っていない。** 現在の状態は `api/batch/state.json` で確認する。

- [今回のsystem](prompts/templates/system.txt)
- [今回のuser](prompts/templates/user.txt)（short v1から変更なし）
- [プロンプトとdiffの案内](prompts/README.md)
- [設計理由・資料との関係・実行方法](../../commentary_experiment/prompt_changes/tactical_short_v2/README.md)

`manifest.json` は今回の設定と本文、`windows/` は61時点の履歴データ、`prompts/` は入力と比較資料の保存コピー、`api/batch/*.input.jsonl` は送信用入力、`evaluation/` は回答・評価用ファイルの保存先。dry run時点の `evaluation/responses.json` はpendingであり、生成結果ではない。

履歴は旧条件と同じ中心step 45/85/145/245/310の前後30step（前後6秒）。目的の説明を短縮する比較実験であり、時刻t以前の情報だけを使うオンライン条件ではない。
