# 08 Reasoning採点の準備

採点依頼文: [Prompt.md](Prompt.md)。入力CSV: [reasoning_for_scoring.csv](../reasoning_for_scoring.csv)。まだ採点はしていない。

元データは `../results/api/records.jsonl` の144回答。Reasoning見出し直後からPrediction見出し直前までを抽出し、前後の空白と見出しのみ除去した。本文の数式・Markdown・改行は保持。全144件で抽出成功。CSVはUTF-8 BOM付き、scoreは全行空欄。

列: `query_id, model, Task, Metrics, H, shots, score_pattern, ordinal, Reasoning, score`。行順は元recordsと同じ。ordinalは0始まり。query_idで採点後の結果を結合する。

今回の採点番号は07と逆向き:

| 新score | 推論の分類 | 07の旧score |
| --- | --- | --- |
| 1 | 一般的な推測・物理直感 | 4 |
| 2 | Agent / policyの一般的傾向 | 3 |
| 3 | 具体的な観測・履歴の根拠 | 2 |
| 4 | 明示的な定量的・数学的推論 | 1 |

優先順位は4 > 3 > 2 > 1。採点依頼文の定義・例・参照番号も反転済み。既存の07のscore、08のfew-shot候補・送信プロンプト・score_pattern・実行結果は変更していない。既存Scoreと比較するときだけ `新score = 5 - 旧score` に揃える。今回の新scoreをさらに反転させない。

採点結果の指定保存先は `../reasoning_for_scoring_scored_astra_high.csv`。採点対象は今回の出力Reasoningであり、few-shot例のScoreとは別の値。
