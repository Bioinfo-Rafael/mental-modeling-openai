# 09 Reasoning採点の準備

採点依頼文は [Prompt.md](Prompt.md)、入力は [reasoning_for_scoring.csv](../reasoning_for_scoring.csv)。入力96件のscoreは空欄のまま保持。採点済み結果は [reasoning_for_scoring_scored_astra_high.csv](../reasoning_for_scoring_scored_astra_high.csv)。全96件をこのセッションのJudgeが個別に採点済み（Score 1: 1件、2: 58件、3: 35件、4: 2件）。

08と同じ基準・番号体系を使う。

| score | 推論の分類 |
| --- | --- |
| 1 | 一般的な推測・物理直感 |
| 2 | Agent / policyの一般的傾向 |
| 3 | 具体的な観測・履歴を根拠にした推論 |
| 4 | 明示的な定量的・数学的推論 |

07の旧番号と比較する場合だけ `新score = 5 - 旧score` に変換する。08の採点結果は今回と同じ向き。few-shot選択の `score_pattern=pattern1` は旧番号体系の条件名であり、今回記入するscoreとは別。

## 入力の作成

元データは `../results/api/records.jsonl` の96回答。Reasoning見出し直後からPrediction見出し直前までを抽出した。見出しと前後の空白以外は変更せず、本文中の数式・Markdown・改行を保持している。few-shot入力や内部推論トークンは抽出していない。

`prepare_scoring.py` は抽出専用で、採点やAPI呼び出しは行わない。全96件の抽出、query_idの一意性、実行manifestとの一致、CSVへの書き出し・読み戻しによる本文と行順の一致を確認済み。検証結果と元データのSHA256は `validation.json` に保存。

CSVは08と同じUTF-8 BOM付き。既存列 `query_id, model, Task, Metrics, H, shots, score_pattern, ordinal, Reasoning, score` を維持し、09の条件として以下を追加した。

- `history_start`: 履歴開始時点（0・100）。
- `query_index`: 予測対象時点（5・20・105・120）。
- `repeat_id`: 同一問題への反復番号（1・2・3）。ordinalは0始まり。
- `reward_present`: 評価対象の履歴のreward提示（1＝あり、0＝なし）。
- `reward_scope`: `target_history_only`。few-shot内のrewardは維持。
- `episode_path`: 元episode_9のファイル。

32条件×3反復＝96行。行順はrecordsと同じ。正誤・正解値は採点用CSVに含めず、Reasoning本文だけを評価する。

## 採点の依頼

評価を担当するセッションに `Prompt.md` の内容を渡す。Judge自身が全行を評価し、score以外の列・行順・本文は変更しない。

保存先は08と同じ命名の `../reasoning_for_scoring_scored_astra_high.csv`。採点とCSV整合性の記録は `scoring_validation.json` に保存した。今回はこのセッションのassistant自身が判定し、採点用APIや別のJudgeは使用していない。正確な実行モデルID・reasoning effortは未検証として記録した。ファイル名だけでは使用モデルを保証しない。

採点後はquery_idで元回答と結合し、model/H/開始時点/shot別にreward有無を比較できる。同一問題の3反復を異なる3問題として扱わない。

抽出を初回実行するコマンド（リポジトリルートから）:

```bash
.venv/bin/python experiments/09_ablation_reward/reasoning_score_analysis/prepare_scoring.py
```

既存の採点用CSVを保護するため、ファイルがある場合は上書きせず停止する。
