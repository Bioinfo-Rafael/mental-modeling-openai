# 10_ICRL Reasoning採点の準備

[採点用CSV](../reasoning_for_scoring.csv) は72回答から抽出したReasoningと、空欄の `score` を含む。採点依頼文は [Prompt.md](Prompt.md)。受領した [採点済みCSV](../reasoning_for_scoring_scored_astra_high.csv) は全72件（Score 1: 0件、2: 56件、3: 16件、4: 0件）で、[HTML](../view_rawdata.html) と [比較図](../analysis/README.md) に反映済み。可視化時の再採点は行っていない。

## Codexに採点を依頼する文面

採点を行うCodexの会話に、次をそのまま貼り付ける。

```text
/Users/cls-lab/Git/Matsuo/mental-modeling-openai/experiments/10_ICRL/reasoning_score_analysis/Prompt.md
を読み、その指示に従って10_ICRLのreasoning_for_scoring.csv全72行を、あなた自身がJudgeとして評価してください。
score以外の列・本文・改行・行順は保持し、元CSVを上書きせず、
experiments/10_ICRL/reasoning_for_scoring_scored_astra_high.csv に保存してください。
外部API・別のLLM Judge・自動採点コードは使用しないでください。
完了後、評価件数、未評価件数、Score 1〜4の件数、保存先を報告してください。
```

これは保存済み回答への推論レベル採点の依頼であり、実験の `run.py` を再実行する手順ではない。CSVの読書きと整合性検証のコードは使用できるが、scoreの決定はJudge自身が行う。採点済みファイルが既にある場合は上書きせず停止する。

保存先の `astra_high` は08・09に合わせた命名。ファイル名だけで実際のJudgeモデルやreasoning effortを保証しない。特定の設定で評価したい場合は、その設定を選んだ会話で上の文面を実行する。

## 基準と対象

08・09と同じ番号体系で、大きいほど具体的・定量的な推論を示す。

| score | 分類 |
| --- | --- |
| 1 | 一般的な推測・物理直感 |
| 2 | Agent / policyの一般的傾向 |
| 3 | 具体的な観測・履歴に基づく推論 |
| 4 | 明示的な定量・数学的推論 |

07と比較する際だけ `新score = 5 - 旧score` に変換する。08・09のscoreは反転しない。

対象は `Pendulum-v1 / next-action / H20 / t=0 / episode_9_seed3407.npz / query index 20` の72回答。2 models × 4 methods × E=1/3/9 × 3 repeatsで、同一条件の3回答を個別に評価する。

method・K・Eやモデル名、最終回答の正誤から採点を推測しない。PPO/FQIへの言及、式の引用、「内部で計算した」という宣言だけでは加点しない。ICR-FQIは中間計算を表示しない指示なので、公開Reasoningの評価から内部のFQI実行有無は判断しない。Q-table等を表示しないこと自体も減点理由にしない。

## CSVと抽出

元データは `../results/api/records.jsonl`。09の `prepare_scoring.extract()` を直接再利用し、Reasoning見出しの直後からPrediction見出しの直前までを抽出する。見出しと前後の空白以外は変更せず、数式・Markdown・改行を保持する。入力prompt、内部推論トークン、最終回答、正解値、正誤情報は採点用CSVへ追加しない。

列は `query_id, condition_id, model, Task, Metrics, H, method, K, E, ordinal, history_start, query_index, repeat_id, reward_present, target_episode, episode_path, Reasoning, score`。

- `query_id` は採点後に元回答と結合するキー。行順はrecordsと同じ。
- `model` は `terra` / `luna`。`K` はFQI条件のみ2または5、他は空欄。`E` はtarget込みのepisode数。
- `ordinal` は0始まり、`repeat_id` は1始まり。`reward_present` は全行1。
- UTF-8 BOM付き、全フィールドを引用。`score` は全72行空欄。

[validation.json](validation.json) に元records・manifest・再利用した抽出コード・出力CSVのSHA256と検証結果を保存する。72件の一意なID、予定した条件展開、manifestとの一致、raw responseと回答本文の一致、Reasoningの抽出、CSV読み戻しでの全列・行順一致を検証した。

抽出専用スクリプトの初回実行コマンド（採点は行わない）：

```bash
cd /Users/cls-lab/Git/Matsuo/mental-modeling-openai
.venv/bin/python -B experiments/10_ICRL/reasoning_score_analysis/prepare_scoring.py
```

CSV・検証ファイル・採点済みCSVのいずれかが存在する場合は上書きせず停止する。今回の準備ファイルは作成済みなので、採点時は再抽出せず上の依頼文を使う。
