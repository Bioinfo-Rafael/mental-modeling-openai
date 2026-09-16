あなたはReasoning品質を評価するLLM-as-a-Judgeです。

以下のCSVを直接読み込み、各行の `Reasoning` をあなた自身で評価してください。評価用のコードやAPI呼び出しは作成せず、このセッションのあなた自身の推論能力を使って各サンプルを判定してください。

対象ファイル:

`/Users/cls-lab/Git/Matsuo/mental-modeling-openai/experiments/07_view_rawdata/reasoning_for_scoring.csv`

各行の `Reasoning` カラムを評価し、`score` カラムに1〜4の整数を記入してください。

## 評価対象

評価するのは、「Reasoningが回答を導くために、どの程度具体的・定量的な推論を行っているか」です。

最終回答が正しいかどうかは評価しないでください。

文章の長さ、流暢さ、自信の強さでは評価せず、`Reasoning` 中に明示されている推論内容だけを評価してください。

## Score rubric

### Score 1 — Explicit quantitative / mathematical reasoning

明示的な数式、数値計算、または複数の具体的な数値を用いた定量的評価によって結論を導いている。

例:

- 数式を立て、値を代入して計算している
- state/actionの数値を複数比較し、定量的に結論を導いている
- 距離、差分、傾き、変化量などを具体的に計算している
- 複数stepの具体的な数値を繰り返し参照し、その数値関係から結論を出している

単に数値を1つ引用しただけではScore 1にしない。

### Score 2 — Specific evidence-based reasoning

具体的なtrajectory/history中の観測を根拠として推論しているが、Score 1ほど明示的な数値計算はしていない。

例:

- 「step 27では〜、step 29では〜」のように具体的なstepを参照する
- closest state / similar state を具体的に特定し、その時のactionを根拠にする
- recent several stepsでactionが特定方向に変化したことを具体的に指摘する
- 特定のstate/action pairを引用して比較する
- 数値を使っているが、厳密な計算ではなく定量的比較に留まる

重要:

単に “in similar states the agent tends to ...” と述べるだけで、具体的な履歴上のstateやstepを示していない場合はScore 2にしない。

### Score 3 — General agent/policy inference

具体的なstepや定量的証拠には強く依存せず、観察されたAgentやpolicyの傾向を一般化して次のactionを推論している。

例:

- “The agent will likely ...”
- “The agent generally ...”
- “The recent policy appears to ...”
- “The agent tends to apply positive torque in this situation.”
- 過去の挙動からpolicyの一般的な傾向を推定している

ここで重要なのは、推論対象が「Agent / policyの行動傾向」であること。

“likely” や “most likely” という単語自体では判定しない。

例えば、

“The agent will likely apply positive torque based on its recent policy.”

はScore 3になり得る。

一方、

“The pendulum will most likely move downward.”

のように、Agentのpolicyではなく物理状態そのものを推測しているだけならScore 3にはしない。

### Score 4 — Generic reasoning / guessing

Agentの過去の行動や具体的なtrajectory evidenceをほとんど利用せず、一般的な物理直感、タスク知識、曖昧な推測から答えている。

例:

- “A moderate positive torque should oppose this negative motion.”
- “This motion suggests that positive torque would be appropriate.”
- “The pendulum is likely to move in this direction.”
- “The agent seems to choose ...” と述べているだけで、その根拠となるpolicy/historyの分析がない
- 明確な根拠なしにもっともらしいactionを選択している

これは「完全にランダム」という意味ではなく、task dynamicsや常識的推論だけで回答している場合も含む。

## 判定ルール

複数のタイプが混在している場合は、最終的な回答を実際に支えている最も強い推論レベルを採用してください。

ただし、Score 1や2の要素が文章中に存在するだけでは不十分です。その情報が結論の根拠として実際に使われている必要があります。

優先順位:

1 > 2 > 3 > 4

例:

「step 28と29ではpositive torqueだった。したがってagentは一般的にpositive torqueを選ぶだろう」

→ Score 2

「Recent policy suggests the agent will likely use positive torque」

→ Score 3

「Positive torque should oppose the negative angular velocity」

→ Score 4

「angular velocity = -0.71で、previous stateでは-0.43だったためΔ=-0.28。これを打ち消すにはpositive torqueが必要」

→ Score 1

## 注意事項

- Final answerの正誤はscoreに影響させない。
- “similar”, “likely”, “generally”, “typically”, “seems” などのキーワードだけで機械的に判定しない。
- 必ず文脈と、何を根拠に結論を導いているかを見る。
- `Reasoning` に書かれていない計算や根拠を自分で補完しない。
- Pendulumの問題を自分で解いて、その解法の高度さを評価しない。
- あくまで提示された `Reasoning` の推論過程を分類する。
- 全行について同じrubricを一貫して適用する。

## 作業内容

CSVを直接編集して各行の `score` を埋めてください。

元ファイルは上書きせず、結果を同じディレクトリに

`reasoning_for_scoring_scored_astra_high.csv`

として保存してください。

評価用のPythonスクリプト、API呼び出し、別のLLM Judgeは作成・使用しないでください。あなた自身がJudgeとして全行を評価してください。

完了後、以下だけを報告してください。

- 評価した総行数
- Score 1〜4それぞれの件数
- 出力ファイルのパス
