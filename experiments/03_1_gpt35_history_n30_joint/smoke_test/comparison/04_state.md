# State promptの直接比較

[比較indexへ戻る](../README.md#文章を左右に並べた比較)

MountainCarとPendulumは同じState question classを使う。引用は`feedback.py`のclass templateであり、smoke出力では`{i}`、`{j}`、`{k}`が実際のindexへ置換される。

## 比較を二段階にする理由

Experiment 01は`INC / DEC`二択版を使用したが、03.1 Jointのbaseは既存の`MoreOptions`三択版である。

1. Experiment 01二択版 対 既存MoreOptions：`UNCH`とthresholdの出所を確認
2. 既存MoreOptions 対 Joint：本当に新規のraw value/deltaを確認

## Next State：二択版対既存MoreOptions

左は`feedback.py:850-872`、右は`feedback.py:795-817`である。

<table>
<tr><th width="50%">Experiment 01：NextStatePrediction（左）</th><th width="50%">既存NextStatePredictionMoreOptions（右）</th></tr>
<tr><td><pre>You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC") or decreases (symbolized as "DEC") compared to the current state s{j} = {state} after taking action a{j}.</pre></td><td><pre>You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}.

The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.</pre></td></tr>
<tr><td><pre>State element [i] INC (increase) or DEC (decrease)

predictions = ["INC", "DEC"]

Please choose either INC or DEC...</pre></td><td><pre>State element [i] INC (increase), DEC (decrease), or UNCH (unchange)

predictions = ["INC", "DEC", "UNCH"]

Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH"...</pre></td></tr>
</table>

右列はJoint追加前から存在した。`UNCH`、threshold `1e-4`、三択例は新規ではない。

## Next State：既存MoreOptions対Joint

左は`feedback.py:795-817`、右は`feedback.py:821-848`である。

<table>
<tr><th width="50%">既存MoreOptions（左）</th><th width="50%">03.1 Joint（右）</th></tr>
<tr><td><pre>Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}.</pre></td><td><pre>[SAME]
Using the history of states, actions, and rewards up to step {j} (indexed from 0), predict the next state s{i} (the agent will transition to) that follows from the current state s{j} = {state}, action a{j} = {action}, and reward r{j} = {reward} received at step {j}.</pre></td></tr>
<tr><td><pre>You only need to deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}.</pre></td><td><pre>[MODIFIED]
Deduce whether each element of the next state s{i} increases (symbolized as "INC"), decreases (symbolized as "DEC"), or remains unchanged (symbolized as "UNCH") compared to the current state s{j} = {state} after taking action a{j}.</pre></td></tr>
<tr><td><pre>The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.

Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:</pre></td><td><pre>[SAME]
The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.

Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction. Begin with a compact reasoning, followed by a step-by-step prediction for each element of s{i}, using the template in your provided answer:</pre></td></tr>
<tr><td><pre>1. [Reasoning]:
2. [Prediction]:
3. [Formatting]:
    State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
    Return a list with the following example format,
    ```python
    # element [0] increases, element [1] decreases, and element [2] remains unchanged
    predictions = ["INC", "DEC", "UNCH"]
    ```</pre></td><td><pre>[SAME]
1. [Reasoning]:
2. [Prediction]:
3. [Formatting]:
    State element [i] INC (increase), DEC (decrease), or UNCH (unchange), where "i" indicates the index of the state element (indexed from 0).
    Return a list with the following example format,
    ```python
    # element [0] increases, element [1] decreases, and element [2] remains unchanged
    predictions = ["INC", "DEC", "UNCH"]
    ```</pre></td></tr>
<tr><td><pre>（raw value/deltaの文章なし）</pre></td><td><pre>[NEW]
Also predict the numerical value of each element of the target state. Also predict the signed change of each state element, defined in the forward time direction as delta_state = s{i} - s{j}.
Return the state values and state deltas using these final markers:
&gt;&gt;Final state values: [v0, v1, v2]
&gt;&gt;Final state deltas: [d0, d1, d2]
All three lists must follow the same state-dimension order and contain one element for each state dimension.</pre></td></tr>
<tr><td><pre>Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.</pre></td><td><pre>[SAME]
Please ensure each state element prediction explicitly states either "INC", "DEC", or "UNCH", even in cases of uncertainty or multiple possibilities.</pre></td></tr>
</table>

`You only need to deduce`から`only`を外したのは、方向以外にraw valueとdeltaも要求するためである。

## Last State：二択版対既存MoreOptions

左は`feedback.py:1176-1198`、右は`feedback.py:1118-1141`である。

<table>
<tr><th width="50%">Experiment 01：LastStatePrediction（左）</th><th width="50%">既存LastStatePredictionMoreOptions（右）</th></tr>
<tr><td><pre>You only need to deduce whether each element of the state s{i} was lower or higher, compared to s{k}, before the action a{i} was taken at state s{i}.

Predict if each element of s{i} increased (symbolized as "INC") or decreased (symbolized as "DEC") to reach s{k}.</pre></td><td><pre>You only need to deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}.

The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.

Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}.</pre></td></tr>
<tr><td><pre>predictions = ["INC", "DEC"]</pre></td><td><pre>predictions = ["INC", "DEC", "UNCH"]</pre></td></tr>
</table>

ここでも三択化は既存MoreOptions由来である。

## Last State：既存MoreOptions対Joint

左は`feedback.py:1118-1141`、右は`feedback.py:1146-1174`である。

<table>
<tr><th width="50%">既存MoreOptions（左）</th><th width="50%">03.1 Joint（右）</th></tr>
<tr><td><pre>In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}.</pre></td><td><pre>[SAME]
In step {i} (indexed from 0), the state was s{i}. Then the agent took an action a{i} = {action} and the state transitted to s{k} = {next_state} from s{i}. Deduce the previous state s{i} by comparing it to s{k}.</pre></td></tr>
<tr><td><pre>You only need to deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}.</pre></td><td><pre>[MODIFIED]
Deduce whether each element of the state s{i} was lower, higher, or the same, compared to s{k}, before the action a{i} was taken at state s{i}.</pre></td></tr>
<tr><td><pre>The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.
Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction.
Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}.</pre></td><td><pre>[SAME]
The state element can stay unchanged if the difference between elements of two successive states is less than a threshold of 1e-4.
Consider the patterns and transition dynamics observed in the historical data up to step {j} to inform your prediction.
Predict if each element of s{i} increased (symbolized as "INC"), decreased (symbolized as "DEC"), or stayed unchanged (symbolized as "UNCH") to reach s{k}.</pre></td></tr>
<tr><td><pre>predictions = ["INC", "DEC", "UNCH"]</pre></td><td><pre>[SAME]
predictions = ["INC", "DEC", "UNCH"]</pre></td></tr>
<tr><td><pre>（raw value/deltaの文章なし）</pre></td><td><pre>[NEW]
Also predict the numerical value of each element of the target previous state s{i}. Also predict the signed change of each state element, defined in the forward time direction as delta_state = s{k} - s{i}.
Return the state values and state deltas using these final markers:
&gt;&gt;Final state values: [v0, v1, v2]
&gt;&gt;Final state deltas: [d0, d1, d2]
All three lists must follow the same state-dimension order and contain one element for each state dimension.</pre></td></tr>
<tr><td><pre>Please choose either INC, DEC, or UNCH for each element, even in cases of uncertainty or multiple possibilities.</pre></td><td><pre>[SAME]
Please choose either INC, DEC, or UNCH for each element, even in cases of uncertainty or multiple possibilities.</pre></td></tr>
</table>

## NextとLastのdelta定義

<table>
<tr><th width="50%">Next State（左）</th><th width="50%">Last State（右）</th></tr>
<tr><td><pre>delta_state = s{i} - s{j}

smoke例：s5 - s4</pre></td><td><pre>delta_state = s{k} - s{i}

smoke例：s6 - s5</pre></td></tr>
</table>

どちらも`later state - earlier state`である。Last Stateは過去状態を推定するが、deltaの符号は逆向きにしていない。

結論：Stateで別の既存箇所から持ってきたのはMoreOptionsの三択方向判定部分である。raw state value、signed forward delta、2個のfinal marker、3リストのdimension順序制約は完全新規である。
