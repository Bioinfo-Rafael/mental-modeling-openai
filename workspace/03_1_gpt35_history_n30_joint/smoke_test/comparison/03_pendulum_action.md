# Pendulum Actionの直接比較

[比較indexへ戻る](../README.md#文章を左右に並べた比較)

Next/Lastとも、continuous no-binsをbaseにし、Experiment 01で使ったcontinuous binsから10-bin定義を移植している。表中の文章は`feedback.py`から直接引用したclass templateである。

## Next Action：no-bins base対Joint

左は`feedback.py:166-189`、右は`feedback.py:194-220`である。

<table>
<tr><th width="50%">既存continuous no-bins（左）</th><th width="50%">03.1 Joint（右）</th></tr>
<tr><td><pre>[SAME]
In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.</pre></td><td><pre>[SAME]
In next step {i} (indexed from 0), the agent transitted to the state s{i} = {state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} that the RL agent will most likely take at state s{i}.</pre></td></tr>
<tr><td><pre>[REUSED]
The action space dimension is {action_dim}, with each dimension ranging from [-2, 2].</pre></td><td><pre>[REUSED]
The action space dimension is {action_dim}, with each dimension ranging from [-2, 2].</pre></td></tr>
<tr><td><pre>（bin定義なし）</pre></td><td><pre>[COPIED＋MODIFIED]
Each dimension has 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. Also predict which bin (indexed from 0 to 9) each element of action a{i} will fall into.</pre></td></tr>
<tr><td><pre>[SAME]
1. [Reasoning]:
2. [Prediction]:
3. [Formatting]:
    Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
    Return a list with the following example format,
    ```python
    # For example, if the action dim is 2, and the predicted action element [0] is -1.52 and element [1] is 1.25, then the predictions would be [-1.52, 1.25]
    predictions = [-1.52, 1.25]
    ```</pre></td><td><pre>[SAME]
1. [Reasoning]:
2. [Prediction]:
3. [Formatting]:
    Action element [i] is a real value from the action range above, where "i" indicates the i-th action element (also indexed from 0).
    Return a list with the following example format,
    ```python
    # For example, if the action dim is 2, and the predicted action element [0] is -1.52 and element [1] is 1.25, then the predictions would be [-1.52, 1.25]
    predictions = [-1.52, 1.25]
    ```</pre></td></tr>
<tr><td><pre>（Joint markerなし）</pre></td><td><pre>[NEW]
Action element [i] is also assigned a bin index from 0 to 9. After the real-valued predictions list, return the corresponding bin indices using this final marker:
&gt;&gt;Final action bins: [1, 8]</pre></td></tr>
<tr><td><pre>Note that the provided example needs to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places for each action dim.</pre></td><td><pre>[MODIFIED]
Note that the provided examples need to be adapted to the current action dim, which is {action_dim}. Predict a real value with up to two decimal places and a single integer from 0 to 9 for each action dim.</pre></td></tr>
</table>

## Experiment 01のbins部分対Joint

左は`feedback.py:91-114`の`NextActionPredictionContinuousBins`で、Experiment 01が実際に使った文章である。

<table>
<tr><th width="50%">既存continuous bins（左）</th><th width="50%">03.1 Joint（右）</th></tr>
<tr><td><pre>The action space dimension is {action_dim}, with each dimension having 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. You only need to predict which bin (indexed from 0 to 9) each element of action a{i} will fall into.</pre></td><td><pre>Each dimension has 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. Also predict which bin (indexed from 0 to 9) each element of action a{i} will fall into.</pre></td></tr>
<tr><td><pre>[MODIFIED]
You only need to predict which bin ...</pre></td><td><pre>[MODIFIED]
Also predict which bin ...</pre></td></tr>
<tr><td><pre>predictions = [1, 8]</pre></td><td><pre>predictions = [-1.52, 1.25]

[NEW]
&gt;&gt;Final action bins: [1, 8]</pre></td></tr>
</table>

bin境界の数値列はコピーしているが、文全体をそのまま貼ったのではない。raw値も要求するため`only`を外し、bin結果を別markerへ移した。

## Last Action：no-bins base対Joint

左は`feedback.py:489-512`、右は`feedback.py:517-543`である。

<table>
<tr><th width="50%">既存continuous no-bins（左）</th><th width="50%">03.1 Joint（右）</th></tr>
<tr><td><pre>[SAME]
In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.</pre></td><td><pre>[SAME]
In next step {i} (indexed from 0), the agent's state was s{i} = {state}. Then the agent took an action a{i} and the state transitted to s{k} = {next_state}. Based on your observation and understanding of the agent's behaviour, predict the action a{i} taken by the RL agent.</pre></td></tr>
<tr><td><pre>The action space dimension is {action_dim}, with each dimension ranging from [-2, 2].

（bin定義・bin markerなし）</pre></td><td><pre>[REUSED]
The action space dimension is {action_dim}, with each dimension ranging from [-2, 2].

[COPIED＋MODIFIED]
Each dimension has 10 discrete bins ranging from [-2, 2]: [-2., -1.6), [-1.6, -1.2), [-1.2, -0.8), [-0.8, -0.4), [-0.4, 0.), [0., 0.4), [0.4, 0.8), [0.8, 1.2), [1.2, 1.6), [1.6, 2.]. Also predict which bin (indexed from 0 to 9) each element of action a{i} will fall into.</pre></td></tr>
<tr><td><pre>predictions = [-1.52, 1.25]

（Joint markerなし）</pre></td><td><pre>[REUSED]
predictions = [-1.52, 1.25]

[NEW]
&gt;&gt;Final action bins: [1, 8]</pre></td></tr>
</table>

Last Actionの10-bin移植元は`feedback.py:413-436`の`LastActionPredictionContinuousBins`である。raw値用formatting、最終注意文の変更、新規markerはNext Actionと同じ構造である。

## render後の例

<table>
<tr><th width="50%">class template（左）</th><th width="50%">pendulum_next_action.txt（右）</th></tr>
<tr><td><pre>action a{i}
state s{i} = {state}
action dimension = {action_dim}</pre></td><td><pre>action a5
state s5 = [[ 0.9906,  0.1366, -0.1319]]
action dimension = 1</pre></td></tr>
</table>

結論：raw valueの文章は既存no-bins、10-bin定義は既存binsから来ている。完全新規なのは、両方を同時に返させる接続文と`>>Final action bins` markerである。
