# MountainCar Actionの直接比較

[比較indexへ戻る](../README.md#文章を左右に並べた比較)

MountainCar ActionはJoint化していない。Experiment 01と03.1は同じclassを指定しており、render後のquestionも同じである。

## Next Action：[SAME]

出所は両方とも`feedback.py:24-41`の`NextActionPrediction`である。

<table>
<tr><th width="50%">Experiment 01（左）</th><th width="50%">03.1 smoke（右）</th></tr>
<tr><td><pre>In next step 5 (indexed from 0), the agent transitted to the state s5 = [[-0.4858,  0.0039]]. Based on your observation and understanding of the agent's behaviour, can you predict the action a5 (an integer from the given range) the RL agent will most likely take at step 5? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. &gt;&gt;Final action choice: []

        Please choose only one action, even if multiple actions seem possible.</pre></td><td><pre>In next step 5 (indexed from 0), the agent transitted to the state s5 = [[-0.4858,  0.0039]]. Based on your observation and understanding of the agent's behaviour, can you predict the action a5 (an integer from the given range) the RL agent will most likely take at step 5? Please first provide a compact reasoning before your answer to the action choice. Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. &gt;&gt;Final action choice: []

        Please choose only one action, even if multiple actions seem possible.</pre></td></tr>
</table>

## Last Action：[SAME]

出所は両方とも`feedback.py:348-365`の`LastActionPrediction`である。

<table>
<tr><th width="50%">Experiment 01（左）</th><th width="50%">03.1 smoke（右）</th></tr>
<tr><td><pre>In subsequent step 5 (indexed from 0), the state was s5 = [[-0.4858,  0.0039]]. Then the agent took an action a5 and the state transitted to s6 = [[-0.4811,  0.0046]], predict the taken action a5 (an integer from the given range). Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. &gt;&gt;Final action choice: []

        Please choose only one action, even if multiple actions seem possible.</pre></td><td><pre>In subsequent step 5 (indexed from 0), the state was s5 = [[-0.4858,  0.0039]]. Then the agent took an action a5 and the state transitted to s6 = [[-0.4811,  0.0046]], predict the taken action a5 (an integer from the given range). Think step by step and use the following template in your provided answer:

        1. [Reasoning]:
        2. [Prediction]:
        3. &gt;&gt;Final action choice: []

        Please choose only one action, even if multiple actions seem possible.</pre></td></tr>
</table>

結論：MountainCar Actionについて03.1で貼り付けた文章、修正した文章、完全新規の文章はない。
