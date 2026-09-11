# Experiment 03.1: GPT-3.5 N30 Joint Prediction

## 目的

Experiment 03のmodel、task、metric、history長、query数、データ選択、backend、loggingを維持し、continuous actionとstate predictionに使用するquestion variantだけをJoint版へ変更する。

既存のExperiment 03自体は変更していない。

## Prompt構造

```text
MountainCar Action
既存Discrete Action Prompt
        ↓
Action ID

Pendulum Action
既存Continuous Value Prompt
        +
既存Bin Instruction
        ↓
Raw Action Value + 10-bin ID

MountainCar / Pendulum State
既存More-Options Prompt
        +
Absolute Value Instruction
        +
Forward Delta Instruction
        ↓
INC / DEC / UNCH + Raw State Value + Delta State
```

Next StateとLast Stateのどちらでも、deltaは常に時間のforward方向として次のように定義する。

```text
later state - earlier state
```

Stateの10-bin分類は含めない。

## 既存実装の再利用

Action Joint：

- base：既存のcontinuous real-value/no-bins prompt
- copied：既存の10-bin境界とbin-index instruction
- new：既存raw-value listとbin markerを含むJoint output

State Joint：

- base：既存のmore-options state prompt
- reused：`INC` / `DEC` / `UNCH`とUNCH threshold `1e-4`
- new：target stateのraw value
- new：forward方向のsigned state delta

system prompt、history format、query index、user prompt wrapperはすべてLLM-Xavierの既存実行経路を利用する。

## 追加したquestion_name

1. `next_action_prediction_continuous_joint`
2. `last_action_prediction_continuous_joint`
3. `next_state_prediction_more_options_joint`
4. `last_state_prediction_more_options_joint`

## Task × Metric mapping

| Task | Next Action | Last Action | Next State | Last State |
| --- | --- | --- | --- | --- |
| MountainCar-v0 | `next_action_prediction` | `last_action_prediction` | `next_state_prediction_more_options_joint` | `last_state_prediction_more_options_joint` |
| Pendulum-v1 | `next_action_prediction_continuous_joint` | `last_action_prediction_continuous_joint` | `next_state_prediction_more_options_joint` | `last_state_prediction_more_options_joint` |

## Smoke test

API clientを作成せず、8種類のAPI message previewを生成・検証する。

```bash
.venv/bin/python experiments/03_1_gpt35_history_n30_joint/smoke_test/run_smoke_test.py
```

生成結果は`smoke_test/prompts/`に保存される。

## Dry run

APIを呼ばずにN30の全実行計画と前提条件を検証し、manifestを保存する。

```bash
.venv/bin/python experiments/03_1_gpt35_history_n30_joint/run.py
```

## 本実行

事前に`OPENAI_API_KEY`を安全な方法で環境変数へ設定する。API keyをcommand lineやREADMEへ直接書かないこと。

以下は本実行用commandであり、今回の実装・検証時には実行していない。

```bash
source .venv/bin/activate
python experiments/03_1_gpt35_history_n30_joint/run.py --execute --confirm-paid-api
```

中断した実行は、既存の順序付きreplay/resume機構を利用できる。

```bash
python experiments/03_1_gpt35_history_n30_joint/run.py \
  --resume --execute --confirm-paid-api
```

request、完全なraw response、query metadataとID、timing、token usage、上流scoreはExperiment 03と同じファイル・logging経路に保存される。

## 実行時の旧scorerと共通オフライン評価

実行時の上流parser/scoringは変更していない。raw responseは上流scoringより先に保存される。

- State JointのDirectionは旧scorerでも扱えるが、absolute/deltaは共通解析で評価する。
- Pendulum Actionのtorque/binは旧scorerでは同時に正しく扱えないため、共通解析でfinal markerから独立に再parseする。

上流AccuracyをJoint全体のmetricとして解釈しないこと。
[analysis/analyze.py](analysis/analyze.py)は[common_analysis](../common_analysis/README.md)を呼び、N=10/20/30の正式な評価を行う。
05/06も同じ共通評価と[Joint question対応表](../joint_questions.py)を使い、06はN=10だけを生成する。

## 重要事項

Experiment 03は変更していない。Experiment 03.1はplanning、dataset loading、loop、backend構築、安全確認、response logging、timing、token収集、output layoutを`experiments/common.py`へ委譲している。実験設定上の差分は上記question mappingのみである。
