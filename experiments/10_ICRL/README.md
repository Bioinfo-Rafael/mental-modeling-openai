# 10_ICRL — Pendulum Next Action

同じ学習済みagentの次の行動を予測する際に、Direct、PPOというprior、ICR-FQI型の推論手順を比較する。条件は [run.py](run.py) に集約。

- **72 calls** = 2 models (`terra`, `luna`) × 4 methods × 3 context sizes (`E=1,3,9`) × 3 repeats。
- `Pendulum-v1` / `next-action` / rewardあり / `H=20` / history start `t=0`。
- targetは常に `episode_9_seed3407.npz`。queryは `s20`、正解は `a20`（raw値 `1.9112794399261475`、bin `9`）。
- repeatは**同一promptの独立request**。異なるqueryではなく、過去の回答も次のrepeatへ渡さない。

## Contextとprompt差分

`E`はtargetを含むepisode数。`E=1`はtargetのみ、`E=3`は補助2本、`E=9`は補助8本を加える。
補助episodeは `discover_episodes()` のPendulum episodeからtargetを除き、pathの辞書順の先頭 `E-1` 本を選ぶ。action/reward/正解値による選別はしない。現在は `episode_0.npz`, `episode_1.npz`, …, `episode_7_seed1024.npz` の順で、contextは入れ子になる。

全episodeの `Step 0…19` にstate/action/rewardを提示。補助episodeには `State at step 20` のみを追加し、targetの `s20` は既存questionで提示する。**どのepisodeもa20/reward20は入力に含めない**。補助episodeは開始・終了ラベルで区切り、独立trajectoryであり境界を跨ぐtransitionを作らないと明記する。

| method | Directからの追加 |
| --- | --- |
| `direct` | なし。E=1は既存Joint system/user文字列と完全一致。E=3/9だけ補助episodeをtargetの前へ追加。 |
| `ppo_prior` | 指定されたPPO priorの英文1段落をuser先頭へ追加。FQIの指示は加えない。 |
| `icrfqi_k2` | 下記の推論手順をuser先頭へ追加、`K = 2`。PPO priorなし。 |
| `icrfqi_k5` | K2と同じ文字列で、`K = 2` の箇所だけ `K = 5`。 |

全methodで既存Pendulum system prompt、target history、question、出力形式を維持する。追加英文の全文は [prompts.py](prompts.py)。出力は既存のcompact reasoning、`predictions = [...]`、`>>Final action bins: [...]`。

ICR-FQIではepisodeごとに `(s_t,a_t,r_t,s_(t+1))`, `t=0…19` を復元し、観測actionを既存の10 bins（`[-2,2]`、ID `0…9`）に対応づける。

```text
Q_0(s,b) = 0; gamma = 0.99
for k = 1, ..., K:
    y_t^(k) = r_t + gamma * max_b Q_(k-1)(s_(t+1), b)
    ((s_t, action_bin(a_t)), y_t^(k)) からin-context regressionでQ_kを推定
b* = argmax_b Q_K(s_query,b)
```

未観測のstate/binも回帰・補間で推定し、不可能扱いしない。最後に選択bin内のcontinuous actionを観測actionとpolicy behaviorから推定する。中間Bellman targetやQ-table全体は表示させない。
[ICR-RL論文](https://proceedings.mlr.press/v306/schiff26a.html)を参考にした**LLMへの推論指示の比較実験**であり、TabPFN等による数値FQIの再現実装ではない。LLM内部で指示どおり計算されたかはこの出力形式だけでは検証しない。

## PPO provenance

**PPO由来であることは確認できなかった。`ppo_prior` は実験的に注入するPPO prior / assumptionである。**
repo、`upstream/LLM-Xavier` のコード・文書、[data/README.md](../../data/README.md)、datasetのファイル配置を確認した。むしろ同梱artifactに次のDDPG名がある（checkpointはdeserializeしていない）：

```text
data/llmx_data/offline_data/physics_data/runs/
  Pendulum-v1__ddpg_continuous_action__1__1709999421/ddpg_continuous_action.cleanrl_model
```

これはDDPGを示唆するが、target NPZとcheckpointの対応は確認できていない。dataset固定revisionは [configs/sources.json](../../configs/sources.json) の `dc2b798f72bc02f7285949ccfcdcb42e5ff326fd`。promptには依頼されたPPO英文をそのまま使用し、optimal policyとは記さない。

## 実行と出力

repo rootで既存 `.venv` をactivateした環境から実行する。

```bash
python experiments/10_ICRL/run.py --dry-run  # API client生成・送信なし
python experiments/10_ICRL/run.py            # 本番: 72 requests（API keyを非表示入力）
python experiments/10_ICRL/run.py --resume   # 未試行requestだけ再開
python -m pytest experiments/10_ICRL/tests experiments/09_ablation_reward/tests -q
```

09と同じくSDK `max_retries=0`, timeout 180秒、request前のjournal、排他lockを使用する。送信開始済みの失敗・成否不明requestはresumeでも再送しない。保存済みresponseの未採点分はローカルで回収する。`results/api` があれば通常実行を拒否し、resumeは保存した計画・入力hashとの一致を要求する。異なるdry-run manifestも上書きせず拒否するため、再計画時は既存の `results/dry_run` を別の場所へ退避する。

| 出力（`results/`以下） | 内容 |
| --- | --- |
| `dry_run/manifest.json`, `.csv` | 72件。JSONに全文prompt/request、target、E、補助path、各episodeのSHA256、正解（採点用metadataのみ）。 |
| `dry_run/prompts/<model>__<method>__E<E>__repeat<R>.txt` | system/user全文、72ファイル。 |
| `api/manifest.json`, `requests.jsonl`, `responses.jsonl` | 実行計画、送信前journal、raw response/例外/elapsed time。 |
| `api/records.jsonl`, `.csv`, `summary.json` | 既存Joint scorerの採点、予測値/正解/absolute error、予測bin/正解bin/status、tokens/elapsed、assistant text。JSONLにはraw responseも保存。 |
| `analysis/per_query.csv`, `conditions.csv`, `summary.json` | method/K/E/repeat別の記録と24条件の集計。`bin_parse_success`, `action_parse_success`, 両方成功の `parse_success` を明示。 |

bin accuracyの分母はbin parse成功数、MAEの分母はnumeric action parse成功数。parse成功数、全予定数に対するmatch率、未試行・未解決数も保存する。tokensとelapsedは保存済み採点recordについて合計・平均・有効件数を集計する。raw responseのない失敗は `responses.jsonl` で確認する。

代表prompt（同条件のmodel/repeat間でpromptは共通）：

| method | E=1 | E=9 |
| --- | --- | --- |
| Direct | [全文](results/dry_run/prompts/terra__direct__E1__repeat1.txt) | [全文](results/dry_run/prompts/terra__direct__E9__repeat1.txt) |
| PPO prior | [全文](results/dry_run/prompts/terra__ppo_prior__E1__repeat1.txt) | [全文](results/dry_run/prompts/terra__ppo_prior__E9__repeat1.txt) |
| ICR-FQI K2 | [全文](results/dry_run/prompts/terra__icrfqi_k2__E1__repeat1.txt) | [全文](results/dry_run/prompts/terra__icrfqi_k2__E9__repeat1.txt) |
| ICR-FQI K5 | [全文](results/dry_run/prompts/terra__icrfqi_k5__E1__repeat1.txt) | [全文](results/dry_run/prompts/terra__icrfqi_k5__E9__repeat1.txt) |

## 実装経路

`run.py → runner.run() → make_plan()`。`prompts.py`だけが追加指示と補助episode表示を所有する。

- `preprocessing/llmx_original.py`: discovery、upstreamに一致するtarget prompt生成。`llm_x.data.Episode`: load/SHA256/history表示。
- `experiments/joint_questions.py` → `upstream/LLM-Xavier/llm_x/feedback.py`: 既存Joint question。`llm_x.prompting`: 既存system/user。
- `experiments/common.py`: Terra/Luna model名、`reasoning_effort='medium'` のrequest、hash、JSON/CSV writer。
- `experiments/split_execution.py`: `execution_lock`。
- `09_ablation_reward/runner/execution.py`: API/logging/resumeを直接再利用。既存既定値を維持した `source_experiment` / `csv_fields` 引数だけを追加。
- `08_fewshot/runner/scoring.py` → `common_analysis/joint_data.py`: continuous actionとfinal binの既存parser/scorerをそのまま使用。新しいparserはない。




# 生成するのに用いたプロンプト
- chatGPTに生成してもらった
```text
/Users/cls-lab/Git/Matsuo/mental-modeling-openai
で、新しい実験 `experiments/10_ICRL` を実装してください。

まず既存実装を十分に読んでから実装してください。特に以下を優先して確認してください。

- experiments/09_ablation_reward
- experiments/08_fewshot
- experiments/06_new_models_n10
- experiments/05_new_models_single
- experiments/common.py
- experiments/split_execution.py
- experiments/joint_questions.py
- preprocessing/llmx_original.py
- upstream/LLM-Xavier/llm_x/feedback.py

今回の目的は、Pendulum-v1 の Next Action predictionについて、
通常のDirect predictionと、
PPOで学習されたpolicyであるというpriorを与える方法と、
ICR-RL論文のICR-FQIを模したreasoning procedureを明示する方法
を比較することです。

==================================================
最重要：既存実装・可読性
==================================================

今回もっとも重要なのは、既存experimentsの設計を踏襲することです。

run.py に Experiment/dataclass 相当の実験条件を簡潔に定義し、
runner側の run() → make_plan() で実際のquery/prompt/manifestを構築する、
という既存の構造を維持してください。

09, 08, 06, 05などに既に存在する
- episode discovery
- prompt生成
- OpenAI API request生成
- request/response logging
- scoring
- resume
- dry-run
- JSON/CSV writer
- lock
などを可能な限り再利用してください。

既存コードを大規模に書き直したり、
10_ICRLのためだけに汎用frameworkを新しく作ったりしないでください。

一方で、一つの巨大ファイルに全部詰め込むのも避けてください。

人間がreviewしやすいことを最優先にしてください。
directoryやfileを不必要に増やさず、
依存関係を何階層にも深くしないでください。

目安として、

experiments/10_ICRL/
    README.md
    run.py
    runner.py
    prompts.py
    tests/
    results/

程度の浅い構成を第一候補にしてください。

ただし既存コードをそのままimportして再利用できるなら、
新しいexecution/scoring moduleを複製して作らないでください。

既存moduleの変更が必要な場合も、
既存experimentの挙動を一切変えない小さい変更に限定してください。
不要なrefactorは禁止です。

==================================================
実験条件
==================================================

Experimentの条件はrun.pyだけを見れば分かるようにしてください。

固定条件：

task = Pendulum-v1
metric = next-action
history start t = 0
H = 20
target episode = episode_9_seed3407.npz
query state = target episode の s20
ground truth = target episode の a20
reward = 常にあり
models = terra, luna
repeats = 3

比較するmethod：

1. direct
2. ppo_prior
3. icrfqi_k2
4. icrfqi_k5

context episode count E：

1
3
9

したがって総API request数は

2 models
× 4 methods
× 3 E
× 3 repeats
= 72

です。

同じconditionのrepeat 3回は、
同じpromptを独立したAPI requestとして3回送るものです。
異なるtarget queryではありません。

==================================================
E=1 / 3 / 9 の定義
==================================================

target episodeは常に

episode_9_seed3407.npz

で固定してください。

EはLLMへ与えるepisode数です。

E=1:
target episodeのみ

E=3:
target episode + auxiliary episode 2本

E=9:
target episode + auxiliary episode 8本

auxiliary episodeは、
targetを除いたPendulum episodeをpath/nameでdeterministicにsortし、
先頭 E-1 本を使用してください。

したがって必ず

C(E=1) ⊂ C(E=3) ⊂ C(E=9)

になるようにしてください。

actionやrewardやground truthを見てauxiliary episodeを選択してはいけません。

manifestには、
target_episode
context_episode_count
auxiliary_episode_paths
各episodeのSHA256
を保存してください。

==================================================
各episodeからLLMへ与える範囲
==================================================

全episodeについて t=0 からの20 transitionsを使用します。

target episodeについては、既存のH=20 Next Action promptと同じく

step 0 ... step 19:
    state
    action
    reward

をhistoryとして提示し、

queryとして s20

を提示してください。

targetのa20/reward20は絶対に入力へ入れないでください。

auxiliary episodeについてはFQIで

(s19, a19, r19, s20)

まで構成できる必要があります。

したがって各auxiliary episodeについて、

step 0 ... step 19:
    state
    action
    reward

に加えて、

state at step 20

だけを提示してください。

auxiliary episodeのa20およびreward20は入力へ入れないでください。

各episodeが独立trajectoryであることをprompt上でも明確にしてください。
episode Aのstep19からepisode Bのstep0へtransitionを作ってはいけません。

==================================================
複数episodeのprompt構造
==================================================

E=1のDirectについては、
現在使っている
`next_action_prediction_continuous_joint`
のsystem/user promptを可能な限り完全にそのまま使ってください。

特にE=1/directについて、
既存promptから意味のない文章変更やreformatをしないでください。

E=3/9の場合のみ、
既存target promptの前に

"Additional reference episodes generated by the same trained agent"

という位置づけでauxiliary episodesを追加してください。

auxiliaryとtargetの境界は明示してください。

例えば概念的には、

[Additional reference episode 1]
Step 0 ...
...
Step 19 ...
State at step 20: ...

[Additional reference episode 2]
...

[Target episode]
<既存のtarget history + 既存のnext-action question>

とします。

target episodeの既存question/output formatは変更しないでください。

==================================================
method 1: direct
==================================================

現在のNext Action predictionそのままです。

追加のreasoning instructionは与えません。

E=1では現在の既存promptそのもの。

E=3/9ではadditional reference episodesを前置きしますが、
next actionの質問文、reasoning instruction、output format自体は
既存promptを維持してください。

==================================================
method 2: ppo_prior
==================================================

Directと同じprompt/inputを使用します。

違いは以下のPPO priorだけです。

以下の英文を、questionに入る前の自然な位置に追加してください。

"The trajectories shown below were generated by the same reinforcement-learning policy trained with Proximal Policy Optimization (PPO) to maximize expected cumulative reward. Use this fact as additional information when inferring the trained agent's action-selection behavior."

重要：
- optimal policyとは書かない
- FQIについては一切指示しない
- Directとの差分はこのPPO情報だけにする
- final output formatはDirectと完全に同じ

また、repo/upstream/data documentationを検索して、
このdatasetを生成したagentがPPOであることを確認してください。

確認できた場合はREADMEに根拠となるfile/pathを記載してください。

確認できなかった場合でも今回のrequested experiment自体は実装してください。
ただしREADME上では、
`ppo_prior` は「dataset provenanceとして確認済み」と勝手に断定せず、
実験的に注入するPPO prior / assumptionであることを明記してください。

prompt condition自体には上記英文を使用してください。

==================================================
method 3/4: ICR-FQI
==================================================

ICR-RL論文
"ICR-RL: Deep Reinforcement Learning via In-Context Regression"
のFitted Q Iteration型reasoningを、
LLMに明示的に実行させます。

Kだけが異なります。

icrfqi_k2:
K = 2

icrfqi_k5:
K = 5

Directの既存questionとoutput formatは維持したまま、
回答前のreasoning procedureとして以下を追加してください。

英語promptは以下の意味・式をそのまま満たすものにしてください。
文章上の微調整は可能ですが、algorithmの意味を変更しないでください。

---
Before answering the next-action question, treat the supplied trajectories as an offline reinforcement-learning dataset and perform the following Fitted Q-Iteration style reasoning internally.

Each episode is an independent trajectory. Never create a transition across episode boundaries.

1. Reconstruct the transitions
   (s_t, a_t, r_t, s_{t+1})
   from every supplied episode.

2. Use the 10 action bins already defined in the question as the candidate action set.
   Map each observed continuous action a_t to its corresponding action bin.
   Define Q_k(s, b) as the estimated value of taking action-bin b at state s.

3. Initialize
   Q_0(s, b) = 0
   and use gamma = 0.99.

4. For k = 1, ..., K, perform:
   
   y_t^(k) = r_t + gamma * max_b Q_(k-1)(s_(t+1), b)

   Then treat
   ((s_t, action_bin(a_t)), y_t^(k))
   as an in-context regression dataset.

   Infer Q_k(s, b) from these examples without updating model parameters.

   When an exact state-action-bin pair was not observed, estimate its value by regression/interpolation from the supplied examples rather than treating the unobserved action as impossible.

5. At the query state s_query, estimate
   Q_K(s_query, b)
   for all 10 action bins and select

   b* = argmax_b Q_K(s_query, b).

6. After selecting the action bin, estimate a real-valued action inside that bin using the observed continuous actions and policy behavior in the supplied trajectories.

Perform the iterative calculations internally. Do not print a large Q-table or all intermediate Bellman targets. In the visible answer, provide only the compact reasoning and the exact answer format required by the original next-action question.
---

Kの部分はconditionに応じて2または5を明示してください。

重要：
- K=2/5以外のprompt差分を作らない
- gamma=0.99固定
- action candidatesは既存の10 bins
- episode boundaryを跨がない
- Q-table全体を出力させない
- 元のfinal answer formatは変更しない
- raw continuous actionとaction binの両方を既存Joint parserで採点可能にする

ICR-FQI conditionにはPPO-priorの文章を追加しないでください。
PPO-priorとFQI reasoningの効果を分離したいためです。

==================================================
system prompt
==================================================

Pendulum-v1について現在使われているsystem prompt
(task description / observation / action / reward / dynamics etc.)
をそのまま使用してください。

今回独自のalgorithm instructionはuser prompt側へ入れてください。

system promptまでmethodごとに書き換えないでください。

==================================================
scoring
==================================================

Pendulum Next Action Jointの既存scoring/parserをそのまま再利用してください。

最低限保存・集計するもの：

- raw continuous action prediction
- ground-truth action
- action MAE
- predicted action bin
- ground-truth action bin
- bin accuracy / match
- parse success
- input tokens
- output tokens
- total tokens
- request elapsed time
- assistant_text
- raw response
- model
- method
- K
- E
- repeat_id

新しい独自parserを作らないでください。
08/09等で使っている既存Joint scoringを再利用してください。

==================================================
run.py
==================================================

run.pyは短くし、
実験条件を見れば72件の構成が分かるようにしてください。

概念的には以下程度です。

EXPERIMENT = Experiment(
    models=('terra', 'luna'),
    methods=('direct', 'ppo_prior', 'icrfqi_k2', 'icrfqi_k5'),
    context_episode_counts=(1, 3, 9),
    repeats=3,
    task='Pendulum-v1',
    H=20,
    start=0,
    target_episode='episode_9_seed3407.npz',
)

if __name__ == '__main__':
    raise SystemExit(runner.run(EXPERIMENT))

実際のfield名は既存styleに合わせてよいです。

==================================================
dry-run / execution
==================================================

既存09等と同様に、

python experiments/10_ICRL/run.py --dry-run

でAPI requestを一切行わず、
72 queryのmanifestとprompt全文を確認できるようにしてください。

各promptを人間が確認できる形で保存してください。

少なくとも以下の代表promptを簡単に確認できるようにしてください。

direct E=1
direct E=9
ppo_prior E=1
ppo_prior E=9
icrfqi_k2 E=1
icrfqi_k2 E=9
icrfqi_k5 E=1
icrfqi_k5 E=9

通常実行およびresumeについては09の挙動を優先して踏襲してください。

既存結果を黙って上書きしないでください。
SDK retry等も既存experimentの方針を踏襲してください。

==================================================
tests
==================================================

最低限以下をtestしてください。

- query総数が72
- 2 model × 4 method × 3 E × 3 repeatになっている
- 全queryがPendulum-v1 / next-action / H20 / t=0
- target episodeが全てepisode_9_seed3407.npz
- query indexが20
- ground truthがtarget a20
- E=1/3/9でcontext episode数が正しい
- E=1 ⊂ E=3 ⊂ E=9
- auxiliary episode選択がdeterministic
- auxiliary episodeにa20/reward20が含まれない
- target a20/reward20がpromptにleakしていない
- auxiliaryのs20は含まれている
- episode boundaryを跨ぐtransitionを作らない構造になっている
- direct E=1のbase promptが既存Next Action promptから不要に変更されていない
- ppo_priorとdirectの差分がPPO priorだけ
- icrfqi_k2とk5の差分がKだけ
- system promptは全methodで同一
- repeatごとにquery_idはunique
- API request model/argumentsは既存Terra/Luna experimentと一致
- dry-runでAPI client/requestが発生しない

==================================================
README
==================================================

READMEには簡潔に以下を書いてください。

- 目的
- 72 callsのfactorization
- target / H / t
- E=1/3/9の定義
- Direct / PPO-prior / ICR-FQI K2/K5の違い
- auxiliary episodeの選択規則
- ICR-FQIの式
- gamma=0.99
- actionを10 binsへ離散化してFQIすること
- continuous predictionは最後にbin内で出すこと
- PPO provenanceを確認できたかどうか
- dry-run / execution / resume command
- 出力ファイル
- 既存実装のどこを再利用しているか

長大な設計文書にはしないでください。
review時に実験仕様と実装経路を短時間で確認できるREADMEにしてください。

==================================================
今回やらないこと
==================================================

以下は実装しないでください。

- rewardなし条件
- H sweep
- t=100
- few-shot
- MountainCar
- Sol
- plottingの大量追加
- 新しい汎用実験framework
- 既存experimentの大規模refactor
- PPOとICR-FQIを同時に与えるcondition
- 新しい独自scorer

==================================================
作業手順
==================================================

1. まず既存05/06/08/09/commonの実装経路を確認する
2. どの既存moduleをそのまま再利用するか決める
3. 最小構成で10_ICRLを実装する
4. testsを実行する
5. dry-runだけ実行する
6. 72 promptsが予定通り生成されたことを確認する
7. representative promptを目視確認する
8. READMEを最終実装と一致させる

実APIは絶対に送信しないでください。
OpenAI APIを使う本番実験はユーザーが後で実行します。

最後に、
- 作成/変更したfile
- 既存から再利用した実装
- 72条件がどう展開されるか
- Direct / PPO-prior / ICR-FQIの実prompt差分
- test結果
- dry-run結果
- PPO provenance確認結果
を簡潔に報告してください。
```

## 解析コードを生成するためのプロンプト
```text
実行が終わったんで可視化とhtmlを作成してほしい。
# htmlについて
- htmlは09_ablation_rewardで作成したのと同じような形式で作成して。
# 可視化について
- 作図せずにまずは何をどうやって作図するのかの計画を立てて。09で作図されたものを見て真似てほしい。まずはそのコードを読んで考えて。
- 今回の振っている条件としてはプロンプトをdirect, PPO, ICRL(K=2,5)の4通りと、Episode数のE=1,3,9
- 2モデルで条件ごとにn=3
- 作図する評価指標は正解率・誤差MAE・推論レベルの３つなので以下全てこの３つの指標を作ってね
## (1) 
for i in {prompt, Episode数}
についてiの条件ごとに群を作ってそれ以外の条件はpoolして。まずはモデルを区別せずに棒グラフで比較して。次にモデルごとに比較して棒グラフを作成して。
モデルを区別してprompt群の比較でn=72/8が棒一つに対応かな
## (2) 
次に色付けしたmatrixを書いてほしい。行方向がPromptの条件で列方向がEpisodeの数。4*3のmatrixになるはず。で、(i,j)成分はその条件での値
モデルごとに分けずに作ったものと分けて作ったものを書いて。
モデルごとに分けない場合は72/12=6通りが1マスに対応し、分けた場合はn=3に対応するのでこれ以上分解できない。

まずはコードだけ作成して。実行しなくていい（まだ推論の評価が完成してない）
```