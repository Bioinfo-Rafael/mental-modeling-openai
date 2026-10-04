# 12: Few-shot demonstrations versus reference episodes

同じPendulum Next Action N10に、固定4件のQuestion / Answer / Labelを前置する方法と、同じagentの別episodeのtrajectoryを前置する方法を比較する。
**2 models × 2 context methods × N10 = 40 calls**。Terra / Luna、H=20、異なる10問でありrepeatではない。

評価対象は05 / 06 / 03_1のPendulum-v1 / next-action / H20の先頭10問と同一。
`episode_0.npz`のordinal・history_startは0..9、history_end_exclusive・query_indexは20..29。
`build_prompt_queries()`から生成し、保存済みmanifestとのepisode / SHA / window / system・target prompt hash一致をテストする。過去manifestは実行時には不要。
両条件のsystem、元のtarget user prompt、正解、API parametersは同一で、`pair_id`でmodelごとに対応づける。

## Context

`fewshot4`は[09の固定snapshot](../09_ablation_reward/data_prep/fewshot_p1_k4.json)の`prefix`をSHA-256検証後にそのまま連結する。
08由来のpattern1 / k4であり、再選択・Question / Answer / Labelの変更は行わない。`New Question:`もsnapshotに含まれる。
prefix SHA-256: `ca649dc7c14bbde933358baae255fd2333db73ba0989afcd3470c8ea6debc7ca`。

使用順の4 example IDと監査情報（元レコードは[08 selected_candidates.json](../08_fewshot/results/dry_run/selected_candidates.json)）：

| Example ID | metric | query_index | H | example_type | reasoning_score |
|---|---|---:|---:|---|---:|
| `28590891f37e7729bcb2296d0bcc0d82fd0f76001aa0db521e303c01113cfa37` | last-action | 10 | 5 | correct | 1 |
| `f3b4df245b5ab0fabf0a7fcfdc6a3d8f7fed36605361a7f70b3608e157379fae` | next-action | 8 | 5 | incorrect | 3 |
| `672e10aab266a24e559da7f1ea9bddf518a1bc0db730c0c997b239665bc48f59` | last-action | 10 | 5 | correct | 1 |
| `68d34bbc532c5290da08f6031c0a458e0c32933f7675fc21895c33e191e49868` | next-action | 13 | 5 | incorrect | 3 |

**4例も評価対象も同じepisode_0由来**。共通episode_pathは
`data/llmx_data/offline_data/physics_data/raw_transitions/Pendulum-v1/episodes/episode_0.npz`。
08のheld-out episode評価とは異なる。例の履歴stepは順に5..9、3..7、5..9、8..12で、予測対象はa10、a8、a10、a13。
Question / Answer / Labelの完全一致と、評価対象a20..a29の直接ラベルが例に入っていないことを検査する。
N10自体の履歴窓は重なり、後の問の通常履歴には前の問の正解actionが含まれる。

`episodes_E9`は10の`method=direct, E=9`方式。**E=9 = target 1 + auxiliary 8**。
Pendulumのpath順でepisode_0を除いた先頭8本を、action / rewardを読む前に選択する。同じepisodesディレクトリの次の8本：

1. `episode_1.npz`
2. `episode_2_seed42.npz`
3. `episode_3_seed157.npz`
4. `episode_4_seed123.npz`
5. `episode_5_seed407.npz`
6. `episode_6_seed512.npz`
7. `episode_7_seed1024.npz`
8. `episode_8_seed1123.npz`

各auxiliaryのwindow startはtargetと同じ。ordinal0はstep 0..19 + s20、ordinal9はstep 9..28 + s29。
20件のstate/action/rewardと最後のstateだけを提示し、query stepのaction/rewardは含めない。
各episodeは独立したblockで、boundaryを跨ぐtransitionを作らない。paths / SHA / windowsをmanifestに保存する。

全条件は既存のDirect Joint prompt（`next_action_prediction_continuous_joint`）。continuous actionと`>>Final action bins: [...]`を維持し、ICR-FQI / PPO priorは入れない。

## Reuse and commands

[run.py](run.py) → [runner.py](runner.py) → 既存helperを呼ぶ浅い構成。
08のJoint scorer、09のfrozen prefixと`execution.execute()`（journal / 180秒timeout / SDK retry=0 / resume）、10の`reference_prefix()` / `compose_user(..., 'direct')` / `summarize()`を直接利用する。
10の集計にはCSV列・条件列・説明文の省略可能引数だけを追加し、既定の動作を維持する。モデル設定・request・writer・hashは`common`を使用する。

リポジトリルートで実行（今回実行するのはtestsとdry-runのみ）：

```sh
.venv/bin/python -m pytest -q experiments/12_fewshots_episodes/tests
.venv/bin/python experiments/12_fewshots_episodes/run.py --dry-run
# 以下は実API送信を行う操作。今回の実装検証では実行しない。
.venv/bin/python experiments/12_fewshots_episodes/run.py
.venv/bin/python experiments/12_fewshots_episodes/run.py --resume
```

既存API結果がある場合は`--resume`必須。resumeは未試行queryのみ送信し、失敗・不確実な試行を再送しない。
保存済みdry-runと異なる計画は上書きを拒否する。同一計画のdry-run再生成は可能。

## Outputs

- `results/dry_run/manifest.json` / `.csv`: 40 query、paired identity、正解、request全文、snapshot provenance、入力SHA。
- `results/dry_run/prompts/*.txt`: system / user全文40件。代表例は[Terra fewshot N0](results/dry_run/prompts/terra__fewshot4__N0.txt)、[N9](results/dry_run/prompts/terra__fewshot4__N9.txt)、[Terra E9 N0](results/dry_run/prompts/terra__episodes_E9__N0.txt)、[N9](results/dry_run/prompts/terra__episodes_E9__N9.txt)。Lunaの本文は同じ。
- 実API実行時のみ`results/api/`: manifest、requests / responses / records JSONL、records CSV、実行summary。assistant text / raw response / bin判定 / continuous error / tokens / elapsedを保存。
- 実API実行時のみ`results/analysis/`: 4条件のsummary JSON / conditions CSV、parse successを含むper_query CSV。bin accuracyの分母はbin parse成功数、match rate all plannedは10問、MAEはcontinuous parse成功数。tokensはmean / sum、elapsedはmean / sumを保存。

テストはAPI client・通信を禁止し、実行・中断・resume・集計は偽clientのみで検証する。

今回の確認結果：新規26件＋既存回帰169件＝195件成功。dry-runで40 promptを保存し、保存後のmanifest / promptにも上記検査を実施済み。
[validation.json](results/dry_run/validation.json)に検査項目とmanifest SHAを記録。実API送信は0件。

## 棒グラフ

保存済み40回答から11と同じ4指標（正解率、bin MAE、action NRMSE、Pearson r）を描画する。
赤線は同一モデル・同じN10の通常Prompt（05→06/07）の成績で、11と同様に相関係数には赤線を付けない。
正解率とbin MAEのエラーバーはquery間の標本標準偏差（ddof=1）。NRMSE / Pearsonには付けない。

```sh
.venv/bin/python experiments/12_fewshots_episodes/plot_results.py
```

追加API送信なし。11の集計・描画コードを直接再利用する。
[4指標一覧PNG](results/analysis/plots/context_comparison.png) / [SVG](results/analysis/plots/context_comparison.svg)、
各指標のPNG / SVG、数値表`results/analysis/bar_metrics.csv`、赤線の値`bar_baselines.csv`、入力SHAを含む`bar_metrics.json`を生成する。
