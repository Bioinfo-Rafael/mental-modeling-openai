# 10_ICRL — Pendulum Next Action

同じ学習済みagentの次の行動を予測する際に、Direct、PPOというprior、ICR-FQI型の推論手順を比較する。条件は [run.py](run.py) に集約。

保存済み72回答の推論レベルは [採点済みCSV](reasoning_for_scoring_scored_astra_high.csv) に記録済み。[採点用CSV](reasoning_for_scoring.csv) のscoreは全行空欄のまま保持している。[Prompt.md](reasoning_score_analysis/Prompt.md) が採点基準で、[採点準備README](reasoning_score_analysis/README.md) に依頼手順を記載している。

推論評価72件を反映した [09形式のHTML](view_rawdata.html) と [比較図7枚・実行手順](analysis/README.md) を用意した。Prompt別／E別の棒グラフと4×3 matrixで、正解率・MAE・推論レベルの3指標を表示する。

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
